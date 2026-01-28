use std::mem;

use anyhow::{anyhow, bail, Result};

use crate::types::AudioClip;

/// VAD aggressiveness mode (our own enum for config)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadMode {
    /// Quality mode (least aggressive, fewer false positives)
    Quality,
    /// Low bitrate mode
    LowBitrate,
    /// Aggressive mode (recommended for most use cases)
    Aggressive,
    /// Very aggressive mode (most aggressive, may have false positives)
    VeryAggressive,
}

impl VadMode {
    /// Convert to WebRTC VAD mode
    pub fn to_webrtc_mode(&self) -> WebRtcVadMode {
        match self {
            VadMode::Quality => WebRtcVadMode::Quality,
            VadMode::LowBitrate => WebRtcVadMode::LowBitrate,
            VadMode::Aggressive => WebRtcVadMode::Aggressive,
            VadMode::VeryAggressive => WebRtcVadMode::VeryAggressive,
        }
    }
}

/// Configuration for Voice Activity Detection
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Sample rate in Hz (must be 8000, 16000, 32000, or 48000 for WebRTC VAD)
    pub sample_rate: u32,
    /// Frame duration in milliseconds (must be 10, 20, or 30 for WebRTC VAD)
    pub frame_duration_ms: usize,
    /// VAD aggressiveness mode
    pub mode: VadMode,
    /// Duration of silence in seconds before resetting context
    pub silence_reset_secs: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_duration_ms: 30,
            mode: VadMode::Aggressive,
            silence_reset_secs: 1.0,
        }
    }
}

/// Swappable VAD interface so implementations can change without affecting
/// the transcription pipeline.
pub trait VoiceActivityDetector {
    fn frame_length(&self) -> usize;
    fn sample_rate(&self) -> u32;
    fn is_speech_frame(&mut self, frame: &[i16]) -> Result<bool>;
    fn reset(&mut self);
}

/// Stateful helper that feeds audio through a VAD and emits speech-only
/// segments while tracking silence gaps.
pub struct VadSegmenter<D: VoiceActivityDetector> {
    detector: D,
    silence_reset_frames: usize,
    pending_samples: Vec<f32>,
    speech_buffer: Vec<f32>,
    in_speech: bool,
    current_segment_start: usize,
    current_segment_requires_reset: bool,
    pending_reset: bool,
    silence_frames: usize,
    processed_samples: usize,
}

impl<D: VoiceActivityDetector> VadSegmenter<D> {
    pub fn new(detector: D, frame_duration_ms: usize, silence_reset_secs: f32) -> Self {
        let frame_duration_secs = frame_duration_ms as f32 / 1000.0;
        let silence_reset_frames =
            (silence_reset_secs / frame_duration_secs).ceil().max(1.0) as usize;

        Self {
            detector,
            silence_reset_frames,
            pending_samples: Vec::new(),
            speech_buffer: Vec::new(),
            in_speech: false,
            current_segment_start: 0,
            current_segment_requires_reset: false,
            pending_reset: false,
            silence_frames: 0,
            processed_samples: 0,
        }
    }

    /// Feed more samples into the segmenter and return any completed speech
    /// segments detected within this batch.
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<Vec<AudioClip>> {
        self.pending_samples.extend_from_slice(samples);
        self.process_frames(false)
    }

    /// Finalize processing and flush any in-flight speech segment.
    pub fn flush(&mut self) -> Result<Vec<AudioClip>> {
        self.process_frames(true)
    }

    /// Reset the timestamp position counter to 0.
    /// Use this after flushing to start timestamp tracking from 0 again
    /// (e.g., for push-to-talk applications where each session should start from 0).
    pub fn reset_position(&mut self) {
        self.processed_samples = 0;
    }

    fn process_frames(&mut self, is_final_flush: bool) -> Result<Vec<AudioClip>> {
        let mut segments = Vec::new();
        let frame_len = self.detector.frame_length();

        while self.pending_samples.len() >= frame_len {
            let frame: Vec<f32> = self.pending_samples.drain(..frame_len).collect();
            let frame_i16 = to_pcm_i16(&frame);
            let is_speech = self.detector.is_speech_frame(&frame_i16)?;

            let frame_start_sample = self.processed_samples;
            let frame_end_sample = frame_start_sample + frame_len;
            self.processed_samples = frame_end_sample;

            if is_speech {
                if !self.in_speech {
                    self.in_speech = true;
                    self.current_segment_start = frame_start_sample;
                    self.current_segment_requires_reset = self.pending_reset;
                    self.pending_reset = false;
                }
                self.speech_buffer.extend_from_slice(&frame);
                self.silence_frames = 0;
            } else {
                self.silence_frames += 1;
                if self.in_speech {
                    let end_sample = frame_start_sample;
                    let samples = mem::take(&mut self.speech_buffer);
                    segments.push(AudioClip::new(
                        samples,
                        self.current_segment_start,
                        end_sample,
                        self.current_segment_requires_reset,
                    ));
                    self.in_speech = false;
                }
                if self.silence_frames == self.silence_reset_frames {
                    self.pending_reset = true;
                    self.detector.reset();
                }
            }
        }

        if is_final_flush {
            if self.in_speech {
                let end_sample = self.processed_samples + self.pending_samples.len();
                self.speech_buffer.append(&mut self.pending_samples);
                segments.push(AudioClip::new(
                    mem::take(&mut self.speech_buffer),
                    self.current_segment_start,
                    end_sample,
                    self.current_segment_requires_reset,
                ));
                self.in_speech = false;
            } else if !self.pending_samples.is_empty() {
                self.processed_samples += self.pending_samples.len();
                self.pending_samples.clear();
            }
        }

        Ok(segments)
    }
}

/// Concrete VAD based on the WebRTC implementation.
pub struct WebRtcVad {
    inner: webrtc_vad::Vad,
    frame_length: usize,
    sample_rate: u32,
    mode: WebRtcVadMode,
}

pub type WebRtcVadMode = webrtc_vad::VadMode;

impl WebRtcVad {
    pub fn new(sample_rate: u32, frame_duration_ms: usize, mode: WebRtcVadMode) -> Result<Self> {
        if !matches!(frame_duration_ms, 10 | 20 | 30) {
            bail!("WebRTC VAD only supports 10, 20, or 30 ms frames");
        }

        let frame_length = (sample_rate as usize * frame_duration_ms) / 1000;
        let mut vad = webrtc_vad::Vad::new();
        let rate = make_sample_rate_enum(sample_rate)?;

        vad.set_sample_rate(rate);
        vad.set_mode(duplicate_mode(&mode));

        Ok(Self {
            inner: vad,
            frame_length,
            sample_rate,
            mode,
        })
    }
}

impl VoiceActivityDetector for WebRtcVad {
    fn frame_length(&self) -> usize {
        self.frame_length
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn is_speech_frame(&mut self, frame: &[i16]) -> Result<bool> {
        if frame.len() != self.frame_length {
            bail!(
                "Unexpected frame length for VAD: got {}, expected {}",
                frame.len(),
                self.frame_length
            );
        }

        self.inner
            .is_voice_segment(frame)
            .map_err(|_| anyhow!("VAD failed to process frame"))
    }

    fn reset(&mut self) {
        self.inner.reset();
        // Reset clears mode/sample rate, so restore them to keep frame sizing valid.
        if let Ok(rate) = make_sample_rate_enum(self.sample_rate) {
            self.inner.set_sample_rate(rate);
        }
        self.inner.set_mode(duplicate_mode(&self.mode));
    }
}

fn to_pcm_i16(frame: &[f32]) -> Vec<i16> {
    frame
        .iter()
        .map(|sample| {
            let scaled = sample * i16::MAX as f32;
            scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect()
}

fn duplicate_mode(mode: &WebRtcVadMode) -> WebRtcVadMode {
    match mode {
        WebRtcVadMode::Quality => WebRtcVadMode::Quality,
        WebRtcVadMode::LowBitrate => WebRtcVadMode::LowBitrate,
        WebRtcVadMode::Aggressive => WebRtcVadMode::Aggressive,
        WebRtcVadMode::VeryAggressive => WebRtcVadMode::VeryAggressive,
    }
}

fn make_sample_rate_enum(sample_rate: u32) -> Result<webrtc_vad::SampleRate> {
    webrtc_vad::SampleRate::try_from(sample_rate as i32)
        .map_err(|e| anyhow!("Unsupported VAD sample rate {}: {}", sample_rate, e))
}
