use std::path::Path;

use tracing::debug;

extern crate ffmpeg_next as ffmpeg;

use crate::error::{AudioError, Result};
use crate::types::{AudioSegment, FileInfo, SendMode, StreamConfig};

/// Streaming audio reader that decodes audio directly from media files.
/// Yields chunks as AudioSegment with embedded timestamps.
/// This avoids writing to a temp file and reduces memory usage.
pub struct AudioStream {
    ictx: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Audio,
    resampler: ffmpeg::software::resampling::Context,
    audio_stream_index: usize,
    time_base: ffmpeg::Rational,
    /// Duration of each chunk in samples (e.g., 30 seconds * 16000 Hz = 480,000)
    chunk_samples: usize,
    /// Sample rate (from config)
    sample_rate: u32,
    /// File info
    file_info: FileInfo,
    /// Buffer for samples that haven't been yielded yet
    sample_buffer: Vec<f32>,
    /// Whether we've sent EOF to the decoder
    eof_sent: bool,
    /// Whether we've flushed the resampler
    resampler_flushed: bool,
    /// Current position in microseconds (for progress tracking)
    current_position_us: i64,
    /// Total samples processed so far (for timestamp tracking)
    samples_processed: usize,
}

impl AudioStream {
    /// Open a media file for streaming audio decoding.
    /// Returns an iterator that yields chunks of audio as AudioSegment.
    pub fn open(input: impl AsRef<Path>, config: Option<StreamConfig>) -> Result<Self> {
        let input = input.as_ref();
        let config = config.unwrap_or_default();

        // Initialize ffmpeg (safe to call multiple times)
        ffmpeg::init().map_err(|e| AudioError::FfmpegInit(e.to_string()))?;

        // Suppress ffmpeg warnings
        ffmpeg::log::set_level(ffmpeg::log::Level::Error);

        // Open input file
        let ictx = ffmpeg::format::input(input)
            .map_err(|e| AudioError::Decode(format!("Failed to open input file: {}", e)))?;

        // Get duration for progress tracking (in microseconds)
        let total_duration_us = ictx.duration();
        let duration_secs = total_duration_us as f64 / 1_000_000.0;

        // Find the best audio stream
        let audio_stream_index = ictx
            .streams()
            .best(ffmpeg::media::Type::Audio)
            .ok_or(AudioError::NoAudioStream)?
            .index();

        let audio_stream = ictx.stream(audio_stream_index).unwrap();
        let time_base = audio_stream.time_base();
        let audio_params = audio_stream.parameters();

        // Check if there are any video streams
        let has_video = ictx.streams().best(ffmpeg::media::Type::Video).is_some();

        // Create decoder for the audio stream
        let decoder_context = ffmpeg::codec::context::Context::from_parameters(audio_params)
            .map_err(|e| AudioError::Decode(format!("Failed to create decoder context: {}", e)))?;
        let decoder = decoder_context
            .decoder()
            .audio()
            .map_err(|e| AudioError::Decode(format!("Failed to create audio decoder: {}", e)))?;

        // Get source channel count from decoder
        let source_channels = decoder.channels();

        debug!(
            "Input audio: {} Hz, {} channels",
            decoder.rate(),
            decoder.channels()
        );

        // Set up resampler to convert to target sample rate, mono, f32
        let resampler = ffmpeg::software::resampling::context::Context::get(
            decoder.format(),
            decoder.channel_layout(),
            decoder.rate(),
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Packed),
            ffmpeg::ChannelLayout::MONO,
            config.sample_rate,
        )
        .map_err(|e| AudioError::Resample(format!("Failed to create audio resampler: {}", e)))?;

        let chunk_samples = config.chunk_duration_secs * config.sample_rate as usize;

        // Calculate total samples (approximate based on duration)
        let total_samples = (duration_secs * config.sample_rate as f64) as usize;

        let file_info = FileInfo {
            duration_secs,
            sample_rate: config.sample_rate,
            channels: source_channels,
            total_samples,
            has_video,
        };

        Ok(Self {
            ictx,
            decoder,
            resampler,
            audio_stream_index,
            time_base,
            chunk_samples,
            sample_rate: config.sample_rate,
            file_info,
            sample_buffer: Vec::new(),
            eof_sent: false,
            resampler_flushed: false,
            current_position_us: 0,
            samples_processed: 0,
        })
    }

    /// Get file information
    pub fn file_info(&self) -> &FileInfo {
        &self.file_info
    }

    /// Get the total duration in seconds
    pub fn duration_secs(&self) -> f64 {
        self.file_info.duration_secs
    }

    /// Get the current position in microseconds (for progress tracking)
    pub fn position_us(&self) -> i64 {
        self.current_position_us
    }

    /// Decode more samples into the buffer
    fn decode_more(&mut self) -> Result<bool> {
        // Try to get more packets
        while let Some((stream, packet)) = self.ictx.packets().next() {
            if stream.index() == self.audio_stream_index {
                // Update progress based on packet timestamp
                if let Some(pts) = packet.pts() {
                    let time_us = pts * 1_000_000 * self.time_base.numerator() as i64
                        / self.time_base.denominator() as i64;
                    if time_us > 0 {
                        self.current_position_us = time_us;
                    }
                }

                self.decoder.send_packet(&packet).ok();

                let mut decoded_frame = ffmpeg::frame::Audio::empty();
                while self.decoder.receive_frame(&mut decoded_frame).is_ok() {
                    // Resample the frame
                    let mut resampled_frame = ffmpeg::frame::Audio::empty();
                    self.resampler
                        .run(&decoded_frame, &mut resampled_frame)
                        .map_err(|e| {
                            AudioError::Resample(format!("Failed to resample audio frame: {}", e))
                        })?;

                    // Extract samples from the resampled frame
                    if resampled_frame.samples() > 0 {
                        let data = resampled_frame.data(0);
                        let samples: &[f32] = bytemuck::cast_slice(data);
                        self.sample_buffer
                            .extend_from_slice(&samples[..resampled_frame.samples()]);
                    }
                }

                // Return if we have enough samples for a chunk
                if self.sample_buffer.len() >= self.chunk_samples {
                    return Ok(true);
                }
            }
        }

        // No more packets, flush decoder
        if !self.eof_sent {
            self.decoder.send_eof().ok();
            self.eof_sent = true;

            let mut decoded_frame = ffmpeg::frame::Audio::empty();
            while self.decoder.receive_frame(&mut decoded_frame).is_ok() {
                let mut resampled_frame = ffmpeg::frame::Audio::empty();
                if self
                    .resampler
                    .run(&decoded_frame, &mut resampled_frame)
                    .is_ok()
                    && resampled_frame.samples() > 0
                {
                    let data = resampled_frame.data(0);
                    let samples: &[f32] = bytemuck::cast_slice(data);
                    self.sample_buffer
                        .extend_from_slice(&samples[..resampled_frame.samples()]);
                }
            }
        }

        // Flush resampler
        if !self.resampler_flushed {
            self.resampler_flushed = true;
            loop {
                let mut resampled_frame = ffmpeg::frame::Audio::empty();
                match self.resampler.flush(&mut resampled_frame) {
                    Ok(_) if resampled_frame.samples() > 0 => {
                        let data = resampled_frame.data(0);
                        let samples: &[f32] = bytemuck::cast_slice(data);
                        self.sample_buffer
                            .extend_from_slice(&samples[..resampled_frame.samples()]);
                    }
                    _ => break,
                }
            }
        }

        // Return true if we have any samples left
        Ok(!self.sample_buffer.is_empty())
    }

    /// Get the next chunk of audio samples as AudioSegment.
    /// Returns None when all audio has been read.
    pub fn next_chunk(&mut self) -> Result<Option<AudioSegment>> {
        // Decode more if we don't have enough samples
        if self.sample_buffer.len() < self.chunk_samples {
            self.decode_more()?;
        }

        if self.sample_buffer.is_empty() {
            return Ok(None);
        }

        // Take up to chunk_samples from the buffer
        let take_count = self.sample_buffer.len().min(self.chunk_samples);
        let samples: Vec<f32> = self.sample_buffer.drain(..take_count).collect();

        let start_sample = self.samples_processed;
        let end_sample = self.samples_processed + samples.len();
        self.samples_processed = end_sample;

        let segment = AudioSegment::new(samples, start_sample, end_sample, self.sample_rate);

        Ok(Some(segment))
    }

    /// Stream audio segments to a tokio mpsc channel
    ///
    /// # Arguments
    /// * `tx` - Tokio mpsc sender channel
    /// * `mode` - Send mode (blocking/non-blocking/drop-oldest)
    ///
    /// # Returns
    /// Number of segments sent
    pub fn stream_to_channel(
        mut self,
        tx: tokio::sync::mpsc::Sender<AudioSegment>,
        mode: SendMode,
    ) -> Result<usize> {
        let mut count = 0;

        while let Some(segment) = self.next_chunk()? {
            match mode {
                SendMode::Blocking => {
                    tx.blocking_send(segment)
                        .map_err(|e| AudioError::ChannelSend(e.to_string()))?;
                }
                SendMode::NonBlocking => {
                    tx.try_send(segment)
                        .map_err(|e| AudioError::ChannelSend(e.to_string()))?;
                }
                SendMode::DropOldest => {
                    // Try to send, if fails, try to make room by dropping oldest
                    if tx.try_send(segment.clone()).is_err() {
                        // Channel is full, we can't actually drop oldest from mpsc
                        // So we just skip this segment (drop newest instead)
                        debug!("Channel full, dropping segment at {}s", segment.start_time);
                        continue;
                    }
                }
            }
            count += 1;
        }

        Ok(count)
    }

    /// Stream audio segments to a std mpsc channel
    ///
    /// Note: std::sync::mpsc channels are unbounded by default, so SendMode
    /// is less relevant here. All modes will use blocking send.
    pub fn stream_to_std_channel(
        mut self,
        tx: std::sync::mpsc::Sender<AudioSegment>,
        _mode: SendMode,
    ) -> Result<usize> {
        let mut count = 0;

        while let Some(segment) = self.next_chunk()? {
            // std::sync::mpsc::Sender doesn't have try_send, only send (blocking)
            tx.send(segment)
                .map_err(|e| AudioError::ChannelSend(e.to_string()))?;
            count += 1;
        }

        Ok(count)
    }
}

impl Iterator for AudioStream {
    type Item = Result<AudioSegment>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(segment)) => Some(Ok(segment)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
