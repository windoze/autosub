use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use tracing::info;

extern crate ffmpeg_next as ffmpeg;

pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Known audio file extensions
const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "wma", "aiff", "aif",
];

/// Known video file extensions
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v", "mpeg", "mpg", "3gp",
];

/// Check if the file is an audio file based on extension
pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ext| AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Check if the file is a video file based on extension
pub fn is_video_file(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|ext| VIDEO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Check if the file is a supported media file (audio or video)
pub fn is_media_file(path: &Path) -> bool {
    is_audio_file(path) || is_video_file(path)
}

/// A handle to an extracted audio file in a temp directory.
/// The file is automatically cleaned up when this struct is dropped.
pub struct ExtractedAudio {
    path: PathBuf,
    duration_secs: f64,
}

impl ExtractedAudio {
    /// Get the path to the extracted WAV file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the duration of the audio in seconds
    pub fn duration_secs(&self) -> f64 {
        self.duration_secs
    }
}

impl Drop for ExtractedAudio {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Clean up any orphaned temp files from previous runs.
/// This is called at startup to remove any temp files that may have been left behind
/// if the program was killed unexpectedly.
pub fn cleanup_orphaned_temp_files() {
    let temp_dir = std::env::temp_dir();
    let current_pid = std::process::id();

    if let Ok(entries) = std::fs::read_dir(&temp_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Clean up autosub temp files from other processes
                if name.starts_with("autosub_") && name.ends_with(".wav") {
                    // Extract PID from filename
                    if let Some(pid_str) = name.strip_prefix("autosub_").and_then(|s| s.strip_suffix(".wav")) {
                        if let Ok(pid) = pid_str.parse::<u32>() {
                            // Don't delete our own temp file
                            if pid == current_pid {
                                continue;
                            }

                            // Check if the process is still running by looking at /proc on Linux
                            // or using ps on macOS/Unix
                            let process_exists = is_process_running(pid);

                            if !process_exists {
                                let _ = std::fs::remove_file(&path);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Check if a process with the given PID is still running
fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        // Use kill with signal 0 to check if process exists
        // This doesn't actually send a signal, just checks existence
        unsafe { libc::kill(pid as i32, 0) == 0 }
    }
    #[cfg(not(unix))]
    {
        // On Windows, use OpenProcess to check if process exists
        const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
        extern "system" {
            fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut std::ffi::c_void;
            fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
        }
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if handle.is_null() {
                false
            } else {
                CloseHandle(handle);
                true
            }
        }
    }
}

/// Extract/convert audio from a media file to a temporary WAV file at 16kHz mono.
/// This is the format required by Whisper models.
///
/// Works with both video files (extracts audio track) and audio files (converts format).
/// The returned `ExtractedAudio` handle will automatically clean up the temp file when dropped.
/// Uses ffmpeg-next crate for extraction (links against system FFmpeg libraries).
/// If a progress bar is provided, it will be updated during extraction.
pub fn extract_audio(input: &Path, progress: Option<&ProgressBar>) -> Result<ExtractedAudio> {
    // Create a temporary WAV file
    let temp_dir = std::env::temp_dir();
    let temp_wav = temp_dir.join(format!("autosub_{}.wav", std::process::id()));

    // Initialize ffmpeg (safe to call multiple times)
    ffmpeg::init().context("Failed to initialize FFmpeg")?;

    // Open input file
    let mut ictx = ffmpeg::format::input(input)
        .context("Failed to open input file with FFmpeg")?;

    // Get duration for progress tracking (in microseconds)
    let duration_us = ictx.duration();
    if let Some(pb) = progress {
        if duration_us > 0 {
            pb.set_length(duration_us as u64);
        }
    }

    // Find the best audio stream
    let audio_stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .context("No audio stream found in input file")?
        .index();

    let audio_stream = ictx.stream(audio_stream_index).unwrap();
    let time_base = audio_stream.time_base();
    let audio_params = audio_stream.parameters();

    // Create decoder for the audio stream
    let decoder_context = ffmpeg::codec::context::Context::from_parameters(audio_params)
        .context("Failed to create decoder context")?;
    let mut decoder = decoder_context
        .decoder()
        .audio()
        .context("Failed to create audio decoder")?;

    // Set up resampler to convert to 16kHz mono s16
    let mut resampler = ffmpeg::software::resampling::context::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        ffmpeg::format::Sample::I16(ffmpeg::format::sample::Type::Packed),
        ffmpeg::ChannelLayout::MONO,
        WHISPER_SAMPLE_RATE,
    )
    .context("Failed to create audio resampler")?;

    // Collect all audio samples
    let mut all_samples: Vec<i16> = Vec::new();

    // Process packets
    for (stream, packet) in ictx.packets() {
        if stream.index() == audio_stream_index {
            // Update progress based on packet timestamp
            if let (Some(pb), Some(pts)) = (progress, packet.pts()) {
                // Convert pts to microseconds
                let time_us = pts * 1_000_000 * time_base.numerator() as i64 / time_base.denominator() as i64;
                if time_us > 0 {
                    pb.set_position(time_us as u64);
                }
            }

            decoder.send_packet(&packet).ok();

            let mut decoded_frame = ffmpeg::frame::Audio::empty();
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                // Resample the frame
                let mut resampled_frame = ffmpeg::frame::Audio::empty();
                resampler
                    .run(&decoded_frame, &mut resampled_frame)
                    .context("Failed to resample audio frame")?;

                // Extract samples from the resampled frame
                if resampled_frame.samples() > 0 {
                    let data = resampled_frame.data(0);
                    let samples: &[i16] = bytemuck::cast_slice(data);
                    all_samples.extend_from_slice(&samples[..resampled_frame.samples()]);
                }
            }
        }
    }

    // Flush the decoder
    decoder.send_eof().ok();
    let mut decoded_frame = ffmpeg::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let mut resampled_frame = ffmpeg::frame::Audio::empty();
        if resampler.run(&decoded_frame, &mut resampled_frame).is_ok() && resampled_frame.samples() > 0 {
            let data = resampled_frame.data(0);
            let samples: &[i16] = bytemuck::cast_slice(data);
            all_samples.extend_from_slice(&samples[..resampled_frame.samples()]);
        }
    }

    // Flush the resampler (get any remaining samples)
    loop {
        let mut resampled_frame = ffmpeg::frame::Audio::empty();
        match resampler.flush(&mut resampled_frame) {
            Ok(_) if resampled_frame.samples() > 0 => {
                let data = resampled_frame.data(0);
                let samples: &[i16] = bytemuck::cast_slice(data);
                all_samples.extend_from_slice(&samples[..resampled_frame.samples()]);
            }
            _ => break,
        }
    }

    // Write to WAV file using hound
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: WHISPER_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&temp_wav, spec)
        .context("Failed to create output WAV file")?;

    for sample in &all_samples {
        writer.write_sample(*sample).context("Failed to write audio sample")?;
    }

    writer.finalize().context("Failed to finalize WAV file")?;

    let duration_secs = all_samples.len() as f64 / WHISPER_SAMPLE_RATE as f64;

    Ok(ExtractedAudio {
        path: temp_wav,
        duration_secs,
    })
}

/// Audio chunk reader for streaming transcription.
/// Reads audio in fixed-size chunks without loading the entire file into memory.
pub struct AudioChunkReader {
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
    chunk_samples: usize,
    total_samples: u32,
    samples_read: u32,
}

impl AudioChunkReader {
    /// Open a WAV file for streaming chunk reading.
    ///
    /// # Arguments
    /// * `path` - Path to the WAV file
    /// * `chunk_duration_secs` - Duration of each chunk in seconds
    pub fn open(path: &Path, chunk_duration_secs: usize) -> Result<Self> {
        let reader = hound::WavReader::open(path)
            .context("Failed to open WAV file")?;

        let spec = reader.spec();
        info!(
            "WAV file: {} Hz, {} channels, {} bits",
            spec.sample_rate, spec.channels, spec.bits_per_sample
        );

        let chunk_samples = chunk_duration_secs * spec.sample_rate as usize;
        let total_samples = reader.len();

        Ok(Self {
            reader,
            chunk_samples,
            total_samples,
            samples_read: 0,
        })
    }

    /// Get the total number of samples in the file
    pub fn total_samples(&self) -> u32 {
        self.total_samples
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.reader.spec().sample_rate
    }

    /// Get the total duration in seconds
    pub fn duration_secs(&self) -> f64 {
        self.total_samples as f64 / self.reader.spec().sample_rate as f64
    }

    /// Get the number of chunks that will be returned
    pub fn num_chunks(&self) -> usize {
        (self.total_samples as usize).div_ceil(self.chunk_samples)
    }

    /// Read the next chunk of audio samples.
    /// Returns None when all samples have been read.
    /// Each chunk contains f32 samples normalized to [-1, 1].
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f32>>> {
        if self.samples_read >= self.total_samples {
            return Ok(None);
        }

        let spec = self.reader.spec();
        let max_value = (1i32 << (spec.bits_per_sample - 1)) as f32;
        let remaining = (self.total_samples - self.samples_read) as usize;
        let to_read = remaining.min(self.chunk_samples);

        let mut samples = Vec::with_capacity(to_read);

        match spec.sample_format {
            hound::SampleFormat::Int => {
                for _ in 0..to_read {
                    if let Some(sample) = self.reader.samples::<i32>().next() {
                        let value = sample.context("Failed to read sample")?;
                        samples.push(value as f32 / max_value);
                        self.samples_read += 1;
                    } else {
                        break;
                    }
                }
            }
            hound::SampleFormat::Float => {
                for _ in 0..to_read {
                    if let Some(sample) = self.reader.samples::<f32>().next() {
                        let value = sample.context("Failed to read sample")?;
                        samples.push(value);
                        self.samples_read += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        if samples.is_empty() {
            Ok(None)
        } else {
            Ok(Some(samples))
        }
    }
}

/// Iterator adapter for AudioChunkReader
impl Iterator for AudioChunkReader {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffmpeg_init() {
        // Test that ffmpeg-next initializes successfully
        let result = ffmpeg::init();
        assert!(result.is_ok(), "FFmpeg should initialize successfully");
    }
}
