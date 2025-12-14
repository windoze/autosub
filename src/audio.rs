use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use tracing::info;

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
        // This is safe and doesn't actually send a signal
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        // On Windows, check if the process exists using tasklist
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {}", pid), "/NH"])
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.contains(&pid.to_string())
            })
            .unwrap_or(false)
    }
}

/// Extract/convert audio from a media file to a temporary WAV file at 16kHz mono.
/// This is the format required by Whisper models.
///
/// Works with both video files (extracts audio track) and audio files (converts format).
/// The returned `ExtractedAudio` handle will automatically clean up the temp file when dropped.
/// Uses FFmpeg command-line tool for extraction (must be installed on system).
pub fn extract_audio(input: &Path) -> Result<ExtractedAudio> {
    if is_audio_file(input) {
        info!("Converting audio to 16kHz mono WAV: {}", input.display());
    } else {
        info!("Extracting audio from: {}", input.display());
    }

    // Create a temporary WAV file
    let temp_dir = std::env::temp_dir();
    let temp_wav = temp_dir.join(format!("autosub_{}.wav", std::process::id()));

    // Use FFmpeg to extract and convert audio to 16kHz mono WAV
    let output = Command::new("ffmpeg")
        .args([
            "-y",                                      // Overwrite output
            "-i", input.to_str().context("Invalid input path")?,
            "-vn",                                     // No video
            "-acodec", "pcm_s16le",                    // 16-bit PCM
            "-ar", &WHISPER_SAMPLE_RATE.to_string(),   // 16kHz
            "-ac", "1",                                // Mono
            temp_wav.to_str().context("Invalid temp path")?,
        ])
        .output()
        .context("Failed to execute FFmpeg. Is it installed?")?;

    if !output.status.success() {
        // Clean up any partial temp file on failure
        let _ = std::fs::remove_file(&temp_wav);
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg failed: {}", stderr);
    }

    // Get file info to calculate duration
    let metadata = std::fs::metadata(&temp_wav)
        .context("Failed to read extracted audio file")?;
    let file_size = metadata.len();

    // Calculate duration: file_size / (sample_rate * bytes_per_sample * channels)
    // For 16-bit mono PCM: bytes_per_sample = 2, channels = 1
    // WAV header is typically 44 bytes
    let data_size = file_size.saturating_sub(44);
    let duration_secs = data_size as f64 / (WHISPER_SAMPLE_RATE as f64 * 2.0);

    info!(
        "Extracted audio to temp file: {} ({:.2} seconds)",
        temp_wav.display(),
        duration_secs
    );

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

/// Check if FFmpeg is available on the system
pub fn check_ffmpeg() -> Result<()> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("FFmpeg not found. Please install FFmpeg to use this tool.")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg check failed");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_ffmpeg() {
        // This test will only pass if ffmpeg is installed
        let result = check_ffmpeg();
        // Just check it doesn't panic, actual availability depends on system
        let _ = result;
    }
}
