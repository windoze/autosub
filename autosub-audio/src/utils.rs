use std::ffi::OsStr;
use std::path::Path;

use tracing::debug;

extern crate ffmpeg_next as ffmpeg;

use crate::error::{AudioError, Result};
use crate::types::FileInfo;

/// Known audio file extensions
const AUDIO_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "flac", "m4a", "aac", "ogg", "opus", "wma", "aiff", "aif",
];

/// Known video file extensions
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v", "mpeg", "mpg", "3gp",
];

/// Check if the file is an audio file based on extension
pub fn is_audio_file(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(OsStr::to_str)
        .map(|ext| AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Check if the file is a video file based on extension
pub fn is_video_file(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(OsStr::to_str)
        .map(|ext| VIDEO_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Check if the file is a supported media file (audio or video)
pub fn is_media_file(path: impl AsRef<Path>) -> bool {
    let path = path.as_ref();
    is_audio_file(path) || is_video_file(path)
}

/// Get list of supported audio formats
pub fn supported_audio_formats() -> &'static [&'static str] {
    AUDIO_EXTENSIONS
}

/// Get list of supported video formats
pub fn supported_video_formats() -> &'static [&'static str] {
    VIDEO_EXTENSIONS
}

/// Clean up any orphaned temp files from previous runs.
/// This is called at startup to remove any temp files that may have been left behind
/// if the program was killed unexpectedly.
pub fn cleanup_temp_files() {
    let temp_dir = std::env::temp_dir();
    let current_pid = std::process::id();

    if let Ok(entries) = std::fs::read_dir(&temp_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Clean up autosub_audio temp files from other processes
                if name.starts_with("autosub_audio_") && name.ends_with(".wav") {
                    // Extract PID from filename
                    if let Some(pid_str) = name
                        .strip_prefix("autosub_audio_")
                        .and_then(|s| s.strip_suffix(".wav"))
                    {
                        if let Ok(pid) = pid_str.parse::<u32>() {
                            // Don't delete our own temp file
                            if pid == current_pid {
                                continue;
                            }

                            // Check if the process is still running
                            let process_exists = is_process_running(pid);

                            if !process_exists {
                                debug!("Cleaning up orphaned temp file: {}", name);
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

/// Get file info without opening a full stream.
/// This probes the media file to extract metadata.
pub fn probe_file(path: impl AsRef<Path>) -> Result<FileInfo> {
    let path = path.as_ref();

    // Initialize ffmpeg (safe to call multiple times)
    ffmpeg::init().map_err(|e| AudioError::FfmpegInit(e.to_string()))?;

    // Suppress ffmpeg warnings
    ffmpeg::log::set_level(ffmpeg::log::Level::Error);

    // Open input file
    let ictx = ffmpeg::format::input(path)
        .map_err(|e| AudioError::Decode(format!("Failed to open input file: {}", e)))?;

    // Get duration
    let duration_us = ictx.duration();
    let duration_secs = if duration_us > 0 {
        duration_us as f64 / 1_000_000.0
    } else {
        0.0
    };

    // Find the best audio stream
    let audio_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or(AudioError::NoAudioStream)?;

    let audio_params = audio_stream.parameters();

    // Create decoder to get audio properties
    let decoder_context = ffmpeg::codec::context::Context::from_parameters(audio_params)
        .map_err(|e| AudioError::Decode(format!("Failed to create decoder context: {}", e)))?;
    let decoder = decoder_context
        .decoder()
        .audio()
        .map_err(|e| AudioError::Decode(format!("Failed to create audio decoder: {}", e)))?;

    // Get audio properties from decoder
    let sample_rate = decoder.rate();
    let channels = decoder.channels();

    // Check if there are video streams
    let has_video = ictx.streams().best(ffmpeg::media::Type::Video).is_some();

    // Calculate total samples (approximate)
    let total_samples = (duration_secs * sample_rate as f64) as usize;

    Ok(FileInfo {
        duration_secs,
        sample_rate,
        channels,
        total_samples,
        has_video,
    })
}

// Platform-specific dependency for libc
#[cfg(unix)]
extern crate libc;
