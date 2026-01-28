//! Audio extraction and streaming from media files for ASR applications.
//!
//! This crate provides functionality to extract and stream audio from video and audio files
//! using FFmpeg. It's designed for ASR (Automatic Speech Recognition) applications that need
//! to process audio in chunks or streams.
//!
//! # Features
//!
//! - **Streaming**: Stream audio in configurable chunks without loading entire files
//! - **Channel support**: Send audio segments via tokio or std channels
//! - **Timestamps**: Audio segments include precise timing information
//! - **Format support**: Supports all audio/video formats that FFmpeg supports
//! - **Metadata extraction**: Get file info without decoding audio
//!
//! # Examples
//!
//! ## Basic streaming
//!
//! ```no_run
//! use autosub_audio::{AudioStream, StreamConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut stream = AudioStream::open("video.mp4", None)?;
//!
//! for segment in stream {
//!     let segment = segment?;
//!     println!("Segment: {:.2}s - {:.2}s ({} samples)",
//!         segment.start_time,
//!         segment.end_time,
//!         segment.samples.len()
//!     );
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Channel-based streaming
//!
//! ```no_run
//! use autosub_audio::{AudioStream, AudioSegment, SendMode};
//! use tokio::sync::mpsc;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let stream = AudioStream::open("audio.mp3", None)?;
//! let (tx, mut rx) = mpsc::channel::<AudioSegment>(100);
//!
//! // Stream in background thread
//! std::thread::spawn(move || {
//!     stream.stream_to_channel(tx, SendMode::Blocking)
//! });
//!
//! // Process segments asynchronously
//! while let Some(segment) = rx.recv().await {
//!     // Process audio...
//! }
//! # Ok(())
//! # }
//! ```

mod error;
mod extract;
mod reader;
mod stream;
mod types;
mod utils;

pub use error::{AudioError, Result};
pub use extract::{extract_audio, ExtractedAudio};
pub use reader::AudioChunkReader;
pub use stream::AudioStream;
pub use types::{AudioSegment, FileInfo, SendMode, StreamConfig};
pub use utils::{
    cleanup_temp_files, is_audio_file, is_media_file, is_video_file, probe_file,
    supported_audio_formats, supported_video_formats,
};

/// Initialize FFmpeg library.
///
/// This is automatically called by other functions, but can be called explicitly
/// at program startup if desired. Safe to call multiple times.
///
/// # Errors
///
/// Returns an error if FFmpeg initialization fails.
pub fn init() -> Result<()> {
    extern crate ffmpeg_next as ffmpeg;
    ffmpeg::init().map_err(|e| AudioError::FfmpegInit(e.to_string()))?;
    ffmpeg::log::set_level(ffmpeg::log::Level::Error);
    Ok(())
}

/// Default sample rate for audio extraction (16kHz, Whisper compatible)
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Default chunk duration in seconds
pub const DEFAULT_CHUNK_DURATION_SECS: usize = 30;
