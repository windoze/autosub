# autosub-audio Crate Guide

## Purpose

This crate handles audio extraction and streaming from media files (video/audio) using FFmpeg. It provides a clean, channel-based interface for feeding audio to ASR engines without loading entire files into memory.

## Key Architecture

### Core Components

1. **AudioStream** (`src/stream.rs`)
   - High-level streaming API: implements `Iterator<Item = Result<AudioSegment>>`
   - Opens any FFmpeg-supported media file
   - Extracts audio in configurable chunks (default: 30s at 16kHz)
   - Can stream to channels (tokio or std) with blocking or async send modes
   - Uses temporary files under the hood for reliable streaming

2. **AudioChunkReader** (`src/reader.rs`)
   - Low-level reader for extracted audio files
   - Reads PCM f32 samples in chunks
   - Tracks sample positions for precise timestamps
   - Used internally by AudioStream

3. **Audio Extraction** (`src/extract.rs`)
   - `extract_audio()` - Converts any media file to PCM WAV
   - Uses FFmpeg via `ffmpeg-next` Rust bindings
   - Outputs: 16kHz, mono, f32 PCM samples normalized to [-1, 1]
   - Handles all FFmpeg-supported formats (MP4, MKV, MP3, WAV, etc.)

4. **Utilities** (`src/utils.rs`)
   - `probe_file()` - Get file metadata without decoding
   - `is_audio_file()`, `is_video_file()`, `is_media_file()` - File type detection
   - `cleanup_temp_files()` - Clean up extracted audio files
   - Format listing: `supported_audio_formats()`, `supported_video_formats()`

### Data Flow

```
Media File (MP4, MKV, MP3, etc.)
    ↓
extract_audio() → Temp WAV file (16kHz mono f32)
    ↓
AudioChunkReader (reads chunks)
    ↓
AudioStream (iterator)
    ↓
AudioSegment (samples + timestamps)
    ↓
Consumer (ASR engine, channel, etc.)
```

### Important Files

- **`src/stream.rs`** - AudioStream iterator and channel streaming
- **`src/extract.rs`** - FFmpeg-based audio extraction
- **`src/reader.rs`** - AudioChunkReader for reading extracted audio
- **`src/types.rs`** - Public types (AudioSegment, FileInfo, StreamConfig, SendMode)
- **`src/utils.rs`** - File probing and utilities
- **`src/error.rs`** - Error types

### Key Types

#### AudioSegment
```rust
pub struct AudioSegment {
    pub samples: Vec<f32>,      // Audio samples (16kHz mono, normalized to [-1, 1])
    pub start_sample: usize,    // Start position in original stream
    pub end_sample: usize,      // End position in original stream
    pub sample_rate: u32,       // Always 16000 for Whisper compatibility
    pub start_time: f64,        // Start time in seconds
    pub end_time: f64,          // End time in seconds
}
```

#### StreamConfig
```rust
pub struct StreamConfig {
    pub sample_rate: u32,              // Default: 16000 (Whisper requirement)
    pub chunk_duration_secs: usize,    // Default: 30 (Whisper's max context)
}
```

#### SendMode
```rust
pub enum SendMode {
    Blocking,  // Use blocking_send (for std::thread)
    Async,     // Use async send (for tokio tasks)
}
```

## Common Tasks

### Basic Audio Streaming

```rust
use autosub_audio::AudioStream;

let mut stream = AudioStream::open("video.mp4", None)?;
for segment in stream {
    let segment = segment?;
    // Process segment.samples
}
```

### Channel-Based Streaming

```rust
use autosub_audio::{AudioStream, SendMode};
use tokio::sync::mpsc;

let stream = AudioStream::open("audio.mp3", None)?;
let (tx, mut rx) = mpsc::channel(100);

std::thread::spawn(move || {
    stream.stream_to_channel(tx, SendMode::Blocking)
});

while let Some(segment) = rx.recv().await {
    // Process segment
}
```

### File Probing

```rust
use autosub_audio::probe_file;

let info = probe_file("video.mp4")?;
println!("Duration: {:.2}s", info.duration_secs);
println!("Has video: {}", info.has_video);
```

### Cleanup

```rust
use autosub_audio::cleanup_temp_files;

// Clean up temp files older than 1 hour
cleanup_temp_files(3600)?;
```

## Testing

### Unit Tests
```bash
cargo test --package autosub-audio
```

### Integration Test with ASR
```bash
cargo test --package autosub-audio -- --ignored
```

The integration test:
- Extracts audio from `test_data/video.mp4`
- Streams to autosub-asr engine
- Verifies transcription accuracy (96%+ with Tiny model)
- Requires test video file (gitignored)

## Performance Considerations

1. **Chunk Size** - 30s is optimal for Whisper (max context window)
2. **Temp Files** - Audio is extracted to temp files in system temp directory
3. **Memory** - Only one chunk in memory at a time (streaming design)
4. **FFmpeg** - Uses system FFmpeg libraries (fast, native)

## Common Pitfalls

1. **FFmpeg Not Installed** - Requires FFmpeg dev libraries at build time
2. **Temp File Cleanup** - Call `cleanup_temp_files()` periodically in long-running apps
3. **Sample Rate Mismatch** - Always outputs 16kHz for Whisper; don't change this
4. **Blocking vs Async Send** - Use correct SendMode for your runtime (Blocking for threads, Async for tokio tasks)
5. **Iterator Exhaustion** - AudioStream is consumed after iteration; create new stream to re-read

## Error Handling

All functions return `Result<T, AudioError>`:

- `AudioError::FfmpegInit` - FFmpeg initialization failed
- `AudioError::Extraction` - Audio extraction failed (invalid file, unsupported format)
- `AudioError::Io` - File I/O error
- `AudioError::Probe` - Failed to probe file metadata
- `AudioError::ChannelClosed` - Channel closed during streaming

Always check error messages - they include FFmpeg output for debugging.

## Dependencies

- **ffmpeg-next** - Rust bindings to FFmpeg libraries
- **tokio** - Async runtime for channel support
- **hound** - WAV file reading
- **tempfile** - Temporary file management

## FFmpeg Integration

This crate requires FFmpeg development libraries:
- macOS: `brew install ffmpeg`
- Ubuntu: `apt install libavcodec-dev libavformat-dev ...`
- Windows: vcpkg or pre-built binaries

The crate links to:
- libavcodec (audio/video codecs)
- libavformat (container formats)
- libavutil (utilities)
- libswresample (audio resampling)

## Related Crates

- **autosub-asr** - Consumes AudioSegments for transcription
- **autosub** (root) - CLI that orchestrates both crates

## Testing Guidelines

- Keep test media files small (<10s) and gitignored
- Use `#[ignore]` for integration tests that require large models
- Test with various formats (MP4, MKV, MP3, WAV)
- Verify timestamps are accurate and continuous

## Recent Changes

- 2026-01-28: Integration test achieves 96%+ accuracy with Whisper Tiny
- Earlier: Migrated from file-based to streaming architecture
