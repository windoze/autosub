# autosub CLI Crate Guide

## Purpose

This is the main CLI application that orchestrates audio extraction, transcription, and optional translation to produce SRT subtitle files. It ties together `autosub-audio` (FFmpeg extraction) and `autosub-asr` (Whisper transcription) with LLM-powered translation.

## Key Architecture

### Core Flow

1. **Input**: Media file (MP4, MKV, MP3, etc.) or existing SRT file
2. **Audio Extraction**: `autosub-audio` extracts 16kHz mono audio
3. **Transcription**: `autosub-asr` transcribes using Whisper + VAD
4. **SRT Writing**: Results streamed to `.srt` file in real-time
5. **Translation** (optional): LLM translates subtitles to target language

### Main Components

1. **CLI Entrypoint** (`src/main.rs`)
   - Parses command-line arguments via `clap`
   - Initializes FFmpeg and logging
   - Orchestrates the full pipeline: extract → transcribe → translate
   - Handles errors and exit codes

2. **Configuration** (`src/config.rs`)
   - `Config` struct with clap derive
   - Command-line options: model size, language, VAD settings, translation
   - Environment variable support (e.g., `AUTOSUB_LLM_API_KEY`)
   - Output path generation (`.srt`, `.zh-CN.srt`, etc.)

3. **Whisper CLI** (`src/whisper_cli.rs`)
   - `transcribe_to_file()` - Main transcription pipeline
   - Coordinates AudioStream → ASR engine → SRT writer
   - ASR engine runs on main thread (critical for Metal/GPU backends)
   - SRT writer runs in separate thread for concurrent I/O
   - Real-time progress bar updates via callback
   - Returns complete Subtitle for optional translation

4. **SRT Handling** (`src/srt.rs`)
   - `Subtitle` - In-memory representation of SRT file
   - `SrtWriter` - Streams subtitle entries to file as they arrive
   - `SrtEntry` - Individual subtitle with timing and text
   - Parsing, formatting, merging consecutive entries

5. **Translation** (`src/translate.rs`)
   - LLM-powered translation (OpenAI, Anthropic, Google, Ollama, DeepSeek)
   - Batch translation (configurable batch size)
   - Streaming output to translated SRT file
   - Context preservation across batches for consistency

### Data Flow

```
Media File (video.mp4)
    ↓
AudioStream (autosub-audio)
    ↓
Channel (AsrInput)
    ↓
AsrEngine (autosub-asr + VAD)
    ↓
Channel (TranscriptionResult)
    ↓
SrtWriter → video.srt
    ↓ (optional)
translate_subtitles_to_file()
    ↓
LLM (OpenAI/Anthropic/etc.)
    ↓
video.zh-CN.srt
```

### Threading Architecture

The application uses a carefully designed threading model to handle Metal/GPU compatibility:

1. **Main Thread** (in `tokio::task::spawn_blocking`)
   - Creates Candle device and loads Whisper model
   - Runs ASR engine synchronously with `run_blocking()`
   - Critical: Model stays on same thread to avoid Metal threading issues

2. **Audio Extraction Thread** (`std::thread`)
   - Extracts audio from media file via FFmpeg
   - Sends audio samples to ASR engine via channel
   - Runs concurrently with transcription

3. **SRT Writer Thread** (`std::thread`)
   - Receives transcription results from ASR engine
   - Writes SRT entries to disk in real-time
   - Accumulates in-memory Subtitle for translation

**Why This Design?**
- Metal (and some other GPU backends) can hang when models are moved between threads
- Solution: Create device + model + engine all in same thread, keep execution there
- Trade-off: ASR is synchronous but still performant (GPU-accelerated)
- Benefit: Audio extraction and SRT writing happen concurrently for maximum throughput

### Important Files

- **`src/main.rs`** - CLI entrypoint, orchestration
- **`src/config.rs`** - Command-line arguments and configuration
- **`src/whisper_cli.rs`** - Transcription pipeline
- **`src/srt.rs`** - SRT file format handling
- **`src/translate.rs`** - LLM-powered translation
- **`src/lib.rs`** - Re-exports for library usage

## Command-Line Usage

### Basic Transcription
```bash
# Transcribe video to SRT
autosub video.mp4

# Specify language
autosub video.mp4 --language en

# Use larger model for better accuracy
autosub video.mp4 --model medium

# Enable verbose logging
autosub video.mp4 -v
```

### VAD Options
```bash
# Enable VAD (recommended for noisy audio)
autosub video.mp4 --enable-vad

# Adjust VAD silence threshold (seconds before resetting context)
autosub video.mp4 --enable-vad --vad-reset-secs 2.0
```

### Translation
```bash
# Transcribe and translate to Chinese
autosub video.mp4 --translate zh-CN

# Translate existing SRT file
autosub video.srt --translate zh-CN

# Use different LLM provider
autosub video.mp4 --translate zh-CN --llm-provider anthropic

# Custom model
autosub video.mp4 --translate zh-CN --llm-model gpt-4-turbo
```

### GPU Acceleration
```bash
# Metal (macOS - automatic)
autosub video.mp4

# CUDA (NVIDIA GPU)
autosub video.mp4 --device cuda
```

## Configuration

### Environment Variables

- `AUTOSUB_LLM_API_KEY` - LLM API key for translation
- `OPENAI_API_KEY` - OpenAI API key (alternative)
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GEMINI_API_KEY` - Google Gemini API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `HF_ENDPOINT` - Custom HuggingFace endpoint
- `AUTOSUB_CACHE_DIR` - Cache directory for Whisper models

### Output Files

- Input: `video.mp4`
- Transcription: `video.srt`
- Translation: `video.zh-CN.srt` (or other language code)

## Common Tasks

### Adding a New LLM Provider

1. Add enum variant to `LlmProvider` in `src/config.rs`
2. Add matching in `translate.rs` to handle the new provider
3. Update API key environment variable handling
4. Test with a sample subtitle file

### Adjusting VAD Behavior

VAD settings are configured in `src/whisper_cli.rs` via `VadConfig`:
- `mode: VadMode::Aggressive` - Filters out non-speech aggressively
- `silence_reset_secs` - Seconds of silence before resetting ASR context
- Lower values = more segments, higher values = more context but potential hallucinations
- See `autosub-asr/src/vad.rs` for VadSegmenter implementation

### Customizing SRT Output

Edit `src/srt.rs`:
- `SrtEntry::format()` - SRT entry formatting
- `merge_consecutive()` - Merging logic for duplicate subtitles
- Timestamp formatting (`format_timestamp()`)

## Testing

### Manual Testing
```bash
# Test with a short video
cargo run -- test/video.mp4 -v

# Test translation
cargo run -- test/video.mp4 --translate zh-CN

# Test SRT-only translation
cargo run -- test/video.srt --translate en
```

### Unit Tests
```bash
cargo test
```

Most testing is done via integration tests in `autosub-audio` and `autosub-asr`.

## Performance Considerations

1. **Model Size vs Speed**
   - Tiny: ~1x realtime, 75MB, ~90% accuracy
   - Base: ~0.5x realtime, 150MB, ~95% accuracy
   - Small: ~0.25x realtime, 500MB, ~97% accuracy
   - Medium: ~0.1x realtime, 1.5GB, ~98% accuracy
   - Large: ~0.05x realtime, 3GB, ~99% accuracy

2. **VAD Impact**
   - Enables: Faster (skips silence), less hallucination
   - Disables: Slower, processes all audio, may hallucinate on silence

3. **GPU Acceleration**
   - Metal (macOS): 2-5x faster than CPU
   - CUDA (NVIDIA): 3-10x faster than CPU

4. **Translation Batching**
   - Larger batches = fewer API calls but more latency
   - Default: 20 entries per batch

## Common Pitfalls

1. **FFmpeg Not Found** - Install FFmpeg dev libraries before building
2. **Model Download Fails** - Check HuggingFace access, use VPN if needed
3. **Translation API Key** - Set environment variable before running
4. **VAD Too Aggressive** - May cut off valid speech; try `--vad-reset-secs 2.0`
5. **Large Files OOM** - Streaming design should handle this, but very large models on small RAM may fail
6. **Metal Threading Issues** - Do NOT move model/device between threads; architecture handles this correctly

## Error Handling

All errors use `anyhow::Result` with context:
```rust
audio_stream.open(&config.input, None)
    .context("Failed to open audio stream from input file")?
```

Errors propagate to `main()` which prints them and exits with error code.

## Dependencies

### Core
- **clap** - Command-line parsing
- **tokio** - Async runtime
- **anyhow** - Error handling
- **tracing** - Logging

### Domain-Specific
- **autosub-audio** - Audio extraction
- **autosub-asr** - Whisper transcription
- **indicatif** - Progress bars
- **reqwest** - HTTP client for LLM APIs
- **serde/serde_json** - Serialization for LLM requests

## Build & Release

### Development Build
```bash
cargo build
```

### Release Build
```bash
cargo build --release
```

### Static FFmpeg (Self-Contained Binary)
```bash
cargo build --release --features ffmpeg-static
```

### CUDA Support
```bash
cargo build --release --features cuda
```

### CI/CD
- `.github/workflows/` - GitHub Actions for CI and releases
- Builds for Linux, macOS, Windows
- Produces standalone binaries with static FFmpeg

## Related Documentation

- See `AGENTS.md` for repository-wide guidelines
- See `autosub-asr/CLAUDE.md` for ASR engine details
- See `autosub-audio/CLAUDE.md` for audio extraction details
- See `README.md` for user documentation

## Recent Changes

- 2026-01-28: Fixed Metal threading issue - ASR engine now runs on main thread (same as model creation)
- 2026-01-28: Added initial prompt support to ASR engine for better context
- Earlier: Refactored to library crates, added VAD, added translation streaming
