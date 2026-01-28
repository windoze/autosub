# Refactoring Summary: ASR Crate Extraction

## Overview

The code has been successfully refactored to extract the ASR (Automatic Speech Recognition) functionality into a separate crate with async channel-based communication. All file I/O, audio processing, and CLI-specific operations have been moved to the CLI wrapper.

## Changes

### 1. New ASR Crate (`autosub-asr/`)

Created a new library crate containing pure ASR functionality:

**Structure:**
```
autosub-asr/
├── Cargo.toml              # Crate configuration with minimal dependencies
├── README.md               # Documentation and usage examples
└── src/
    ├── lib.rs              # Public API exports
    ├── types.rs            # AudioClip and TranscriptionResult types
    ├── model.rs            # Core Whisper model and ASR engine
    ├── vad.rs              # Voice Activity Detection
    ├── melfilters.bytes    # Pre-computed 80-bin mel filters
    └── melfilters128.bytes # Pre-computed 128-bin mel filters
```

**Key Components:**

- **`AudioClip`**: Represents audio data with timing metadata
  - `samples: Vec<f32>` - Audio samples at 16kHz mono
  - `start_sample`, `end_sample` - Position in stream
  - `reset_context` - Context reset flag for long silences

- **`TranscriptionResult`**: Transcribed text with timestamps
  - `text: String` - Transcribed text
  - `start`, `end: f64` - Time in seconds

- **`AsrEngine`**: Async engine processing clips via channels
  - Takes `mpsc::Receiver<AudioClip>` for input
  - Sends `TranscriptionResult` via `mpsc::Sender`
  - Runs asynchronously using Tokio

- **`WhisperModel`**: Core Whisper model wrapper
  - Model loading and management
  - Token decoding and language detection
  - Hallucination filtering
  - Sentence segmentation

- **`VadSegmenter`**: Voice Activity Detection
  - Speech/silence segmentation
  - Context reset tracking for long silences
  - Produces `AudioClip` directly from audio samples
  - WebRTC VAD backend implementation

### 2. CLI Wrapper Changes

**New File: `src/whisper_cli.rs`**

Contains all CLI-specific operations:
- Audio stream processing from files (using FFmpeg)
- Progress bar management
- SRT file writing
- Channel orchestration between audio source and ASR engine
- Uses VAD from the ASR crate for speech segmentation

**Key Function:**
```rust
pub async fn transcribe_stream_to_file(
    audio_stream: AudioStream,
    output_path: &Path,
    model_size: ConfigModelSize,
    // ... other parameters
) -> Result<Subtitle>
```

This function:
1. Loads the Whisper model
2. Creates async channels for audio/results
3. Spawns ASR engine task
4. Processes audio stream (with or without VAD)
5. Writes results to SRT file as they arrive
6. Returns completed subtitle

### 3. Updated Files

**`src/lib.rs`**
- Removed `whisper` module
- Added `whisper_cli` module
- Updated exports

**`src/main.rs`**
- Changed import from `whisper::` to `whisper_cli::`
- Added `.await` to `transcribe_stream_to_file` call

**`Cargo.toml`**
- Added `autosub-asr` as path dependency
- Simplified feature flags (propagate to ASR crate)
- Removed redundant Candle dependencies

**`src/whisper.rs`**
- Renamed to `whisper.rs.old` (backup)
- Original functionality split between ASR crate and CLI wrapper

### 4. Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                  CLI Wrapper                         │
│  (File I/O, Audio Stream, Progress)                 │
│                                                      │
│  ┌────────────┐                                     │
│  │ AudioStream│                                     │
│  │  (FFmpeg)  │                                     │
│  └──────┬─────┘                                     │
│         │                                            │
│         v                                            │
│    Raw Audio Samples                                │
└──────────────┬──────────────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────────────┐
│              ASR Crate (autosub-asr)                 │
│                                                       │
│  ┌──────────────┐                                   │
│  │VadSegmenter  │ (optional)                        │
│  │  (WebRTC)    │                                   │
│  └──────┬───────┘                                   │
│         │                                            │
│         v                                            │
│  mpsc::channel<AudioClip>                           │
│         │                                            │
│         v                                            │
│  ┌──────────┐         ┌──────────────┐             │
│  │AsrEngine │────────>│WhisperModel  │             │
│  └────┬─────┘         └──────────────┘             │
│       │                                              │
│       v                                              │
│  mpsc::channel<TranscriptionResult>                 │
└───────────┬──────────────────────────────────────────┘
            │
            v
┌──────────────────────────────────────────────────────┐
│            CLI Wrapper                                │
│  (SRT Writer, Result Collection)                     │
└──────────────────────────────────────────────────────┘
```

## Key Design Decision: VAD in ASR Crate

Voice Activity Detection (VAD) has been moved into the `autosub-asr` crate rather than staying in the CLI wrapper. This decision was made because:

1. **VAD is core to ASR pipeline**: Speech segmentation is a fundamental preprocessing step for transcription
2. **Simplified interface**: VAD now produces `AudioClip` directly, eliminating the need for intermediate `SpeechSegment` type
3. **Better encapsulation**: All audio processing logic related to ASR is in one place
4. **Reusability**: Other applications using the ASR crate can benefit from the VAD functionality
5. **Cleaner CLI code**: The CLI wrapper just pipes audio samples to the ASR crate, which handles segmentation internally

## Benefits

### 1. **Separation of Concerns**
- ASR logic (including VAD) is independent of file formats, audio codecs, and UI
- Easier to test ASR functionality in isolation
- Can reuse ASR crate in other projects with VAD included

### 2. **Async Channel Architecture**
- Efficient concurrent processing
- Audio decoding and transcription run in parallel
- Results can be processed as they arrive (streaming output)

### 3. **Cleaner Dependencies**
- ASR crate has minimal dependencies (no FFmpeg, no CLI tools)
- CLI wrapper handles all platform-specific code
- Better separation of ML logic from I/O operations

### 4. **Maintainability**
- Each module has a clear, focused responsibility
- Easier to add new audio sources or output formats
- ASR improvements don't affect CLI code

### 5. **Reusability**
- ASR crate can be used in other applications
- Different frontends can use the same ASR engine
- Potential for WebAssembly or embedded use

## Migration Notes

### For Users
No changes to the CLI interface - the tool works the same way.

### For Developers

**Using the ASR crate directly:**
```rust
use autosub_asr::{AsrEngine, AudioClip, WhisperModel, WhisperModelConfig};
use tokio::sync::mpsc;

// Load model
let model = WhisperModel::load(config)?;

// Create channels
let (audio_tx, audio_rx) = mpsc::channel(10);
let (result_tx, result_rx) = mpsc::channel(100);

// Spawn ASR engine
let engine = AsrEngine::new(model, Some("en".to_string()));
tokio::spawn(async move {
    engine.run(audio_rx, result_tx).await
});

// Send audio clips
audio_tx.send(AudioClip::new(samples, 0, samples.len(), true)).await?;

// Receive results
while let Some(result) = result_rx.recv().await {
    println!("{}", result.text);
}
```

**Extending with new audio sources:**
Implement a producer that sends `AudioClip` to the channel - no need to modify ASR code.

**Adding new output formats:**
Implement a consumer that processes `TranscriptionResult` - no need to modify ASR code.

## Testing

The refactored code compiles successfully with:
```bash
cargo check
```

All warnings are minor (unused constants in ASR crate that may be needed later).

## Next Steps

Potential improvements:
1. Add unit tests for ASR crate
2. Add integration tests using sample audio
3. Benchmark performance vs. old implementation
4. Consider adding batch processing mode
5. Add metrics/telemetry hooks in AsrEngine

## Files Changed

**Added:**
- `autosub-asr/` - New crate directory
- `autosub-asr/Cargo.toml`
- `autosub-asr/README.md`
- `autosub-asr/src/lib.rs`
- `autosub-asr/src/types.rs`
- `autosub-asr/src/model.rs`
- `autosub-asr/src/vad.rs` - Moved from `src/vad.rs`
- `autosub-asr/src/melfilters.bytes` - Copied from `src/`
- `autosub-asr/src/melfilters128.bytes` - Copied from `src/`
- `src/whisper_cli.rs`
- `REFACTORING_SUMMARY.md`

**Modified:**
- `Cargo.toml` - Added ASR crate dependency, removed webrtc-vad
- `autosub-asr/Cargo.toml` - Added webrtc-vad dependency
- `src/lib.rs` - Updated module exports, removed vad module
- `src/main.rs` - Updated imports and async call
- `src/whisper_cli.rs` - Now imports VAD from ASR crate

**Renamed/Backed up:**
- `src/whisper.rs` → `src/whisper.rs.old`
- `src/vad.rs` → `src/vad.rs.old`

## Conclusion

The refactoring successfully achieves the goals:
✅ ASR functionality extracted to separate crate
✅ Async channel interface for audio clips and results
✅ File I/O, audio processing, and UI moved to CLI wrapper
✅ Clean separation of concerns
✅ Code compiles successfully
