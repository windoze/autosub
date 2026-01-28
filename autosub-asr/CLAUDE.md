# autosub-asr Crate Guide

## Purpose

This crate provides the ASR (Automatic Speech Recognition) engine for the autosub project. It wraps OpenAI's Whisper model using the Candle ML framework and provides a clean, channel-based interface for transcription.

## Key Architecture

### Core Components

1. **WhisperModel** (`src/model.rs:78-413`)
   - Loads and runs Whisper models (Tiny, Base, Small, Medium, Large)
   - Handles mel spectrogram conversion and audio encoding
   - Manages decoder with timestamps for word-level alignment
   - Supports initial prompts to guide transcription

2. **AsrEngine** (`src/model.rs:415-540`)
   - High-level API that integrates VAD + Whisper
   - Processes audio through channels: `AsrInput` → `TranscriptionResult`
   - Manages model lifecycle and context (KV cache)
   - Runs in a dedicated thread/task

3. **VAD (Voice Activity Detection)** (`src/vad.rs`)
   - Uses WebRTC VAD to segment speech from silence
   - Prevents hallucinations on silent audio
   - Configurable aggressiveness modes (Quality/LowBitrate/Aggressive/VeryAggressive)
   - Automatically resets ASR context after long silence

4. **Hallucination Filter** (`src/filter.rs`)
   - Plugin system to filter out hallucinated transcriptions
   - DefaultHallucinationFilter: catches common patterns (repeated phrases, music descriptions)
   - NoFilter: for when you need raw output
   - Custom filters can be implemented via `HallucinationFilter` trait

### Data Flow

```
Raw Audio Samples (f32, 16kHz, mono)
    ↓
AsrInput::Samples → Channel
    ↓
VadSegmenter (WebRtcVad)
    ↓
AudioClip (speech segments)
    ↓
WhisperModel::transcribe_clip()
    ↓
HallucinationFilter (optional)
    ↓
TranscriptionResult → Channel
    ↓
Consumer (CLI/UI)
```

### Important Files

- **`src/model.rs`** - Core Whisper model and ASR engine
- **`src/vad.rs`** - Voice activity detection using WebRTC VAD
- **`src/filter.rs`** - Hallucination filtering system
- **`src/types.rs`** - Public types (AudioClip, TranscriptionResult, AsrInput)
- **`src/melfilters.bytes`** / **`melfilters128.bytes`** - Precomputed mel filter banks (DO NOT MODIFY)

### Key Features

1. **Initial Prompt Support** (Added 2026-01-28)
   - Guide the model with context via `initial_prompt` parameter
   - Helps with domain-specific terminology, names, technical content
   - Example: `Some("Technical discussion about machine learning.")"`
   - Tokens are prepended to decoder input after SOT/lang/transcribe tokens

2. **Context Management**
   - KV cache persists across consecutive VAD segments for better accuracy
   - Only resets after long silence (configurable via `silence_reset_secs`)
   - `AudioClip.reset_context` signals when to clear cache

3. **Timestamp Extraction**
   - Whisper's timestamp tokens (50364-51864) are parsed
   - Each segment has precise start/end times
   - Sentences are split and timestamps distributed proportionally

## Common Tasks

### Adding a New Model Size

1. Add variant to `WhisperModelSize` enum in `src/model.rs:57`
2. Implement `repo_id()` to return HuggingFace model ID
3. Ensure mel filter banks match the model's `num_mel_bins` (80 or 128)

### Modifying VAD Behavior

- Adjust `VadConfig` parameters (aggressiveness, silence threshold)
- Edit `VadSegmenter` logic in `src/vad.rs`
- Be careful: too aggressive = lost speech, too lenient = hallucinations

### Custom Hallucination Filters

Implement the `HallucinationFilter` trait:
```rust
pub trait HallucinationFilter: Send + Sync {
    fn is_hallucinated(&self, text: &str) -> bool;
}
```

Pass to `AsrEngine::with_filter()`.

## Testing

- Unit tests are sparse (this is primarily integration-tested via `autosub-audio`)
- Integration tests live in `tests/` and `../autosub-audio/tests/`
- Use `WhisperModelSize::Tiny` for tests (fast, ~75MB)
- Test files should be short (<10s) to keep CI fast

## Performance Considerations

- **Model Size**: Tiny (75MB) → Large-v3 (3GB)
- **Device**: CPU is default, Metal on macOS, CUDA with `--features cuda`
- **Mel Filters**: Precomputed to avoid runtime calculation
- **KV Cache**: Persisting cache across segments reduces recomputation

## Common Pitfalls

1. **Don't modify mel filter assets** - Regenerate via `scripts/generate_melfilters.py` if needed
2. **Hallucination vs. Valid Text** - Default filter may have false positives; tune as needed
3. **VAD Segment Length** - Very long segments (>30s) get truncated to N_FRAMES (3000 frames = 30s)
4. **Thread Safety** - `AsrEngine` is `!Send` due to WebRtcVad; use `unsafe impl Send` with care
5. **Initial Prompt Length** - Keep prompts concise; very long prompts may affect quality

## Dependencies

- **candle-core/candle-nn/candle-transformers** - ML framework
- **hf-hub** - Download models from HuggingFace
- **tokenizers** - Whisper tokenizer
- **tokio** - Async channels
- **webrtc-vad** - Voice activity detection (C bindings)

## Related Crates

- **autosub-audio** - FFmpeg-based audio extraction, feeds this crate
- **autosub** (root) - CLI that orchestrates both crates

## Recent Changes

- 2026-01-28: Added `initial_prompt` parameter to guide transcription
- Earlier: Integrated VAD into AsrEngine, added hallucination filtering
