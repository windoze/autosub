# autosub-asr

ASR (Automatic Speech Recognition) engine library for automatic transcription using OpenAI Whisper models.

## Overview

This crate provides a clean, channel-based interface for speech recognition. It's designed to be used as a library component separate from file I/O, audio decoding, and UI concerns.

## Features

- **Channel-based async interface**: Process audio clips via async channels
- **Pure ASR logic**: No file I/O, no progress bars, no CLI dependencies
- **Whisper model support**: Tiny, Base, Small, Medium, and Large variants
- **Language detection**: Auto-detect or specify language
- **Hallucination filtering**: Built-in detection of common transcription errors
- **Sentence segmentation**: Automatically split long segments into sentences
- **Voice Activity Detection (VAD)**: Built-in WebRTC VAD for speech segmentation

## Architecture

The crate exposes the following main types:

### `AudioClip`
Represents a chunk of audio to be transcribed:
```rust
pub struct AudioClip {
    pub samples: Vec<f32>,        // Audio samples at 16kHz, mono, normalized to [-1, 1]
    pub start_sample: usize,      // Start position in the original stream
    pub end_sample: usize,        // End position in the original stream
    pub reset_context: bool,      // Whether to reset ASR context before processing
}
```

### `TranscriptionResult`
Represents a transcribed segment:
```rust
pub struct TranscriptionResult {
    pub text: String,
    pub start: f64,               // Start time in seconds
    pub end: f64,                 // End time in seconds
}
```

### `AsrEngine`
The main ASR engine that processes audio clips:
```rust
pub struct AsrEngine {
    // ...
}

impl AsrEngine {
    pub fn new(model: WhisperModel, language: Option<String>) -> Self;

    pub async fn run(
        self,
        input: mpsc::Receiver<AudioClip>,
        output: mpsc::Sender<TranscriptionResult>,
    ) -> Result<()>;
}
```

### VAD (Voice Activity Detection)

The crate includes VAD functionality for speech segmentation:

```rust
pub struct VadSegmenter<D: VoiceActivityDetector> {
    // ...
}

impl<D: VoiceActivityDetector> VadSegmenter<D> {
    pub fn new(detector: D, frame_duration_ms: usize, silence_reset_secs: f32) -> Self;
    pub fn push_samples(&mut self, samples: &[f32]) -> Result<Vec<AudioClip>>;
    pub fn flush(&mut self) -> Result<Vec<AudioClip>>;
}

// WebRTC VAD implementation
pub struct WebRtcVad { /* ... */ }
pub type WebRtcVadMode = webrtc_vad::VadMode;
```

The VAD segmenter processes audio samples and produces `AudioClip` instances containing only speech segments, filtering out silence and non-speech audio.

## Usage Example

```rust
use autosub_asr::{AsrEngine, AudioClip, WhisperModel, WhisperModelConfig, WhisperModelSize};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load model
    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Base,
        cache_dir: None,
        device: candle_core::Device::Cpu,
    };
    let model = WhisperModel::load(config)?;

    // Create channels
    let (audio_tx, audio_rx) = mpsc::channel::<AudioClip>(10);
    let (result_tx, mut result_rx) = mpsc::channel(100);

    // Create and spawn ASR engine
    let engine = AsrEngine::new(model, Some("en".to_string()));
    let asr_task = tokio::spawn(async move {
        engine.run(audio_rx, result_tx).await
    });

    // Send audio clips
    tokio::spawn(async move {
        // Example: send audio clips
        let clip = AudioClip::new(
            vec![0.0; 16000], // 1 second of silence
            0,
            16000,
            true,
        );
        audio_tx.send(clip).await.unwrap();
    });

    // Receive results
    while let Some(result) = result_rx.recv().await {
        println!("{:.2}-{:.2}: {}", result.start, result.end, result.text);
    }

    asr_task.await??;
    Ok(())
}
```

## Design Principles

1. **Separation of concerns**: ASR logic is completely separate from:
   - File format handling (WAV, MP3, etc.)
   - Audio decoding (FFmpeg)
   - Progress indication (progress bars)
   - CLI interface

2. **Async-first**: Uses Tokio channels for efficient concurrent processing

3. **Zero-copy where possible**: Audio samples are passed by ownership through channels

4. **Flexible device support**: Works with CPU, CUDA, and Metal backends

## Files

The crate includes pre-computed mel filter banks required by Whisper models:
- `src/melfilters.bytes` - 80-bin mel filters (for smaller models)
- `src/melfilters128.bytes` - 128-bin mel filters (for larger models)

These are embedded directly into the binary at compile time.

## Dependencies

- `candle-*`: ML framework for running Whisper models
- `tokio`: Async runtime for channel-based communication
- `hf-hub`: Download models from HuggingFace
- `tokenizers`: Tokenization for Whisper
- `byteorder`: Reading mel filter banks

## License

MIT
