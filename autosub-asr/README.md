# autosub-asr

ASR (Automatic Speech Recognition) engine library for automatic transcription using OpenAI Whisper models.

## Overview

This crate provides a clean, channel-based interface for speech recognition. It's designed to be used as a library component separate from file I/O, audio decoding, and UI concerns.

## Features

- **Channel-based async interface**: Process audio clips via async channels
- **Pure ASR logic**: No file I/O, no progress bars, no CLI dependencies
- **Whisper model support**: Tiny, Base, Small, Medium, and Large variants
- **Language detection**: Auto-detect or specify language
- **Initial prompt support**: Guide the model with context to improve transcription quality
- **Pluggable hallucination filtering**: Customizable filtering via plugin system
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

### Hallucination Filtering

The crate provides a plugin-based hallucination filtering system that allows you to control how transcription artifacts are detected and filtered:

```rust
pub trait HallucinationFilter: Send {
    fn is_hallucinated(&self, text: &str) -> bool;
}

// Built-in filters
pub struct DefaultHallucinationFilter;  // Filters common hallucination patterns
pub struct NoFilter;                    // Disables filtering (passes everything through)
```

You can create custom filters by implementing the `HallucinationFilter` trait, or use the built-in filters:

```rust
use autosub_asr::{AsrEngine, DefaultHallucinationFilter, NoFilter};

// Use default hallucination filter
let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;

// Disable filtering completely
let engine = AsrEngine::with_filter(
    model,
    Some("en".to_string()),
    None, // No initial prompt
    vad_config,
    Some(Box::new(NoFilter)),
)?;

// Use with initial prompt to guide the model
let engine = AsrEngine::with_filter(
    model,
    Some("en".to_string()),
    Some("Technical discussion about machine learning and artificial intelligence.".to_string()),
    vad_config,
    Some(Box::new(DefaultHallucinationFilter::new())),
)?;

// Use custom filter
struct MyCustomFilter;
impl HallucinationFilter for MyCustomFilter {
    fn is_hallucinated(&self, text: &str) -> bool {
        // Your custom logic here
        text.len() < 3
    }
}

let engine = AsrEngine::with_filter(
    model,
    Some("en".to_string()),
    None, // No initial prompt
    vad_config,
    Some(Box::new(MyCustomFilter)),
)?;
```

**Note**: The default filter may occasionally produce false positives (filtering valid text). If you need complete transcription output, use `NoFilter` or implement a custom filter tuned to your use case.

### Initial Prompt

You can provide an initial prompt to guide the Whisper model and improve transcription quality. The initial prompt helps the model understand the context, expected terminology, and speaking style of the audio.

**Benefits of using an initial prompt:**
- Improves recognition of domain-specific terminology
- Helps maintain consistent spelling and formatting
- Can guide the model on expected speaking style
- Useful for technical content, names, or specialized vocabulary

**Example use cases:**
```rust
// For technical discussions
let prompt = Some("This is a technical discussion about machine learning, neural networks, and artificial intelligence.".to_string());

// For medical content
let prompt = Some("Medical terminology: diagnosis, treatment, patient care.".to_string());

// For names and proper nouns
let prompt = Some("Interview with Dr. Jane Smith about her research.".to_string());

let engine = AsrEngine::with_filter(
    model,
    Some("en".to_string()),
    prompt,
    vad_config,
    Some(Box::new(DefaultHallucinationFilter::new())),
)?;
```

**Tips:**
- Keep prompts relevant to your audio content
- Include key terminology or names you expect to appear
- The prompt should be in the same language as the audio
- Prompts work best when they match the speaking style and context

## Usage Example

```rust
use autosub_asr::{
    AsrEngine, AsrInput, VadConfig, VadMode, WhisperModel,
    WhisperModelConfig, WhisperModelSize,
};
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

    // Configure VAD
    let vad_config = VadConfig {
        sample_rate: 16000,
        frame_duration_ms: 30,
        mode: VadMode::Quality,
        silence_reset_secs: 1.0,
    };

    // Create channels
    let (audio_tx, audio_rx) = mpsc::channel::<AsrInput>(100);
    let (result_tx, mut result_rx) = mpsc::channel(100);

    // Create and spawn ASR engine (with default hallucination filter)
    let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;
    let asr_task = tokio::spawn(async move {
        engine.run(audio_rx, result_tx).await
    });

    // Send audio samples
    tokio::spawn(async move {
        // Send raw audio samples (VAD will segment them)
        let samples = vec![0.0; 16000]; // 1 second of audio at 16kHz
        audio_tx.send(AsrInput::Samples(samples)).await.unwrap();

        // Signal end of stream
        audio_tx.send(AsrInput::Flush).await.unwrap();
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
