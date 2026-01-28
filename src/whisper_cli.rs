use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, Result};
use autosub_asr::{
    AsrEngine, AsrInput, DefaultHallucinationFilter, TranscriptionResult, VadConfig, VadMode,
    WhisperModel, WhisperModelConfig, WhisperModelSize,
};
use autosub_audio::AudioStream;
use candle_core::Device;
use tracing::info;

use crate::config::WhisperModelSize as ConfigModelSize;
use crate::srt::{SrtWriter, Subtitle};

/// Configuration for transcription
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub model_size: ConfigModelSize,
    pub cache_dir: Option<std::path::PathBuf>,
    pub device: Device,
    pub language: Option<String>,
    pub enable_vad: bool,
    pub vad_silence_secs: f32,
    pub enable_hallucination_filter: bool,
}

/// Progress callback trait for reporting transcription progress
pub trait ProgressCallback: Send {
    /// Called when progress is updated
    /// - position: current position in microseconds
    /// - total: total duration in microseconds
    fn on_progress(&self, position: u64, total: u64);

    /// Called when transcription is complete
    fn on_complete(&self);
}

/// Convert config model size to ASR crate model size
fn convert_model_size(size: ConfigModelSize) -> WhisperModelSize {
    match size {
        ConfigModelSize::Tiny => WhisperModelSize::Tiny,
        ConfigModelSize::Base => WhisperModelSize::Base,
        ConfigModelSize::Small => WhisperModelSize::Small,
        ConfigModelSize::Medium => WhisperModelSize::Medium,
        ConfigModelSize::Large => WhisperModelSize::Large,
    }
}

/// High-level transcription API that orchestrates audio extraction and ASR
///
/// This function handles the complete transcription pipeline:
/// 1. Loads the Whisper model with specified configuration
/// 2. Extracts audio using autosub-audio (AudioStream)
/// 3. Processes audio through VAD and ASR engine (autosub-asr)
/// 4. Streams results to SRT file as they arrive
/// 5. Reports progress via callback
///
/// This is a fully synchronous/blocking function.
///
/// # Arguments
/// * `audio_stream` - Audio stream from autosub-audio
/// * `output_path` - Path to write SRT file
/// * `config` - Transcription configuration (model, VAD, language, etc.)
/// * `progress_callback` - Optional callback for progress updates
///
/// # Returns
/// Complete subtitle with all transcription results
pub fn transcribe_to_file(
    audio_stream: AudioStream,
    output_path: &Path,
    config: TranscriptionConfig,
    progress_callback: Option<Box<dyn ProgressCallback>>,
) -> Result<Subtitle> {
    info!("Loading Whisper model ({:?})...", config.model_size);

    // Load the Whisper model
    let model_config = WhisperModelConfig {
        model_size: convert_model_size(config.model_size),
        cache_dir: config.cache_dir,
        device: config.device,
    };
    let model = WhisperModel::load(model_config)?;

    // Configure VAD parameters
    let vad_config = VadConfig {
        sample_rate: autosub_audio::DEFAULT_SAMPLE_RATE,
        frame_duration_ms: 30,
        mode: if config.enable_vad {
            VadMode::Aggressive
        } else {
            VadMode::Quality
        },
        silence_reset_secs: if config.enable_vad {
            config.vad_silence_secs
        } else {
            999999.0 // Effectively disable VAD segmentation
        },
    };

    // Create ASR engine with VAD and hallucination filter
    let asr_engine = if config.enable_hallucination_filter {
        AsrEngine::with_filter(
            model,
            config.language.clone(),
            None, // No initial prompt for now
            vad_config,
            Some(Box::new(DefaultHallucinationFilter::new())),
        )?
    } else {
        AsrEngine::with_filter(
            model,
            config.language.clone(),
            None,
            vad_config,
            None, // No filtering
        )?
    };

    // Create standard library channels for communication (not tokio async)
    let (audio_tx, audio_rx) = mpsc::channel::<AsrInput>();
    let (result_tx, result_rx) = mpsc::channel::<TranscriptionResult>();

    // Spawn ASR engine in a thread
    info!("Starting ASR engine");
    let asr_task = std::thread::spawn(move || {
        // Convert std::sync::mpsc to the blocking mpsc that AsrEngine expects
        asr_engine.run_blocking(audio_rx, result_tx)
    });

    // Spawn audio extraction in a thread
    let audio_task = std::thread::spawn(move || {
        extract_and_stream_audio(audio_stream, audio_tx, progress_callback)
    });

    // Create SRT writer and accumulate results in the main thread
    let mut writer = SrtWriter::create(output_path)?;
    let mut subtitle = Subtitle::new();

    // Collect transcription results as they arrive (blocking)
    for result in result_rx {
        writer.write_entry(result.start, result.end, &result.text)?;
        subtitle.push(result.start, result.end, result.text);
    }

    // Wait for audio extraction to complete
    audio_task
        .join()
        .map_err(|_| anyhow::anyhow!("Audio thread panicked"))??;

    writer.finish()?;

    // Wait for ASR engine to complete
    asr_task
        .join()
        .map_err(|_| anyhow::anyhow!("ASR thread panicked"))??;

    info!("Transcription complete: {} segments", subtitle.len());
    Ok(subtitle)
}

/// Extract audio from stream and send to ASR engine via channel
/// This function is internal and handles all low-level audio processing
fn extract_and_stream_audio(
    mut audio_stream: AudioStream,
    audio_tx: mpsc::Sender<AsrInput>,
    progress_callback: Option<Box<dyn ProgressCallback>>,
) -> Result<()> {
    let file_info = audio_stream.file_info();
    let total_duration_us = (file_info.duration_secs * 1_000_000.0) as u64;

    info!("Extracting audio: {:.2} seconds", file_info.duration_secs);

    // Stream audio samples to ASR engine
    while let Some(chunk) = audio_stream.next() {
        let segment = chunk.context("Failed to read audio chunk")?;

        // Send samples to ASR engine (blocking send on standard channel)
        audio_tx
            .send(AsrInput::Samples(segment.samples))
            .map_err(|_| anyhow::anyhow!("ASR engine channel closed unexpectedly"))?;

        // Report progress
        if let Some(ref callback) = progress_callback {
            let position_us = audio_stream.position_us() as u64;
            callback.on_progress(position_us, total_duration_us);
        }
    }

    info!("Audio extraction complete, flushing ASR engine");

    // Send flush signal to process any remaining buffered audio
    audio_tx
        .send(AsrInput::Flush)
        .map_err(|_| anyhow::anyhow!("Failed to send flush signal"))?;

    // Close the channel to signal end of input
    drop(audio_tx);

    // Report completion
    if let Some(callback) = progress_callback {
        callback.on_complete();
    }

    Ok(())
}


