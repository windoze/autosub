use std::path::Path;
use std::sync::mpsc;

use anyhow::{Context, Result};
use autosub_asr::{
    AsrEngine, AsrInput, DefaultHallucinationFilter, TranscriptionResult, VadConfig, VadMode,
    WhisperModel, WhisperModelConfig, WhisperModelSize,
};
use autosub_audio::AudioStream;
use candle_core::Device;
use tracing::{debug, info};

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

    // Create standard library channels for communication (not tokio async).
    //
    // IMPORTANT: We intentionally run the ASR engine on the current thread (the same
    // thread that created/loaded the Whisper model). Some device backends (notably Metal)
    // can hang when a model is moved to a different thread for execution.
    //
    // Use a bounded channel (capacity 5) for audio input to prevent the audio extraction
    // thread from getting too far ahead of transcription. This ensures the progress bar
    // accurately reflects transcription progress rather than just audio extraction progress.
    let (audio_tx, audio_rx) = mpsc::sync_channel::<AsrInput>(2);
    let (result_tx, result_rx) = mpsc::channel::<TranscriptionResult>();

    // Spawn SRT writing in a separate thread so we can keep streaming results to disk
    // while the ASR engine runs on this thread.
    let output_path = output_path.to_path_buf();
    let writer_task = std::thread::spawn(move || -> Result<Subtitle> {
        let mut writer = SrtWriter::create(&output_path)?;
        let mut subtitle = Subtitle::new();

        for result in result_rx {
            writer.write_entry(result.start, result.end, &result.text)?;
            subtitle.push(result.start, result.end, result.text);
        }

        writer.finish()?;
        Ok(subtitle)
    });

    // Spawn audio extraction in a thread
    let audio_task = std::thread::spawn(move || {
        extract_and_stream_audio(audio_stream, audio_tx, progress_callback)
    });

    // Run ASR engine in this thread (blocking).
    debug!("Starting ASR engine (same thread as model)");
    let asr_result = asr_engine.run_blocking(audio_rx, result_tx);

    // Join threads (always attempt to join so we don't leave threads running on error).
    let audio_result = audio_task
        .join()
        .map_err(|_| anyhow::anyhow!("Audio thread panicked"))?;
    let subtitle_result = writer_task
        .join()
        .map_err(|_| anyhow::anyhow!("SRT writer thread panicked"))?;

    // Surface any errors from each stage.
    asr_result?;
    audio_result?;
    let subtitle = subtitle_result?;

    debug!("Transcription complete: {} segments", subtitle.len());
    Ok(subtitle)
}

/// Extract audio from stream and send to ASR engine via channel
/// This function is internal and handles all low-level audio processing
fn extract_and_stream_audio(
    mut audio_stream: AudioStream,
    audio_tx: mpsc::SyncSender<AsrInput>,
    progress_callback: Option<Box<dyn ProgressCallback>>,
) -> Result<()> {
    let file_info = audio_stream.file_info();
    let total_duration_us = (file_info.duration_secs * 1_000_000.0) as u64;

    debug!("Extracting audio: {:.2} seconds", file_info.duration_secs);

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

    debug!("Audio extraction complete, flushing ASR engine");

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
