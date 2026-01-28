use std::path::Path;

use anyhow::{Context, Result};
use autosub_asr::{
    AsrEngine, AsrInput, TranscriptionResult, VadConfig, VadMode, WhisperModel,
    WhisperModelConfig, WhisperModelSize,
};
use candle_core::Device;
use indicatif::ProgressBar;
use tokio::sync::mpsc;
use tracing::info;

use crate::audio::AudioStream;
use crate::config::WhisperModelSize as ConfigModelSize;
use crate::srt::{SrtWriter, Subtitle};

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

/// Transcribe audio using streaming mode with async channels.
/// The audio stream is processed through VAD if enabled, then sent to the ASR engine
/// via async channels. Results are written to the SRT file as they arrive.
#[allow(clippy::too_many_arguments)]
pub async fn transcribe_stream_to_file<F>(
    audio_stream: AudioStream,
    output_path: &Path,
    model_size: ConfigModelSize,
    cache_dir: Option<std::path::PathBuf>,
    device: Device,
    language: Option<&str>,
    vad_enabled: bool,
    vad_silence_secs: f32,
    create_progress: F,
) -> Result<Subtitle>
where
    F: FnOnce() -> Option<ProgressBar>,
{
    info!("Loading Whisper model...");

    // Load the ASR model
    let config = WhisperModelConfig {
        model_size: convert_model_size(model_size),
        cache_dir,
        device,
    };

    let model = WhisperModel::load(config)?;

    // Create progress bar before spawning tasks
    let progress = create_progress();

    // Create async channels for audio input and transcription results
    // Use larger buffers to avoid any potential backpressure issues
    let (audio_tx, audio_rx) = mpsc::channel::<AsrInput>(1000);
    let (result_tx, mut result_rx) = mpsc::channel::<TranscriptionResult>(1000);

    // Configure VAD
    let vad_config = if vad_enabled {
        VadConfig {
            sample_rate: crate::audio::WHISPER_SAMPLE_RATE,
            frame_duration_ms: 30,
            mode: VadMode::Aggressive,
            silence_reset_secs: vad_silence_secs,
        }
    } else {
        // Even without VAD enabled, we still need a config
        // Set silence_reset_secs to a very large value to effectively disable auto-segmentation
        VadConfig {
            sample_rate: crate::audio::WHISPER_SAMPLE_RATE,
            frame_duration_ms: 30,
            mode: VadMode::Quality,
            silence_reset_secs: 999999.0,
        }
    };

    // Create ASR engine with VAD
    let asr_engine = AsrEngine::new(model, language.map(|s| s.to_string()), vad_config)?;

    // Spawn ASR engine in a plain thread since it's fully synchronous
    info!("Spawning ASR engine task");
    let asr_task = std::thread::spawn(move || {
        info!("ASR engine task started, calling run()");
        let result = asr_engine.run(audio_rx, result_tx);
        info!("ASR engine run() completed with result: {:?}", result.as_ref().map(|_| "Ok").map_err(|e| format!("{}", e)));
        result
    });
    info!("ASR task spawned successfully");

    // Yield to allow the spawned task to start
    tokio::task::yield_now().await;
    info!("After yield_now()");

    // Create SRT writer and subtitle accumulator
    let mut writer = SrtWriter::create(output_path)?;
    let mut subtitle = Subtitle::new();

    // Spawn audio processing in a blocking thread
    let audio_task = std::thread::spawn(move || process_audio_stream_blocking(audio_stream, audio_tx, progress));

    // Collect transcription results
    while let Some(result) = result_rx.recv().await {
        writer.write_entry(result.start, result.end, &result.text)?;
        subtitle.push(result.start, result.end, result.text);
    }

    // Wait for audio thread to complete
    tokio::task::spawn_blocking(move || audio_task.join())
        .await
        .context("Failed to join audio thread")?
        .map_err(|_| anyhow::anyhow!("Audio thread panicked"))??;

    writer.finish()?;

    // Wait for ASR thread to complete
    tokio::task::spawn_blocking(move || asr_task.join())
        .await
        .context("Failed to join ASR thread")?
        .map_err(|_| anyhow::anyhow!("ASR thread panicked"))??;

    info!("Transcription complete: {} segments", subtitle.len());
    Ok(subtitle)
}

/// Process audio stream by sending raw samples to the ASR engine (blocking version)
fn process_audio_stream_blocking(
    mut audio_stream: AudioStream,
    audio_tx: mpsc::Sender<AsrInput>,
    progress: Option<ProgressBar>,
) -> Result<()> {
    let duration_secs = audio_stream.duration_secs();
    let total_duration_us = audio_stream.total_duration_us();

    info!(
        "Streaming transcription: {:.2} seconds of audio",
        duration_secs
    );

    // Setup progress bar
    if let Some(pb) = progress.as_ref() {
        if total_duration_us > 0 {
            pb.set_length(total_duration_us as u64);
        }
        pb.set_position(0);
    }

    info!("Starting audio processing");

    // Stream audio samples directly to the ASR engine
    // AudioStream is synchronous, so this is straightforward
    let mut chunk_count = 0;
    while let Some(chunk) = audio_stream.next() {
        chunk_count += 1;
        info!("Read audio chunk {} from stream", chunk_count);

        match chunk {
            Ok(samples) => {
                info!("Forwarding {} samples to ASR engine", samples.len());
                audio_tx
                    .blocking_send(AsrInput::Samples(samples))
                    .map_err(|_| anyhow::anyhow!("Failed to send audio samples to ASR engine"))?;
                info!("Successfully sent samples to ASR engine");

                if let Some(pb) = progress.as_ref() {
                    let pos = audio_stream.current_position_us();
                    pb.set_position(pos as u64);
                }
            }
            Err(e) => {
                return Err(e).context("Failed to read audio chunk");
            }
        }
    }

    let chunk_word = if chunk_count == 1 { "chunk" } else { "chunks" };
    info!("Audio reading complete ({} {}), processing...", chunk_count, chunk_word);

    // Send flush signal to emit any remaining buffered speech
    audio_tx
        .blocking_send(AsrInput::Flush)
        .map_err(|_| anyhow::anyhow!("Failed to send flush signal to ASR engine"))?;

    // Drop the sender to signal end of input
    drop(audio_tx);

    if let Some(pb) = progress {
        pb.finish_with_message("Audio processing complete");
    }

    Ok(())
}


