use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use autosub::{
    config::Config,
    srt::Subtitle,
    translate::translate_subtitles_to_file,
    whisper_cli::{transcribe_to_file, ProgressCallback, TranscriptionConfig},
};
use autosub_audio::{cleanup_temp_files, AudioStream};

fn main() -> ExitCode {
    // Initialize FFmpeg
    if let Err(e) = autosub_audio::init() {
        eprintln!("Failed to initialize FFmpeg: {}", e);
        return ExitCode::FAILURE;
    }

    // Clean up any orphaned temp files from previous runs that were killed
    cleanup_temp_files();

    let config = Config::parse();

    // Set up logging
    let filter = if config.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    // Run the async main
    let runtime = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");

    match runtime.block_on(run(config)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("Error: {:#}", e);
            ExitCode::FAILURE
        }
    }
}

async fn run(mut config: Config) -> Result<()> {
    // Validate input file
    if !config.input.exists() {
        anyhow::bail!("Input file does not exist: {}", config.input.display());
    }

    info!("Processing: {}", config.input.display());

    // Auto-detect SRT files and enable translate-only mode
    if !config.translate_only {
        if let Some(ext) = config.input.extension() {
            if ext.eq_ignore_ascii_case("srt") {
                if config.translate.is_none() {
                    anyhow::bail!(
                        "Input is an SRT file. Please specify target language with --translate <LANG>"
                    );
                }
                info!("Detected SRT file, enabling translate-only mode");
                config.translate_only = true;
            }
        }
    }

    // Handle translate-only mode
    if config.translate_only {
        return run_translate_only(&config).await;
    }

    // Step 1: Open audio stream from input file
    let audio_stream = AudioStream::open(&config.input, None)
        .context("Failed to open audio stream from input file")?;

    info!("Audio duration: {:.2} seconds", audio_stream.duration_secs());

    // Step 2: Transcribe with Whisper (fully blocking, no async)
    let output_path = config.output_path();
    info!("Transcribing to: {}", output_path.display());

    let transcription_config = TranscriptionConfig {
        model_size: config.model,
        cache_dir: Some(config.cache_dir()),
        device: config.device.to_candle_device()?,
        language: config.language.clone(),
        enable_vad: config.enable_vad,
        vad_silence_secs: config.vad_reset_secs,
        enable_hallucination_filter: true,
    };

    let progress_bar = create_progress_bar("Transcribing");
    let progress_callback = ProgressBarCallback::new(progress_bar);
    let output_path_clone = output_path.clone();

    // Run transcription in a blocking task since it's fully synchronous
    let subtitle = tokio::task::spawn_blocking(move || {
        transcribe_to_file(
            audio_stream,
            &output_path_clone,
            transcription_config,
            Some(Box::new(progress_callback)),
        )
    })
    .await
    .context("Transcription task failed")??;

    info!("Transcription complete: {} segments written to {}", subtitle.len(), output_path.display());

    // Step 3: Translate if requested (this needs async for HTTP requests)
    if let Some(ref target_lang) = config.translate {
        translate_subtitle(&subtitle, target_lang, &config).await?;
    }

    info!("Done!");
    Ok(())
}

/// Run translate-only mode: read existing SRT and translate it
async fn run_translate_only(config: &Config) -> Result<()> {
    let target_lang = config
        .translate
        .as_ref()
        .context("--translate is required when using --translate-only")?;

    info!("Translate-only mode: reading existing SRT file");

    // Read the existing SRT file
    let mut subtitle = Subtitle::from_file(&config.input).context("Failed to read SRT file")?;

    // Merge consecutive entries with same text before translation
    subtitle.merge_consecutive(0.1);

    info!("Loaded {} subtitle entries", subtitle.len());

    // Translate
    translate_subtitle(&subtitle, target_lang, config).await?;

    info!("Done!");
    Ok(())
}

/// Translate subtitle and save to file (streaming output)
async fn translate_subtitle(subtitle: &Subtitle, target_lang: &str, config: &Config) -> Result<()> {
    let api_key = config.llm_api_key.as_ref().context(
        "LLM API key required for translation. Set --llm-api-key or AUTOSUB_LLM_API_KEY",
    )?;

    let translated_path = config
        .translated_output_path()
        .context("Could not determine output path for translated subtitles")?;

    info!(
        "Translating to {} using {:?} (streaming to {})...",
        target_lang,
        config.llm_provider,
        translated_path.display()
    );

    translate_subtitles_to_file(
        subtitle,
        &translated_path,
        target_lang,
        config.llm_provider,
        api_key,
        &config.llm_model,
        config.llm_url.as_deref(),
        config.translation_batch_size,
    )
    .await
    .context("Failed to translate subtitles")?;

    info!(
        "Saved translated subtitles to: {}",
        translated_path.display()
    );

    Ok(())
}

/// Progress bar implementation for transcription progress
struct ProgressBarCallback {
    progress_bar: ProgressBar,
}

impl ProgressBarCallback {
    fn new(progress_bar: ProgressBar) -> Self {
        Self { progress_bar }
    }
}

impl ProgressCallback for ProgressBarCallback {
    fn on_progress(&self, position: u64, total: u64) {
        if self.progress_bar.length().is_none() || self.progress_bar.length() == Some(100) {
            self.progress_bar.set_length(total);
        }
        self.progress_bar.set_position(position);
    }

    fn on_complete(&self) {
        self.progress_bar.finish_with_message("Transcription complete");
    }
}

fn create_progress_bar(message: &str) -> ProgressBar {
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:43}] {percent}%")
            .unwrap()
            .progress_chars("█░"),
    );
    pb.set_message(message.to_string());
    pb
}
