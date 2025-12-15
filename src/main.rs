use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use autosub::{
    audio::{cleanup_orphaned_temp_files, extract_audio, is_audio_file},
    config::Config,
    srt::Subtitle,
    translate::translate_subtitles_to_file,
    whisper::transcribe_to_file,
};

fn main() -> ExitCode {
    // Clean up any orphaned temp files from previous runs that were killed
    cleanup_orphaned_temp_files();

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

    // Step 1: Extract/convert audio to temp file
    let is_audio = is_audio_file(&config.input);
    let progress_msg = if is_audio {
        "Converting audio"
    } else {
        "Extracting audio"
    };
    let progress = create_progress_bar(progress_msg);
    let extracted_audio = extract_audio(&config.input, Some(&progress))
        .context("Failed to process audio from input file")?;
    let done_msg = if is_audio {
        format!("Audio converted ({:.2} seconds)", extracted_audio.duration_secs())
    } else {
        format!("Audio extracted ({:.2} seconds)", extracted_audio.duration_secs())
    };
    // Set position to length to ensure bar is fully filled before finishing
    if let Some(len) = progress.length() {
        progress.set_position(len);
    }
    progress.finish();
    println!("{}", done_msg);

    // Step 2: Transcribe with Whisper using streaming, writing SRT as we go
    let output_path = config.output_path();
    let device = config.device.to_candle_device()?;

    info!("Transcribing to: {}", output_path.display());
    let subtitle = transcribe_to_file(
        extracted_audio.path(),
        &output_path,
        config.model,
        Some(config.cache_dir()),
        device,
        config.language.as_deref(),
        || Some(create_progress_bar("Transcribing")),
    )
    .context("Failed to transcribe audio")?;

    info!("Transcription complete: {} segments written to {}", subtitle.len(), output_path.display());

    // The extracted_audio temp file is automatically cleaned up when it goes out of scope

    // Step 3: Translate if requested
    if let Some(ref target_lang) = config.translate {
        translate_subtitle(&subtitle, target_lang, &config).await?;
    }

    info!("Done!");
    Ok(())
}

/// Run translate-only mode: read existing SRT and translate it
async fn run_translate_only(config: &Config) -> Result<()> {
    let target_lang = config.translate.as_ref().context(
        "--translate is required when using --translate-only",
    )?;

    info!("Translate-only mode: reading existing SRT file");

    // Read the existing SRT file
    let mut subtitle = Subtitle::from_file(&config.input)
        .context("Failed to read SRT file")?;

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

    let translated_path = config.translated_output_path().context(
        "Could not determine output path for translated subtitles",
    )?;

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

    info!("Saved translated subtitles to: {}", translated_path.display());

    Ok(())
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
