use std::path::PathBuf;

use anyhow::Result;
use autosub_asr::{
    AsrEngine, AsrInput, TranscriptionResult, VadConfig, VadMode, WhisperModel,
    WhisperModelConfig, WhisperModelSize,
};
use candle_core::Device;
use tokio::sync::mpsc;

/// Load WAV file and convert to f32 samples
fn load_wav_samples(path: &str) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    println!(
        "WAV file: {} Hz, {} channels, {} bits",
        spec.sample_rate, spec.channels, spec.bits_per_sample
    );

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
    };

    Ok(samples)
}

#[test]
fn test_asr_engine_short_audio() -> Result<()> {
    // Set up logging for debugging
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    // Load test audio
    let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("test_short.wav");

    println!("Loading test file: {:?}", test_file);
    let samples = load_wav_samples(test_file.to_str().unwrap())?;
    println!("Loaded {} samples", samples.len());

    // Load Whisper model (use tiny for faster tests)
    let device = if cfg!(target_os = "macos") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Tiny,
        cache_dir: None,
        device,
    };

    println!("Loading Whisper Tiny model...");
    let model = WhisperModel::load(config)?;
    println!("Model loaded successfully");

    // Configure VAD
    let vad_config = VadConfig {
        sample_rate: 16000,
        frame_duration_ms: 30,
        mode: VadMode::Aggressive,
        silence_reset_secs: 1.0,
    };

    // Create channels
    let (input_tx, input_rx) = mpsc::channel::<AsrInput>(100);
    let (output_tx, mut output_rx) = mpsc::channel::<TranscriptionResult>(100);

    // Create ASR engine
    let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;
    println!("ASR engine created");

    // Run engine in a separate thread
    let engine_handle = std::thread::spawn(move || {
        println!("ASR engine thread started");
        engine.run(input_rx, output_tx)
    });

    // Send audio samples
    println!("Sending {} samples to ASR engine", samples.len());
    input_tx.blocking_send(AsrInput::Samples(samples))?;
    input_tx.blocking_send(AsrInput::Flush)?;
    drop(input_tx); // Close channel to signal end

    // Collect results
    let mut results = Vec::new();
    while let Some(result) = output_rx.blocking_recv() {
        println!(
            "Result: [{:.2}s - {:.2}s] {}",
            result.start, result.end, result.text
        );
        results.push(result);
    }

    // Wait for engine to finish
    engine_handle.join().unwrap()?;

    // Verify we got some results
    assert!(
        !results.is_empty(),
        "Expected some transcription results, got none"
    );

    // Verify the transcription contains expected words
    let full_text: String = results.iter().map(|r| r.text.as_str()).collect();
    println!("\n========================================");
    println!("TRANSCRIPTION RESULT:");
    println!("\"{}\"", full_text);
    println!("========================================\n");

    // The audio says "Hello world, this is a test"
    // Allow some flexibility in transcription
    let lower = full_text.to_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("world") || lower.contains("test"),
        "Expected transcription to contain 'hello', 'world', or 'test', got: {}",
        full_text
    );

    Ok(())
}

#[test]
fn test_asr_engine_longer_audio() -> Result<()> {
    // Set up logging
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    // Load test audio
    let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("test_longer.wav");

    println!("Loading test file: {:?}", test_file);
    let samples = load_wav_samples(test_file.to_str().unwrap())?;
    println!("Loaded {} samples", samples.len());

    // Load Whisper model
    let device = if cfg!(target_os = "macos") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Tiny,
        cache_dir: None,
        device,
    };

    println!("Loading Whisper Tiny model...");
    let model = WhisperModel::load(config)?;

    // Configure VAD
    let vad_config = VadConfig {
        sample_rate: 16000,
        frame_duration_ms: 30,
        mode: VadMode::Aggressive,
        silence_reset_secs: 1.0,
    };

    // Create channels
    let (input_tx, input_rx) = mpsc::channel::<AsrInput>(100);
    let (output_tx, mut output_rx) = mpsc::channel::<TranscriptionResult>(100);

    // Create and run ASR engine
    let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;

    let engine_handle = std::thread::spawn(move || engine.run(input_rx, output_tx));

    // Send audio samples
    println!("Sending samples to ASR engine");
    input_tx.blocking_send(AsrInput::Samples(samples))?;
    input_tx.blocking_send(AsrInput::Flush)?;
    drop(input_tx);

    // Collect results
    let mut results = Vec::new();
    while let Some(result) = output_rx.blocking_recv() {
        println!(
            "Result: [{:.2}s - {:.2}s] {}",
            result.start, result.end, result.text
        );
        results.push(result);
    }

    engine_handle.join().unwrap()?;

    // Verify results
    assert!(!results.is_empty(), "Expected transcription results");

    let full_text: String = results.iter().map(|r| r.text.as_str()).collect();
    println!("\n========================================");
    println!("TRANSCRIPTION RESULT:");
    println!("\"{}\"", full_text);
    println!("========================================\n");

    // The audio contains "quick brown fox"
    let lower = full_text.to_lowercase();
    assert!(
        lower.contains("quick") || lower.contains("brown") || lower.contains("fox"),
        "Expected transcription to contain words from test phrase, got: {}",
        full_text
    );

    Ok(())
}

#[test]
fn test_vad_segmentation() -> Result<()> {
    // Test that VAD properly segments speech
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("test_longer.wav");

    let samples = load_wav_samples(test_file.to_str().unwrap())?;

    let device = if cfg!(target_os = "macos") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Tiny,
        cache_dir: None,
        device,
    };

    let model = WhisperModel::load(config)?;

    // Use aggressive VAD to ensure segmentation
    let vad_config = VadConfig {
        sample_rate: 16000,
        frame_duration_ms: 30,
        mode: VadMode::Aggressive,
        silence_reset_secs: 0.5, // Shorter silence threshold for more segments
    };

    let (input_tx, input_rx) = mpsc::channel::<AsrInput>(100);
    let (output_tx, mut output_rx) = mpsc::channel::<TranscriptionResult>(100);

    let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;
    let engine_handle = std::thread::spawn(move || engine.run(input_rx, output_tx));

    input_tx.blocking_send(AsrInput::Samples(samples))?;
    input_tx.blocking_send(AsrInput::Flush)?;
    drop(input_tx);

    let mut results = Vec::new();
    while let Some(result) = output_rx.blocking_recv() {
        println!("Segment: [{:.2}s - {:.2}s]", result.start, result.end);
        results.push(result);
    }

    engine_handle.join().unwrap()?;

    // With a longer audio file and VAD, we should get multiple segments
    println!("Got {} segments", results.len());
    assert!(
        !results.is_empty(),
        "Expected at least one segment from VAD"
    );

    // Verify timestamps are monotonically increasing
    for i in 1..results.len() {
        assert!(
            results[i].start >= results[i - 1].end,
            "Segment timestamps should be monotonically increasing"
        );
    }

    Ok(())
}
