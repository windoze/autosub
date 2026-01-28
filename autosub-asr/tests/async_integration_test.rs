use std::path::PathBuf;

use anyhow::Result;
use autosub_asr::{
    AsrEngine, AsrInput, TranscriptionResult, VadConfig, VadMode, WhisperModel,
    WhisperModelConfig, WhisperModelSize,
};
use candle_core::Device;
use indicatif::ProgressBar;
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

/// Test that mimics the CLI pattern: async runtime + std::thread + Metal
/// This test spawns the ASR engine in a std::thread (like CLI does)
/// and uses async channels to communicate (like CLI does)
#[tokio::test]
async fn test_async_runtime_with_metal() -> Result<()> {
    // Set up logging
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

    // Load Whisper model with Metal if available
    let device = if cfg!(target_os = "macos") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    println!("Using device: {:?}", device);

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

    // Create async channels (like CLI does)
    let (input_tx, input_rx) = mpsc::channel::<AsrInput>(1000);
    let (output_tx, mut output_rx) = mpsc::channel::<TranscriptionResult>(1000);

    // Create ASR engine
    let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;
    println!("ASR engine created");

    // Spawn engine in std::thread (like CLI does)
    println!("Spawning ASR engine in std::thread");
    let engine_handle = std::thread::spawn(move || {
        println!("ASR engine thread started");
        engine.run(input_rx, output_tx)
    });

    // Spawn audio sender in std::thread (like CLI does)
    let sender_handle = std::thread::spawn(move || {
        println!("Sending {} samples to ASR engine", samples.len());
        input_tx.blocking_send(AsrInput::Samples(samples))?;
        input_tx.blocking_send(AsrInput::Flush)?;
        drop(input_tx); // Close channel
        println!("Audio sender finished");
        Ok::<(), anyhow::Error>(())
    });

    // Collect results in async task (like CLI does)
    let mut results = Vec::new();
    println!("Starting to collect results asynchronously");
    while let Some(result) = output_rx.recv().await {
        println!(
            "Received result: [{:.2}s - {:.2}s] {}",
            result.start, result.end, result.text
        );
        results.push(result);
    }
    println!("Result collection complete");

    // Wait for threads to complete
    tokio::task::spawn_blocking(move || {
        sender_handle.join().unwrap()?;
        engine_handle.join().unwrap()?;
        Ok::<(), anyhow::Error>(())
    })
    .await??;

    // Verify results
    assert!(
        !results.is_empty(),
        "Expected some transcription results, got none"
    );

    let full_text: String = results.iter().map(|r| r.text.as_str()).collect();
    println!("\n========================================");
    println!("TRANSCRIPTION RESULT:");
    println!("\"{}\"", full_text);
    println!("========================================\n");

    let lower = full_text.to_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("world") || lower.contains("test"),
        "Expected transcription to contain expected words, got: {}",
        full_text
    );

    println!("Test passed!");
    Ok(())
}

/// Test with longer audio to stress test the async/Metal interaction
#[tokio::test]
async fn test_async_runtime_with_metal_longer() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("test_longer.wav");

    println!("Loading test file: {:?}", test_file);
    let samples = load_wav_samples(test_file.to_str().unwrap())?;
    println!("Loaded {} samples", samples.len());

    let device = if cfg!(target_os = "macos") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    println!("Using device: {:?}", device);

    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Tiny,
        cache_dir: None,
        device,
    };

    let model = WhisperModel::load(config)?;

    let vad_config = VadConfig {
        sample_rate: 16000,
        frame_duration_ms: 30,
        mode: VadMode::Aggressive,
        silence_reset_secs: 1.0,
    };

    let (input_tx, input_rx) = mpsc::channel::<AsrInput>(1000);
    let (output_tx, mut output_rx) = mpsc::channel::<TranscriptionResult>(1000);

    let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;

    // Spawn in std::thread
    let engine_handle = std::thread::spawn(move || engine.run(input_rx, output_tx));

    let sender_handle = std::thread::spawn(move || {
        input_tx.blocking_send(AsrInput::Samples(samples))?;
        input_tx.blocking_send(AsrInput::Flush)?;
        drop(input_tx);
        Ok::<(), anyhow::Error>(())
    });

    // Collect results asynchronously
    let mut results = Vec::new();
    while let Some(result) = output_rx.recv().await {
        println!("Result: [{:.2}s - {:.2}s]", result.start, result.end);
        results.push(result);
    }

    tokio::task::spawn_blocking(move || {
        sender_handle.join().unwrap()?;
        engine_handle.join().unwrap()?;
        Ok::<(), anyhow::Error>(())
    })
    .await??;

    assert!(!results.is_empty(), "Expected transcription results");

    let full_text: String = results.iter().map(|r| r.text.as_str()).collect();
    println!("\n========================================");
    println!("TRANSCRIPTION RESULT:");
    println!("\"{}\"", full_text);
    println!("Got {} segments", results.len());
    println!("========================================\n");

    Ok(())
}

/// Test with ProgressBar like the CLI uses
#[tokio::test]
async fn test_async_with_progressbar_and_metal() -> Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init();

    let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("test_short.wav");

    println!("Loading test file with ProgressBar: {:?}", test_file);
    let samples = load_wav_samples(test_file.to_str().unwrap())?;
    println!("Loaded {} samples", samples.len());

    let device = if cfg!(target_os = "macos") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    println!("Using device: {:?}", device);

    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Tiny,
        cache_dir: None,
        device,
    };

    let model = WhisperModel::load(config)?;

    let vad_config = VadConfig {
        sample_rate: 16000,
        frame_duration_ms: 30,
        mode: VadMode::Aggressive,
        silence_reset_secs: 1.0,
    };

    let (input_tx, input_rx) = mpsc::channel::<AsrInput>(1000);
    let (output_tx, mut output_rx) = mpsc::channel::<TranscriptionResult>(1000);

    let engine = AsrEngine::new(model, Some("en".to_string()), vad_config)?;

    // Create ProgressBar (like CLI)
    let pb = ProgressBar::new(samples.len() as u64);
    println!("Created ProgressBar");

    // Spawn engine in std::thread
    let engine_handle = std::thread::spawn(move || {
        println!("ASR engine with ProgressBar started");
        engine.run(input_rx, output_tx)
    });

    // Spawn sender with ProgressBar updates
    let sender_handle = std::thread::spawn(move || {
        println!("Sending samples with ProgressBar updates");
        pb.set_position(0);
        input_tx.blocking_send(AsrInput::Samples(samples))?;
        pb.set_position(pb.length().unwrap());
        input_tx.blocking_send(AsrInput::Flush)?;
        drop(input_tx);
        pb.finish_with_message("Complete");
        println!("Audio sender with ProgressBar finished");
        Ok::<(), anyhow::Error>(())
    });

    // Collect results asynchronously
    let mut results = Vec::new();
    while let Some(result) = output_rx.recv().await {
        println!(
            "Result: [{:.2}s - {:.2}s] {}",
            result.start, result.end, result.text
        );
        results.push(result);
    }

    tokio::task::spawn_blocking(move || {
        sender_handle.join().unwrap()?;
        engine_handle.join().unwrap()?;
        Ok::<(), anyhow::Error>(())
    })
    .await??;

    assert!(!results.is_empty());

    let full_text: String = results.iter().map(|r| r.text.as_str()).collect();
    println!("\n========================================");
    println!("TRANSCRIPTION RESULT (with ProgressBar):");
    println!("\"{}\"", full_text);
    println!("========================================\n");
    println!("ProgressBar test passed!");

    Ok(())
}
