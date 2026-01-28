use autosub_asr::{AudioClip, WhisperModel, WhisperModelConfig, WhisperModelSize};
use ort::execution_providers::{CoreMLExecutionProvider, CPUExecutionProvider};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // Load a simple sine wave for testing (1 second at 16kHz)
    let sample_rate = 16000;
    let duration = 1.0;
    let frequency = 440.0; // A4 note

    let samples: Vec<f32> = (0..((sample_rate as f32 * duration) as usize))
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.1
        })
        .collect();

    println!("Generated {} samples", samples.len());

    // Load model
    let execution_providers = if cfg!(target_os = "macos") {
        vec![
            CoreMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ]
    } else {
        vec![CPUExecutionProvider::default().build()]
    };

    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Tiny,
        cache_dir: None,
        execution_providers,
    };

    println!("Loading Whisper model...");
    let mut model = WhisperModel::load(config)?;
    println!("Model loaded");

    // Create audio clip
    let clip = AudioClip {
        samples,
        start_time: 0.0,
        reset_context: false,
    };

    println!("Transcribing...");
    let results = model.transcribe_clip(&clip, Some("en"), None, None)?;

    println!("\nResults: {} segments", results.len());
    for (i, result) in results.iter().enumerate() {
        println!("  Segment {}: [{:.2}s - {:.2}s] '{}'",
            i + 1, result.start, result.end, result.text);
    }

    Ok(())
}
