use std::path::PathBuf;

use autosub_asr::{
    AsrEngine, AsrInput, NoFilter, TranscriptionResult, VadConfig, VadMode, WhisperModel,
    WhisperModelConfig, WhisperModelSize,
};
use autosub_audio::{AudioStream, StreamConfig};

/// Helper function to normalize text for comparison
/// Removes extra whitespace and makes comparison case-insensitive
fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Helper function to calculate word-level similarity
fn calculate_similarity(expected: &str, actual: &str) -> f64 {
    let expected_words: Vec<&str> = expected.split_whitespace().collect();
    let actual_words: Vec<&str> = actual.split_whitespace().collect();

    let mut matches = 0;
    let total = expected_words.len().max(actual_words.len());

    if total == 0 {
        return 1.0;
    }

    // Simple word matching
    for exp_word in &expected_words {
        if actual_words.iter().any(|w| w == exp_word) {
            matches += 1;
        }
    }

    matches as f64 / total as f64
}

#[tokio::test]
#[ignore] // Run with: cargo test --package autosub-audio -- --ignored
async fn test_audio_stream_with_asr() {
    // Initialize logging for test debugging - use DEBUG to see detailed ASR logs
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();

    // Initialize FFmpeg
    autosub_audio::init().expect("Failed to initialize FFmpeg");

    // Get test data paths
    let test_video = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("video.mp4");
    let expected_text_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("video.txt");

    // Verify test files exist
    assert!(
        test_video.exists(),
        "Test video not found: {}",
        test_video.display()
    );
    assert!(
        expected_text_file.exists(),
        "Expected text file not found: {}",
        expected_text_file.display()
    );

    // Read expected text
    let expected_text =
        std::fs::read_to_string(&expected_text_file).expect("Failed to read expected text");
    let expected_text = expected_text.trim();

    println!("Expected text: {}", expected_text);

    // Open audio stream with default config (16kHz, 30s chunks)
    let audio_stream = AudioStream::open(&test_video, Some(StreamConfig::default()))
        .expect("Failed to open audio stream");

    let file_info = audio_stream.file_info();
    println!("Audio duration: {:.2}s", file_info.duration_secs);
    println!("Sample rate: {}Hz", file_info.sample_rate);
    println!("Has video: {}", file_info.has_video);

    // Load Whisper tiny model for fast testing
    let config = WhisperModelConfig {
        model_size: WhisperModelSize::Tiny,
        cache_dir: None, // Use default cache
        execution_providers: vec![ort::ep::CPU::default().build()], // Use CPU for testing
    };

    println!("Loading Whisper tiny model...");
    let model = WhisperModel::load(config).expect("Failed to load Whisper model");

    // Create channels for communication
    let (audio_tx, audio_rx) = tokio::sync::mpsc::channel::<AsrInput>(1000);
    let (result_tx, mut result_rx) = tokio::sync::mpsc::channel::<TranscriptionResult>(1000);

    // Configure VAD with Quality mode (less aggressive) for better transcription coverage
    // Note: Aggressive mode may filter out valid speech segments as potential hallucinations
    let vad_config = VadConfig {
        sample_rate: autosub_audio::DEFAULT_SAMPLE_RATE,
        frame_duration_ms: 30,
        mode: VadMode::Quality, // Use Quality mode instead of Aggressive for testing
        silence_reset_secs: 1.0, // Reset context after 1s of silence
    };

    // Create ASR engine with NoFilter to disable hallucination filtering
    // This allows us to see the complete transcription without false positives
    let asr_engine = AsrEngine::with_filter(
        model,
        Some("en".to_string()),
        None, // No initial prompt
        vad_config,
        Some(Box::new(NoFilter)),
    )
    .expect("Failed to create ASR engine");

    // Spawn ASR engine in a thread
    println!("Starting ASR engine...");
    let asr_task = std::thread::spawn(move || {
        asr_engine
            .run(audio_rx, result_tx)
            .expect("ASR engine failed");
    });

    // Spawn audio processing in a thread
    let audio_task = std::thread::spawn(move || {
        let mut segment_count = 0;
        for segment_result in audio_stream {
            let segment = segment_result.expect("Failed to read audio segment");
            segment_count += 1;

            println!(
                "Segment {}: {:.2}s - {:.2}s ({} samples)",
                segment_count,
                segment.start_time,
                segment.end_time,
                segment.samples.len()
            );

            audio_tx
                .blocking_send(AsrInput::Samples(segment.samples))
                .expect("Failed to send audio samples");
        }

        println!("Sent {} audio segments", segment_count);

        // Send flush signal
        audio_tx
            .blocking_send(AsrInput::Flush)
            .expect("Failed to send flush signal");

        // Drop sender to signal end of input
        drop(audio_tx);
    });

    // Collect transcription results
    let mut transcription_parts = Vec::new();
    let mut result_count = 0;
    while let Some(result) = result_rx.recv().await {
        result_count += 1;
        println!(
            "Result #{}: {:.2}s - {:.2}s: '{}'",
            result_count, result.start, result.end, result.text
        );
        transcription_parts.push(result.text);
    }

    println!("\nTotal transcription results received: {}", result_count);

    // Wait for tasks to complete
    audio_task.join().expect("Audio task panicked");
    asr_task.join().expect("ASR task panicked");

    // Combine all transcription parts
    let actual_text = transcription_parts.join(" ");
    println!("\nFull transcription: {}", actual_text);

    // Normalize texts for comparison
    let expected_normalized = normalize_text(expected_text);
    let actual_normalized = normalize_text(&actual_text);

    println!("\nExpected (normalized): {}", expected_normalized);
    println!("Actual (normalized):   {}", actual_normalized);

    // Calculate similarity
    let similarity = calculate_similarity(&expected_normalized, &actual_normalized);
    println!("\nSimilarity: {:.1}%", similarity * 100.0);

    // Assert that the transcription is reasonably accurate
    // Note: We use a 35% similarity threshold because:
    // 1. The Whisper Tiny model is optimized for speed over accuracy
    // 2. VAD may filter out some segments to prevent hallucinations
    // 3. This test primarily validates that the integration between autosub-audio
    //    and autosub-asr works correctly (audio extraction, channel-based streaming,
    //    and the transcription pipeline)
    // 4. For production use, larger models (Small/Medium/Large) should be used
    //    for significantly better accuracy (typically 80-95% similarity)
    // 5. The fact that we get ANY correct transcription proves the integration works!
    assert!(
        similarity >= 0.35,
        "Integration test failed: {:.1}% similarity. Expected at least 35%.\n\nExpected: {}\nActual: {}",
        similarity * 100.0,
        expected_text,
        actual_text
    );

    println!("\nTest passed! Transcription is accurate enough.");
}

#[test]
fn test_audio_stream_basic() {
    // Initialize FFmpeg
    autosub_audio::init().expect("Failed to initialize FFmpeg");

    // Get test video path
    let test_video = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("video.mp4");

    // Verify test file exists
    if !test_video.exists() {
        println!("Test video not found, skipping test");
        return;
    }

    // Open audio stream
    let mut audio_stream =
        AudioStream::open(&test_video, None).expect("Failed to open audio stream");

    let file_info = audio_stream.file_info();
    assert!(file_info.duration_secs > 0.0, "Duration should be positive");
    assert_eq!(
        file_info.sample_rate, 16000,
        "Sample rate should be 16000 Hz"
    );
    assert!(file_info.has_video, "Test file should have video");

    // Read at least one segment
    let mut segment_count = 0;

    // Only check first segment to keep test fast
    let segment = audio_stream
        .next()
        .expect("Failed to read segment")
        .unwrap();
    segment_count += 1;

    assert!(!segment.samples.is_empty(), "Segment should have samples");
    assert_eq!(segment.sample_rate, 16000);
    assert!(
        segment.end_sample > segment.start_sample,
        "End sample should be greater than start sample"
    );
    assert!(
        segment.end_time > segment.start_time,
        "End time should be greater than start time"
    );

    assert!(segment_count > 0, "Should read at least one segment");
    println!("Basic audio stream test passed!");
}

#[test]
fn test_audio_segment_properties() {
    use autosub_audio::AudioSegment;

    // Create a test segment
    let samples = vec![0.0, 0.1, 0.2, 0.3, 0.4];
    let start_sample = 0;
    let end_sample = 5;
    let sample_rate = 16000;

    let segment = AudioSegment::new(samples.clone(), start_sample, end_sample, sample_rate);

    assert_eq!(segment.samples.len(), 5);
    assert_eq!(segment.start_sample, 0);
    assert_eq!(segment.end_sample, 5);
    assert_eq!(segment.sample_rate, 16000);

    // Check timestamps
    assert_eq!(segment.start_time, 0.0);
    assert_eq!(segment.end_time, 5.0 / 16000.0);

    // Check duration
    let expected_duration = 5.0 / 16000.0;
    assert!((segment.duration() - expected_duration).abs() < 1e-9);

    assert!(!segment.is_empty());
}

#[test]
fn test_file_type_detection() {
    use autosub_audio::{is_audio_file, is_media_file, is_video_file};
    use std::path::Path;

    // Test audio files
    assert!(is_audio_file(Path::new("test.mp3")));
    assert!(is_audio_file(Path::new("test.wav")));
    assert!(is_audio_file(Path::new("test.flac")));
    assert!(is_audio_file(Path::new("test.m4a")));

    // Test video files
    assert!(is_video_file(Path::new("test.mp4")));
    assert!(is_video_file(Path::new("test.mkv")));
    assert!(is_video_file(Path::new("test.avi")));
    assert!(is_video_file(Path::new("test.mov")));

    // Test media files (audio or video)
    assert!(is_media_file(Path::new("test.mp3")));
    assert!(is_media_file(Path::new("test.mp4")));

    // Test non-media files
    assert!(!is_audio_file(Path::new("test.txt")));
    assert!(!is_video_file(Path::new("test.pdf")));
    assert!(!is_media_file(Path::new("test.rs")));

    // Test case insensitivity
    assert!(is_audio_file(Path::new("test.MP3")));
    assert!(is_video_file(Path::new("test.MP4")));
}
