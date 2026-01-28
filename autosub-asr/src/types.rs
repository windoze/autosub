use serde::{Deserialize, Serialize};

/// An audio clip to be transcribed.
/// Contains audio samples and metadata about timing.
#[derive(Debug, Clone)]
pub struct AudioClip {
    /// Audio samples as f32, normalized to [-1, 1], mono channel at 16kHz
    pub samples: Vec<f32>,

    /// Start position of this clip in the original audio stream (in samples)
    pub start_sample: usize,

    /// End position of this clip in the original audio stream (in samples)
    pub end_sample: usize,

    /// Whether to reset the ASR context before processing this clip.
    /// Set to true after long silence periods to prevent context leakage.
    pub reset_context: bool,
}

impl AudioClip {
    /// Create a new audio clip
    pub fn new(
        samples: Vec<f32>,
        start_sample: usize,
        end_sample: usize,
        reset_context: bool,
    ) -> Self {
        Self {
            samples,
            start_sample,
            end_sample,
            reset_context,
        }
    }

    /// Get the duration of this clip in seconds (at 16kHz sample rate)
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / 16000.0
    }

    /// Get the start time in seconds (at 16kHz sample rate)
    pub fn start_time_secs(&self) -> f64 {
        self.start_sample as f64 / 16000.0
    }

    /// Get the end time in seconds (at 16kHz sample rate)
    pub fn end_time_secs(&self) -> f64 {
        self.end_sample as f64 / 16000.0
    }
}

/// A transcription result from the ASR engine.
/// Contains transcribed text with timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Transcribed text
    pub text: String,

    /// Start time in seconds
    pub start: f64,

    /// End time in seconds
    pub end: f64,
}

impl TranscriptionResult {
    /// Create a new transcription result
    pub fn new(text: String, start: f64, end: f64) -> Self {
        Self { text, start, end }
    }
}

/// Input message for the ASR engine
#[derive(Debug, Clone)]
pub enum AsrInput {
    /// Raw audio samples (f32 mono @ 16kHz) to be processed through VAD
    Samples(Vec<f32>),
    /// Explicit flush signal to emit any buffered speech
    /// (e.g., on PTT button release, end-of-stream)
    Flush,
}
