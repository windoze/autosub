/// A segment of audio with samples and timing metadata
#[derive(Debug, Clone)]
pub struct AudioSegment {
    /// Audio samples as f32, normalized to [-1, 1], mono channel
    pub samples: Vec<f32>,

    /// Start time in seconds
    pub start_time: f64,

    /// End time in seconds
    pub end_time: f64,

    /// Sample rate (typically 16000 Hz)
    pub sample_rate: u32,

    /// Start position in samples (absolute position from file start)
    pub start_sample: usize,

    /// End position in samples (absolute position from file start)
    pub end_sample: usize,
}

impl AudioSegment {
    /// Create a new AudioSegment
    pub fn new(samples: Vec<f32>, start_sample: usize, end_sample: usize, sample_rate: u32) -> Self {
        let start_time = start_sample as f64 / sample_rate as f64;
        let end_time = end_sample as f64 / sample_rate as f64;

        Self {
            samples,
            start_time,
            end_time,
            sample_rate,
            start_sample,
            end_sample,
        }
    }

    /// Get the duration of this segment in seconds
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Check if this segment is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Metadata about a media file
#[derive(Debug, Clone)]
pub struct FileInfo {
    /// Total duration in seconds
    pub duration_secs: f64,

    /// Sample rate of the audio stream (after conversion)
    pub sample_rate: u32,

    /// Number of audio channels in source
    pub channels: u16,

    /// Total samples (at target sample rate after conversion)
    pub total_samples: usize,

    /// Whether the file has video streams
    pub has_video: bool,
}

/// Configuration for audio streaming
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Target sample rate (default: 16000)
    pub sample_rate: u32,

    /// Chunk duration in seconds (default: 30)
    pub chunk_duration_secs: usize,

    /// Whether to normalize samples to [-1, 1] (default: true)
    pub normalize: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            chunk_duration_secs: 30,
            normalize: true,
        }
    }
}

/// How to handle channel send operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendMode {
    /// Block until send succeeds
    Blocking,

    /// Try to send, return error on full channel
    NonBlocking,

    /// Try to send, drop oldest on full channel (useful for real-time)
    DropOldest,
}
