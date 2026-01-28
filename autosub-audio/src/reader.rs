use std::path::Path;

use tracing::info;

use crate::error::Result;
use crate::types::{AudioSegment, FileInfo};

/// Audio chunk reader for streaming transcription.
/// Reads audio in fixed-size chunks without loading the entire file into memory.
pub struct AudioChunkReader {
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
    chunk_samples: usize,
    total_samples: u32,
    samples_read: u32,
    sample_rate: u32,
}

impl AudioChunkReader {
    /// Open a WAV file for streaming chunk reading.
    ///
    /// # Arguments
    /// * `path` - Path to the WAV file
    /// * `chunk_duration_secs` - Duration of each chunk in seconds
    pub fn open(path: impl AsRef<Path>, chunk_duration_secs: usize) -> Result<Self> {
        let path = path.as_ref();
        let reader = hound::WavReader::open(path)?;

        let spec = reader.spec();
        info!(
            "WAV file: {} Hz, {} channels, {} bits",
            spec.sample_rate, spec.channels, spec.bits_per_sample
        );

        let chunk_samples = chunk_duration_secs * spec.sample_rate as usize;
        let total_samples = reader.len();
        let sample_rate = spec.sample_rate;

        Ok(Self {
            reader,
            chunk_samples,
            total_samples,
            samples_read: 0,
            sample_rate,
        })
    }

    /// Get file information
    pub fn file_info(&self) -> FileInfo {
        let spec = self.reader.spec();
        FileInfo {
            duration_secs: self.total_samples as f64 / self.sample_rate as f64,
            sample_rate: self.sample_rate,
            channels: spec.channels,
            total_samples: self.total_samples as usize,
            has_video: false, // WAV files don't have video
        }
    }

    /// Get the total number of samples in the file
    pub fn total_samples(&self) -> u32 {
        self.total_samples
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the total duration in seconds
    pub fn duration_secs(&self) -> f64 {
        self.total_samples as f64 / self.sample_rate as f64
    }

    /// Get the number of chunks that will be returned
    pub fn num_chunks(&self) -> usize {
        (self.total_samples as usize).div_ceil(self.chunk_samples)
    }

    /// Read the next chunk of audio samples as AudioSegment.
    /// Returns None when all samples have been read.
    pub fn next_chunk(&mut self) -> Result<Option<AudioSegment>> {
        if self.samples_read >= self.total_samples {
            return Ok(None);
        }

        let spec = self.reader.spec();
        let max_value = (1i32 << (spec.bits_per_sample - 1)) as f32;
        let remaining = (self.total_samples - self.samples_read) as usize;
        let to_read = remaining.min(self.chunk_samples);

        let mut samples = Vec::with_capacity(to_read);

        let start_sample = self.samples_read as usize;

        match spec.sample_format {
            hound::SampleFormat::Int => {
                for _ in 0..to_read {
                    if let Some(sample) = self.reader.samples::<i32>().next() {
                        let value = sample?;
                        samples.push(value as f32 / max_value);
                        self.samples_read += 1;
                    } else {
                        break;
                    }
                }
            }
            hound::SampleFormat::Float => {
                for _ in 0..to_read {
                    if let Some(sample) = self.reader.samples::<f32>().next() {
                        let value = sample?;
                        samples.push(value);
                        self.samples_read += 1;
                    } else {
                        break;
                    }
                }
            }
        }

        if samples.is_empty() {
            Ok(None)
        } else {
            let end_sample = self.samples_read as usize;
            let segment = AudioSegment::new(samples, start_sample, end_sample, self.sample_rate);
            Ok(Some(segment))
        }
    }
}

/// Iterator adapter for AudioChunkReader
impl Iterator for AudioChunkReader {
    type Item = Result<AudioSegment>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(segment)) => Some(Ok(segment)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
