use std::path::{Path, PathBuf};

use tracing::info;

extern crate ffmpeg_next as ffmpeg;

use crate::error::{AudioError, Result};
use crate::types::FileInfo;

/// A handle to an extracted audio file in a temp directory.
/// The file is automatically cleaned up when this struct is dropped.
pub struct ExtractedAudio {
    path: PathBuf,
    file_info: FileInfo,
}

impl ExtractedAudio {
    /// Get the path to the extracted WAV file
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get file information
    pub fn file_info(&self) -> &FileInfo {
        &self.file_info
    }

    /// Get the duration of the audio in seconds
    pub fn duration_secs(&self) -> f64 {
        self.file_info.duration_secs
    }
}

impl Drop for ExtractedAudio {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Extract/convert audio from a media file to a temporary WAV file.
/// This is useful when you need a complete audio file rather than streaming.
///
/// Works with both video files (extracts audio track) and audio files (converts format).
/// The returned `ExtractedAudio` handle will automatically clean up the temp file when dropped.
///
/// # Arguments
/// * `input` - Path to input media file
/// * `sample_rate` - Target sample rate (default: 16000 if None)
/// * `progress` - Optional progress callback: fn(current_microseconds, total_microseconds)
///
/// # Returns
/// ExtractedAudio handle that auto-cleans temp file on drop
pub fn extract_audio(
    input: impl AsRef<Path>,
    sample_rate: Option<u32>,
    progress: Option<Box<dyn Fn(u64, u64) + Send>>,
) -> Result<ExtractedAudio> {
    let input = input.as_ref();
    let sample_rate = sample_rate.unwrap_or(16000);

    // Create a temporary WAV file
    let temp_dir = std::env::temp_dir();
    let temp_wav = temp_dir.join(format!("autosub_audio_{}.wav", std::process::id()));

    info!("Extracting audio to: {}", temp_wav.display());

    // Initialize ffmpeg (safe to call multiple times)
    ffmpeg::init().map_err(|e| AudioError::FfmpegInit(e.to_string()))?;

    // Suppress ffmpeg warnings
    ffmpeg::log::set_level(ffmpeg::log::Level::Error);

    // Open input file
    let mut ictx = ffmpeg::format::input(input)
        .map_err(|e| AudioError::Decode(format!("Failed to open input file: {}", e)))?;

    // Get duration for progress tracking (in microseconds)
    let duration_us = ictx.duration();
    let total_duration_us = if duration_us > 0 {
        duration_us as u64
    } else {
        0
    };

    // Check if there are video streams
    let has_video = ictx.streams().best(ffmpeg::media::Type::Video).is_some();

    // Find the best audio stream
    let audio_stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or(AudioError::NoAudioStream)?
        .index();

    let audio_stream = ictx.stream(audio_stream_index).unwrap();
    let time_base = audio_stream.time_base();
    let audio_params = audio_stream.parameters();

    // Create decoder for the audio stream
    let decoder_context = ffmpeg::codec::context::Context::from_parameters(audio_params)
        .map_err(|e| AudioError::Decode(format!("Failed to create decoder context: {}", e)))?;
    let mut decoder = decoder_context
        .decoder()
        .audio()
        .map_err(|e| AudioError::Decode(format!("Failed to create audio decoder: {}", e)))?;

    // Get source channel count from decoder
    let source_channels = decoder.channels();

    info!(
        "Input audio: {} Hz, {} channels",
        decoder.rate(),
        decoder.channels()
    );

    // Set up resampler to convert to target sample rate, mono, i16
    let mut resampler = ffmpeg::software::resampling::context::Context::get(
        decoder.format(),
        decoder.channel_layout(),
        decoder.rate(),
        ffmpeg::format::Sample::I16(ffmpeg::format::sample::Type::Packed),
        ffmpeg::ChannelLayout::MONO,
        sample_rate,
    )
    .map_err(|e| AudioError::Resample(format!("Failed to create audio resampler: {}", e)))?;

    // Collect all audio samples
    let mut all_samples: Vec<i16> = Vec::new();

    // Process packets
    for (stream, packet) in ictx.packets() {
        if stream.index() == audio_stream_index {
            // Update progress based on packet timestamp
            if let (Some(ref callback), Some(pts)) = (&progress, packet.pts()) {
                // Convert pts to microseconds
                let time_us =
                    pts * 1_000_000 * time_base.numerator() as i64 / time_base.denominator() as i64;
                if time_us > 0 && total_duration_us > 0 {
                    callback(time_us as u64, total_duration_us);
                }
            }

            decoder.send_packet(&packet).ok();

            let mut decoded_frame = ffmpeg::frame::Audio::empty();
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                // Resample the frame
                let mut resampled_frame = ffmpeg::frame::Audio::empty();
                resampler
                    .run(&decoded_frame, &mut resampled_frame)
                    .map_err(|e| {
                        AudioError::Resample(format!("Failed to resample audio frame: {}", e))
                    })?;

                // Extract samples from the resampled frame
                if resampled_frame.samples() > 0 {
                    let data = resampled_frame.data(0);
                    let samples: &[i16] = bytemuck::cast_slice(data);
                    all_samples.extend_from_slice(&samples[..resampled_frame.samples()]);
                }
            }
        }
    }

    // Flush the decoder
    decoder.send_eof().ok();
    let mut decoded_frame = ffmpeg::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let mut resampled_frame = ffmpeg::frame::Audio::empty();
        if resampler.run(&decoded_frame, &mut resampled_frame).is_ok()
            && resampled_frame.samples() > 0
        {
            let data = resampled_frame.data(0);
            let samples: &[i16] = bytemuck::cast_slice(data);
            all_samples.extend_from_slice(&samples[..resampled_frame.samples()]);
        }
    }

    // Flush the resampler (get any remaining samples)
    loop {
        let mut resampled_frame = ffmpeg::frame::Audio::empty();
        match resampler.flush(&mut resampled_frame) {
            Ok(_) if resampled_frame.samples() > 0 => {
                let data = resampled_frame.data(0);
                let samples: &[i16] = bytemuck::cast_slice(data);
                all_samples.extend_from_slice(&samples[..resampled_frame.samples()]);
            }
            _ => break,
        }
    }

    info!("Total samples: {}", all_samples.len());

    // Write to WAV file using hound
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&temp_wav, spec)?;

    for sample in &all_samples {
        writer.write_sample(*sample)?;
    }

    writer.finalize()?;

    let duration_secs = all_samples.len() as f64 / sample_rate as f64;

    let file_info = FileInfo {
        duration_secs,
        sample_rate,
        channels: source_channels,
        total_samples: all_samples.len(),
        has_video,
    };

    info!("Extraction complete: {:.2}s", duration_secs);

    Ok(ExtractedAudio {
        path: temp_wav,
        file_info,
    })
}
