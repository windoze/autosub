// Audio preprocessing for Whisper model
// Implements mel spectrogram extraction compatible with OpenAI's Whisper preprocessing

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

// Whisper audio preprocessing constants
const N_FFT: usize = 400; // FFT size
const HOP_LENGTH: usize = 160; // 10ms hop at 16kHz
const N_SAMPLES: usize = 480000; // 30 seconds at 16kHz
const CHUNK_LENGTH: usize = N_SAMPLES;

/// Whisper model configuration (subset needed for audio preprocessing)
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub num_mel_bins: usize,
}

/// Convert PCM samples to mel spectrogram
///
/// # Arguments
/// * `config` - Audio configuration (primarily num_mel_bins)
/// * `pcm` - Input audio samples (f32, mono, 16kHz, normalized to [-1, 1])
/// * `mel_filters` - Pre-computed mel filter bank (num_mel_bins * (n_fft/2 + 1))
///
/// # Returns
/// Flattened mel spectrogram as Vec<f32> with shape (num_mel_bins, n_frames)
/// where n_frames = (len(pcm) - n_fft) / hop_length + 1
pub fn pcm_to_mel(config: &AudioConfig, pcm: &[f32], mel_filters: &[f32]) -> Vec<f32> {
    let n_mels = config.num_mel_bins;

    // Pad or truncate audio to expected length
    let mut samples = pcm.to_vec();
    if samples.len() < CHUNK_LENGTH {
        samples.resize(CHUNK_LENGTH, 0.0);
    } else if samples.len() > CHUNK_LENGTH {
        samples.truncate(CHUNK_LENGTH);
    }

    // Compute STFT (Short-Time Fourier Transform)
    let stft = stft(&samples, N_FFT, HOP_LENGTH);

    // Compute power spectrum
    let magnitudes = stft_magnitudes(&stft);

    // Apply mel filter banks
    let mel_spec = apply_mel_filters(&magnitudes, mel_filters, n_mels);

    // Apply log scaling
    log_mel_spectrogram(&mel_spec)
}

/// Compute Short-Time Fourier Transform
fn stft(samples: &[f32], n_fft: usize, hop_length: usize) -> Vec<Vec<Complex<f32>>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Create Hann window
    let window = hann_window(n_fft);

    // Whisper uses `torch.stft(..., center=True, pad_mode="reflect")`.
    // That implies padding by n_fft/2 on both sides, then dropping the last frame so the
    // resulting spectrogram has exactly `CHUNK_LENGTH / HOP_LENGTH` frames (3000 for 30s audio).
    let pad = n_fft / 2;
    let padded = reflect_pad(samples, pad);

    let n_frames = (padded.len() - n_fft) / hop_length + 1;
    let mut result = Vec::with_capacity(n_frames.saturating_sub(1));

    for i in 0..n_frames {
        let start = i * hop_length;
        let end = start + n_fft;

        if end > padded.len() {
            break;
        }

        // Apply window and prepare complex buffer
        let mut buffer: Vec<Complex<f32>> = padded[start..end]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Compute FFT
        fft.process(&mut buffer);

        result.push(buffer);
    }

    // Drop the last frame to match Whisper's 3000-frame convention.
    // (With center padding, torch.stft produces 3001 frames for 30s audio.)
    result.pop();

    result
}

/// Reflect-pad a 1D signal on both sides, matching PyTorch's `pad_mode="reflect"`.
///
/// For a padding of `pad`, the left pad is `samples[pad..1]` (reversed) and the right pad is
/// `samples[len-2 .. len-2-pad]` (reversed). This does not repeat edge samples.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    if pad == 0 {
        return samples.to_vec();
    }

    // `reflect` padding requires at least `pad + 1` samples. Whisper inputs are always
    // 30 seconds (480k samples), so this is a safe guard for tests and edge cases.
    if samples.len() <= pad {
        let mut out = Vec::with_capacity(samples.len() + 2 * pad);
        out.resize(pad, 0.0);
        out.extend_from_slice(samples);
        out.resize(out.len() + pad, 0.0);
        return out;
    }

    let mut out = Vec::with_capacity(samples.len() + 2 * pad);

    // Left pad: samples[pad], samples[pad-1], ..., samples[1]
    for i in (1..=pad).rev() {
        out.push(samples[i]);
    }

    out.extend_from_slice(samples);

    // Right pad: samples[len-2], samples[len-3], ..., samples[len-1-pad]
    let len = samples.len();
    for i in 0..pad {
        out.push(samples[len - 2 - i]);
    }

    out
}

/// Compute power spectrum from STFT (magnitude squared)
fn stft_magnitudes(stft: &[Vec<Complex<f32>>]) -> Vec<Vec<f32>> {
    stft.iter()
        .map(|frame| {
            // Only take first half of FFT (positive frequencies)
            // For n_fft=400, we get 201 frequency bins (0..=200)
            frame[..=frame.len() / 2]
                .iter()
                .map(|c| c.norm_sqr())
                .collect()
        })
        .collect()
}

/// Apply mel filter banks to magnitude spectrum
#[allow(clippy::needless_range_loop)]
fn apply_mel_filters(magnitudes: &[Vec<f32>], mel_filters: &[f32], n_mels: usize) -> Vec<Vec<f32>> {
    let n_freqs = magnitudes[0].len();
    let n_frames = magnitudes.len();

    let mut mel_spec = vec![vec![0.0; n_frames]; n_mels];

    for (frame_idx, frame) in magnitudes.iter().enumerate() {
        for mel_idx in 0..n_mels {
            let mut sum = 0.0;
            for freq_idx in 0..n_freqs {
                let filter_idx = mel_idx * n_freqs + freq_idx;
                sum += frame[freq_idx] * mel_filters[filter_idx];
            }
            mel_spec[mel_idx][frame_idx] = sum;
        }
    }

    mel_spec
}

/// Apply log scaling to mel spectrogram (matching OpenAI Whisper preprocessing)
fn log_mel_spectrogram(mel_spec: &[Vec<f32>]) -> Vec<f32> {
    // OpenAI Whisper uses a log10 mel spectrogram with dynamic range compression:
    //   log_spec = log10(max(mel_spec, 1e-10))
    //   log_spec = max(log_spec, log_spec.max() - 8)
    //   log_spec = (log_spec + 4) / 4
    //
    // This yields values roughly in [-1, 1].
    let clamp_value = 1e-10_f32;

    let mut log_mel: Vec<f32> = mel_spec
        .iter()
        .flat_map(|mel_bin| mel_bin.iter().map(|&val| val.max(clamp_value).log10()))
        .collect();

    let max_log = log_mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_log = max_log - 8.0;

    for x in &mut log_mel {
        if *x < min_log {
            *x = min_log;
        }
        *x = (*x + 4.0) / 4.0;
    }

    log_mel
}

/// Create Hann window (periodic, matching `torch.hann_window(n_fft)` default)
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let window = hann_window(400);
        assert_eq!(window.len(), 400);

        // Periodic Hann window should be 0 at the first sample
        assert!(window[0].abs() < 1e-6);

        // Should peak near the middle
        assert!(window[200] > 0.9);
    }

    #[test]
    fn test_stft_shape() {
        // Create 1 second of silent audio
        let samples = vec![0.0; 16000];
        let stft_result = stft(&samples, 400, 160);

        // With center reflection padding and dropping the last frame, we get:
        //   frames = samples.len() / hop_length = 100
        assert!(stft_result.len() >= 98 && stft_result.len() <= 102);

        // Each frame should have 400 complex samples
        assert_eq!(stft_result[0].len(), 400);
    }

    #[test]
    fn test_stft_magnitudes_shape() {
        let samples = vec![0.0; 16000];
        let stft_result = stft(&samples, 400, 160);
        let magnitudes = stft_magnitudes(&stft_result);

        // Should have same number of frames
        assert_eq!(magnitudes.len(), stft_result.len());

        // Each frame should have 201 frequency bins (half of 400 + 1)
        assert_eq!(magnitudes[0].len(), 201);
    }

    #[test]
    fn test_pcm_to_mel_shape() {
        // Create 30 seconds of audio (Whisper's chunk length)
        let samples = vec![0.0; 480000];

        // Mock mel filters (80 mel bins * 201 frequency bins)
        let mel_filters = vec![1.0; 80 * 201];

        let config = AudioConfig { num_mel_bins: 80 };
        let mel_spec = pcm_to_mel(&config, &samples, &mel_filters);

        // Expected shape: 80 mel bins * 3000 frames (for 30-second audio)
        // (480000 - 400) / 160 + 1 ≈ 2996 frames, padded to match Whisper's expectation
        assert!(mel_spec.len() > 200000); // At least 80 * 2500
    }

    #[test]
    fn test_log_mel_spectrogram_no_nan() {
        // Create mel spectrogram with zeros (edge case)
        let mel_spec = vec![vec![0.0; 100]; 80];
        let log_mel = log_mel_spectrogram(&mel_spec);

        // Should not contain NaN or infinity
        assert!(log_mel.iter().all(|&v| v.is_finite()));

        // Should clamp to -10.0 for very small values
        assert!(log_mel.iter().all(|&v| v >= -10.0));
    }
}
