use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::api::sync::Api;
use hound::WavReader;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::config::WhisperModelSize;
use crate::srt::{SrtWriter, Subtitle};
use std::io::Write;

const SAMPLE_RATE: usize = 16000;
const N_FRAMES: usize = 3000; // Frames per 30-second segment
const HOP_LENGTH: usize = 160;

// Pre-computed mel filter banks from OpenAI Whisper
const MEL_FILTERS_80: &[u8] = include_bytes!("melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("melfilters128.bytes");

// Whisper timestamp token range
const TIMESTAMP_BEGIN: u32 = 50364; // <|0.00|>
const TIMESTAMP_END: u32 = 51864;   // <|30.00|>

/// Convert a timestamp token to seconds
fn timestamp_token_to_seconds(token: u32) -> f64 {
    (token - TIMESTAMP_BEGIN) as f64 * 0.02
}

/// Check if a token is a timestamp token
fn is_timestamp_token(token: u32) -> bool {
    (TIMESTAMP_BEGIN..=TIMESTAMP_END).contains(&token)
}

/// A decoded segment with timestamps extracted from tokens
#[derive(Debug, Clone)]
struct DecodedSegment {
    start: f64,
    end: f64,
    tokens: Vec<u32>,
}

/// Split text into sentences and distribute timestamps proportionally.
/// Returns Vec of (start_time, end_time, text) tuples.
fn split_into_sentences(text: &str, start: f64, end: f64) -> Vec<(f64, f64, String)> {
    // Split on sentence-ending punctuation, keeping the punctuation
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        // Split on sentence-ending punctuation followed by space or end
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Don't forget remaining text
    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    // If only one sentence or couldn't split, return as-is
    if sentences.len() <= 1 {
        return vec![(start, end, text.to_string())];
    }

    // Calculate total character count for proportional timing
    let total_chars: usize = sentences.iter().map(|s| s.chars().count()).sum();
    if total_chars == 0 {
        return vec![(start, end, text.to_string())];
    }

    let duration = end - start;
    let mut result = Vec::new();
    let mut current_time = start;

    for (i, sentence) in sentences.iter().enumerate() {
        let char_count = sentence.chars().count();
        let sentence_duration = if i == sentences.len() - 1 {
            // Last sentence takes remaining time to avoid rounding errors
            end - current_time
        } else {
            duration * (char_count as f64 / total_chars as f64)
        };

        let sentence_end = current_time + sentence_duration;
        result.push((current_time, sentence_end, sentence.clone()));
        current_time = sentence_end;
    }

    result
}

/// Check if text appears to be hallucinated garbage.
/// Whisper sometimes hallucinates on silence/music, producing repetitive
/// nonsense text or characters from unrelated scripts.
fn is_hallucinated_text(text: &str) -> bool {
    // Skip very short segments (less than 2 chars)
    if text.chars().count() < 2 {
        return true;
    }

    // Check for Sinhala, Khmer, and other scripts that indicate hallucination
    // when they appear unexpectedly (these are common hallucination targets)
    let suspicious_scripts = text.chars().any(|c| {
        matches!(c,
            '\u{0D80}'..='\u{0DFF}' |  // Sinhala
            '\u{1780}'..='\u{17FF}' |  // Khmer
            '\u{1200}'..='\u{137F}'    // Ethiopic
        )
    });

    if suspicious_scripts {
        return true;
    }

    // Check for repetitive sentences (common hallucination pattern)
    // e.g., "Yn ymwneud, yw'r cyffredin yn ymwneud." repeated many times
    let sentences: Vec<&str> = text
        .split(['.', '!', '?', '。', '！', '？'])
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if sentences.len() >= 3 {
        // Check if any sentence appears more than twice
        let mut seen = std::collections::HashMap::new();
        for sentence in &sentences {
            *seen.entry(*sentence).or_insert(0) += 1;
        }
        // If any sentence repeats 3+ times, it's likely hallucination
        if seen.values().any(|&count| count >= 3) {
            return true;
        }
    }

    false
}

/// A segment of transcribed text with timing
#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Whisper model wrapper for transcription
pub struct WhisperModel {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
}

impl WhisperModel {
    /// Download and load a Whisper model
    pub fn load(
        model_size: WhisperModelSize,
        _cache_dir: Option<PathBuf>,
        device: Device,
    ) -> Result<Self> {
        info!("Loading Whisper {} model...", model_size);

        let api = Api::new().context("Failed to create HuggingFace API")?;
        let api_repo = api.model(model_size.repo_id().to_string());

        // Download model files
        info!("Downloading model files (this may take a while on first run)...");

        let config_path = api_repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let tokenizer_path = api_repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let weights_path = api_repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;

        debug!("Config: {}", config_path.display());
        debug!("Tokenizer: {}", tokenizer_path.display());
        debug!("Weights: {}", weights_path.display());

        // Load config
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Failed to read config.json")?,
        )
        .context("Failed to parse config.json")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &device)
                .context("Failed to load model weights")?
        };

        let model = m::model::Whisper::load(&vb, config.clone())
            .context("Failed to create Whisper model")?;

        // Load pre-computed mel filters based on model config
        let mel_bytes = match config.num_mel_bins {
            80 => MEL_FILTERS_80,
            128 => MEL_FILTERS_128,
            n => anyhow::bail!("Unsupported num_mel_bins: {}", n),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

        info!("Model loaded successfully (device: {:?})", device);

        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            mel_filters,
        })
    }

    /// Load audio from WAV file and return PCM samples as f32
    fn load_audio(audio_path: &Path) -> Result<Vec<f32>> {
        let reader = WavReader::open(audio_path).context("Failed to open WAV file")?;
        let spec = reader.spec();

        info!(
            "WAV file: {} Hz, {} channels, {} bits",
            spec.sample_rate, spec.channels, spec.bits_per_sample
        );

        let max_value = (1i32 << (spec.bits_per_sample - 1)) as f32;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max_value)
                .collect(),
            hound::SampleFormat::Float => reader
                .into_samples::<f32>()
                .filter_map(Result::ok)
                .collect(),
        };

        Ok(samples)
    }

    /// Transcribe audio from a file and write segments to an SrtWriter as they're decoded.
    /// Each segment is flushed immediately, providing real-time output.
    /// Also returns the complete Subtitle in memory.
    pub fn transcribe_to_writer<W: Write>(
        &mut self,
        audio_path: &Path,
        language: Option<&str>,
        writer: &mut SrtWriter<W>,
    ) -> Result<Subtitle> {
        // Load all audio samples
        let samples = Self::load_audio(audio_path)?;
        let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;

        info!("Loaded {:.2} seconds of audio ({} samples)", duration_secs, samples.len());

        // Compute mel spectrogram for the ENTIRE audio file
        let mel = audio::pcm_to_mel(&self.config, &samples, &self.mel_filters);
        let mel_len = mel.len();
        let n_mels = self.config.num_mel_bins;
        let content_frames = mel_len / n_mels;

        debug!(
            "Mel spectrogram: {} samples -> {} values, {} bins, {} frames",
            samples.len(),
            mel_len,
            n_mels,
            content_frames
        );

        // Create mel tensor with shape (1, n_mels, content_frames)
        let mel = Tensor::from_vec(mel, (1, n_mels, content_frames), &self.device)?;

        // Process in segments of N_FRAMES
        let mut subtitle = Subtitle::new();
        let mut seek = 0;
        let num_segments = content_frames.div_ceil(N_FRAMES);

        info!("Processing {} segments...", num_segments);

        while seek < content_frames {
            let segment_size = usize::min(content_frames - seek, N_FRAMES);
            let time_offset = (seek * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;

            debug!(
                "Processing segment at frame {}/{} (offset: {:.2}s, size: {} frames)",
                seek, content_frames, time_offset, segment_size
            );

            // Extract mel segment using narrow
            let mel_segment = mel.narrow(2, seek, segment_size)?;

            // Pad to N_FRAMES if needed
            let mel_segment = if segment_size < N_FRAMES {
                // Pad with zeros
                let padding = Tensor::zeros((1, n_mels, N_FRAMES - segment_size), candle_core::DType::F32, &self.device)?;
                Tensor::cat(&[&mel_segment, &padding], 2)?
            } else {
                mel_segment
            };

            // Reset KV cache before processing a new segment
            self.model.reset_kv_cache();

            // Encode audio segment
            let audio_features = self.model.encoder.forward(&mel_segment, true)?;

            // Decode and write, getting the last timestamp for seeking
            let last_timestamp = self.decode_segment_to_writer(&audio_features, language, time_offset, writer, &mut subtitle)?;

            // Advance seek based on the last timestamp from the model
            // Convert timestamp (seconds within segment) to frames
            let seek_advance = if last_timestamp > 0.0 {
                // Use the last timestamp to determine how far to advance
                let frames_from_timestamp = (last_timestamp * SAMPLE_RATE as f64 / HOP_LENGTH as f64) as usize;
                // Advance at least 1 frame to avoid infinite loops, but use timestamp when available
                frames_from_timestamp.max(1)
            } else {
                // No valid timestamp, advance by full segment
                segment_size
            };

            seek += seek_advance;
        }

        info!("Transcription complete: {} segments", subtitle.len());
        Ok(subtitle)
    }

    /// Decode a single segment and write to writer.
    /// Returns the last timestamp (in seconds, relative to segment start) for seek advancement.
    fn decode_segment_to_writer<W: Write>(
        &mut self,
        audio_features: &Tensor,
        language: Option<&str>,
        time_offset: f64,
        writer: &mut SrtWriter<W>,
        subtitle: &mut Subtitle,
    ) -> Result<f64> {
        // Decode with timestamps
        let (segments, last_timestamp) = self.decode_segment_with_timestamps(audio_features, language)?;

        for seg in segments {
            let text = self.decode_tokens(&seg.tokens)?;
            let text = text.trim();

            // Skip empty, blank audio markers, and hallucinated garbage
            if text.is_empty() || text == "[BLANK_AUDIO]" || is_hallucinated_text(text) {
                continue;
            }

            let start_time = time_offset + seg.start;
            let end_time = time_offset + seg.end;

            // Split long segments into sentences
            let sentences = split_into_sentences(text, start_time, end_time);

            for (sent_start, sent_end, sent_text) in sentences {
                debug!("Segment: {:.2}-{:.2}: {}", sent_start, sent_end, sent_text);
                writer.write_entry(sent_start, sent_end, &sent_text)?;
                subtitle.push(sent_start, sent_end, sent_text);
            }
        }

        Ok(last_timestamp)
    }

    /// Decode a segment with timestamps, returning multiple timed segments and the last timestamp.
    /// The last timestamp is used to determine how far to seek for the next segment.
    fn decode_segment_with_timestamps(
        &mut self,
        audio_features: &Tensor,
        language: Option<&str>,
    ) -> Result<(Vec<DecodedSegment>, f64)> {
        // Get special token IDs
        let sot_token = self.token_id("<|startoftranscript|>")?;
        let transcribe_token = self.token_id("<|transcribe|>")?;
        let eot_token = self.token_id("<|endoftext|>")?;

        // Language token - either use specified language or auto-detect
        let language_token = if let Some(lang) = language {
            self.token_id(&format!("<|{}|>", lang)).unwrap_or_else(|_| {
                self.token_id("<|en|>").unwrap()
            })
        } else {
            // Auto-detect language by letting model predict after SOT token
            self.detect_language(audio_features)?
        };

        // Initial tokens: SOT, language, transcribe, and first timestamp <|0.00|>
        // Adding the initial timestamp token forces the model into timestamp mode
        let mut tokens = vec![sot_token, language_token, transcribe_token, TIMESTAMP_BEGIN];

        let sample_len = self.config.max_target_positions / 2;
        // Include the initial timestamp in all_tokens for proper parsing
        let mut all_tokens = vec![TIMESTAMP_BEGIN];
        let mut last_timestamp = 0.0_f64;

        for i in 0..sample_len {
            let tokens_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            // flush_kv_cache should be true only on the first iteration
            let ys = self.model.decoder.forward(&tokens_tensor, audio_features, i == 0)?;

            // Get the last position's hidden states and apply final linear layer to get logits
            let (_, seq_len, _) = ys.dims3()?;
            let ys_last = ys.narrow(1, seq_len - 1, 1)?;
            let logits = self.model.decoder.final_linear(&ys_last)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;

            // Greedy decode
            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

            if next_token == eot_token {
                break;
            }

            // Track the last timestamp token for seek advancement
            if is_timestamp_token(next_token) {
                last_timestamp = timestamp_token_to_seconds(next_token);
            }

            all_tokens.push(next_token);
            tokens.push(next_token);

            // Prevent infinite loops on repetition (check last 4 tokens)
            if all_tokens.len() >= 4 {
                let len = all_tokens.len();
                if all_tokens[len - 1] == all_tokens[len - 2]
                    && all_tokens[len - 2] == all_tokens[len - 3]
                    && all_tokens[len - 3] == all_tokens[len - 4]
                {
                    // Remove the repeated tokens
                    while all_tokens.len() > 1
                        && all_tokens[all_tokens.len() - 1] == all_tokens[all_tokens.len() - 2]
                    {
                        all_tokens.pop();
                    }
                    break;
                }
            }
        }

        // Parse tokens into segments based on timestamps
        // Format: <|start_time|> text tokens... <|end_time|> <|start_time|> text...
        let segments = self.parse_timestamped_tokens(&all_tokens);

        // If we got segments, use the last segment's end time as the seek position
        if let Some(last_seg) = segments.last() {
            last_timestamp = last_timestamp.max(last_seg.end);
        }

        Ok((segments, last_timestamp))
    }

    /// Parse tokens containing timestamps into segments
    fn parse_timestamped_tokens(&self, tokens: &[u32]) -> Vec<DecodedSegment> {
        let mut segments = Vec::new();
        let mut current_start: Option<f64> = None;
        let mut current_tokens = Vec::new();

        for &token in tokens {
            if is_timestamp_token(token) {
                let time = timestamp_token_to_seconds(token);

                if current_start.is_none() {
                    // This is a start timestamp
                    current_start = Some(time);
                } else {
                    // This is an end timestamp - create a segment
                    if !current_tokens.is_empty() {
                        segments.push(DecodedSegment {
                            start: current_start.unwrap(),
                            end: time,
                            tokens: current_tokens.clone(),
                        });
                    }
                    current_tokens.clear();
                    // The end timestamp also serves as start of next segment
                    current_start = Some(time);
                }
            } else if current_start.is_some() && token < 50257 {
                // Text token (not a special token)
                current_tokens.push(token);
            }
        }

        // Handle any remaining tokens without end timestamp
        if !current_tokens.is_empty() {
            if let Some(start) = current_start {
                // Estimate end time: 30 seconds is max segment, or use last token position
                let end = (start + 5.0).min(30.0); // Default 5 second duration
                segments.push(DecodedSegment {
                    start,
                    end,
                    tokens: current_tokens,
                });
            }
        }

        segments
    }

    /// Auto-detect language from audio features.
    /// Returns the language token ID.
    fn detect_language(&mut self, audio_features: &Tensor) -> Result<u32> {
        let sot_token = self.token_id("<|startoftranscript|>")?;

        // Feed SOT token to decoder and get prediction for language
        let tokens = Tensor::new(&[sot_token], &self.device)?.unsqueeze(0)?;
        let ys = self.model.decoder.forward(&tokens, audio_features, true)?;

        // Get logits for the language position
        let (_, seq_len, _) = ys.dims3()?;
        let ys_last = ys.narrow(1, seq_len - 1, 1)?;
        let logits = self.model.decoder.final_linear(&ys_last)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        // Language tokens are in range 50259-50358 (99 languages)
        // We need to find the highest probability language token
        let lang_token_start = 50259u32;
        let lang_token_end = 50358u32;

        // Get the token with highest probability among language tokens
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let mut best_lang_token = self.token_id("<|en|>")?; // Default to English
        let mut best_prob = f32::NEG_INFINITY;

        for token_id in lang_token_start..=lang_token_end {
            if let Some(&prob) = logits_vec.get(token_id as usize) {
                if prob > best_prob {
                    best_prob = prob;
                    best_lang_token = token_id;
                }
            }
        }

        // Log detected language
        if let Some(lang_str) = self.tokenizer.id_to_token(best_lang_token) {
            debug!("Detected language: {}", lang_str);
        }

        // Reset KV cache after language detection
        self.model.reset_kv_cache();

        Ok(best_lang_token)
    }

    fn token_id(&self, token: &str) -> Result<u32> {
        self.tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow::anyhow!("Token not found: {}", token))
    }

    fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))
    }
}

/// Transcribe audio from a file and return a Subtitle.
pub fn transcribe_file(
    audio_path: &Path,
    model_size: WhisperModelSize,
    cache_dir: Option<PathBuf>,
    device: Device,
    language: Option<&str>,
) -> Result<Subtitle> {
    let mut model = WhisperModel::load(model_size, cache_dir, device)?;
    // Use a dummy writer that discards output
    let mut buffer = Vec::new();
    let mut writer = SrtWriter::new(&mut buffer);
    model.transcribe_to_writer(audio_path, language, &mut writer)
}

/// Transcribe audio from a file and write directly to an SRT file.
/// Each segment is written and flushed immediately, providing real-time output.
/// Returns the complete Subtitle (also kept in memory).
pub fn transcribe_to_file(
    audio_path: &Path,
    output_path: &Path,
    model_size: WhisperModelSize,
    cache_dir: Option<PathBuf>,
    device: Device,
    language: Option<&str>,
) -> Result<Subtitle> {
    let mut model = WhisperModel::load(model_size, cache_dir, device)?;
    let mut writer = SrtWriter::create(output_path)?;
    let subtitle = model.transcribe_to_writer(audio_path, language, &mut writer)?;
    writer.finish()?;
    Ok(subtitle)
}
