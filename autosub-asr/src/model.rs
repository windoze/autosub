use std::path::PathBuf;

use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{debug, info};

use crate::filter::HallucinationFilter;
use crate::types::{AsrInput, AudioClip, TranscriptionResult};
use crate::vad::{VadConfig, VadSegmenter, WebRtcVad};

const N_FRAMES: usize = 3000; // Frames per 30-second segment

// Pre-computed mel filter banks from OpenAI Whisper
const MEL_FILTERS_80: &[u8] = include_bytes!("melfilters.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("melfilters128.bytes");

// Whisper timestamp token range
const TIMESTAMP_BEGIN: u32 = 50364; // <|0.00|>
const TIMESTAMP_END: u32 = 51864; // <|30.00|>

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

/// Configuration for loading a Whisper model
#[derive(Debug, Clone)]
pub struct WhisperModelConfig {
    /// Model size/variant (tiny, base, small, medium, large)
    pub model_size: WhisperModelSize,
    /// Optional cache directory for model files
    pub cache_dir: Option<PathBuf>,
    /// Device to run the model on
    pub device: Device,
}

/// Whisper model size variants
#[derive(Debug, Clone, Copy)]
pub enum WhisperModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl WhisperModelSize {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::Tiny => "openai/whisper-tiny",
            Self::Base => "openai/whisper-base",
            Self::Small => "openai/whisper-small",
            Self::Medium => "openai/whisper-medium",
            Self::Large => "openai/whisper-large-v3",
        }
    }
}

/// Core Whisper model for ASR
pub struct WhisperModel {
    model: m::model::Whisper,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
}

impl WhisperModel {
    /// Download and load a Whisper model
    pub fn load(config: WhisperModelConfig) -> Result<Self> {
        info!(
            "Loading Whisper {:?} model...",
            config.model_size
        );

        let api = Api::new().context("Failed to create HuggingFace API")?;
        let api_repo = api.model(config.model_size.repo_id().to_string());

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
        let model_config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Failed to read config.json")?,
        )
        .context("Failed to parse config.json")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, &config.device)
                .context("Failed to load model weights")?
        };

        let model = m::model::Whisper::load(&vb, model_config.clone())
            .context("Failed to create Whisper model")?;

        // Load pre-computed mel filters based on model config
        let mel_bytes = match model_config.num_mel_bins {
            80 => MEL_FILTERS_80,
            128 => MEL_FILTERS_128,
            n => anyhow::bail!("Unsupported num_mel_bins: {}", n),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

        info!("Model loaded successfully (device: {:?})", config.device);

        Ok(Self {
            model,
            tokenizer,
            config: model_config,
            device: config.device,
            mel_filters,
        })
    }

    /// Transcribe a single audio clip and return transcription results
    pub fn transcribe_clip(
        &mut self,
        clip: &AudioClip,
        language: Option<&str>,
        initial_prompt: Option<&str>,
        filter: Option<&dyn HallucinationFilter>,
    ) -> Result<Vec<TranscriptionResult>> {
        let n_mels = self.config.num_mel_bins;

        if clip.samples.is_empty() {
            return Ok(Vec::new());
        }

        // Only reset KV cache when explicitly signaled (after long silence)
        // This preserves context across consecutive VAD segments for better accuracy
        if clip.reset_context {
            debug!("Resetting ASR context due to long silence");
            self.model.reset_kv_cache();
        }

        let mel = audio::pcm_to_mel(&self.config, &clip.samples, &self.mel_filters);
        let mel_len = mel.len();
        let content_frames = mel_len / n_mels;

        if content_frames == 0 {
            return Ok(Vec::new());
        }

        let mel = Tensor::from_vec(mel, (1, n_mels, content_frames), &self.device)?;

        // Pad or truncate to N_FRAMES
        let mel = if content_frames < N_FRAMES {
            let padding = Tensor::zeros(
                (1, n_mels, N_FRAMES - content_frames),
                candle_core::DType::F32,
                &self.device,
            )?;
            Tensor::cat(&[&mel, &padding], 2)?
        } else if content_frames > N_FRAMES {
            mel.narrow(2, 0, N_FRAMES)?
        } else {
            mel
        };

        let audio_features = self.model.encoder.forward(&mel, true)?;

        let segments = self.decode_segment_with_timestamps(&audio_features, language, initial_prompt)?;

        let mut results = Vec::new();
        let mut last_end_time = 0.0_f64;

        for seg in segments {
            let text = self.decode_tokens(&seg.tokens)?;
            let text = text.trim();

            // Skip empty or blank audio markers
            if text.is_empty() {
                debug!("Decoded empty text for segment");
                continue;
            }

            if text == "[BLANK_AUDIO]" {
                debug!("Decoded [BLANK_AUDIO] marker");
                continue;
            }

            // Apply hallucination filter if configured
            if let Some(f) = filter {
                if f.is_hallucinated(text) {
                    debug!("Filtered as hallucination: '{}'", text);
                    continue;
                }
            }

            debug!("Valid segment text: '{}'", text);

            let time_offset = clip.start_time_secs();
            let mut start_time = time_offset + seg.start;
            let mut end_time = time_offset + seg.end;

            if start_time < last_end_time {
                start_time = last_end_time;
            }
            if end_time <= start_time {
                end_time = start_time + 0.01;
            }

            // Split long segments into sentences
            let sentences = split_into_sentences(text, start_time, end_time);

            for (sent_start, sent_end, sent_text) in sentences {
                debug!("Segment: {:.2}-{:.2}: {}", sent_start, sent_end, sent_text);
                results.push(TranscriptionResult::new(sent_text, sent_start, sent_end));
                last_end_time = last_end_time.max(sent_end);
            }
        }

        Ok(results)
    }

    /// Decode a segment with timestamps, returning multiple timed segments
    fn decode_segment_with_timestamps(
        &mut self,
        audio_features: &Tensor,
        language: Option<&str>,
        initial_prompt: Option<&str>,
    ) -> Result<Vec<DecodedSegment>> {
        // Get special token IDs
        let sot_token = self.token_id("<|startoftranscript|>")?;
        let transcribe_token = self.token_id("<|transcribe|>")?;
        let eot_token = self.token_id("<|endoftext|>")?;

        // Language token - either use specified language or auto-detect
        let language_token = if let Some(lang) = language {
            self.token_id(&format!("<|{}|>", lang))
                .unwrap_or_else(|_| self.token_id("<|en|>").unwrap())
        } else {
            self.detect_language(audio_features)?
        };

        // Initial tokens: SOT, language, transcribe
        let mut tokens = vec![sot_token, language_token, transcribe_token];

        // Add initial prompt tokens if provided
        if let Some(prompt_text) = initial_prompt {
            if let Ok(encoding) = self.tokenizer.encode(prompt_text, false) {
                let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
                // Only add prompt tokens that are valid (< 50257, not special tokens)
                for &token in &prompt_tokens {
                    if token < 50257 {
                        tokens.push(token);
                    }
                }
            }
        }

        // Add first timestamp <|0.00|>
        tokens.push(TIMESTAMP_BEGIN);

        let sample_len = self.config.max_target_positions / 2;
        let mut all_tokens = vec![TIMESTAMP_BEGIN];

        for i in 0..sample_len {
            let tokens_tensor = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            let ys = self
                .model
                .decoder
                .forward(&tokens_tensor, audio_features, i == 0)?;

            let (_, seq_len, _) = ys.dims3()?;
            let ys_last = ys.narrow(1, seq_len - 1, 1)?;
            let logits = self.model.decoder.final_linear(&ys_last)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;

            let next_token = logits.argmax(0)?.to_scalar::<u32>()?;

            if next_token == eot_token {
                break;
            }

            all_tokens.push(next_token);
            tokens.push(next_token);

            // Prevent infinite loops on repetition
            if all_tokens.len() >= 4 {
                let len = all_tokens.len();
                if all_tokens[len - 1] == all_tokens[len - 2]
                    && all_tokens[len - 2] == all_tokens[len - 3]
                    && all_tokens[len - 3] == all_tokens[len - 4]
                {
                    while all_tokens.len() > 1
                        && all_tokens[all_tokens.len() - 1] == all_tokens[all_tokens.len() - 2]
                    {
                        all_tokens.pop();
                    }
                    break;
                }
            }
        }

        let segments = self.parse_timestamped_tokens(&all_tokens);
        Ok(segments)
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
                    current_start = Some(time);
                } else {
                    if !current_tokens.is_empty() {
                        segments.push(DecodedSegment {
                            start: current_start.unwrap(),
                            end: time,
                            tokens: current_tokens.clone(),
                        });
                    }
                    current_tokens.clear();
                    current_start = Some(time);
                }
            } else if current_start.is_some() && token < 50257 {
                current_tokens.push(token);
            }
        }

        // Handle remaining tokens
        if !current_tokens.is_empty() {
            if let Some(start) = current_start {
                let end = (start + 5.0).min(30.0);
                segments.push(DecodedSegment {
                    start,
                    end,
                    tokens: current_tokens,
                });
            }
        }

        segments
    }

    /// Auto-detect language from audio features
    fn detect_language(&mut self, audio_features: &Tensor) -> Result<u32> {
        let sot_token = self.token_id("<|startoftranscript|>")?;

        let tokens = Tensor::new(&[sot_token], &self.device)?.unsqueeze(0)?;
        let ys = self.model.decoder.forward(&tokens, audio_features, true)?;

        let (_, seq_len, _) = ys.dims3()?;
        let ys_last = ys.narrow(1, seq_len - 1, 1)?;
        let logits = self.model.decoder.final_linear(&ys_last)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;

        let lang_token_start = 50259u32;
        let lang_token_end = 50358u32;

        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let mut best_lang_token = self.token_id("<|en|>")?;
        let mut best_prob = f32::NEG_INFINITY;

        for token_id in lang_token_start..=lang_token_end {
            if let Some(&prob) = logits_vec.get(token_id as usize) {
                if prob > best_prob {
                    best_prob = prob;
                    best_lang_token = token_id;
                }
            }
        }

        if let Some(lang_str) = self.tokenizer.id_to_token(best_lang_token) {
            debug!("Detected language: {}", lang_str);
        }

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

/// ASR Engine that processes raw audio samples through VAD and transcription
pub struct AsrEngine {
    model: WhisperModel,
    language: Option<String>,
    initial_prompt: Option<String>,
    segmenter: VadSegmenter<WebRtcVad>,
    filter: Option<Box<dyn crate::filter::HallucinationFilter>>,
}

// SAFETY: AsrEngine is only used within a single tokio task and never shared across threads.
// The WebRtcVad pointer is never accessed concurrently. The async operations in run()
// do not leak the VAD state across threads.
unsafe impl Send for AsrEngine {}

impl AsrEngine {
    /// Create a new ASR engine with VAD and default hallucination filter
    pub fn new(model: WhisperModel, language: Option<String>, vad_config: VadConfig) -> Result<Self> {
        Self::with_filter(
            model,
            language,
            None,
            vad_config,
            Some(Box::new(crate::filter::DefaultHallucinationFilter::new())),
        )
    }

    /// Create a new ASR engine with custom hallucination filter
    ///
    /// # Arguments
    /// * `model` - Whisper model to use for transcription
    /// * `language` - Optional language code (e.g., "en")
    /// * `initial_prompt` - Optional initial prompt to guide the model (helps with context, terminology, and style)
    /// * `vad_config` - Voice activity detection configuration
    /// * `filter` - Optional custom hallucination filter (None = no filtering)
    pub fn with_filter(
        model: WhisperModel,
        language: Option<String>,
        initial_prompt: Option<String>,
        vad_config: VadConfig,
        filter: Option<Box<dyn crate::filter::HallucinationFilter>>,
    ) -> Result<Self> {
        let vad = WebRtcVad::new(
            vad_config.sample_rate,
            vad_config.frame_duration_ms,
            vad_config.mode.to_webrtc_mode(),
        )?;

        let segmenter = VadSegmenter::new(
            vad,
            vad_config.frame_duration_ms,
            vad_config.silence_reset_secs,
        );

        Ok(Self {
            model,
            language,
            initial_prompt,
            segmenter,
            filter,
        })
    }

    /// Run the ASR engine, consuming raw audio samples or flush signals from input channel,
    /// processing through VAD, and sending transcription results to output channel
    pub fn run(
        mut self,
        mut input: mpsc::Receiver<AsrInput>,
        output: mpsc::Sender<TranscriptionResult>,
    ) -> Result<()> {
        info!("ASR engine started with integrated VAD");

        loop {
            info!("ASR engine about to call input.recv()");

            let msg = match input.blocking_recv() {
                Some(m) => {
                    info!("ASR engine received a message!");
                    m
                }
                None => {
                    info!("ASR engine input channel closed");
                    break;
                }
            };
            let clips = match msg {
                AsrInput::Samples(samples) => {
                    info!("Processing {} audio samples through VAD", samples.len());
                    let clips = self.segmenter.push_samples(&samples)?;
                    info!("VAD processing returned {} clips", clips.len());
                    clips
                }
                AsrInput::Flush => {
                    info!("Flushing VAD segmenter and resetting timestamp position");
                    let clips = self.segmenter.flush()?;
                    info!("VAD flush returned {} clips", clips.len());
                    // Reset timestamp position so next samples start from 0
                    // This is important for push-to-talk applications where each session
                    // should have timestamps starting from 0
                    self.segmenter.reset_position();
                    clips
                }
            };

            // Process each clip from the VAD segmenter
            for (idx, clip) in clips.iter().enumerate() {
                info!(
                    "VAD segment {}/{}: {:.2}s - {:.2}s ({} samples)",
                    idx + 1,
                    clips.len(),
                    clip.start_time_secs(),
                    clip.end_time_secs(),
                    clip.samples.len()
                );

                info!("Starting Whisper transcription for segment {}...", idx + 1);
                let results = self.model.transcribe_clip(clip, self.language.as_deref(), self.initial_prompt.as_deref(), self.filter.as_deref())?;
                info!("Whisper transcription completed for segment {}, got {} results", idx + 1, results.len());

                for result in results {
                    if output.blocking_send(result).is_err() {
                        info!("Output channel closed, stopping ASR engine");
                        return Ok(());
                    }
                }
            }
        }

        info!("ASR engine finished");
        Ok(())
    }
}

/// Split text into sentences and distribute timestamps proportionally
fn split_into_sentences(text: &str, start: f64, end: f64) -> Vec<(f64, f64, String)> {
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let remaining = current.trim().to_string();
    if !remaining.is_empty() {
        sentences.push(remaining);
    }

    if sentences.len() <= 1 {
        return vec![(start, end, text.to_string())];
    }

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
