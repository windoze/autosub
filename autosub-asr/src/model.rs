use std::path::PathBuf;

use anyhow::{Context, Result};
use byteorder::{ByteOrder, LittleEndian};
use hf_hub::api::sync::Api;
use ndarray::{Array3, ArrayD};
use ort::{
    execution_providers::ExecutionProviderDispatch, session::builder::GraphOptimizationLevel,
    memory::Allocator,
    session::{Session, SessionInputValue},
    tensor::TensorElementType,
    value::{DynTensor, Value as OrtValue, ValueType},
};
use serde::{Deserialize, Serialize};
use std::sync::mpsc as std_mpsc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::audio::{pcm_to_mel, AudioConfig};

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

#[derive(Debug, Clone)]
struct DecoderCacheInput {
    input_index: usize,
    name: String,
    element_type: TensorElementType,
    shape: Vec<i64>,
}

#[derive(Debug, Clone)]
struct DecoderUseCacheInput {
    input_index: usize,
    name: String,
    element_type: TensorElementType,
    shape: Vec<i64>,
}

#[derive(Debug, Clone)]
struct DecoderPlan {
    input_count: usize,
    use_cache_input: Option<DecoderUseCacheInput>,
    cache_inputs: Vec<DecoderCacheInput>,
    cache_pos_by_input_index: Vec<Option<usize>>,
}

impl DecoderPlan {
    fn new(session: &Session) -> Result<Self> {
        let inputs = session.inputs();
        if inputs.len() < 2 {
            anyhow::bail!("Decoder model must have at least 2 inputs (got {})", inputs.len());
        }

        let mut use_cache_input: Option<DecoderUseCacheInput> = None;
        let mut cache_inputs: Vec<DecoderCacheInput> = Vec::new();
        let mut unknown_inputs: Vec<String> = Vec::new();

        for (idx, outlet) in inputs.iter().enumerate().skip(2) {
            let name = outlet.name();

            if name.contains("use_cache") {
                let (element_type, shape) = match outlet.dtype() {
                    ValueType::Tensor { ty, shape, .. } => (*ty, shape.as_ref().to_vec()),
                    other => anyhow::bail!(
                        "Decoder input '{}' expected tensor type for use_cache, got {:?}",
                        name,
                        other
                    ),
                };
                use_cache_input = Some(DecoderUseCacheInput {
                    input_index: idx,
                    name: name.to_string(),
                    element_type,
                    shape,
                });
                continue;
            }

            if name.contains("past") {
                let (element_type, shape) = match outlet.dtype() {
                    ValueType::Tensor { ty, shape, .. } => (*ty, shape.as_ref().to_vec()),
                    other => anyhow::bail!(
                        "Decoder input '{}' expected tensor type for cache, got {:?}",
                        name,
                        other
                    ),
                };
                cache_inputs.push(DecoderCacheInput {
                    input_index: idx,
                    name: name.to_string(),
                    element_type,
                    shape,
                });
                continue;
            }

            unknown_inputs.push(name.to_string());
        }

        if !unknown_inputs.is_empty() {
            anyhow::bail!(
                "Unsupported decoder inputs (expected only input_ids, encoder_hidden_states, optional use_cache*, and past*): {:?}",
                unknown_inputs
            );
        }

        let mut cache_pos_by_input_index = vec![None; inputs.len()];
        for (pos, cache) in cache_inputs.iter().enumerate() {
            if cache.input_index < cache_pos_by_input_index.len() {
                cache_pos_by_input_index[cache.input_index] = Some(pos);
            }
        }

        Ok(Self {
            input_count: inputs.len(),
            use_cache_input,
            cache_inputs,
            cache_pos_by_input_index,
        })
    }

    fn supports_kv_cache(&self, output_count: usize) -> bool {
        // Expect: [logits, present_*...] where present count matches past count.
        !self.cache_inputs.is_empty() && output_count > self.cache_inputs.len()
    }
}

fn resolve_decoder_cache_shape(
    name: &str,
    shape: &[i64],
    encoder_seq_len: usize,
) -> Result<Vec<usize>> {
    let mut resolved: Vec<usize> = shape
        .iter()
        .map(|&d| if d < 0 { 1 } else { d as usize })
        .collect();

    // Common Whisper cache layouts:
    // - Self-attn (decoder): [batch, heads, past_seq_len, head_dim]  (past_seq_len dynamic)
    // - Cross-attn (encoder): [batch, heads, encoder_seq_len, head_dim]
    //
    // The `onnx-community` exports use names that include "decoder"/"encoder" in the cache input name.
    let is_cache = name.contains("past");
    if is_cache {
        if name.contains("decoder") || name.contains("self") {
            // Decoder self-attention KV cache: we want an *empty* past at the beginning.
            let dynamic_idxs: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| (d < 0).then_some(i))
                .collect();
            if dynamic_idxs.len() >= 2 {
                resolved[dynamic_idxs[1]] = 0;
            } else if resolved.len() >= 3 {
                resolved[2] = 0;
            }
        } else if (name.contains("encoder") || name.contains("cross"))
            && resolved.len() >= 3 && shape.get(2) == Some(&-1) {
                resolved[2] = encoder_seq_len;
            }
    }

    Ok(resolved)
}

fn make_zero_tensor(element_type: TensorElementType, shape: Vec<usize>) -> Result<OrtValue> {
    // `Tensor::from_array` rejects dimensions of 0, but Whisper merged decoder models require
    // zero-length past KV caches for correct prefill behavior. Use `DynTensor::new`, which
    // supports zero-sized tensors.
    let allocator = Allocator::default();
    let tensor = DynTensor::new(&allocator, element_type, shape)?;
    Ok(tensor.into_dyn())
}

fn make_filled_tensor_i64(shape: Vec<usize>, value: i64) -> Result<OrtValue> {
    let allocator = Allocator::default();
    let mut tensor = DynTensor::new(&allocator, TensorElementType::Int64, shape.clone())?;

    let elem_count = shape.iter().copied().product::<usize>();
    if elem_count == 0 {
        return Ok(tensor.into_dyn());
    }

    let ptr = tensor.data_ptr_mut().cast::<i64>();
    if ptr.is_null() {
        anyhow::bail!("Got null data pointer for non-empty int64 tensor shape {:?}", shape);
    }
    for i in 0..elem_count {
        unsafe { *ptr.add(i) = value };
    }

    Ok(tensor.into_dyn())
}

fn make_filled_tensor_bool(shape: Vec<usize>, value: bool) -> Result<OrtValue> {
    let allocator = Allocator::default();
    let mut tensor = DynTensor::new(&allocator, TensorElementType::Bool, shape.clone())?;

    let elem_count = shape.iter().copied().product::<usize>();
    if elem_count == 0 {
        return Ok(tensor.into_dyn());
    }

    let ptr = tensor.data_ptr_mut().cast::<bool>();
    if ptr.is_null() {
        anyhow::bail!("Got null data pointer for non-empty bool tensor shape {:?}", shape);
    }
    for i in 0..elem_count {
        unsafe { *ptr.add(i) = value };
    }

    Ok(tensor.into_dyn())
}

fn resolve_use_cache_shape(shape: &[i64]) -> Vec<usize> {
    shape
        .iter()
        .map(|&d| if d < 0 { 1 } else { d as usize })
        .collect()
}

fn make_use_cache_value(
    input: &DecoderUseCacheInput,
    enabled: bool,
) -> Result<OrtValue> {
    let shape = resolve_use_cache_shape(&input.shape);
    match input.element_type {
        TensorElementType::Int64 => make_filled_tensor_i64(shape, if enabled { 1 } else { 0 }),
        TensorElementType::Bool => make_filled_tensor_bool(shape, enabled),
        other => anyhow::bail!(
            "Unsupported '{}' input type {:?} (expected int64 or bool)",
            input.name,
            other
        ),
    }
}

/// Whisper model configuration from HuggingFace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,
    pub max_target_positions: usize,
}

/// Configuration for loading a Whisper model
#[derive(Debug, Clone)]
pub struct WhisperModelConfig {
    /// Model size/variant (tiny, base, small, medium, large)
    pub model_size: WhisperModelSize,
    /// Optional cache directory for model files
    pub cache_dir: Option<PathBuf>,
    /// Execution providers for ONNX Runtime (e.g., CoreML, CUDA, CPU)
    pub execution_providers: Vec<ExecutionProviderDispatch>,
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
            Self::Tiny => "onnx-community/whisper-tiny",
            Self::Base => "onnx-community/whisper-base",
            Self::Small => "onnx-community/whisper-small",
            Self::Medium => "onnx-community/whisper-medium",
            Self::Large => "onnx-community/whisper-large-v3",
        }
    }
}

/// Core Whisper model for ASR using ONNX Runtime
pub struct WhisperModel {
    encoder_session: Session,
    decoder_session: Session,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
}

impl WhisperModel {
    /// Download and load a Whisper ONNX model
    pub fn load(config: WhisperModelConfig) -> Result<Self> {
        info!("Loading Whisper {:?} ONNX model...", config.model_size);

        let api = Api::new().context("Failed to create HuggingFace API")?;
        let api_repo = api.model(config.model_size.repo_id().to_string());

        // Download model files
        info!("Downloading ONNX model files (this may take a while on first run)...");

        let config_path = api_repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let tokenizer_path = api_repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;

        // Prefer KV-cache merged decoder for performance, and fall back to non-merged decoder.
        let decoder_path = api_repo
            .get("onnx/decoder_model_merged.onnx")
            .or_else(|_| api_repo.get("onnx/decoder_model.onnx"))
            .context("Failed to download decoder model (tried decoder_model_merged.onnx and decoder_model.onnx)")?;

        let encoder_path = api_repo
            .get("onnx/encoder_model.onnx")
            .context("Failed to download encoder_model.onnx")?;

        debug!("Config: {}", config_path.display());
        debug!("Tokenizer: {}", tokenizer_path.display());
        debug!("Encoder: {}", encoder_path.display());
        debug!("Decoder: {}", decoder_path.display());

        // Load config
        let model_config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Failed to read config.json")?,
        )
        .context("Failed to parse config.json")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Create ONNX Runtime sessions
        info!("Creating ONNX Runtime sessions");

        let mut session_builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;

        // Apply execution providers
        if !config.execution_providers.is_empty() {
            session_builder = session_builder
                .with_execution_providers(config.execution_providers.clone())?;
        }

        let encoder_session = session_builder
            .clone()
            .commit_from_file(&encoder_path)
            .context("Failed to load encoder ONNX model")?;

        let decoder_session = session_builder
            .commit_from_file(&decoder_path)
            .context("Failed to load decoder ONNX model")?;

        debug!("Decoder model inputs:");
        for (idx, input) in decoder_session.inputs().iter().enumerate() {
            debug!("  [{}] {}: {:?}", idx, input.name(), input.dtype());
        }
        debug!("Decoder model outputs:");
        for (idx, output) in decoder_session.outputs().iter().enumerate() {
            debug!("  [{}] {}: {:?}", idx, output.name(), output.dtype());
        }

        // Load pre-computed mel filters based on model config
        let mel_bytes = match model_config.num_mel_bins {
            80 => MEL_FILTERS_80,
            128 => MEL_FILTERS_128,
            n => anyhow::bail!("Unsupported num_mel_bins: {}", n),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

        info!("ONNX model loaded successfully");

        Ok(Self {
            encoder_session,
            decoder_session,
            tokenizer,
            config: model_config,
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

        // Note: ONNX decoder models handle KV cache internally
        // reset_context flag is noted but doesn't require explicit cache reset
        if clip.reset_context {
            debug!("Context reset signaled (handled by ONNX model)");
        }

        // Convert PCM to mel spectrogram
        let audio_config = AudioConfig { num_mel_bins: n_mels };
        let mel = pcm_to_mel(&audio_config, &clip.samples, &self.mel_filters);
        let mel_len = mel.len();
        let content_frames = mel_len / n_mels;

        debug!("Transcribe: {} samples -> {} bins x {} frames", clip.samples.len(), n_mels, content_frames);

        if content_frames == 0 {
            warn!("No content frames, returning empty");
            return Ok(Vec::new());
        }

        // Reshape mel spectrogram to 3D tensor [batch, n_mels, frames]
        let mut mel_array = Array3::<f32>::zeros((1, n_mels, content_frames));
        for i in 0..n_mels {
            for j in 0..content_frames {
                mel_array[[0, i, j]] = mel[i * content_frames + j];
            }
        }

        // Pad or truncate to N_FRAMES
        let mel_array = if content_frames < N_FRAMES {
            let mut padded = Array3::<f32>::zeros((1, n_mels, N_FRAMES));
            padded.slice_mut(ndarray::s![.., .., ..content_frames]).assign(&mel_array);
            padded
        } else if content_frames > N_FRAMES {
            mel_array.slice(ndarray::s![.., .., ..N_FRAMES]).to_owned()
        } else {
            mel_array
        };

        // Run encoder to get audio features
        // Convert array to tuple format: (shape, data)
        let mel_shape = mel_array.shape().to_vec();
        let (mel_data, _offset) = mel_array.into_raw_vec_and_offset();

        // Check mel spectrogram statistics
        let mel_min = mel_data.iter().copied().fold(f32::INFINITY, f32::min);
        let mel_max = mel_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mel_mean = mel_data.iter().sum::<f32>() / mel_data.len() as f32;
        debug!("Mel stats: min={:.2}, max={:.2}, mean={:.2}", mel_min, mel_max, mel_mean);

        let mel_value = OrtValue::from_array((mel_shape, mel_data))?;

        let audio_features = {
            // Use positional inputs for ONNX encoder (onnx-community models)
            let encoder_outputs = self.encoder_session
                .run([SessionInputValue::from(mel_value)])
                .context("Failed to run encoder")?;

            let (audio_features_shape, audio_features_data) = encoder_outputs[0]
                .try_extract_tensor::<f32>()?;

            // Convert to ArrayD for easier handling
            let shape_vec: Vec<usize> = audio_features_shape.as_ref().iter().map(|&d| d as usize).collect();
            let data_vec = audio_features_data.to_vec();
            let features = ArrayD::from_shape_vec(shape_vec, data_vec)?;

            // Check encoder output
            let first_10: Vec<f32> = features.iter().take(10).copied().collect();
            debug!("Encoder output shape: {:?}, sample values: {:?}", features.shape(), first_10);
            features
        };

        let segments =
            self.decode_segment_with_timestamps(&audio_features, language, initial_prompt)?;

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
        audio_features: &ndarray::ArrayD<f32>,
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

        debug!("Initial decoder tokens: {:?} (SOT={}, lang={}, transcribe={}, timestamp_begin={})",
            tokens, sot_token, language_token, transcribe_token, TIMESTAMP_BEGIN);

        let sample_len = self.config.max_target_positions / 2;
        let mut all_tokens = vec![TIMESTAMP_BEGIN];

        let features_shape = audio_features.shape().to_vec();
        if features_shape.len() != 3 {
            anyhow::bail!(
                "Expected encoder_hidden_states to be rank-3 [batch, seq, hidden], got shape {:?}",
                features_shape
            );
        }
        let encoder_seq_len = features_shape[1];
        let features_data = audio_features.iter().copied().collect::<Vec<f32>>();
        let features_value_f32 = OrtValue::from_array((features_shape.clone(), features_data))?;

        let decoder_plan = DecoderPlan::new(&self.decoder_session)?;
        let decoder_outputs_count = self.decoder_session.outputs().len();
        let use_kv_cache = decoder_plan.supports_kv_cache(decoder_outputs_count);

        fn trim_on_repeated_ngram(tokens: &mut Vec<u32>) -> bool {
            // If the model starts looping (common failure mode in greedy decoding),
            // truncate the most recent repeated n-gram and stop.
            //
            // We purposely keep this lightweight and token-based to avoid decoding strings
            // inside the inner loop.
            const MIN_NGRAM: usize = 4;
            const MAX_NGRAM: usize = 12;

            let len = tokens.len();
            if len < MIN_NGRAM * 2 {
                return false;
            }

            let max_n = MAX_NGRAM.min(len / 2);
            for n in (MIN_NGRAM..=max_n).rev() {
                if tokens[len - n..len] == tokens[len - 2 * n..len - n] {
                    tokens.truncate(len - n);
                    return true;
                }
            }

            false
        }

        fn extract_last_logits<'a>(
            logits_shape: &[i64],
            logits_data: &'a [f32],
            input_seq_len: usize,
        ) -> Result<&'a [f32]> {
            match logits_shape {
                [1, _seq_len, vocab] => {
                    let vocab_size: usize = (*vocab)
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("Invalid vocab size in logits shape: {:?}", logits_shape))?;
                    let last_logits_start = (input_seq_len - 1) * vocab_size;
                    let last_logits_end = last_logits_start + vocab_size;
                    logits_data
                        .get(last_logits_start..last_logits_end)
                        .ok_or_else(|| anyhow::anyhow!("Logits slice out of range for shape {:?}", logits_shape))
                }
                [1, vocab] => {
                    let vocab_size: usize = (*vocab)
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("Invalid vocab size in logits shape: {:?}", logits_shape))?;
                    logits_data
                        .get(..vocab_size)
                        .ok_or_else(|| anyhow::anyhow!("Logits slice out of range for shape {:?}", logits_shape))
                }
                _ => anyhow::bail!("Unexpected logits tensor shape {:?}", logits_shape),
            }
        }

        if use_kv_cache {
            if decoder_outputs_count != 1 + decoder_plan.cache_inputs.len() {
                let out_names: Vec<&str> = self.decoder_session.outputs().iter().map(|o| o.name()).collect();
                anyhow::bail!(
                    "Decoder KV-cache output mismatch: expected {} outputs (logits + {} present), got {}: {:?}",
                    1 + decoder_plan.cache_inputs.len(),
                    decoder_plan.cache_inputs.len(),
                    decoder_outputs_count,
                    out_names
                );
            }

            let (use_cache_off, use_cache_on) = match decoder_plan.use_cache_input.as_ref() {
                Some(input) => (
                    Some(make_use_cache_value(input, false)?),
                    Some(make_use_cache_value(input, true)?),
                ),
                None => (None, None),
            };

            // Initialize past KV cache inputs (zeros). The decoder will produce present_* outputs we can feed back.
            let mut past_values: Vec<OrtValue> = Vec::with_capacity(decoder_plan.cache_inputs.len());
            for cache in &decoder_plan.cache_inputs {
                let resolved_shape =
                    resolve_decoder_cache_shape(&cache.name, &cache.shape, encoder_seq_len)?;
                past_values.push(make_zero_tensor(cache.element_type, resolved_shape)?);
            }

            let mut run_decoder = |input_ids_value: OrtValue,
                                   use_cache_value: Option<&OrtValue>,
                                   past_values: &[OrtValue]|
             -> Result<Vec<OrtValue>> {
                if past_values.len() != decoder_plan.cache_inputs.len() {
                    anyhow::bail!(
                        "Decoder KV-cache inputs mismatch: expected {} past tensors, got {}",
                        decoder_plan.cache_inputs.len(),
                        past_values.len()
                    );
                }

                let mut run_inputs = Vec::with_capacity(decoder_plan.input_count);
                run_inputs.push(SessionInputValue::from(input_ids_value));
                run_inputs.push(SessionInputValue::from(&features_value_f32));

                for idx in 2..decoder_plan.input_count {
                    if let Some(use_cache_input) = decoder_plan.use_cache_input.as_ref() {
                        if idx == use_cache_input.input_index {
                            let v = use_cache_value.ok_or_else(|| {
                                anyhow::anyhow!("Decoder requires '{}' input", self.decoder_session.inputs()[idx].name())
                            })?;
                            run_inputs.push(SessionInputValue::from(v));
                            continue;
                        }
                    }

                    if let Some(past_pos) = decoder_plan
                        .cache_pos_by_input_index
                        .get(idx)
                        .and_then(|p| *p)
                    {
                        run_inputs.push(SessionInputValue::from(&past_values[past_pos]));
                        continue;
                    }

                    anyhow::bail!(
                        "Unsupported decoder input '{}' (index {})",
                        self.decoder_session.inputs()[idx].name(),
                        idx
                    );
                }

                let outputs = self.decoder_session
                    .run(run_inputs.as_slice())
                    .context("Failed to run decoder (KV-cache mode)") // add context for debugging
                    ?;

                Ok(outputs.into_iter().map(|(_name, value)| value).collect())
            };

            debug!(
                "Decoder input shapes: features={:?} (kv_cache=true, past_inputs={})",
                features_shape,
                decoder_plan.cache_inputs.len()
            );

            let cache_prefill = use_cache_off.as_ref();
            let cache_step = use_cache_on.as_ref().or(use_cache_off.as_ref());

            // Prefill phase 1: run the prompt once with `use_cache_branch=false` to let the model
            // compute cross-attention (encoder) KV caches from `encoder_hidden_states`.
            let prompt_tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
            let prompt_shape = vec![1, tokens.len()];
            let prompt_value =
                OrtValue::from_array((prompt_shape.clone(), prompt_tokens_i64))?.into_dyn();

            let prefill_outputs = run_decoder(prompt_value, cache_prefill, &past_values)?;

            // We don't trust the prefill branch to return decoder self-attention caches in the
            // exact format expected by the cache branch. But we *do* need the encoder (cross-attn)
            // KV caches, because the cache branch may ignore `encoder_hidden_states` entirely.
            past_values = prefill_outputs.into_iter().skip(1).collect();

            // Reset decoder self-attention caches back to empty, while keeping the computed
            // encoder caches from the prefill branch.
            for (pos, cache) in decoder_plan.cache_inputs.iter().enumerate() {
                if cache.name.contains(".decoder.") || cache.name.contains("decoder.key") || cache.name.contains("decoder.value") {
                    let resolved_shape =
                        resolve_decoder_cache_shape(&cache.name, &cache.shape, encoder_seq_len)?;
                    past_values[pos] = make_zero_tensor(cache.element_type, resolved_shape)?;
                }
            }

            let decoder_cache_positions: Vec<usize> = decoder_plan
                .cache_inputs
                .iter()
                .enumerate()
                .filter_map(|(pos, cache)| cache.name.contains(".decoder.").then_some(pos))
                .collect();

            // Prefill phase 2: build decoder self-attention caches on the cache branch by feeding
            // the prompt tokens token-by-token, using the encoder caches we just computed.
            let mut next_token: u32 = eot_token;
            for (pos, &prompt_token) in tokens.iter().enumerate() {
                let step_tokens_i64 = vec![prompt_token as i64];
                let step_shape = vec![1, 1];
                let step_value =
                    OrtValue::from_array((step_shape, step_tokens_i64))?.into_dyn();

                let step_outputs = run_decoder(step_value, cache_step, &past_values)?;

                if pos + 1 == tokens.len() {
                    let (logits_shape, logits_data) = step_outputs[0].try_extract_tensor::<f32>()?;
                    debug!(
                        "Decoder output logits shape: {:?}, total elements: {}",
                        logits_shape,
                        logits_data.len()
                    );

                    let last_logits =
                        extract_last_logits(logits_shape.as_ref(), logits_data, 1)?;

                    // Debug: show top 5 tokens on the first prediction
                    let mut top_tokens: Vec<(usize, f32)> = last_logits
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (i, v))
                        .collect();
                    top_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    debug!(
                        "Top 5 predicted tokens (prefill): {:?}",
                        &top_tokens[..5.min(top_tokens.len())]
                    );

                    next_token = last_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as u32)
                        .ok_or_else(|| anyhow::anyhow!("Failed to get next token"))?;
                }

                let mut present_values: Vec<OrtValue> = step_outputs.into_iter().skip(1).collect();
                if present_values.len() != past_values.len() {
                    anyhow::bail!(
                        "Decoder KV-cache outputs mismatch: expected {} present tensors, got {}",
                        past_values.len(),
                        present_values.len()
                    );
                }

                // Only update decoder caches; encoder caches are constant and the cache branch may
                // return them as passthrough/dummy values.
                for &cache_pos in &decoder_cache_positions {
                    std::mem::swap(&mut past_values[cache_pos], &mut present_values[cache_pos]);
                }
            }

            if next_token == eot_token {
                debug!("Hit EOT token immediately after prefill; stopping");
            } else {
                all_tokens.push(next_token);
                if trim_on_repeated_ngram(&mut all_tokens) {
                    debug!("Detected repeated n-gram after prefill; truncating and stopping");
                    let segments = self.parse_timestamped_tokens(&all_tokens);
                    debug!("Parsed {} segments from tokens", segments.len());
                    return Ok(segments);
                }
            }

            if all_tokens.len() < 5 {
                debug!(
                    "Token {}: {} (is_timestamp={}, is_text={}, EOT={})",
                    all_tokens.len().saturating_sub(1),
                    next_token,
                    is_timestamp_token(next_token),
                    next_token < 50257,
                    eot_token
                );
            }

            // Continue token-by-token with cache enabled (use_cache=on).
            for _ in 1..sample_len {
                if next_token == eot_token {
                    break;
                }

                let step_tokens_i64 = vec![next_token as i64];
                let step_shape = vec![1, 1];
                let step_value = OrtValue::from_array((step_shape, step_tokens_i64))?.into_dyn();

                let step_outputs = run_decoder(step_value, cache_step, &past_values)?;

                next_token = {
                    let (logits_shape, logits_data) = step_outputs[0].try_extract_tensor::<f32>()?;
                    let last_logits = extract_last_logits(logits_shape.as_ref(), logits_data, 1)?;

                    last_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx as u32)
                        .ok_or_else(|| anyhow::anyhow!("Failed to get next token"))?
                };

                if all_tokens.len() < 5 {
                    debug!(
                        "Token {}: {} (is_timestamp={}, is_text={}, EOT={})",
                        all_tokens.len(),
                        next_token,
                        is_timestamp_token(next_token),
                        next_token < 50257,
                        eot_token
                    );
                }

                if next_token == eot_token {
                    debug!("Hit EOT token at position {}, stopping", all_tokens.len());
                    break;
                }

                all_tokens.push(next_token);
                let mut present_values: Vec<OrtValue> = step_outputs.into_iter().skip(1).collect();
                if present_values.len() != past_values.len() {
                    anyhow::bail!(
                        "Decoder KV-cache outputs mismatch: expected {} present tensors, got {}",
                        past_values.len(),
                        present_values.len()
                    );
                }
                for &cache_pos in &decoder_cache_positions {
                    std::mem::swap(&mut past_values[cache_pos], &mut present_values[cache_pos]);
                }

                if trim_on_repeated_ngram(&mut all_tokens) {
                    debug!("Detected repeated n-gram during decode; truncating and stopping");
                    break;
                }

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
        } else {
            // Fallback: no KV cache available. Re-run the decoder over the full token sequence each step.
            for _ in 0..sample_len {
                // Create tokens tensor [1, seq_len]
                let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
                let tokens_shape = vec![1, tokens.len()];
                let tokens_value = OrtValue::from_array((tokens_shape.clone(), tokens_i64))?;

                // Debug on first iteration
                if all_tokens.len() == 1 {
                    debug!(
                        "Decoder input shapes: tokens={:?}, features={:?} (kv_cache=false)",
                        tokens_shape, features_shape
                    );
                }

                // Run decoder with positional inputs: [input_ids (i64), encoder_hidden_states (f32)]
                let decoder_outputs = self
                    .decoder_session
                    .run([
                        SessionInputValue::from(tokens_value),
                        SessionInputValue::from(&features_value_f32),
                    ])
                    .context("Failed to run decoder")?;

                // Get logits from output
                let (logits_shape, logits_data) = decoder_outputs[0].try_extract_tensor::<f32>()?;

                // Debug on first iteration
                if all_tokens.len() == 1 {
                    debug!(
                        "Decoder output logits shape: {:?}, total elements: {}",
                        logits_shape,
                        logits_data.len()
                    );
                }

                let last_logits =
                    extract_last_logits(logits_shape.as_ref(), logits_data, tokens.len())?;

                // Find token with highest probability (argmax)
                let next_token = last_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .ok_or_else(|| anyhow::anyhow!("Failed to get next token"))?;

                if all_tokens.len() < 5 {
                    debug!(
                        "Token {}: {} (is_timestamp={}, is_text={}, EOT={})",
                        all_tokens.len(),
                        next_token,
                        is_timestamp_token(next_token),
                        next_token < 50257,
                        eot_token
                    );
                }

                if next_token == eot_token {
                    debug!("Hit EOT token at position {}, stopping", all_tokens.len());
                    break;
                }

                all_tokens.push(next_token);
                tokens.push(next_token);

                if trim_on_repeated_ngram(&mut all_tokens) {
                    debug!("Detected repeated n-gram during decode; truncating and stopping");
                    break;
                }

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
        }

        debug!("Decoder generated {} tokens: {:?}", all_tokens.len(), &all_tokens[..all_tokens.len().min(20)]);

        let segments = self.parse_timestamped_tokens(&all_tokens);
        debug!("Parsed {} segments from tokens", segments.len());

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
    fn detect_language(&mut self, audio_features: &ArrayD<f32>) -> Result<u32> {
        let sot_token = self.token_id("<|startoftranscript|>")?;

        // Create tokens tensor [1, 1]
        let tokens_shape = vec![1, 1];
        let tokens_data = vec![sot_token as i64];
        let tokens_value = OrtValue::from_array((tokens_shape, tokens_data))?;

        let features_shape = audio_features.shape().to_vec();
        if features_shape.len() != 3 {
            anyhow::bail!(
                "Expected encoder_hidden_states to be rank-3 [batch, seq, hidden], got shape {:?}",
                features_shape
            );
        }
        let encoder_seq_len = features_shape[1];
        let features_data = audio_features.iter().copied().collect::<Vec<f32>>();
        let features_value = OrtValue::from_array((features_shape, features_data))?;

        let decoder_plan = DecoderPlan::new(&self.decoder_session)?;

        let use_cache_off = match decoder_plan.use_cache_input.as_ref() {
            Some(input) => Some(make_use_cache_value(input, false)?),
            None => None,
        };

        let mut past_values: Vec<OrtValue> = Vec::with_capacity(decoder_plan.cache_inputs.len());
        for cache in &decoder_plan.cache_inputs {
            let resolved_shape =
                resolve_decoder_cache_shape(&cache.name, &cache.shape, encoder_seq_len)?;
            past_values.push(make_zero_tensor(cache.element_type, resolved_shape)?);
        }

        // Run decoder with single SOT token and extract logits data
        let last_logits: Vec<f32> = {
            let mut run_inputs = Vec::with_capacity(decoder_plan.input_count);
            run_inputs.push(SessionInputValue::from(tokens_value));
            run_inputs.push(SessionInputValue::from(&features_value));

            for idx in 2..decoder_plan.input_count {
                if let Some(use_cache_input) = decoder_plan.use_cache_input.as_ref() {
                    if idx == use_cache_input.input_index {
                        let v = use_cache_off.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("Decoder requires '{}' input", self.decoder_session.inputs()[idx].name())
                        })?;
                        run_inputs.push(SessionInputValue::from(v));
                        continue;
                    }
                }

                if let Some(past_pos) = decoder_plan
                    .cache_pos_by_input_index
                    .get(idx)
                    .and_then(|p| *p)
                {
                    run_inputs.push(SessionInputValue::from(&past_values[past_pos]));
                    continue;
                }

                anyhow::bail!(
                    "Unsupported decoder input '{}' (index {})",
                    self.decoder_session.inputs()[idx].name(),
                    idx
                );
            }

            let decoder_outputs = self
                .decoder_session
                .run(run_inputs.as_slice())
                .context("Failed to run decoder for language detection")?;

            // Get logits for the only token position.
            let (logits_shape, logits_data) = decoder_outputs[0].try_extract_tensor::<f32>()?;
            let last_logits = match logits_shape.as_ref() {
                [1, 1, vocab] => {
                    let vocab_size = (*vocab) as usize;
                    logits_data
                        .get(..vocab_size)
                        .ok_or_else(|| anyhow::anyhow!("Logits slice out of range for shape {:?}", logits_shape))?
                }
                [1, vocab] => {
                    let vocab_size = (*vocab) as usize;
                    logits_data
                        .get(..vocab_size)
                        .ok_or_else(|| anyhow::anyhow!("Logits slice out of range for shape {:?}", logits_shape))?
                }
                other => anyhow::bail!("Unexpected logits tensor shape {:?}", other),
            };
            last_logits.to_vec()
        };

        // Language tokens are in range 50259-50358
        let lang_token_start = 50259u32;
        let lang_token_end = 50358u32;

        let mut best_lang_token = self.token_id("<|en|>")?;
        let mut best_prob = f32::NEG_INFINITY;

        for token_id in lang_token_start..=lang_token_end {
            if let Some(&prob) = last_logits.get(token_id as usize) {
                if prob > best_prob {
                    best_prob = prob;
                    best_lang_token = token_id;
                }
            }
        }

        if let Some(lang_str) = self.tokenizer.id_to_token(best_lang_token) {
            debug!("Detected language: {}", lang_str);
        }

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
    pub fn new(
        model: WhisperModel,
        language: Option<String>,
        vad_config: VadConfig,
    ) -> Result<Self> {
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

    /// Run the ASR engine with tokio async channels (for async applications)
    pub fn run(
        mut self,
        mut input: mpsc::Receiver<AsrInput>,
        output: mpsc::Sender<TranscriptionResult>,
    ) -> Result<()> {
        debug!("ASR engine started with integrated VAD");

        loop {
            let msg = match input.blocking_recv() {
                Some(m) => m,
                None => {
                    debug!("ASR engine input channel closed");
                    break;
                }
            };
            let clips = match msg {
                AsrInput::Samples(samples) => {
                    self.segmenter.push_samples(&samples)?
                }
                AsrInput::Flush => {
                    debug!("Flushing VAD segmenter and resetting timestamp position");
                    let clips = self.segmenter.flush()?;
                    // Reset timestamp position so next samples start from 0
                    // This is important for push-to-talk applications where each session
                    // should have timestamps starting from 0
                    self.segmenter.reset_position();
                    clips
                }
            };

            // Process each clip from the VAD segmenter
            for clip in clips.iter() {
                let results = self.model.transcribe_clip(
                    clip,
                    self.language.as_deref(),
                    self.initial_prompt.as_deref(),
                    self.filter.as_deref(),
                )?;

                for result in results {
                    if output.blocking_send(result).is_err() {
                        debug!("Output channel closed, stopping ASR engine");
                        return Ok(());
                    }
                }
            }
        }

        debug!("ASR engine finished");
        Ok(())
    }

    /// Run the ASR engine with standard library blocking channels (for synchronous applications)
    pub fn run_blocking(
        mut self,
        input: std_mpsc::Receiver<AsrInput>,
        output: std_mpsc::Sender<TranscriptionResult>,
    ) -> Result<()> {
        debug!("ASR engine started with integrated VAD (blocking mode)");

        loop {
            let msg = match input.recv() {
                Ok(m) => m,
                Err(_) => {
                    debug!("ASR engine input channel closed");
                    break;
                }
            };

            let clips = match msg {
                AsrInput::Samples(samples) => {
                    self.segmenter.push_samples(&samples)?
                }
                AsrInput::Flush => {
                    debug!("Flushing VAD segmenter and resetting timestamp position");
                    let clips = self.segmenter.flush()?;
                    self.segmenter.reset_position();
                    clips
                }
            };

            // Process each clip from the VAD segmenter
            for clip in clips.iter() {
                let results = self.model.transcribe_clip(
                    clip,
                    self.language.as_deref(),
                    self.initial_prompt.as_deref(),
                    self.filter.as_deref(),
                )?;

                for result in results {
                    if output.send(result).is_err() {
                        debug!("Output channel closed, stopping ASR engine");
                        return Ok(());
                    }
                }
            }
        }

        debug!("ASR engine finished");
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
