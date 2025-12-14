use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
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

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Base => "base",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }
}

impl std::fmt::Display for WhisperModelSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum Device {
    #[default]
    /// Automatically select the best available device (Metal > CUDA > CPU)
    Auto,
    /// Use CPU for inference
    Cpu,
    #[cfg(feature = "cuda")]
    /// Use CUDA GPU for inference
    Cuda,
    #[cfg(feature = "metal")]
    /// Use Metal GPU for inference (Apple Silicon)
    Metal,
}

impl Device {
    /// Convert to candle device with automatic fallback.
    /// For Auto mode: tries Metal (on macOS) -> CUDA -> CPU
    /// For specific device: tries that device, falls back to CPU on failure
    pub fn to_candle_device(&self) -> anyhow::Result<candle_core::Device> {
        match self {
            Self::Auto => Self::auto_select_device(),
            Self::Cpu => Ok(candle_core::Device::Cpu),
            #[cfg(feature = "cuda")]
            Self::Cuda => Self::try_cuda_with_fallback(),
            #[cfg(feature = "metal")]
            Self::Metal => Self::try_metal_with_fallback(),
        }
    }

    /// Automatically select the best available device
    fn auto_select_device() -> anyhow::Result<candle_core::Device> {
        // Try Metal first (macOS with Apple Silicon)
        #[cfg(feature = "metal")]
        {
            if let Ok(device) = candle_core::Device::new_metal(0) {
                tracing::info!("Using Metal GPU acceleration");
                return Ok(device);
            }
            tracing::debug!("Metal not available, trying next option");
        }

        // Try CUDA (NVIDIA GPU)
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = candle_core::Device::new_cuda(0) {
                tracing::info!("Using CUDA GPU acceleration");
                return Ok(device);
            }
            tracing::debug!("CUDA not available, trying next option");
        }

        // Fall back to CPU
        tracing::info!("Using CPU for inference");
        Ok(candle_core::Device::Cpu)
    }

    #[cfg(feature = "metal")]
    fn try_metal_with_fallback() -> anyhow::Result<candle_core::Device> {
        match candle_core::Device::new_metal(0) {
            Ok(device) => {
                tracing::info!("Using Metal GPU acceleration");
                Ok(device)
            }
            Err(e) => {
                tracing::warn!("Metal not available ({}), falling back to CPU", e);
                Ok(candle_core::Device::Cpu)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn try_cuda_with_fallback() -> anyhow::Result<candle_core::Device> {
        match candle_core::Device::new_cuda(0) {
            Ok(device) => {
                tracing::info!("Using CUDA GPU acceleration");
                Ok(device)
            }
            Err(e) => {
                tracing::warn!("CUDA not available ({}), falling back to CPU", e);
                Ok(candle_core::Device::Cpu)
            }
        }
    }
}

/// LLM provider for translation
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum LlmProvider {
    #[default]
    /// OpenAI API (default)
    Openai,
    /// Anthropic Claude API
    Anthropic,
    /// Google Gemini API
    Google,
    /// Local Ollama server
    Ollama,
    /// DeepSeek API
    Deepseek,
}

#[derive(Parser, Debug)]
#[command(name = "autosub")]
#[command(version, about = "CLI tool for video transcription and subtitle generation using Whisper")]
pub struct Config {
    /// Input video file path
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output SRT file path (default: input filename with .srt extension)
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Whisper model size to use
    #[arg(short, long, value_enum, default_value = "small")]
    pub model: WhisperModelSize,

    /// Source language code (e.g., 'en', 'zh', 'ja'). Auto-detect if not specified
    #[arg(short, long)]
    pub language: Option<String>,

    /// Translate subtitles to target language (e.g., 'en', 'zh', 'ja')
    #[arg(long, value_name = "LANG")]
    pub translate: Option<String>,

    /// Translate an existing SRT file only (skip transcription).
    /// When set, INPUT should be an SRT file instead of a video file.
    #[arg(long)]
    pub translate_only: bool,

    /// LLM provider for translation
    #[arg(long, value_enum, default_value = "openai", env = "AUTOSUB_LLM_PROVIDER")]
    pub llm_provider: LlmProvider,

    /// LLM API base URL (optional, for custom endpoints like Azure OpenAI)
    #[arg(long, env = "AUTOSUB_LLM_URL")]
    pub llm_url: Option<String>,

    /// LLM API key for translation
    #[arg(long, env = "AUTOSUB_LLM_API_KEY")]
    pub llm_api_key: Option<String>,

    /// LLM model name for translation
    #[arg(long, env = "AUTOSUB_LLM_MODEL", default_value = "gpt-4o-mini")]
    pub llm_model: String,

    /// Model cache directory
    #[arg(long, value_name = "DIR")]
    pub cache_dir: Option<PathBuf>,

    /// Device to use for inference (auto selects best available: Metal > CUDA > CPU)
    #[arg(long, value_enum, default_value = "auto")]
    pub device: Device,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,
}

impl Config {
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir.clone().unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("autosub")
                .join("models")
        })
    }

    pub fn output_path(&self) -> PathBuf {
        self.output.clone().unwrap_or_else(|| {
            self.input.with_extension("srt")
        })
    }

    pub fn translated_output_path(&self) -> Option<PathBuf> {
        self.translate.as_ref().map(|lang| {
            let stem = self.input.file_stem().unwrap_or_default().to_string_lossy();
            self.input.with_file_name(format!("{}.{}.srt", stem, lang))
        })
    }
}
