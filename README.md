# autosub

[中文文档](README.zh-CN.md)

A CLI tool for video/audio transcription and subtitle generation using Whisper, with optional LLM-powered translation.

## Features

- Transcribe video and audio files to SRT subtitles using OpenAI's Whisper model
- GPU acceleration support (Metal for Apple Silicon, CUDA for NVIDIA)
- Automatic language detection or manual language specification
- Stream output to SRT file in real-time as transcription progresses
- Translate subtitles using multiple LLM providers (OpenAI, Anthropic, Google, Ollama, DeepSeek)
- Translate existing SRT files without re-transcription

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) (1.70 or later)
- [FFmpeg](https://ffmpeg.org/) (required for audio extraction)

### Build from source

```bash
# Clone the repository
git clone https://github.com/user/autosub.git
cd autosub

# Build with default settings (Metal enabled on macOS)
cargo build --release

# Build with CUDA acceleration (NVIDIA GPU, requires CUDA toolkit)
cargo build --release --features cuda --no-default-features

# Build for CPU only
cargo build --release --no-default-features
```

The binary will be available at `target/release/autosub`.

## Usage

### Basic Transcription

```bash
# Transcribe a video file (auto-detect language)
autosub video.mp4

# Transcribe with specific language
autosub video.mp4 --language en

# Transcribe an audio file
autosub audio.mp3

# Specify output file
autosub video.mp4 --output subtitles.srt

# Use a different Whisper model
autosub video.mp4 --model large
```

### Translation

```bash
# Transcribe and translate to Chinese
autosub video.mp4 --translate zh

# Translate using Anthropic Claude
autosub video.mp4 --translate zh --llm-provider anthropic --llm-api-key YOUR_KEY

# Translate using DeepSeek
autosub video.mp4 --translate en --llm-provider deepseek --llm-api-key YOUR_KEY

# Translate an existing SRT file only
autosub existing.srt --translate-only --translate en --llm-api-key YOUR_KEY
```

### Environment Variables

You can set API keys and preferences via environment variables:

```bash
export AUTOSUB_LLM_PROVIDER=openai
export AUTOSUB_LLM_API_KEY=your-api-key
export AUTOSUB_LLM_MODEL=gpt-4o-mini
export AUTOSUB_LLM_URL=https://api.openai.com/v1  # Optional, for custom endpoints
```

## Command Line Options

```
Usage: autosub [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input video/audio file path (or SRT file with --translate-only)

Options:
  -o, --output <FILE>           Output SRT file path (default: input.srt)
  -m, --model <MODEL>           Whisper model size [default: small]
                                [possible values: tiny, base, small, medium, large]
  -l, --language <LANG>         Source language code (e.g., 'en', 'zh', 'ja')
      --translate <LANG>        Translate subtitles to target language
      --translate-only          Translate existing SRT file only (skip transcription)
      --llm-provider <PROVIDER> LLM provider [default: openai]
                                [possible values: openai, anthropic, google, ollama, deepseek]
      --llm-url <URL>           LLM API base URL (for custom endpoints)
      --llm-api-key <KEY>       LLM API key for translation
      --llm-model <MODEL>       LLM model name [default: gpt-4o-mini]
      --cache-dir <DIR>         Model cache directory
      --device <DEVICE>         Device to use [default: auto]
                                [possible values: auto, cpu, cuda, metal]
  -v, --verbose                 Enable verbose output
  -h, --help                    Print help
  -V, --version                 Print version
```

The `auto` device option automatically selects the best available device with fallback:
- On macOS: Metal GPU → CPU
- On Linux with NVIDIA GPU: CUDA → CPU
- Otherwise: CPU

## Whisper Models

| Model  | Size   | Speed  | Quality | VRAM Required |
|--------|--------|--------|---------|---------------|
| tiny   | ~39MB  | Fastest | Lower   | ~1GB          |
| base   | ~74MB  | Fast    | Basic   | ~1GB          |
| small  | ~244MB | Medium  | Good    | ~2GB          |
| medium | ~769MB | Slower  | Better  | ~5GB          |
| large  | ~1.5GB | Slowest | Best    | ~10GB         |

Models are automatically downloaded from Hugging Face on first use and cached locally.

## Using Hugging Face Mirror

If you're in a region with slow access to Hugging Face, you can use a mirror by setting the `HF_ENDPOINT` environment variable:

```bash
# Use hf-mirror.com (commonly used in China)
export HF_ENDPOINT=https://hf-mirror.com

# Then run autosub normally
autosub video.mp4
```

The `hf-hub` crate respects this environment variable and will download models from the specified mirror.

## Supported File Formats

### Video
mp4, mkv, avi, mov, wmv, flv, webm, m4v, mpeg, mpg, 3gp

### Audio
wav, mp3, flac, m4a, aac, ogg, opus, wma, aiff, aif

## LLM Providers for Translation

| Provider  | Environment Variable | Default Model |
|-----------|---------------------|---------------|
| OpenAI    | `OPENAI_API_KEY` or `AUTOSUB_LLM_API_KEY` | gpt-4o-mini |
| Anthropic | `ANTHROPIC_API_KEY` or `AUTOSUB_LLM_API_KEY` | claude-3-haiku-20240307 |
| Google    | `GOOGLE_API_KEY` or `AUTOSUB_LLM_API_KEY` | gemini-pro |
| DeepSeek  | `DEEPSEEK_API_KEY` or `AUTOSUB_LLM_API_KEY` | deepseek-chat |
| Ollama    | N/A (local) | llama2 |

## Examples

```bash
# Transcribe a Chinese video and translate to English
autosub chinese_video.mp4 --language zh --translate en --llm-api-key $OPENAI_API_KEY

# Transcribe with the large model for better accuracy
autosub lecture.mp4 --model large --language en

# Use Ollama for local translation (no API key needed)
autosub video.mp4 --translate zh --llm-provider ollama --llm-model qwen2.5

# Translate existing subtitles using DeepSeek
autosub movie.srt --translate-only --translate zh \
  --llm-provider deepseek --llm-api-key $DEEPSEEK_API_KEY
```

## License

MIT
