# autosub

[English](README.md)

一个使用 Whisper 进行视频/音频转录和字幕生成的命令行工具，支持 LLM 驱动的翻译功能。

## 功能特点

- 使用 OpenAI Whisper 模型将视频和音频文件转录为 SRT 字幕
- 支持 GPU 加速（Apple Silicon 使用 Metal，NVIDIA 显卡使用 CUDA）
- 自动语言检测或手动指定语言
- 实时流式输出 SRT 文件，转录过程中即时写入
- 支持多个 LLM 提供商进行字幕翻译（OpenAI、Anthropic、Google、Ollama、DeepSeek）
- 支持翻译现有 SRT 文件，无需重新转录

## 安装

### 前置要求

- [Rust](https://rustup.rs/)（1.70 或更高版本）
- [FFmpeg](https://ffmpeg.org/)（用于音频提取）

### 从源码构建

```bash
# 克隆仓库
git clone https://github.com/user/autosub.git
cd autosub

# 构建（macOS 自动启用 Metal，其他平台使用 CPU）
cargo build --release

# 使用 CUDA 加速构建（NVIDIA GPU，需要安装 CUDA 工具包）
cargo build --release --features cuda
```

编译后的二进制文件位于 `target/release/autosub`。

## 使用方法

### 基本转录

```bash
# 转录视频文件（自动检测语言）
autosub video.mp4

# 指定语言进行转录
autosub video.mp4 --language zh

# 转录音频文件
autosub audio.mp3

# 指定输出文件
autosub video.mp4 --output subtitles.srt

# 使用不同的 Whisper 模型
autosub video.mp4 --model large
```

### 翻译功能

```bash
# 转录并翻译为中文
autosub video.mp4 --translate zh

# 使用 Anthropic Claude 进行翻译
autosub video.mp4 --translate zh --llm-provider anthropic --llm-api-key YOUR_KEY

# 使用 DeepSeek 进行翻译
autosub video.mp4 --translate en --llm-provider deepseek --llm-api-key YOUR_KEY

# 仅翻译现有 SRT 文件
autosub existing.srt --translate-only --translate en --llm-api-key YOUR_KEY
```

### 环境变量

可以通过环境变量设置 API 密钥和偏好：

```bash
export AUTOSUB_LLM_PROVIDER=openai
export AUTOSUB_LLM_API_KEY=your-api-key
export AUTOSUB_LLM_MODEL=gpt-4o-mini
export AUTOSUB_LLM_URL=https://api.openai.com/v1  # 可选，用于自定义端点
```

## 命令行选项

```
用法: autosub [选项] <输入文件>

参数:
  <INPUT>  输入视频/音频文件路径（使用 --translate-only 时为 SRT 文件）

选项:
  -o, --output <FILE>           输出 SRT 文件路径（默认：输入文件名.srt）
  -m, --model <MODEL>           Whisper 模型大小 [默认: small]
                                [可选值: tiny, base, small, medium, large]
  -l, --language <LANG>         源语言代码（如 'en', 'zh', 'ja'）
      --translate <LANG>        翻译字幕到目标语言
      --translate-only          仅翻译现有 SRT 文件（跳过转录）
      --llm-provider <PROVIDER> LLM 提供商 [默认: openai]
                                [可选值: openai, anthropic, google, ollama, deepseek]
      --llm-url <URL>           LLM API 基础 URL（用于自定义端点）
      --llm-api-key <KEY>       翻译用的 LLM API 密钥
      --llm-model <MODEL>       LLM 模型名称 [默认: gpt-4o-mini]
      --cache-dir <DIR>         模型缓存目录
      --device <DEVICE>         使用的设备 [默认: auto]
                                [可选值: auto, cpu, cuda, metal]
  -v, --verbose                 启用详细输出
  -h, --help                    打印帮助信息
  -V, --version                 打印版本信息
```

`auto` 设备选项会自动选择最佳可用设备，并在不可用时回退：
- macOS：Metal GPU → CPU
- Linux（NVIDIA 显卡）：CUDA → CPU
- 其他情况：CPU

## Whisper 模型

| 模型   | 大小    | 速度   | 质量   | 显存需求 |
|--------|---------|--------|--------|----------|
| tiny   | ~39MB   | 最快   | 较低   | ~1GB     |
| base   | ~74MB   | 快     | 基础   | ~1GB     |
| small  | ~244MB  | 中等   | 良好   | ~2GB     |
| medium | ~769MB  | 较慢   | 更好   | ~5GB     |
| large  | ~1.5GB  | 最慢   | 最佳   | ~10GB    |

模型会在首次使用时自动从 Hugging Face 下载并缓存到本地。

## 使用 Hugging Face 镜像

如果您所在地区访问 Hugging Face 较慢，可以通过设置 `HF_ENDPOINT` 环境变量使用镜像：

```bash
# 使用 hf-mirror.com（国内常用镜像）
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常运行 autosub
autosub video.mp4
```

`hf-hub` crate 会自动识别此环境变量，从指定的镜像下载模型。

### 其他可用镜像

```bash
# 使用阿里云镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或者使用其他镜像站点
export HF_ENDPOINT=https://huggingface.sukaka.top
```

## 支持的文件格式

### 视频
mp4, mkv, avi, mov, wmv, flv, webm, m4v, mpeg, mpg, 3gp

### 音频
wav, mp3, flac, m4a, aac, ogg, opus, wma, aiff, aif

## 翻译 LLM 提供商

| 提供商    | 环境变量 | 默认模型 |
|-----------|---------|----------|
| OpenAI    | `OPENAI_API_KEY` 或 `AUTOSUB_LLM_API_KEY` | gpt-4o-mini |
| Anthropic | `ANTHROPIC_API_KEY` 或 `AUTOSUB_LLM_API_KEY` | claude-3-haiku-20240307 |
| Google    | `GOOGLE_API_KEY` 或 `AUTOSUB_LLM_API_KEY` | gemini-pro |
| DeepSeek  | `DEEPSEEK_API_KEY` 或 `AUTOSUB_LLM_API_KEY` | deepseek-chat |
| Ollama    | 无需（本地运行） | llama2 |

## 使用示例

```bash
# 转录中文视频并翻译为英文
autosub chinese_video.mp4 --language zh --translate en --llm-api-key $OPENAI_API_KEY

# 使用 large 模型获得更好的准确性
autosub lecture.mp4 --model large --language en

# 使用 Ollama 进行本地翻译（无需 API 密钥）
autosub video.mp4 --translate zh --llm-provider ollama --llm-model qwen2.5

# 使用 DeepSeek 翻译现有字幕
autosub movie.srt --translate-only --translate zh \
  --llm-provider deepseek --llm-api-key $DEEPSEEK_API_KEY

# 国内用户完整示例（使用镜像 + DeepSeek）
export HF_ENDPOINT=https://hf-mirror.com
export AUTOSUB_LLM_PROVIDER=deepseek
export AUTOSUB_LLM_API_KEY=your-deepseek-key
autosub video.mp4 --translate en
```

## 常见问题

### 模型下载慢怎么办？

设置 Hugging Face 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 如何使用国内的 LLM 服务？

推荐使用 DeepSeek，价格实惠且效果好：
```bash
autosub video.mp4 --translate en --llm-provider deepseek --llm-api-key YOUR_KEY
```

### 如何使用本地 LLM？

可以使用 Ollama 运行本地模型：
```bash
# 先启动 Ollama 并拉取模型
ollama pull qwen2.5

# 使用本地模型翻译
autosub video.mp4 --translate zh --llm-provider ollama --llm-model qwen2.5
```

## 开发

### Mel 滤波器组

Mel 滤波器文件（`src/melfilters.bytes` 和 `src/melfilters128.bytes`）是从 OpenAI Whisper Python 库预计算得到的。它们包含用于将 FFT 频谱图转换为梅尔刻度频谱图的三角滤波器组矩阵。

如需重新生成这些文件（需要 `openai-whisper` Python 包）：

```bash
pip install openai-whisper
python scripts/generate_melfilters.py
```

- `melfilters.bytes` - 80 个梅尔频段，用于 tiny/base/small/medium 模型
- `melfilters128.bytes` - 128 个梅尔频段，用于 large-v3 模型

## 许可证

MIT
