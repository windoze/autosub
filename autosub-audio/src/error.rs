use std::io;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("Failed to initialize FFmpeg: {0}")]
    FfmpegInit(String),

    #[error("Failed to open file: {0}")]
    FileOpen(#[from] io::Error),

    #[error("No audio stream found in file")]
    NoAudioStream,

    #[error("Failed to decode audio: {0}")]
    Decode(String),

    #[error("Failed to resample audio: {0}")]
    Resample(String),

    #[error("Unsupported sample rate: {0}")]
    UnsupportedSampleRate(u32),

    #[error("Channel send failed: {0}")]
    ChannelSend(String),

    #[error("FFmpeg error: {0}")]
    Ffmpeg(#[from] ffmpeg_next::Error),

    #[error("WAV error: {0}")]
    Wav(#[from] hound::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AudioError>;
