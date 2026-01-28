pub mod config;
pub mod srt;
pub mod translate;
pub mod whisper_cli;

// Re-export audio types from autosub_audio for backward compatibility
pub use autosub_audio::{
    cleanup_temp_files, extract_audio, is_audio_file, is_media_file, is_video_file,
    AudioChunkReader, AudioStream, ExtractedAudio,
};
pub use config::Config;
pub use srt::{SrtWriter, Subtitle, SubtitleEntry};
pub use translate::{
    translate_subtitles, translate_subtitles_to_file, translate_subtitles_to_writer,
};
pub use whisper_cli::{transcribe_to_file, ProgressCallback, TranscriptionConfig};
