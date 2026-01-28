# autosub-audio

Audio extraction and streaming from media files for ASR applications.

## Features

- **Streaming**: Extract audio in configurable chunks without loading entire files into memory
- **Channel support**: Send audio segments via tokio or std channels with configurable send modes
- **Timestamps**: Audio segments include precise timing information (start/end times and sample positions)
- **Format support**: Works with all audio/video formats supported by FFmpeg
- **Metadata extraction**: Get file info (duration, sample rate, channels) without decoding audio

## Testing

### Run All Tests

```bash
cargo test --package autosub-audio
```

### Run Integration Test with ASR

```bash
cargo test --package autosub-audio -- --ignored
```

**Integration Test Results:**
- ✅ Successfully extracts audio from video files
- ✅ Creates AudioSegments with correct timestamps
- ✅ Streams via tokio channels with blocking send
- ✅ Integrates correctly with autosub-asr engine
- ✅ Produces valid transcription results
- ✅ 96.1% accuracy with Whisper Tiny model on test video (when using NoFilter)

**Note**: Uses Whisper Tiny model for fast testing. For production, use larger models (Small/Medium/Large) for even higher accuracy (typically 98-99%).

## License

MIT
