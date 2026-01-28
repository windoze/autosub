# Test Audio Files

This directory contains test audio files for the `autosub-asr` crate.

## Generating Test Files

Use the `generate_sample.sh` script to create test audio files:

```bash
# Generate all predefined test files
./generate_sample.sh

# Generate a custom test file
./generate_sample.sh "Your custom text here" output_name [voice]
```

### Examples

```bash
# Generate all standard test files
./generate_sample.sh

# Generate custom test with default voice (Samantha)
./generate_sample.sh "Testing real-time transcription" test_realtime

# Generate custom test with different voice
./generate_sample.sh "Hello from Alex" test_alex Alex
```

## Standard Test Files

The script generates these standard test files:

- **test_short.wav** - Short phrase for quick testing
  - Text: "Hello world, this is a test"
  - Duration: ~0.5 seconds

- **test_longer.wav** - Longer phrase for VAD testing
  - Text: "The quick brown fox jumps over the lazy dog. This is a longer test sentence to verify the voice activity detection and transcription works correctly."
  - Duration: ~1 second

- **test_numbers.wav** - Number recognition testing
  - Text: "One, two, three, four, five."

- **test_pauses.wav** - Multiple sentences with pauses for VAD segmentation testing
  - Text: "Testing push to talk functionality with multiple pauses. First sentence. Second sentence. Third sentence."

## Audio Format

All test files are generated in Whisper-compatible format:
- Sample rate: 16000 Hz
- Channels: 1 (mono)
- Format: WAV (PCM 16-bit)

## Available Voices

To see all available voices on your system:

```bash
say -v ?
```

Common voices:
- Samantha (default, female, US English)
- Alex (male, US English)
- Victoria (female, UK English)
- Karen (female, Australian English)

## Requirements

- macOS with `say` command
- `ffmpeg` for audio conversion
- `ffprobe` for duration checking (included with ffmpeg)

## Usage in Tests

Example test code:

```rust
use std::path::PathBuf;

let test_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("test_data")
    .join("test_short.wav");

let samples = load_wav_samples(test_file.to_str().unwrap())?;
```
