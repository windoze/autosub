#!/bin/bash
# Script to generate test audio files using macOS 'say' command
# Converts output to 16kHz mono WAV format suitable for Whisper

set -e

# Check if ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not installed"
    exit 1
fi

# Function to generate a test audio file
generate_audio() {
    local text="$1"
    local output_name="$2"
    local voice="${3:-Samantha}"  # Default voice

    echo "Generating: $output_name"
    echo "Text: \"$text\""

    # Generate AIFF file with say
    say -v "$voice" -o "${output_name}.aiff" "$text"

    # Convert to 16kHz mono WAV (Whisper format)
    ffmpeg -i "${output_name}.aiff" -ar 16000 -ac 1 -y "${output_name}.wav" \
        -loglevel error -stats

    # Remove intermediate AIFF file
    rm "${output_name}.aiff"

    # Show file info
    local duration=$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "${output_name}.wav")
    local size=$(ls -lh "${output_name}.wav" | awk '{print $5}')

    echo "✓ Created ${output_name}.wav (${duration}s, $size)"
    echo ""
}

# Change to script directory
cd "$(dirname "$0")"

# Check if custom arguments provided
if [ $# -ge 2 ]; then
    # Custom mode: generate single file with provided text
    TEXT="$1"
    OUTPUT_NAME="$2"
    VOICE="${3:-Samantha}"

    echo "=========================================="
    echo "Generating Custom Audio File"
    echo "=========================================="
    echo ""

    generate_audio "$TEXT" "$OUTPUT_NAME" "$VOICE"

    echo "✓ Done!"
    exit 0
fi

# Default mode: generate all predefined test files
echo "=========================================="
echo "Generating Test Audio Files"
echo "=========================================="
echo ""

# Generate predefined test files
generate_audio "Hello world, this is a test" "test_short" "Samantha"

generate_audio "The quick brown fox jumps over the lazy dog. This is a longer test sentence to verify the voice activity detection and transcription works correctly." "test_longer" "Samantha"

# Generate additional test files with different characteristics
generate_audio "One, two, three, four, five." "test_numbers" "Samantha"

generate_audio "Testing push to talk functionality with multiple pauses. First sentence. Second sentence. Third sentence." "test_pauses" "Samantha"

echo "=========================================="
echo "All test files generated successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh *.wav 2>/dev/null | awk '{print "  " $9 " - " $5}' || echo "  (no wav files found)"
echo ""
echo "Usage:"
echo "  Generate all test files:    $0"
echo "  Generate custom file:       $0 <text> <output_name> [voice]"
echo ""
echo "Example:"
echo "  $0 \"Custom test phrase\" test_custom Alex"
echo ""
echo "Available voices:"
echo "  Samantha (default), Alex, Victoria, Karen, etc."
echo "  Run 'say -v ?' to see all available voices"
echo ""
