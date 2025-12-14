use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};

/// A single subtitle entry
#[derive(Debug, Clone)]
pub struct SubtitleEntry {
    /// Entry index (1-based)
    pub index: usize,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Subtitle text (can contain multiple lines)
    pub text: String,
}

impl SubtitleEntry {
    pub fn new(index: usize, start: f64, end: f64, text: impl Into<String>) -> Self {
        Self {
            index,
            start,
            end,
            text: text.into(),
        }
    }

    /// Format timestamp as SRT format: HH:MM:SS,mmm
    fn format_timestamp(seconds: f64) -> String {
        let total_ms = (seconds * 1000.0).round() as u64;
        let ms = total_ms % 1000;
        let total_secs = total_ms / 1000;
        let secs = total_secs % 60;
        let total_mins = total_secs / 60;
        let mins = total_mins % 60;
        let hours = total_mins / 60;

        format!("{:02}:{:02}:{:02},{:03}", hours, mins, secs, ms)
    }

    /// Parse timestamp from SRT format: HH:MM:SS,mmm
    fn parse_timestamp(s: &str) -> Result<f64> {
        let s = s.trim();
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 3 {
            anyhow::bail!("Invalid timestamp format: {}", s);
        }

        let hours: f64 = parts[0].parse().context("Invalid hours")?;
        let mins: f64 = parts[1].parse().context("Invalid minutes")?;

        let sec_parts: Vec<&str> = parts[2].split(',').collect();
        if sec_parts.len() != 2 {
            anyhow::bail!("Invalid timestamp format (missing milliseconds): {}", s);
        }

        let secs: f64 = sec_parts[0].parse().context("Invalid seconds")?;
        let ms: f64 = sec_parts[1].parse().context("Invalid milliseconds")?;

        Ok(hours * 3600.0 + mins * 60.0 + secs + ms / 1000.0)
    }
}

impl fmt::Display for SubtitleEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.index)?;
        writeln!(
            f,
            "{} --> {}",
            Self::format_timestamp(self.start),
            Self::format_timestamp(self.end)
        )?;
        writeln!(f, "{}", self.text)?;
        Ok(())
    }
}

/// A collection of subtitle entries
#[derive(Debug, Clone, Default)]
pub struct Subtitle {
    pub entries: Vec<SubtitleEntry>,
}

impl Subtitle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the subtitle
    pub fn push(&mut self, start: f64, end: f64, text: impl Into<String>) {
        let index = self.entries.len() + 1;
        self.entries.push(SubtitleEntry::new(index, start, end, text));
    }

    /// Add an entry with explicit index
    pub fn push_entry(&mut self, entry: SubtitleEntry) {
        self.entries.push(entry);
    }

    /// Parse an SRT file
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open SRT file")?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Parse SRT content from a reader
    pub fn from_reader<R: BufRead>(reader: R) -> Result<Self> {
        let mut subtitle = Self::new();
        let mut lines = reader.lines();

        while let Some(line) = lines.next() {
            let line = line?;
            let line = line.trim();

            // Skip empty lines
            if line.is_empty() {
                continue;
            }

            // Parse index
            let index: usize = line.parse().context("Failed to parse subtitle index")?;

            // Parse timestamp line
            let timestamp_line = lines
                .next()
                .context("Unexpected end of file (expected timestamp)")??;
            let timestamp_parts: Vec<&str> = timestamp_line.split("-->").collect();
            if timestamp_parts.len() != 2 {
                anyhow::bail!("Invalid timestamp line: {}", timestamp_line);
            }

            let start = SubtitleEntry::parse_timestamp(timestamp_parts[0])?;
            let end = SubtitleEntry::parse_timestamp(timestamp_parts[1])?;

            // Parse text (can be multiple lines until empty line)
            let mut text_lines = Vec::new();
            for line in lines.by_ref() {
                let line = line?;
                if line.trim().is_empty() {
                    break;
                }
                text_lines.push(line);
            }

            let text = text_lines.join("\n");
            subtitle.push_entry(SubtitleEntry::new(index, start, end, text));
        }

        Ok(subtitle)
    }

    /// Write subtitle to an SRT file
    pub fn to_file(&self, path: &Path) -> Result<()> {
        let mut file = File::create(path).context("Failed to create SRT file")?;
        self.write_to(&mut file)
    }

    /// Write subtitle to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        for (i, entry) in self.entries.iter().enumerate() {
            // Use sequential index
            writeln!(writer, "{}", i + 1)?;
            writeln!(
                writer,
                "{} --> {}",
                SubtitleEntry::format_timestamp(entry.start),
                SubtitleEntry::format_timestamp(entry.end)
            )?;
            writeln!(writer, "{}", entry.text)?;
            writeln!(writer)?;
        }
        Ok(())
    }

    /// Get the total duration of the subtitle
    pub fn duration(&self) -> f64 {
        self.entries
            .last()
            .map(|e| e.end)
            .unwrap_or(0.0)
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if subtitle is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Display for Subtitle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for entry in &self.entries {
            writeln!(f, "{}", entry)?;
        }
        Ok(())
    }
}

/// Streaming SRT writer that flushes after each entry
pub struct SrtWriter<W: Write> {
    writer: BufWriter<W>,
    index: usize,
}

impl<W: Write> SrtWriter<W> {
    /// Create a new SRT writer
    pub fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::new(writer),
            index: 0,
        }
    }

    /// Write a single subtitle entry and flush immediately
    pub fn write_entry(&mut self, start: f64, end: f64, text: &str) -> Result<()> {
        self.index += 1;
        writeln!(self.writer, "{}", self.index)?;
        writeln!(
            self.writer,
            "{} --> {}",
            SubtitleEntry::format_timestamp(start),
            SubtitleEntry::format_timestamp(end)
        )?;
        writeln!(self.writer, "{}", text)?;
        writeln!(self.writer)?;
        self.writer.flush()?;
        Ok(())
    }

    /// Get the number of entries written
    pub fn count(&self) -> usize {
        self.index
    }

    /// Finish writing and return the inner writer
    pub fn finish(mut self) -> Result<W> {
        self.writer.flush()?;
        self.writer
            .into_inner()
            .map_err(|e| anyhow::anyhow!("Failed to finish SRT writer: {}", e))
    }
}

impl SrtWriter<File> {
    /// Create a new SRT writer that writes to a file
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create(path).context("Failed to create SRT file")?;
        Ok(Self::new(file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp() {
        assert_eq!(SubtitleEntry::format_timestamp(0.0), "00:00:00,000");
        assert_eq!(SubtitleEntry::format_timestamp(1.5), "00:00:01,500");
        assert_eq!(SubtitleEntry::format_timestamp(61.123), "00:01:01,123");
        assert_eq!(SubtitleEntry::format_timestamp(3661.999), "01:01:01,999");
    }

    #[test]
    fn test_parse_timestamp() {
        assert!((SubtitleEntry::parse_timestamp("00:00:00,000").unwrap() - 0.0).abs() < 0.001);
        assert!((SubtitleEntry::parse_timestamp("00:00:01,500").unwrap() - 1.5).abs() < 0.001);
        assert!((SubtitleEntry::parse_timestamp("00:01:01,123").unwrap() - 61.123).abs() < 0.001);
        assert!((SubtitleEntry::parse_timestamp("01:01:01,999").unwrap() - 3661.999).abs() < 0.001);
    }

    #[test]
    fn test_subtitle_roundtrip() {
        let mut subtitle = Subtitle::new();
        subtitle.push(0.0, 2.5, "First line");
        subtitle.push(2.5, 5.0, "Second line\nwith newline");

        let mut buffer = Vec::new();
        subtitle.write_to(&mut buffer).unwrap();

        let parsed = Subtitle::from_reader(buffer.as_slice()).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.entries[0].text, "First line");
        assert_eq!(parsed.entries[1].text, "Second line\nwith newline");
    }
}
