use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use llm::{
    builder::{LLMBackend, LLMBuilder},
    chat::ChatMessage,
};
use tracing::info;

use crate::config::LlmProvider;
use crate::srt::{SrtWriter, Subtitle, SubtitleEntry};

/// Convert our LlmProvider enum to llm crate's LLMBackend
fn to_llm_backend(provider: LlmProvider) -> LLMBackend {
    match provider {
        LlmProvider::Openai => LLMBackend::OpenAI,
        LlmProvider::Anthropic => LLMBackend::Anthropic,
        LlmProvider::Google => LLMBackend::Google,
        LlmProvider::Ollama => LLMBackend::Ollama,
        LlmProvider::Deepseek => LLMBackend::DeepSeek,
    }
}

/// Translate subtitles using an LLM provider and stream to output file
pub async fn translate_subtitles_to_file(
    subtitle: &Subtitle,
    output_path: &Path,
    target_language: &str,
    provider: LlmProvider,
    api_key: &str,
    model: &str,
    base_url: Option<&str>,
    batch_size: usize,
) -> Result<Subtitle> {
    let mut writer = SrtWriter::create(output_path)?;
    let result = translate_subtitles_to_writer(
        subtitle,
        &mut writer,
        target_language,
        provider,
        api_key,
        model,
        base_url,
        batch_size,
    )
    .await?;
    writer.finish()?;
    Ok(result)
}

/// Translate subtitles using an LLM provider, streaming output to writer
pub async fn translate_subtitles_to_writer<W: Write>(
    subtitle: &Subtitle,
    writer: &mut SrtWriter<W>,
    target_language: &str,
    provider: LlmProvider,
    api_key: &str,
    model: &str,
    base_url: Option<&str>,
    batch_size: usize,
) -> Result<Subtitle> {
    info!(
        "Translating {} subtitle entries to {} using {:?}...",
        subtitle.len(),
        target_language,
        provider
    );

    let system_prompt = format!(
        "You are a professional subtitle translator. Translate the following subtitle text to {}.\n\
         Rules:\n\
         - Preserve the [N] markers at the start of each line\n\
         - Keep translations concise and natural for subtitles\n\
         - Maintain the original meaning and tone\n\
         - Do not add explanations or notes\n\
         - Return only the translated text with the same [N] format",
        target_language
    );

    // Build the LLM client with system prompt
    let mut builder = LLMBuilder::new()
        .backend(to_llm_backend(provider))
        .api_key(api_key)
        .model(model)
        .system(&system_prompt);

    // Set custom base URL if provided (for Azure OpenAI, etc.)
    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }

    let llm = builder.build().context("Failed to build LLM client")?;

    let mut translated = Subtitle::new();
    let pb = ProgressBar::new(subtitle.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Batch translations for efficiency
    let batches: Vec<_> = subtitle.entries.chunks(batch_size).collect();

    for batch in batches {
        let batch_texts: Vec<String> = batch
            .iter()
            .enumerate()
            .map(|(i, e)| format!("[{}] {}", i + 1, e.text))
            .collect();
        let batch_text = batch_texts.join("\n");

        let messages = vec![ChatMessage::user().content(&batch_text).build()];

        let response = llm
            .chat(&messages)
            .await
            .context("Failed to call translation API")?;

        let translated_text = response
            .text()
            .context("Empty response from translation API")?;

        // Parse the translated text back to entries
        let translated_lines: Vec<&str> = translated_text.lines().collect();

        for (i, entry) in batch.iter().enumerate() {
            let translated_line = translated_lines
                .iter()
                .find(|line| line.starts_with(&format!("[{}]", i + 1)))
                .map(|line| {
                    // Remove the [N] prefix
                    line.trim_start_matches(&format!("[{}]", i + 1))
                        .trim()
                        .to_string()
                })
                .unwrap_or_else(|| {
                    // Fallback: if parsing fails, use the original line or portion of response
                    translated_lines
                        .get(i)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| entry.text.clone())
                });

            // Write to file immediately and flush
            writer.write_entry(entry.start, entry.end, &translated_line)?;

            translated.push_entry(SubtitleEntry::new(
                entry.index,
                entry.start,
                entry.end,
                translated_line,
            ));
            pb.inc(1);
        }
    }

    pb.finish_with_message("Translation complete");
    info!("Translation complete");

    Ok(translated)
}

/// Translate subtitles using an LLM provider (returns Subtitle without streaming)
pub async fn translate_subtitles(
    subtitle: &Subtitle,
    target_language: &str,
    provider: LlmProvider,
    api_key: &str,
    model: &str,
    base_url: Option<&str>,
    batch_size: usize,
) -> Result<Subtitle> {
    info!(
        "Translating {} subtitle entries to {} using {:?}...",
        subtitle.len(),
        target_language,
        provider
    );

    let system_prompt = format!(
        "You are a professional subtitle translator. Translate the following subtitle text to {}.\n\
         Rules:\n\
         - Preserve the [N] markers at the start of each line\n\
         - Keep translations concise and natural for subtitles\n\
         - Maintain the original meaning and tone\n\
         - Do not add explanations or notes\n\
         - Return only the translated text with the same [N] format",
        target_language
    );

    // Build the LLM client with system prompt
    let mut builder = LLMBuilder::new()
        .backend(to_llm_backend(provider))
        .api_key(api_key)
        .model(model)
        .system(&system_prompt);

    // Set custom base URL if provided (for Azure OpenAI, etc.)
    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }

    let llm = builder.build().context("Failed to build LLM client")?;

    let mut translated = Subtitle::new();
    let pb = ProgressBar::new(subtitle.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Batch translations for efficiency
    let batches: Vec<_> = subtitle.entries.chunks(batch_size).collect();

    for batch in batches {
        let batch_texts: Vec<String> = batch
            .iter()
            .enumerate()
            .map(|(i, e)| format!("[{}] {}", i + 1, e.text))
            .collect();
        let batch_text = batch_texts.join("\n");

        let messages = vec![ChatMessage::user().content(&batch_text).build()];

        let response = llm
            .chat(&messages)
            .await
            .context("Failed to call translation API")?;

        let translated_text = response
            .text()
            .context("Empty response from translation API")?;

        // Parse the translated text back to entries
        let translated_lines: Vec<&str> = translated_text.lines().collect();

        for (i, entry) in batch.iter().enumerate() {
            let translated_line = translated_lines
                .iter()
                .find(|line| line.starts_with(&format!("[{}]", i + 1)))
                .map(|line| {
                    // Remove the [N] prefix
                    line.trim_start_matches(&format!("[{}]", i + 1))
                        .trim()
                        .to_string()
                })
                .unwrap_or_else(|| {
                    // Fallback: if parsing fails, use the original line or portion of response
                    translated_lines
                        .get(i)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| entry.text.clone())
                });

            translated.push_entry(SubtitleEntry::new(
                entry.index,
                entry.start,
                entry.end,
                translated_line,
            ));
            pb.inc(1);
        }
    }

    pb.finish_with_message("Translation complete");
    info!("Translation complete");

    Ok(translated)
}

/// Simple single-entry translation for testing
#[allow(dead_code)]
pub async fn translate_text(
    text: &str,
    target_language: &str,
    provider: LlmProvider,
    api_key: &str,
    model: &str,
    base_url: Option<&str>,
) -> Result<String> {
    let system_prompt = format!(
        "Translate the following text to {}. Return only the translation, nothing else.",
        target_language
    );

    let mut builder = LLMBuilder::new()
        .backend(to_llm_backend(provider))
        .api_key(api_key)
        .model(model)
        .system(&system_prompt);

    if let Some(url) = base_url {
        builder = builder.base_url(url);
    }

    let llm = builder.build().context("Failed to build LLM client")?;

    let messages = vec![ChatMessage::user().content(text).build()];

    let response = llm
        .chat(&messages)
        .await
        .context("Failed to call translation API")?;

    let translated = response
        .text()
        .context("Empty response from translation API")?;

    Ok(translated)
}

#[cfg(test)]
mod tests {
    // Translation tests would require mocking the API
    // or integration tests with actual API credentials
}
