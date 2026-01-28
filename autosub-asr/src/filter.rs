/// Trait for filtering potentially hallucinated transcriptions
pub trait HallucinationFilter: Send {
    /// Returns true if the text should be filtered out as a hallucination
    fn is_hallucinated(&self, text: &str) -> bool;
}

/// Default hallucination filter with heuristics for common patterns
pub struct DefaultHallucinationFilter;

impl DefaultHallucinationFilter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultHallucinationFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl HallucinationFilter for DefaultHallucinationFilter {
    fn is_hallucinated(&self, text: &str) -> bool {
        if text.chars().count() < 2 {
            return true;
        }

        // Check for suspicious scripts that often indicate hallucinations
        let suspicious_scripts = text.chars().any(|c| {
            matches!(c,
                '\u{0D80}'..='\u{0DFF}' |  // Sinhala
                '\u{1780}'..='\u{17FF}' |  // Khmer
                '\u{1200}'..='\u{137F}'    // Ethiopic
            )
        });

        if suspicious_scripts {
            return true;
        }

        // Check for repeated sentences (3+ identical sentences)
        let sentences: Vec<&str> = text
            .split(['.', '!', '?', '。', '！', '？'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if sentences.len() >= 3 {
            let mut seen = std::collections::HashMap::new();
            for sentence in &sentences {
                *seen.entry(*sentence).or_insert(0) += 1;
            }
            if seen.values().any(|&count| count >= 3) {
                return true;
            }
        }

        // Check for repeated substrings (indicates model looping)
        if let Some((longest_substr, repeat_count)) = find_longest_repeated_substring(text) {
            if repeat_count >= 5 && longest_substr.chars().count() >= 3 {
                return true;
            }
        }

        // Check for common hallucination patterns
        if text.contains("字幕")
            || text.contains("(下集")
            || text.contains("謝謝大家")
            || text.contains("www.")
            || text.contains("http")
        {
            return true;
        }

        false
    }
}

/// No-op filter that never filters anything
pub struct NoFilter;

impl HallucinationFilter for NoFilter {
    fn is_hallucinated(&self, _text: &str) -> bool {
        false
    }
}

/// Helper function to find the longest repeated substring in text
fn find_longest_repeated_substring(text: &str) -> Option<(&str, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    if n < 6 {
        return None;
    }

    let mut best_start = 0;
    let mut best_len = 0;
    let mut best_count = 0;

    // Try different substring lengths, starting from longer ones
    for len in (3..=n / 2).rev() {
        for i in 0..=(n - len) {
            let substr: String = chars[i..i + len].iter().collect();
            let count = text.matches(&substr).count();

            if count >= 2 && len > best_len {
                best_start = i;
                best_len = len;
                best_count = count;
            }
        }

        // Once we find a repeated substring, we can stop (greedy approach)
        if best_len > 0 {
            break;
        }
    }

    if best_count >= 2 && best_len > 0 {
        let substr: String = chars[best_start..best_start + best_len].iter().collect();
        // Return a static string by leaking (safe for small test strings)
        // In production, this is only used for counting, not storage
        Some((Box::leak(substr.into_boxed_str()), best_count))
    } else {
        None
    }
}
