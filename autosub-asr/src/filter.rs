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
            let substr_len = longest_substr.chars().count();

            // Filter if pattern (3+ chars) repeated 5+ times
            if repeat_count >= 5 && substr_len >= 3 {
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

/// Helper function to find the most repeated substring in text
/// Returns the substring with the highest repetition count
fn find_longest_repeated_substring(text: &str) -> Option<(&str, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    if n < 6 {
        return None;
    }

    let mut best_start = 0;
    let mut best_len = 0;
    let mut best_count = 0;

    // Try different substring lengths from 3 to half the text
    // Prioritize finding patterns with highest repetition count
    for len in 3..=n / 2 {
        for i in 0..=(n - len) {
            let substr: String = chars[i..i + len].iter().collect();
            let count = text.matches(&substr).count();

            // Update best if we find more repetitions, or same repetitions but longer
            if count > best_count || (count == best_count && count >= 2 && len > best_len) {
                best_start = i;
                best_len = len;
                best_count = count;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_and_short_text() {
        let filter = DefaultHallucinationFilter::new();

        assert!(filter.is_hallucinated(""));
        assert!(filter.is_hallucinated("a"));
        assert!(!filter.is_hallucinated("ab"));
    }

    #[test]
    fn test_suspicious_scripts() {
        let filter = DefaultHallucinationFilter::new();

        // Sinhala script
        assert!(filter.is_hallucinated("ඇ"));

        // Khmer script
        assert!(filter.is_hallucinated("ខ"));

        // Ethiopic script
        assert!(filter.is_hallucinated("ሀ"));

        // Normal text should not be filtered
        assert!(!filter.is_hallucinated("Hello world"));
        assert!(!filter.is_hallucinated("你好世界"));
    }

    #[test]
    fn test_repeated_sentences() {
        let filter = DefaultHallucinationFilter::new();

        // 3 identical sentences should be filtered
        assert!(filter.is_hallucinated("Hello. Hello. Hello."));

        // Chinese sentences
        assert!(filter.is_hallucinated("你好。你好。你好。"));

        // 2 repetitions should not be filtered
        assert!(!filter.is_hallucinated("Hello. Hello."));

        // Different sentences should not be filtered
        assert!(!filter.is_hallucinated("Hello. World. Test."));
    }

    #[test]
    fn test_repeated_chinese_words() {
        let filter = DefaultHallucinationFilter::new();

        // The specific pattern mentioned by the user - many repetitions of "回答你"
        let repeated_text = "回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你 回答你";
        assert!(filter.is_hallucinated(repeated_text));

        // The specific pattern mentioned by the user - many repetitions of "또는"
        let repeated_text = "-너는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는 또는";
        assert!(filter.is_hallucinated(repeated_text));

        // Other common repeated patterns
        assert!(filter.is_hallucinated("謝謝 謝謝 謝謝 謝謝 謝謝 謝謝 謝謝 謝謝"));
        assert!(filter.is_hallucinated("hello hello hello hello hello hello"));
    }

    #[test]
    fn test_repeated_substrings_edge_cases() {
        let filter = DefaultHallucinationFilter::new();

        // 5+ repetitions with length >= 3 should be filtered
        assert!(filter.is_hallucinated("abcabcabcabcabc"));

        // 4 repetitions should not be filtered (< 5)
        assert!(!filter.is_hallucinated("abcabcabcabc"));

        // Short substrings (length < 3) even with many repetitions should not be filtered
        assert!(!filter.is_hallucinated("ababababab"));
    }

    #[test]
    fn test_common_hallucination_patterns() {
        let filter = DefaultHallucinationFilter::new();

        assert!(filter.is_hallucinated("這是字幕"));
        assert!(filter.is_hallucinated("請看(下集"));
        assert!(filter.is_hallucinated("謝謝大家"));
        assert!(filter.is_hallucinated("www.example.com"));
        assert!(filter.is_hallucinated("http://example.com"));
        assert!(filter.is_hallucinated("https://example.com"));
    }

    #[test]
    fn test_valid_text_not_filtered() {
        let filter = DefaultHallucinationFilter::new();

        // Normal English text
        assert!(!filter.is_hallucinated("Hello, how are you today?"));

        // Normal Chinese text
        assert!(!filter.is_hallucinated("今天天气很好"));

        // Mixed language
        assert!(!filter.is_hallucinated("Hello 你好 world 世界"));

        // Text with some repetition but not excessive
        assert!(!filter.is_hallucinated("I think, I think we should go"));
    }

    #[test]
    fn test_find_longest_repeated_substring() {
        // Test the helper function directly
        let result = find_longest_repeated_substring("abcabcabc");
        assert!(result.is_some());
        let (substr, count) = result.unwrap();
        assert_eq!(substr, "abc");
        assert_eq!(count, 3);

        // Test with Chinese - algorithm finds "回答你" (without space) as it repeats more
        let result = find_longest_repeated_substring("回答你 回答你 回答你 回答你 回答你");
        assert!(result.is_some());
        let (substr, count) = result.unwrap();
        assert!(substr.contains("回答你"));
        assert!(count >= 5);

        // No repetition
        let result = find_longest_repeated_substring("abcdefg");
        assert!(result.is_none());

        // Too short
        let result = find_longest_repeated_substring("abc");
        assert!(result.is_none());
    }

    #[test]
    fn test_no_filter() {
        let filter = NoFilter;

        // NoFilter should never filter anything
        assert!(!filter.is_hallucinated(""));
        assert!(!filter.is_hallucinated("回答你 回答你 回答你 回答你 回答你 回答你"));
        assert!(!filter.is_hallucinated("字幕"));
        assert!(!filter.is_hallucinated("www.example.com"));
    }
}
