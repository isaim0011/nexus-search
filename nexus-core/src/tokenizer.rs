use crate::error::Result;
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Supported languages for stemming / stop-word filtering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    English,
    German,
    French,
    Spanish,
    Portuguese,
    Italian,
    Dutch,
    Russian,
    Arabic,
    // Extend as needed. CJK handled separately via char-level n-grams.
    Chinese,
    Japanese,
    Korean,
}

/// A single token emitted by the tokenizer pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The normalized token text (lowercased, NFC, stripped of diacritics).
    pub text: String,
    /// Byte offset of the original token in the source string.
    pub offset_start: usize,
    pub offset_end: usize,
    /// Position in the token stream (for phrase queries).
    pub position: usize,
}

/// Tokenizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub language: Option<Language>,
    /// Minimum token length after normalization.
    pub min_token_len: usize,
    /// Maximum token length.
    pub max_token_len: usize,
    /// Strip ASCII punctuation between words.
    pub strip_punctuation: bool,
    /// Generate n-grams for prefix / partial-match support.
    pub ngram_min: Option<usize>,
    pub ngram_max: Option<usize>,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            language: None,
            min_token_len: 2,
            max_token_len: 100,
            strip_punctuation: true,
            ngram_min: None,
            ngram_max: None,
        }
    }
}

/// The main tokenizer. Stateless; clone freely.
#[derive(Clone)]
pub struct Tokenizer {
    config: TokenizerConfig,
}

impl Tokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Self { config }
    }

    /// Tokenize `text` and return a stream of `Token`s.
    pub fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        // 1. Unicode NFC normalization
        let normalized: String = text.nfc().collect();

        // 2. Word-boundary segmentation (handles CJK implicitly via Unicode rules)
        let mut tokens = Vec::new();
        let mut position = 0usize;

        for (byte_offset, word) in normalized.split_word_bound_indices() {
            let word_trimmed = word.trim();
            if word_trimmed.is_empty() {
                continue;
            }

            // 3. Lowercase
            let lowered = word_trimmed.to_lowercase();

            // 4. Strip punctuation tokens entirely
            if self.config.strip_punctuation
                && lowered.chars().all(|c| c.is_ascii_punctuation())
            {
                continue;
            }

            // 5. Length filter
            let char_count = lowered.chars().count();
            if char_count < self.config.min_token_len
                || char_count > self.config.max_token_len
            {
                continue;
            }

            tokens.push(Token {
                offset_start: byte_offset,
                offset_end: byte_offset + word.len(),
                text: lowered,
                position,
            });
            position += 1;
        }

        // 6. Optional n-gram expansion (for prefix search / CJK)
        if let (Some(min), Some(max)) = (self.config.ngram_min, self.config.ngram_max) {
            let base_tokens = tokens.clone();
            for tok in &base_tokens {
                let chars: Vec<char> = tok.text.chars().collect();
                for n in min..=max {
                    for start in 0..chars.len().saturating_sub(n - 1) {
                        let gram: String = chars[start..start + n].iter().collect();
                        if gram != tok.text {
                            tokens.push(Token {
                                text: gram,
                                offset_start: tok.offset_start,
                                offset_end: tok.offset_end,
                                position: tok.position,
                            });
                        }
                    }
                }
            }
        }

        Ok(tokens)
    }

    /// Tokenize and return just the text strings — common path for query expansion.
    pub fn terms(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.tokenize(text)?.into_iter().map(|t| t.text).collect())
    }
}

/// English stop words. Extend or replace per language.
pub fn english_stop_words() -> &'static [&'static str] {
    &[
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "not", "that",
        "this", "it", "its", "i", "we", "you", "he", "she", "they",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tokenize() {
        let tok = Tokenizer::new(TokenizerConfig::default());
        let tokens = tok.terms("Hello, World! This is Nexus.").unwrap();
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"nexus".to_string()));
        // stop-word-like but not filtered at this layer (filtering is a step above)
        assert!(tokens.contains(&"this".to_string()));
    }

    #[test]
    fn ngram_expansion() {
        let tok = Tokenizer::new(TokenizerConfig {
            ngram_min: Some(2),
            ngram_max: Some(3),
            ..Default::default()
        });
        let tokens = tok.terms("rust").unwrap();
        // should include "ru", "us", "st", "rus", "ust" in addition to "rust"
        assert!(tokens.contains(&"ru".to_string()));
        assert!(tokens.contains(&"ust".to_string()));
    }

    #[test]
    fn short_word_filtered() {
        let tok = Tokenizer::new(TokenizerConfig {
            min_token_len: 3,
            ..Default::default()
        });
        let tokens = tok.terms("I am a Rust developer").unwrap();
        assert!(!tokens.contains(&"i".to_string()));
        assert!(!tokens.contains(&"am".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"rust".to_string()));
    }
}
