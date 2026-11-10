use crate::tokenizer::{Tokenizer, TokenizerConfig};
use crate::query::Highlight;

/// Configuration for the highlight engine.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Characters of context to include before the first match in a snippet.
    pub pre_context_chars: usize,
    /// Characters of context to include after the last match in a snippet.
    pub post_context_chars: usize,
    /// Opening tag wrapped around a match. Default `<mark>`.
    pub open_tag: String,
    /// Closing tag wrapped around a match. Default `</mark>`.
    pub close_tag: String,
    /// Maximum number of snippets to return per field.
    pub max_snippets: usize,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            pre_context_chars: 60,
            post_context_chars: 60,
            open_tag: "<mark>".into(),
            close_tag: "</mark>".into(),
            max_snippets: 3,
        }
    }
}

/// Stateless highlight engine. Clone freely.
#[derive(Clone)]
pub struct Highlighter {
    tokenizer: Tokenizer,
    config: HighlightConfig,
}

impl Highlighter {
    pub fn new(config: HighlightConfig) -> Self {
        Self {
            tokenizer: Tokenizer::new(TokenizerConfig::default()),
            config,
        }
    }

    /// Generate a `Highlight` for `field_name` by scanning `text` for any of
    /// the normalized `query_terms`. Returns `None` if no terms match.
    pub fn highlight_field(
        &self,
        field_name: &str,
        text: &str,
        query_terms: &[String],
    ) -> Option<Highlight> {
        if query_terms.is_empty() || text.is_empty() {
            return None;
        }

        let tokens = self.tokenizer.tokenize(text).ok()?;

        // Find all tokens that match a query term (case-insensitive, normalized).
        let matched_positions: Vec<(usize, usize)> = tokens
            .iter()
            .filter(|tok| query_terms.iter().any(|qt| qt == &tok.text))
            .map(|tok| (tok.offset_start, tok.offset_end))
            .collect();

        if matched_positions.is_empty() {
            return None;
        }

        let matched_tokens: Vec<String> = tokens
            .iter()
            .filter(|tok| query_terms.iter().any(|qt| qt == &tok.text))
            .map(|tok| tok.text.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Build snippets by grouping nearby matches and extracting context windows.
        let text_bytes = text.as_bytes();
        let text_len = text.len();
        let snippet = self.build_snippet(text, text_bytes, text_len, &matched_positions);

        Some(Highlight {
            field: field_name.to_owned(),
            snippet,
            matched_tokens,
        })
    }

    fn build_snippet(
        &self,
        text: &str,
        _text_bytes: &[u8],
        text_len: usize,
        matched: &[(usize, usize)],
    ) -> String {
        // Group overlapping / nearby match spans into clusters.
        let clusters = cluster_spans(matched, self.config.pre_context_chars + self.config.post_context_chars);

        let mut result = String::new();
        let mut snippets_emitted = 0;

        for cluster in &clusters {
            if snippets_emitted >= self.config.max_snippets {
                break;
            }

            // Context window around the cluster.
            let cluster_start = cluster.first().map(|(s, _)| *s).unwrap_or(0);
            let cluster_end = cluster.last().map(|(_, e)| *e).unwrap_or(0);

            let win_start = char_safe_start(text, cluster_start.saturating_sub(self.config.pre_context_chars));
            let win_end = char_safe_end(text, (cluster_end + self.config.post_context_chars).min(text_len));

            let window = &text[win_start..win_end];

            // Insert tags by rebuilding the window with offsets re-based to window start.
            let mut tagged = String::with_capacity(window.len() + cluster.len() * 20);
            let mut cursor = win_start;

            for &(span_start, span_end) in cluster {
                // Skip spans outside the window (shouldn't happen but guard anyway).
                if span_start < win_start || span_end > win_end {
                    continue;
                }
                // Append text before this span.
                tagged.push_str(&text[cursor..span_start]);
                tagged.push_str(&self.config.open_tag);
                tagged.push_str(&text[span_start..span_end]);
                tagged.push_str(&self.config.close_tag);
                cursor = span_end;
            }
            // Append remaining text in window.
            tagged.push_str(&text[cursor..win_end]);

            if win_start > 0 {
                result.push_str("…");
            }
            result.push_str(&tagged);
            if win_end < text_len {
                result.push_str("…");
            }

            snippets_emitted += 1;
        }

        result
    }
}

/// Group spans that are within `merge_gap` bytes of each other into clusters.
fn cluster_spans(spans: &[(usize, usize)], merge_gap: usize) -> Vec<Vec<(usize, usize)>> {
    if spans.is_empty() {
        return vec![];
    }
    let mut clusters: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current: Vec<(usize, usize)> = vec![spans[0]];

    for &span in &spans[1..] {
        let last_end = current.last().map(|(_, e)| *e).unwrap_or(0);
        if span.0 <= last_end + merge_gap {
            current.push(span);
        } else {
            clusters.push(current);
            current = vec![span];
        }
    }
    clusters.push(current);
    clusters
}

/// Walk backwards from `pos` to a valid UTF-8 char boundary.
fn char_safe_start(text: &str, pos: usize) -> usize {
    let mut p = pos.min(text.len());
    while p > 0 && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

/// Walk forwards from `pos` to a valid UTF-8 char boundary.
fn char_safe_end(text: &str, pos: usize) -> usize {
    let mut p = pos.min(text.len());
    while p < text.len() && !text.is_char_boundary(p) {
        p += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highlights_matching_term() {
        let hl = Highlighter::new(HighlightConfig::default());
        let text = "Rust is a systems programming language. It is fast and safe.";
        let result = hl.highlight_field("body", text, &["rust".into()]);
        assert!(result.is_some());
        let h = result.unwrap();
        assert!(h.snippet.contains("<mark>Rust</mark>") || h.snippet.contains("<mark>rust</mark>"));
    }

    #[test]
    fn no_match_returns_none() {
        let hl = Highlighter::new(HighlightConfig::default());
        let text = "Hello world";
        let result = hl.highlight_field("body", text, &["nexus".into()]);
        assert!(result.is_none());
    }

    #[test]
    fn multiple_matches_in_one_snippet() {
        let hl = Highlighter::new(HighlightConfig::default());
        let text = "async rust is great for async programming";
        let result = hl.highlight_field("body", text, &["async".into(), "rust".into()]);
        assert!(result.is_some());
        let h = result.unwrap();
        assert!(h.snippet.contains("<mark>"));
    }

    #[test]
    fn custom_tags() {
        let hl = Highlighter::new(HighlightConfig {
            open_tag: "<b>".into(),
            close_tag: "</b>".into(),
            ..Default::default()
        });
        let text = "search engines are powerful tools";
        let result = hl.highlight_field("body", text, &["search".into()]);
        assert!(result.is_some());
        assert!(result.unwrap().snippet.contains("<b>"));
    }
}
