/// BM25+ scoring (Lv & Zhai, 2011 — avoids the under-counting of BM25 for long docs).
///
/// Parameters:
///   k1 = 1.2  (term saturation)
///   b  = 0.75 (length normalization)
///   δ  = 0.5  (BM25+ lower-bound)
#[derive(Clone, Debug)]
pub struct Bm25Scorer {
    pub k1: f32,
    pub b: f32,
    pub delta: f32,
    /// Average document length (in tokens) across the corpus.
    pub avg_doc_len: f32,
    /// Total number of documents in the index.
    pub doc_count: u64,
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            delta: 0.5,
            avg_doc_len: 100.0,
            doc_count: 1,
        }
    }
}

impl Bm25Scorer {
    /// Score a single (query term, document) pair.
    ///
    /// - `tf`    : term frequency in the document
    /// - `df`    : document frequency of the term across corpus
    /// - `doc_len`: length of the document in tokens
    /// - `field_boost`: per-field boost factor from the schema
    pub fn score(&self, tf: u32, df: u64, doc_len: u32, field_boost: f32) -> f32 {
        let n = self.doc_count as f32;
        let df = df as f32;
        // IDF with smoothing to prevent negative values when df > n/2
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        let tf = tf as f32;
        let dl = doc_len as f32;
        let norm = tf * (self.k1 + 1.0)
            / (tf + self.k1 * (1.0 - self.b + self.b * dl / self.avg_doc_len));
        field_boost * idf * (norm + self.delta)
    }
}

/// Reciprocal Rank Fusion — merges BM25 and vector rankings.
///
/// Each list contributes `1 / (k + rank)` per document.
/// Classic default: k = 60.
pub struct Rrf {
    pub k: f32,
}

impl Default for Rrf {
    fn default() -> Self {
        Self { k: 60.0 }
    }
}

impl Rrf {
    /// Fuse two ranked lists into a single score map.
    ///
    /// `bm25_hits`   : (doc_id, bm25_score) ordered best-first
    /// `vector_hits` : (doc_id, cosine_sim) ordered best-first
    /// `alpha`       : weight on the BM25 side (1-alpha goes to vector)
    pub fn fuse(
        &self,
        bm25_hits: &[(uuid::Uuid, f32)],
        vector_hits: &[(uuid::Uuid, f32)],
        alpha: f32,
    ) -> Vec<(uuid::Uuid, f32)> {
        use std::collections::HashMap;

        let mut scores: HashMap<uuid::Uuid, f32> = HashMap::new();

        for (rank, (id, _)) in bm25_hits.iter().enumerate() {
            *scores.entry(*id).or_default() +=
                alpha / (self.k + rank as f32 + 1.0);
        }
        for (rank, (id, _)) in vector_hits.iter().enumerate() {
            *scores.entry(*id).or_default() +=
                (1.0 - alpha) / (self.k + rank as f32 + 1.0);
        }

        let mut fused: Vec<(uuid::Uuid, f32)> = scores.into_iter().collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused
    }
}

/// Typo-tolerance distance budget for a query term.
///
/// Industry-standard heuristic (matches Meilisearch / Typesense defaults):
///   len < 4  → 0 typos allowed
///   len < 8  → 1 typo
///   len >= 8 → 2 typos
pub fn typo_budget(term_len: usize) -> usize {
    match term_len {
        0..=3 => 0,
        4..=7 => 1,
        _ => 2,
    }
}

/// Fast Levenshtein distance with early-exit at `max_dist`.
/// Returns `None` if distance exceeds `max_dist`.
pub fn edit_distance(a: &str, b: &str, max_dist: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();

    // Quick length check
    if m.abs_diff(n) > max_dist {
        return None;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        curr[0] = i;
        let mut row_min = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j - 1] + cost)
                .min(prev[j] + 1)
                .min(curr[j - 1] + 1);
            row_min = row_min.min(curr[j]);
        }
        if row_min > max_dist {
            return None;
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let d = prev[n];
    if d <= max_dist { Some(d) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_increases_with_tf() {
        let scorer = Bm25Scorer {
            avg_doc_len: 50.0,
            doc_count: 1000,
            ..Default::default()
        };
        let s1 = scorer.score(1, 50, 50, 1.0);
        let s2 = scorer.score(5, 50, 50, 1.0);
        assert!(s2 > s1, "higher TF should yield higher score");
    }

    #[test]
    fn edit_distance_exact() {
        assert_eq!(edit_distance("rust", "rust", 2), Some(0));
    }

    #[test]
    fn edit_distance_one_typo() {
        assert_eq!(edit_distance("ruts", "rust", 2), Some(2)); // transposition = 2 ops
        assert_eq!(edit_distance("rxst", "rust", 2), Some(1));
    }

    #[test]
    fn edit_distance_exceeds_budget() {
        assert_eq!(edit_distance("hello", "world", 2), None);
    }

    #[test]
    fn rrf_merges_correctly() {
        use uuid::Uuid;
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        let rrf = Rrf::default();
        let fused = rrf.fuse(
            &[(a, 1.0), (b, 0.8)],
            &[(b, 0.95), (c, 0.7)],
            0.5,
        );
        // b should rank first because it appears in both lists
        assert_eq!(fused[0].0, b);
    }
}
