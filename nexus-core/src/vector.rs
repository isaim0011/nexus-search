use crate::error::{NexusError, Result};
use serde::{Deserialize, Serialize};

/// A dense embedding vector. The inner `Vec<f32>` is always L2-normalized
/// before storage so that dot-product == cosine similarity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Create a new embedding from raw floats, normalizing in-place.
    pub fn new(raw: Vec<f32>) -> Self {
        let mut e = Self(raw);
        e.normalize();
        e
    }

    /// Dimension of the vector.
    pub fn dim(&self) -> usize {
        self.0.len()
    }

    /// L2-normalize in-place. No-op on zero vectors.
    pub fn normalize(&mut self) {
        let norm: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for x in &mut self.0 {
                *x /= norm;
            }
        }
    }

    /// Dot-product (== cosine sim for normalized vectors).
    pub fn dot(&self, other: &Self) -> f32 {
        #[cfg(feature = "simd")]
        {
            use simsimd::SpatialSimilarity;
            f32::dot(&self.0, &other.0).unwrap_or(0.0) as f32
        }
        #[cfg(not(feature = "simd"))]
        {
            self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
        }
    }

    /// Validate that dimension matches expected.
    pub fn check_dim(&self, expected: usize) -> Result<()> {
        if self.0.len() != expected {
            Err(NexusError::VectorDimMismatch {
                expected,
                actual: self.0.len(),
            })
        } else {
            Ok(())
        }
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(v: Vec<f32>) -> Self {
        Self::new(v)
    }
}

/// A flat (brute-force) in-memory HNSW-like approximate nearest neighbor index.
/// For production use with large corpora, delegate to pgvector or a dedicated ANN lib.
/// This implementation is used for testing and small indexes (< 100k docs).
#[derive(Default)]
pub struct FlatVectorIndex {
    entries: Vec<(uuid::Uuid, Embedding)>,
}

impl FlatVectorIndex {
    pub fn insert(&mut self, id: uuid::Uuid, embedding: Embedding) {
        // Replace if exists
        if let Some(pos) = self.entries.iter().position(|(eid, _)| *eid == id) {
            self.entries[pos].1 = embedding;
        } else {
            self.entries.push((id, embedding));
        }
    }

    pub fn remove(&mut self, id: &uuid::Uuid) {
        self.entries.retain(|(eid, _)| eid != id);
    }

    /// Return the top-k nearest neighbors by cosine similarity.
    pub fn search(&self, query: &Embedding, k: usize) -> Vec<(uuid::Uuid, f32)> {
        let mut scored: Vec<(uuid::Uuid, f32)> = self
            .entries
            .iter()
            .map(|(id, emb)| (*id, emb.dot(query)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization() {
        let e = Embedding::new(vec![3.0, 4.0]); // norm = 5
        let dot = e.dot(&e);
        assert!((dot - 1.0).abs() < 1e-5, "normalized dot-self should be ~1");
    }

    #[test]
    fn flat_index_top_k() {
        use uuid::Uuid;
        let mut idx = FlatVectorIndex::default();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        idx.insert(a, Embedding::new(vec![1.0, 0.0]));
        idx.insert(b, Embedding::new(vec![0.9, 0.1]));
        idx.insert(c, Embedding::new(vec![0.0, 1.0]));

        let query = Embedding::new(vec![1.0, 0.0]);
        let results = idx.search(&query, 2);
        assert_eq!(results[0].0, a); // exact match first
        assert_eq!(results[1].0, b); // closest second
    }
}
