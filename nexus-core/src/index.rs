use crate::{
    document::{Document, DocumentId},
    error::Result,
    query::{Query, SearchResults},
    schema::Schema,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Runtime configuration for an index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    /// Unique name for this index.
    pub name: String,
    pub schema: Schema,
    /// BM25 k1 parameter.
    pub bm25_k1: f32,
    /// BM25 b parameter.
    pub bm25_b: f32,
    /// Typo tolerance globally enabled/disabled.
    pub typo_tolerance: bool,
    /// Maximum number of typos allowed (overrides per-field if set globally).
    pub max_typos: Option<usize>,
    /// Enable prefix search (last term in query matched as prefix).
    pub prefix_search: bool,
    /// Default hybrid alpha (BM25 weight when both text+vector given).
    pub hybrid_alpha: f32,
    /// Enable stop-word filtering.
    pub stop_words: bool,
}

impl IndexConfig {
    pub fn new(name: impl Into<String>, schema: Schema) -> Self {
        Self {
            name: name.into(),
            schema,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            typo_tolerance: true,
            max_typos: None,
            prefix_search: true,
            hybrid_alpha: 0.5,
            stop_words: true,
        }
    }
}

/// Live stats for an index.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexStats {
    pub document_count: u64,
    pub indexing_in_progress: bool,
    /// Average document length in tokens (used by BM25 scorer).
    pub avg_doc_len: f32,
    /// Total tokens across all documents.
    pub total_tokens: u64,
    /// Size of the document store in bytes (compressed).
    pub store_size_bytes: u64,
    /// Number of unique terms in the index.
    pub unique_terms: u64,
}

/// The core trait every backend must implement. Nexus is fully async.
#[async_trait]
pub trait Index: Send + Sync {
    /// Return the index configuration.
    fn config(&self) -> &IndexConfig;

    /// Add or replace a document. The document id is preserved.
    async fn upsert(&self, doc: Document) -> Result<DocumentId>;

    /// Batch upsert. More efficient than calling `upsert` in a loop.
    async fn upsert_many(&self, docs: Vec<Document>) -> Result<Vec<DocumentId>>;

    /// Remove a document by id.
    async fn delete(&self, id: DocumentId) -> Result<()>;

    /// Remove all documents matching a filter. Returns the count removed.
    async fn delete_by_filter(
        &self,
        filter: &crate::facet::FacetFilter,
    ) -> Result<u64>;

    /// Retrieve a document by id.
    async fn get(&self, id: DocumentId) -> Result<Option<Document>>;

    /// Execute a search query and return ranked results.
    async fn search(&self, query: Query) -> Result<SearchResults>;

    /// Current stats.
    async fn stats(&self) -> Result<IndexStats>;

    /// Flush all pending writes to durable storage.
    async fn flush(&self) -> Result<()>;

    /// Completely drop the index and all its data.
    async fn drop_index(&self) -> Result<()>;
}
