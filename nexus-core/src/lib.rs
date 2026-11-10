/// Nexus — production-grade full-text + hybrid (BM25 + vector) search for Rust.
///
/// # Design goals
/// - **Typed documents**: your structs, not stringly-typed maps.
/// - **Hybrid search**: BM25 keyword + dense-vector semantic, fused with RRF.
/// - **Typo tolerance**: Levenshtein-based with configurable distance, prefix-aware.
/// - **Faceted filtering**: zero-cost typed facet fields with pre-aggregated counts.
/// - **Backend-agnostic**: swap Postgres, Redis, or in-memory via `StorageBackend`.
/// - **SIMD-accelerated**: dot-product and cosine similarity via `simsimd`.
/// - **Async-first**: built on Tokio; indexing and search are always async.
pub mod document;
pub mod error;
pub mod facet;
pub mod highlight;
pub mod index;
pub mod query;
pub mod schema;
pub mod score;
pub mod tokenizer;
pub mod vector;

pub use document::{Document, DocumentId, Field, FieldValue};
pub use error::{NexusError, Result};
pub use highlight::{Highlighter, HighlightConfig};
pub use index::{Index, IndexConfig, IndexStats};
pub use query::{Highlight, Query, QueryBuilder, SearchResult, SearchResults, SortOrder};
pub use schema::{FieldType, Schema, SchemaBuilder};
pub use tokenizer::{Language, Token, Tokenizer, TokenizerConfig};
pub use vector::Embedding;
