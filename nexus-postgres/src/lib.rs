/// PostgreSQL backend for Nexus.
///
/// Uses:
/// - `tsvector` / `tsquery` for full-text search with per-language dictionaries.
/// - `jsonb` for document storage with LZ4 compression in a custom column.
/// - `SELECT … FOR UPDATE SKIP LOCKED` for concurrent safe batch ingestion.
/// - `LISTEN/NOTIFY` for near-real-time index sync across nodes.
/// - Optional `pgvector` extension for ANN vector search.
pub mod backend;
pub mod migrations;
pub mod pool;

pub use backend::PgIndex;
pub use pool::{PgPool, PoolConfig};
