/// Redis backend for Nexus.
///
/// Data layout:
///   `nx:{index}:doc:{id}`          → LZ4-compressed document JSON (STRING)
///   `nx:{index}:post:{term}`       → Hash { doc_id → tf }
///   `nx:{index}:scores:{term}`     → Sorted set { doc_id → bm25_score }
///   `nx:{index}:facet:{field}:{v}` → Set of doc_ids
///   `nx:{index}:config`            → JSON-encoded IndexConfig
///   `nx:{index}:stats`             → Hash { doc_count, total_tokens }
///
/// Multi-node sync uses Redis pub/sub channel `nx:{index}:sync`.
pub mod backend;

pub use backend::RedisIndex;
pub use redis::aio::ConnectionManager;

/// Build a Redis connection manager from a URL.
pub async fn connect(url: &str) -> anyhow::Result<ConnectionManager> {
    let client = redis::Client::open(url)?;
    Ok(ConnectionManager::new(client).await?)
}
