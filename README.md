# nexus-search

**Production-grade full-text + hybrid search for Rust** — built on Postgres, designed to live inside your Axum app.

[![crates.io](https://img.shields.io/crates/v/nexus-search-core.svg?label=nexus-search-core)](https://crates.io/crates/nexus-search-core)
[![crates.io](https://img.shields.io/crates/v/nexus-postgres.svg?label=nexus-postgres)](https://crates.io/crates/nexus-postgres)
[![crates.io](https://img.shields.io/crates/v/nexus-redis.svg?label=nexus-redis)](https://crates.io/crates/nexus-redis)
[![crates.io](https://img.shields.io/crates/v/nexus-dashboard.svg?label=nexus-dashboard)](https://crates.io/crates/nexus-dashboard)
[![docs.rs](https://img.shields.io/docsrs/nexus-postgres)](https://docs.rs/nexus-postgres)
[![license](https://img.shields.io/crates/l/nexus-postgres.svg)](https://github.com/isaim0011/nexus-search/blob/main/LICENSE-MIT)

| Crate | crates.io | docs.rs |
|---|---|---|
| `nexus_core` (published as `nexus-search-core`) | [![](https://img.shields.io/crates/v/nexus-search-core.svg)](https://crates.io/crates/nexus-search-core) | [![](https://img.shields.io/docsrs/nexus-search-core)](https://docs.rs/nexus-search-core) |
| `nexus_postgres` | [![](https://img.shields.io/crates/v/nexus-postgres.svg)](https://crates.io/crates/nexus-postgres) | [![](https://img.shields.io/docsrs/nexus-postgres)](https://docs.rs/nexus-postgres) |
| `nexus_redis` | [![](https://img.shields.io/crates/v/nexus-redis.svg)](https://crates.io/crates/nexus-redis) | [![](https://img.shields.io/docsrs/nexus-redis)](https://docs.rs/nexus-redis) |
| `nexus` CLI binary (published as `nexus-search-cli`) | [![](https://img.shields.io/crates/v/nexus-search-cli.svg)](https://crates.io/crates/nexus-search-cli) | [![](https://img.shields.io/docsrs/nexus-search-cli)](https://docs.rs/nexus-search-cli) |
| `nexus_dashboard` | [![](https://img.shields.io/crates/v/nexus-dashboard.svg)](https://crates.io/crates/nexus-dashboard) | [![](https://img.shields.io/docsrs/nexus-dashboard)](https://docs.rs/nexus-dashboard) |

```toml
nexus-core = { package = "nexus-search-core", version = "0.1" }  # typed documents, BM25+, RRF, tokenizer
nexus-postgres = "0.1"          # Postgres backend (pgvector-ready)
nexus-dashboard = "0.1"         # Axum HTTP API + dashboard (optional)
```

> The core crate is published as `nexus-search-core` (plain `nexus-core` was
> already taken on crates.io) — use the `package` key above to import it as
> `nexus_core` in your code, same as every crate in this workspace does.

---

## Why nexus-search?

Most search engines (Meilisearch, Typesense, Tantivy, Elasticsearch) are
standalone engines or dedicated inverted-index storage systems. They are
blazing fast, but they introduce operational overhead: network latency,
out-of-band sync queues, dual-write consistency issues, and extra
infrastructure to monitor.

`nexus-search` takes a different architectural approach:

- **Zero extra infrastructure:** runs directly inside your Axum process,
  backed by your existing Postgres instance (or Redis).
- **Atomic consistency:** search updates happen in the exact same
  transaction as your database writes — no outbox workers, no sync lag,
  no drift.
- **Good-enough performance for most workloads:** BM25+ and `pgvector`
  with RRF hybrid fusion, inside the database you already run.

**When to use it:** you want full-text and vector search in a Rust web
service, your data already lives in Postgres, and you value transactional
correctness and zero-ops simplicity over shaving milliseconds off query
latency.

**When NOT to use it:** you're building an e-commerce search-as-you-type
bar over 10M+ products requiring sub-10ms raw keystroke latency at massive
scale (use Meilisearch/Typesense), or a custom file-backed search cluster
at huge scale (use Tantivy). nexus-search does not claim to out-perform
these engines on raw query speed — no such benchmark has been run, and
given Postgres-backed queries carry SQL parsing and connection-pool
overhead that memory-mapped, purpose-built engines don't, it's not
expected to win one. The trade here is operational simplicity and ACID
consistency, not speed.

---

## 10-line "add search to your Axum app"

```rust
use nexus_core::{Index, IndexConfig, SchemaBuilder};
use nexus_postgres::{pool, PoolConfig, PgIndex};
use nexus_dashboard::{DashboardState, router};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let pool   = pool::connect(&PoolConfig::default()).await.unwrap();
    let schema = SchemaBuilder::new().text("title").text("body").build();
    let index  = Arc::new(PgIndex::open(pool, IndexConfig::new("docs", schema)).await.unwrap());
    let app    = axum::Router::new().nest("/search", router(DashboardState::new(index)));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

You get:
- `GET  /search/indexes/:name/search?q=rust+async&limit=10`
- `POST /search/indexes/:name/documents`  — upsert documents
- `GET  /search/indexes/:name/stats`
- `DELETE /search/indexes/:name/documents/:id`

---

## Features

| Feature | Description |
|---------|-------------|
| **BM25+** | Lv & Zhai (2011) variant with δ=0.5; avoids under-counting on short docs |
| **Hybrid search** | Dense-vector + BM25 fused with Reciprocal Rank Fusion (RRF) |
| **Typo tolerance** | Levenshtein with per-term budget (0/1/2 typos by word length) |
| **Faceted filtering** | Typed pre-aggregated facets — `Eq`, `Range`, `In`, `And`, `Or`, `Not` |
| **Highlights** | `<mark>` snippets with configurable pre/post context |
| **pgvector ANN** | HNSW index via `nexus-postgres` with `pgvector` feature flag |
| **Redis backend** | Inverted index in Redis hashes + sorted sets; pub/sub multi-node sync |
| **Batch ingestion** | `upsert_many` batches to single-transaction bulk insert |
| **SIMD dot-product** | `simsimd` feature for AVX-512 / NEON accelerated cosine similarity |
| **Async-first** | Tokio throughout; `Index` trait is object-safe |

---

## Comparison

| | nexus-search | tantivy | Meilisearch | Typesense |
|---|---|---|---|---|
| **Runs inside your app** | ✅ | ✅ | ❌ external | ❌ external |
| **Postgres backend** | ✅ | ❌ | ❌ | ❌ |
| **Hybrid BM25+vector** | ✅ | ❌ | ✅ | ✅ |
| **Typed documents** | ✅ | partial | JSON | JSON |
| **Transactional upsert** | ✅ | ❌ | ❌ | ❌ |
| **Facets** | ✅ | ✅ | ✅ | ✅ |

**nexus-search** is the choice when you want search **inside a Postgres-backed Axum app** — no sidecar process, transactional indexing, full Rust types.

**tantivy** is lower-level (you build your own indexing pipeline) and excellent for file-based local indexes.

**Meilisearch / Typesense** are managed external search engines — great if you want an ops-free hosted search, but add network latency and a separate process to run.

---

## Index a document

```rust
use nexus_core::{Document, FieldValue};

let doc = Document::builder()
    .field("title", "Async Rust with Tokio")
    .field("body",  "Learn how to write concurrent programs in Rust.")
    .field("lang",  "en")
    .build();

index.upsert(doc).await?;
```

## Search with highlights

```rust
use nexus_core::QueryBuilder;

let results = index.search(
    QueryBuilder::new()
        .text("async rust")
        .highlight(["title", "body"])
        .limit(10)
        .build()
).await?;

for hit in results.hits {
    println!("{} (score: {:.3})", hit.id, hit.score);
    for hl in &hit.highlights {
        println!("  [{}] {}", hl.field, hl.snippet);
    }
}
```

## Hybrid search (BM25 + vector)

```rust
use nexus_core::QueryBuilder;

let query_embedding = my_model.encode("async rust tutorial"); // Vec<f32>

let results = index.search(
    QueryBuilder::new()
        .text("async rust")
        .vector(query_embedding)
        .hybrid_alpha(0.6)   // 60% BM25, 40% vector
        .limit(10)
        .build()
).await?;
```

## Faceted search

```rust
use nexus_core::{QueryBuilder, facet::FacetFilter};

let results = index.search(
    QueryBuilder::new()
        .text("web framework")
        .filter(FacetFilter::Eq {
            field: "lang".into(),
            value: "rust".into(),
        })
        .facets(["lang", "license"])
        .build()
).await?;

for (field, dist) in &results.facet_distributions {
    for bucket in &dist.buckets {
        println!("{field}/{}: {}", bucket.value, bucket.count);
    }
}
```

---

## pgvector ANN

Enable the `pgvector` feature for HNSW approximate nearest-neighbour search:

```toml
nexus-postgres = { version = "0.1", features = ["pgvector"] }
```

Requires `pgvector` extension in your Postgres instance:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The migration runs automatically via `nexus_postgres::pool::connect()`.

---

## Redis backend

```rust
use nexus_redis::{connect, RedisIndex};
use nexus_core::IndexConfig;

let conn  = nexus_redis::connect("redis://127.0.0.1/").await?;
let index = RedisIndex::open(conn, IndexConfig::new("docs", schema)).await?;
```

Multi-node invalidation is handled automatically via Redis pub/sub on the
`nx:{index}:sync` channel.

---

## Published crates

All crates are live on crates.io. Badges above always show the current
published version; as of this writing:

| Crate | Published as | Version | crates.io |
|---|---|---|---|
| `nexus_core` (Rust import path) | `nexus-search-core` | 0.1.1 | [crates.io/crates/nexus-search-core](https://crates.io/crates/nexus-search-core) |
| `nexus_postgres` | `nexus-postgres` | 0.1.1 | [crates.io/crates/nexus-postgres](https://crates.io/crates/nexus-postgres) |
| `nexus_redis` | `nexus-redis` | 0.1.1 | [crates.io/crates/nexus-redis](https://crates.io/crates/nexus-redis) |
| CLI binary `nexus` | `nexus-search-cli` | 0.1.2 | [crates.io/crates/nexus-search-cli](https://crates.io/crates/nexus-search-cli) |
| `nexus_dashboard` | `nexus-dashboard` | 0.1.1 | [crates.io/crates/nexus-dashboard](https://crates.io/crates/nexus-dashboard) |

`nexus-core` and `nexus-cli` were already taken on crates.io by unrelated
projects, so this workspace publishes those two under the
`nexus-search-core` / `nexus-search-cli` names while keeping the Rust
import path (`nexus_core::...`) and the installed binary name (`nexus`)
unchanged — see the `package = "..."` key in each dependent's `Cargo.toml`.

Publish order (core has no Postgres/Redis deps, so it goes first):

```
cargo publish -p nexus-search-core
cargo publish -p nexus-postgres
cargo publish -p nexus-redis
cargo publish -p nexus-search-cli
cargo publish -p nexus-dashboard
```

---

## License

MIT OR Apache-2.0
