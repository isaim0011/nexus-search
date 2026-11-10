# Changelog

All notable changes to this workspace are documented here. Crates are
versioned independently going forward — see each entry for exactly which
crates moved.

## [0.1.1 / 0.1.2] - 2026-11-10

Metadata-only patch release — no behavior changes to any crate.

### Changed
- `nexus-search-core`, `nexus-postgres`, `nexus-redis`, `nexus-dashboard`
  → `0.1.1`: crates.io keywords replaced with higher-intent search terms
  (e.g. `pgvector`, `rag`, `hybrid-search`, `cache`) in place of generic
  ones; `nexus-dashboard`'s swapped `api` for `pgvector`.
- `nexus-search-cli` → `0.1.2`: added `keywords`/`categories` (it had
  none at `0.1.0`); fixed its docs.rs build for real this time — `0.1.1`
  had tried `[package.metadata.docs.rs] cargo-args = ["--bins"]`, which
  turned out not to work because docs.rs hardcodes `--lib` in its build
  invocation regardless of `cargo-args`. `0.1.2` adds an actual stub
  `[lib]` target instead, which docs.rs can build.
- Added 10 GitHub repo topics (`axum`, `postgres`, `pgvector`,
  `full-text-search`, `bm25`, `rag`, `redis`, `search`, `rust`,
  `hybrid-search`) for discoverability alongside the crates.io keywords.

## [0.1.0] - 2026-11-10

### Added
- `nexus-search-core`: typed documents, schema builder, BM25+ scoring (Lv &
  Zhai 2011, δ=0.5), RRF hybrid fusion, tokenizer, highlight generation.
- `nexus-postgres`: Postgres backend using `tsvector`/`tsquery`, LZ4-compressed
  document storage, pgvector HNSW ANN search (`pgvector` feature) with
  brute-force fallback, and `COPY FROM STDIN`-based bulk ingestion for
  `upsert_many`.
- `nexus-redis`: Redis backend with inverted-index hashes, BM25 sorted sets,
  and pub/sub cross-node sync.
- `nexus-search-cli` (binary `nexus`): monitor/admin CLI for indexes.
- `nexus-dashboard`: Axum-based HTTP API and dashboard.
- Criterion benchmarks for BM25 scoring, tokenization, edit distance, RRF
  fusion, and flat vector indexing.
- Live integration test suite (`nexus-postgres/tests/pg_integration.rs`)
  covering single-row upsert, bulk COPY ingestion, and pgvector ANN search
  against a real Postgres instance.

### Fixed (found via live Postgres integration testing)
- Multi-statement schema migration failed under sqlx's prepared-statement
  protocol ("cannot insert multiple commands into a prepared statement");
  switched to `sqlx::raw_sql`.
- `IndexStats` decoding failed on `AVG(doc_len)` (Postgres `NUMERIC` vs.
  expected `FLOAT8`); fixed with an explicit `::float8` cast.
- Race condition in `run_pgvector_upgrade`'s check-then-`ALTER TABLE ADD
  COLUMN` sequence when opening indexes concurrently; fixed with
  `ADD COLUMN IF NOT EXISTS`.

### Notes
- `nexus-core` and `nexus-cli` were already registered on crates.io by
  unrelated projects, so this workspace publishes under `nexus-search-core`
  and `nexus-search-cli` instead. The Rust import path (`nexus_core::...`)
  and the installed CLI binary name (`nexus`) are unaffected.
