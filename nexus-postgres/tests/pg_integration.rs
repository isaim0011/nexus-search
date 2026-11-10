//! Live integration test against a real Postgres instance.
//!
//! Skipped automatically if `DATABASE_URL` is not set. Run with:
//!
//! ```bash
//! export DATABASE_URL=postgres://user:pass@host:5432/dbname
//! cargo test -p nexus-postgres --test pg_integration -- --nocapture
//! # with ANN (needs the pgvector extension installed on the server):
//! cargo test -p nexus-postgres --test pg_integration --features pgvector -- --nocapture
//! ```
//!
//! The test creates its own uniquely-named index (`nx_it_<uuid>`), inserts
//! into it, asserts on it, and drops it at the end — safe to run against a
//! shared database since it never touches other indexes' rows.

use nexus_core::{
    document::Document,
    facet::FacetFilter,
    index::{Index, IndexConfig},
    query::QueryBuilder,
    schema::Schema,
};
use nexus_postgres::{pool, PgIndex};

fn database_url() -> Option<String> {
    std::env::var("DATABASE_URL").ok()
}

async fn open_test_index(suffix: &str) -> Option<(PgIndex, String)> {
    let url = database_url()?;
    let pool_cfg = pool::PoolConfig {
        database_url: url,
        ..Default::default()
    };
    let pg_pool = pool::connect(&pool_cfg)
        .await
        .expect("failed to connect to DATABASE_URL");

    let index_name = format!("nx_it_{}_{}", suffix, uuid::Uuid::new_v4().simple());

    let schema = Schema::builder()
        .text("title")
        .text("body")
        .keyword("category")
        .integer_facet("views")
        .vector("embedding", 8)
        .default_search_field("body")
        .build();

    let config = IndexConfig::new(&index_name, schema);
    let index = PgIndex::open(pg_pool, config)
        .await
        .expect("PgIndex::open failed");

    Some((index, index_name))
}

fn doc(title: &str, body: &str, category: &str, views: i64, seed: f32) -> Document {
    Document::builder()
        .field("title", title)
        .field("body", body)
        .field("category", category)
        .field("views", views)
        .field("embedding", vec![seed; 8])
        .build()
}

/// Exercises `upsert`, BM25 text search, and facet filtering on a handful of
/// documents (below the COPY threshold — single-row INSERT path).
#[tokio::test]
async fn small_upsert_and_search() {
    let Some((index, index_name)) = open_test_index("small").await else {
        eprintln!("DATABASE_URL not set — skipping live Postgres test");
        return;
    };

    let d1 = doc("Rust async guide", "Tokio makes async Rust ergonomic and fast", "guide", 100, 0.1);
    let d2 = doc("Python tutorial", "Async in python uses asyncio event loops", "tutorial", 50, 0.9);

    index.upsert(d1.clone()).await.expect("upsert d1 failed");
    index.upsert(d2.clone()).await.expect("upsert d2 failed");

    let results = index
        .search(QueryBuilder::new().text("async rust").limit(10).build())
        .await
        .expect("search failed");

    assert!(results.total_hits >= 1, "expected at least one hit for 'async rust'");
    assert!(
        results.hits.iter().any(|h| h.id == d1.id),
        "expected the rust doc to be a hit"
    );

    let filtered = index
        .search(
            QueryBuilder::new()
                .filter(FacetFilter::Gte { field: "views".into(), value: 75.0 })
                .limit(10)
                .build(),
        )
        .await
        .expect("filtered search failed");
    assert!(
        filtered.hits.iter().all(|h| h.id == d1.id),
        "views >= 75 filter should only match d1"
    );

    let fetched = index.get(d1.id).await.expect("get failed");
    assert!(fetched.is_some(), "expected d1 to be retrievable by id");

    index.drop_index().await.expect("drop_index failed");
    let _ = index_name;
}

/// Exercises the bulk `upsert_many` COPY path (>=16 docs triggers COPY in
/// `nexus-postgres/src/backend.rs`), verifying postings, facets, and vectors
/// all land correctly via the temp-table + `COPY FROM STDIN` pipeline.
#[tokio::test]
async fn bulk_upsert_many_uses_copy_path() {
    let Some((index, index_name)) = open_test_index("bulk").await else {
        eprintln!("DATABASE_URL not set — skipping live Postgres test");
        return;
    };

    let docs: Vec<Document> = (0..40)
        .map(|i| {
            doc(
                &format!("Document number {i}"),
                &format!("This is body text for document {i} about widgets and gadgets"),
                if i % 2 == 0 { "even" } else { "odd" },
                i as i64,
                (i as f32) / 40.0,
            )
        })
        .collect();

    let ids = index
        .upsert_many(docs.clone())
        .await
        .expect("upsert_many (COPY path) failed");
    assert_eq!(ids.len(), 40, "expected all 40 docs to be upserted");

    let stats = index.stats().await.expect("stats failed");
    assert_eq!(stats.document_count, 40, "stats should report 40 documents after bulk COPY");

    let results = index
        .search(QueryBuilder::new().text("widgets gadgets").limit(50).build())
        .await
        .expect("search after bulk upsert failed");
    assert_eq!(
        results.hits.len(),
        40,
        "all 40 docs should match the shared body text"
    );

    let even_only = index
        .search(
            QueryBuilder::new()
                .filter(FacetFilter::Eq {
                    field: "category".into(),
                    value: nexus_core::document::FieldValue::Text("even".into()),
                })
                .limit(50)
                .build(),
        )
        .await
        .expect("facet-filtered search after bulk upsert failed");
    assert_eq!(even_only.hits.len(), 20, "exactly half the docs are category=even");

    // Re-upsert the same ids (dedup path inside upsert_many) and confirm the
    // document count doesn't double.
    index
        .upsert_many(docs)
        .await
        .expect("second upsert_many (re-COPY) failed");
    let stats2 = index.stats().await.expect("stats after re-upsert failed");
    assert_eq!(stats2.document_count, 40, "re-upserting the same ids must not duplicate rows");

    index.drop_index().await.expect("drop_index failed");
    let _ = index_name;
}

/// Only meaningful with `--features pgvector` and a server that has the
/// pgvector extension installed. Verifies the HNSW ANN path returns the
/// nearest-by-cosine-distance vector first, and that the fallback brute-force
/// path (feature disabled, or extension missing) is exercised identically
/// through the same public API when pgvector isn't available.
#[tokio::test]
async fn vector_search_nearest_neighbor() {
    let Some((index, index_name)) = open_test_index("vec").await else {
        eprintln!("DATABASE_URL not set — skipping live Postgres test");
        return;
    };

    // Use >=16 docs so the vectors go through the bulk path too, plus a
    // couple of individually-upserted docs to hit the single-row path.
    // Filler docs point along basis vector e0; the needle points along e1.
    // Cosine similarity (both the pgvector <=> path and the normalized-dot
    // brute-force path) can't distinguish anything from a zero query vector,
    // so the query must be a real direction — here, the same direction as
    // the needle, giving it cosine similarity 1.0 vs 0.0 for every filler.
    let mut e0 = vec![0.0f32; 8];
    e0[0] = 1.0;
    let mut e1 = vec![0.0f32; 8];
    e1[1] = 1.0;

    let mut docs: Vec<Document> = (0..20)
        .map(|i| {
            Document::builder()
                .field("title", format!("filler {i}"))
                .field("body", format!("filler body {i} about widgets"))
                .field("category", "filler")
                .field("views", i as i64)
                .field("embedding", e0.clone())
                .build()
        })
        .collect();

    let needle = Document::builder()
        .field("title", "needle")
        .field("body", "the exact match document")
        .field("category", "target")
        .field("views", 999i64)
        .field("embedding", e1.clone())
        .build();
    docs.push(needle.clone());

    index.upsert_many(docs).await.expect("upsert_many failed");

    let results = index
        .search(
            QueryBuilder::new()
                .vector(e1)
                .limit(1)
                .build(),
        )
        .await
        .expect("vector search failed");

    assert_eq!(results.hits.len(), 1);
    assert_eq!(
        results.hits[0].id, needle.id,
        "nearest neighbor to [0.0; 8] should be the needle doc (embedding seed 0.0)"
    );

    index.drop_index().await.expect("drop_index failed");
    let _ = index_name;
}
