use anyhow::Result;
use sqlx::PgPool;

/// Idempotent schema migration. Called automatically by `connect()`.
///
/// Tables created:
///
/// `nexus_indexes`   — one row per index, stores the `IndexConfig` as JSONB.
/// `nexus_documents` — per-index document store with compressed JSONB body.
/// `nexus_postings`  — posting list: (index, term, doc_id, field, tf, positions).
/// `nexus_vectors`   — per-document dense vector (uses pgvector `vector` type when
///                     the `pgvector` feature is enabled, otherwise `FLOAT4[]`).
/// `nexus_facets`    — pre-aggregated facet term → doc_id mapping for fast facet counts.
pub async fn run(pool: &PgPool) -> Result<()> {
    // Try to enable pgvector extension. Silently skips if not installed.
    let _ = sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await;

    // Multiple `;`-separated statements can't go through the extended
    // (prepared-statement) protocol that `sqlx::query()` uses — Postgres
    // rejects it with "cannot insert multiple commands into a prepared
    // statement". `raw_sql` uses the simple query protocol instead, which
    // supports multi-statement bodies.
    sqlx::raw_sql(
        r#"
        CREATE TABLE IF NOT EXISTS nexus_indexes (
            name        TEXT PRIMARY KEY,
            config      JSONB NOT NULL,
            stats       JSONB NOT NULL DEFAULT '{}',
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        -- One row per indexed document.
        CREATE TABLE IF NOT EXISTS nexus_documents (
            index_name  TEXT NOT NULL REFERENCES nexus_indexes(name) ON DELETE CASCADE,
            doc_id      UUID NOT NULL,
            -- LZ4-compressed JSONB fields blob. Decompressed on retrieval.
            body        BYTEA NOT NULL,
            -- Postgres native tsvector for fast phrase + ranking queries.
            tsv         TSVECTOR,
            -- Token count for per-doc BM25 length normalization.
            doc_len     INTEGER NOT NULL DEFAULT 0,
            indexed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (index_name, doc_id)
        );
        CREATE INDEX IF NOT EXISTS nexus_documents_tsv
            ON nexus_documents USING GIN (tsv);

        -- Posting list for BM25 scoring.
        -- Stored separately for efficient IDF calculation and posting merges.
        CREATE TABLE IF NOT EXISTS nexus_postings (
            index_name  TEXT NOT NULL,
            term        TEXT NOT NULL,
            doc_id      UUID NOT NULL,
            field_name  TEXT NOT NULL,
            tf          INTEGER NOT NULL DEFAULT 1,
            -- Packed array of token positions for phrase queries.
            positions   INTEGER[] NOT NULL DEFAULT '{}',
            PRIMARY KEY (index_name, term, doc_id, field_name),
            FOREIGN KEY (index_name, doc_id)
                REFERENCES nexus_documents(index_name, doc_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS nexus_postings_term
            ON nexus_postings (index_name, term);

        -- Dense vector store. Uses FLOAT4[] for portability.
        -- When pgvector is available, vectors are stored in a separate pgvector column
        -- added by run_pgvector_upgrade() below.
        CREATE TABLE IF NOT EXISTS nexus_vectors (
            index_name  TEXT NOT NULL,
            doc_id      UUID NOT NULL,
            field_name  TEXT NOT NULL,
            embedding   FLOAT4[] NOT NULL,
            PRIMARY KEY (index_name, doc_id, field_name),
            FOREIGN KEY (index_name, doc_id)
                REFERENCES nexus_documents(index_name, doc_id) ON DELETE CASCADE
        );

        -- Pre-aggregated facet values for O(1) facet count queries.
        CREATE TABLE IF NOT EXISTS nexus_facets (
            index_name  TEXT NOT NULL,
            field_name  TEXT NOT NULL,
            value       TEXT NOT NULL,
            doc_id      UUID NOT NULL,
            PRIMARY KEY (index_name, field_name, value, doc_id),
            FOREIGN KEY (index_name, doc_id)
                REFERENCES nexus_documents(index_name, doc_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS nexus_facets_field
            ON nexus_facets (index_name, field_name, value);
        "#,
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Add a typed `vector(dim)` column to `nexus_vectors` and build an HNSW index.
///
/// pgvector's HNSW index requires a column with a fixed dimension, so this runs
/// per deployment with the dimension declared in the index schema. It is called
/// automatically by `PgIndex::open` when the `pgvector` feature is enabled.
///
/// Returns `true` if the pgvector column is available (extension installed and
/// the column dimension matches), `false` if the extension is missing or the
/// column already exists with a different dimension.
pub async fn run_pgvector_upgrade(pool: &PgPool, dim: usize) -> Result<bool> {
    let has_ext: (bool,) = sqlx::query_as(
        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')",
    )
    .fetch_one(pool)
    .await?;

    if !has_ext.0 {
        return Ok(false);
    }

    // If the column already exists, verify its dimension (atttypmod holds dim for
    // the vector type) rather than failing on a mismatched ALTER.
    let existing: Option<(i32,)> = sqlx::query_as(
        "SELECT atttypmod FROM pg_attribute
         WHERE attrelid = 'nexus_vectors'::regclass
           AND attname = 'vec' AND NOT attisdropped",
    )
    .fetch_optional(pool)
    .await?;

    match existing {
        Some((typmod,)) => {
            if typmod != dim as i32 {
                tracing::warn!(
                    existing_dim = typmod,
                    requested_dim = dim,
                    "nexus_vectors.vec exists with a different dimension; \
                     pgvector ANN disabled for this index"
                );
                return Ok(false);
            }
        }
        None => {
            // `IF NOT EXISTS` guards against a race: two `PgIndex::open` calls
            // (e.g. opening two indexes concurrently) can both see no `vec`
            // column here and both attempt to add it — without the guard,
            // the second one fails with "column already exists".
            sqlx::query(&format!(
                "ALTER TABLE nexus_vectors ADD COLUMN IF NOT EXISTS vec vector({dim})"
            ))
            .execute(pool)
            .await?;
        }
    }

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS nexus_vectors_hnsw
         ON nexus_vectors USING hnsw (vec vector_cosine_ops)
         WITH (m = 16, ef_construction = 64)",
    )
    .execute(pool)
    .await?;

    Ok(true)
}
