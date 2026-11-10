use async_trait::async_trait;
use nexus_core::{
    document::{Document, DocumentId},
    error::{NexusError, Result},
    facet::FacetFilter,
    highlight::{Highlighter, HighlightConfig},
    index::{Index, IndexConfig, IndexStats},
    query::{Query, SearchResult, SearchResults},
    score::{Bm25Scorer, Rrf, typo_budget, edit_distance},
    tokenizer::{Tokenizer, TokenizerConfig},
    vector::Embedding,
};
use std::collections::HashMap;
use tracing::{debug, instrument, warn};
use uuid::Uuid;

use crate::PgPool;

/// Escape a string for PostgreSQL `COPY … FROM STDIN` text format:
/// backslash, tab, newline and carriage return become escape sequences.
fn copy_escape(s: &str, out: &mut String) {
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
}

/// PostgreSQL-backed Nexus index.
///
/// One `PgIndex` instance is cheap to clone (just an `Arc<PgPool>` inside).
/// Create one per index name; use `Arc<PgIndex>` in your Axum state.
pub struct PgIndex {
    pool: PgPool,
    config: IndexConfig,
    tokenizer: Tokenizer,
    highlighter: Highlighter,
    /// True when the pgvector `vec` column + HNSW index are usable for this
    /// index's vector dimension. Falls back to brute-force scan when false.
    #[cfg_attr(not(feature = "pgvector"), allow(dead_code))]
    ann_enabled: bool,
}

impl PgIndex {
    /// Open (or create) an index backed by the given pool.
    pub async fn open(pool: PgPool, config: IndexConfig) -> Result<Self> {
        let config_json =
            serde_json::to_value(&config).map_err(NexusError::Serialization)?;

        sqlx::query(
            "INSERT INTO nexus_indexes (name, config) VALUES ($1, $2)
             ON CONFLICT (name) DO UPDATE SET config = $2, updated_at = NOW()",
        )
        .bind(&config.name)
        .bind(&config_json)
        .execute(&pool)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        let tokenizer = Tokenizer::new(TokenizerConfig {
            min_token_len: 2,
            max_token_len: 100,
            ..Default::default()
        });

        // With the pgvector feature, provision the typed vector(N) column +
        // HNSW index for this schema's vector dimension (if any).
        #[allow(unused_mut)]
        let mut ann_enabled = false;
        #[cfg(feature = "pgvector")]
        {
            if let Some(dim) = config.schema.vector_fields().find_map(|f| {
                match f.field_type {
                    nexus_core::schema::FieldType::Vector { dim } => Some(dim),
                    _ => None,
                }
            }) {
                ann_enabled = crate::migrations::run_pgvector_upgrade(&pool, dim)
                    .await
                    .map_err(|e| NexusError::Backend(e.to_string()))?;
            }
        }

        Ok(Self {
            pool,
            config,
            tokenizer,
            highlighter: Highlighter::new(HighlightConfig::default()),
            ann_enabled,
        })
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn compress(doc: &Document) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(doc).map_err(NexusError::Serialization)?;
        Ok(lz4_flex::compress_prepend_size(&json))
    }

    fn decompress(bytes: &[u8]) -> Result<Document> {
        let json = lz4_flex::decompress_size_prepended(bytes)
            .map_err(|e| NexusError::Compression(e.to_string()))?;
        serde_json::from_slice(&json).map_err(NexusError::Serialization)
    }

    async fn fetch_stats_row(&self) -> Result<(u64, f32)> {
        let row: (Option<i64>, Option<f64>) = sqlx::query_as(
            "SELECT COUNT(*), AVG(doc_len)::float8 FROM nexus_documents WHERE index_name = $1",
        )
        .bind(&self.config.name)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        Ok((
            row.0.unwrap_or(0) as u64,
            row.1.unwrap_or(100.0) as f32,
        ))
    }

    /// Run pgvector ANN if the extension + column are available; fall back to brute-force.
    #[cfg(feature = "pgvector")]
    async fn vector_search_pgvector(
        &self,
        query_emb: &Embedding,
        limit: usize,
    ) -> Result<Vec<(Uuid, f32)>> {
        // Cast the Float4[] query to vector and use the <=> cosine-distance operator.
        // 1 - distance = similarity.
        let dim = query_emb.0.len();
        let vec_literal: String = {
            let parts: Vec<String> = query_emb.0.iter().map(|v| v.to_string()).collect();
            format!("[{}]", parts.join(","))
        };

        let rows: Vec<(Uuid, f64)> = sqlx::query_as(&format!(
            "SELECT doc_id, 1.0 - (vec <=> $1::vector({dim})) AS score
             FROM nexus_vectors
             WHERE index_name = $2 AND vec IS NOT NULL
             ORDER BY vec <=> $1::vector({dim})
             LIMIT $3"
        ))
        .bind(&vec_literal)
        .bind(&self.config.name)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        Ok(rows.into_iter().map(|(id, s)| (id, s as f32)).collect())
    }

    async fn vector_search_brute_force(
        &self,
        query_emb: &Embedding,
    ) -> Result<Vec<(Uuid, f32)>> {
        let rows: Vec<(Uuid, Vec<f32>)> = sqlx::query_as(
            "SELECT doc_id, embedding FROM nexus_vectors WHERE index_name = $1",
        )
        .bind(&self.config.name)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        let mut scored: Vec<(Uuid, f32)> = rows
            .into_iter()
            .map(|(id, raw)| {
                let emb = Embedding::new(raw);
                (id, emb.dot(query_emb))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

#[async_trait]
impl Index for PgIndex {
    fn config(&self) -> &IndexConfig {
        &self.config
    }

    #[instrument(skip(self, doc), fields(index = %self.config.name))]
    async fn upsert(&self, mut doc: Document) -> Result<DocumentId> {
        doc.indexed_at = Some(chrono::Utc::now());
        self.config.schema.validate(&doc)?;

        let compressed = Self::compress(&doc)?;

        let mut tsv_text = String::new();
        let mut total_tokens = 0u32;

        for f in self.config.schema.text_fields() {
            if let Some(nexus_core::document::FieldValue::Text(t)) = doc.fields.get(&f.name) {
                tsv_text.push(' ');
                tsv_text.push_str(t);
                total_tokens += self.tokenizer.tokenize(t).map(|v| v.len() as u32).unwrap_or(0);
            }
        }

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

        sqlx::query(
            "INSERT INTO nexus_documents (index_name, doc_id, body, tsv, doc_len)
             VALUES ($1, $2, $3, to_tsvector('english', $4), $5)
             ON CONFLICT (index_name, doc_id) DO UPDATE
               SET body = EXCLUDED.body, tsv = EXCLUDED.tsv,
                   doc_len = EXCLUDED.doc_len, indexed_at = NOW()",
        )
        .bind(&self.config.name)
        .bind(doc.id)
        .bind(&compressed)
        .bind(&tsv_text)
        .bind(total_tokens as i32)
        .execute(&mut *tx)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        sqlx::query(
            "DELETE FROM nexus_postings WHERE index_name = $1 AND doc_id = $2",
        )
        .bind(&self.config.name)
        .bind(doc.id)
        .execute(&mut *tx)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        for field_schema in self.config.schema.text_fields() {
            if let Some(nexus_core::document::FieldValue::Text(text)) =
                doc.fields.get(&field_schema.name)
            {
                let tokens = self.tokenizer.tokenize(text)?;
                let mut tf_map: HashMap<String, (u32, Vec<i32>)> = HashMap::new();
                for tok in &tokens {
                    let e = tf_map.entry(tok.text.clone()).or_default();
                    e.0 += 1;
                    e.1.push(tok.position as i32);
                }
                for (term, (tf, positions)) in tf_map {
                    sqlx::query(
                        "INSERT INTO nexus_postings
                           (index_name, term, doc_id, field_name, tf, positions)
                         VALUES ($1, $2, $3, $4, $5, $6)
                         ON CONFLICT DO NOTHING",
                    )
                    .bind(&self.config.name)
                    .bind(&term)
                    .bind(doc.id)
                    .bind(&field_schema.name)
                    .bind(tf as i32)
                    .bind(&positions)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| NexusError::Backend(e.to_string()))?;
                }
            }
        }

        for field_schema in self.config.schema.vector_fields() {
            if let Some(nexus_core::document::FieldValue::Vector(raw)) =
                doc.fields.get(&field_schema.name)
            {
                let emb = Embedding::new(raw.clone());
                let dim = match field_schema.field_type {
                    nexus_core::schema::FieldType::Vector { dim } => dim,
                    _ => unreachable!(),
                };
                emb.check_dim(dim)?;
                sqlx::query(
                    "INSERT INTO nexus_vectors (index_name, doc_id, field_name, embedding)
                     VALUES ($1, $2, $3, $4)
                     ON CONFLICT (index_name, doc_id, field_name) DO UPDATE
                       SET embedding = EXCLUDED.embedding",
                )
                .bind(&self.config.name)
                .bind(doc.id)
                .bind(&field_schema.name)
                .bind(&emb.0)
                .execute(&mut *tx)
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;

                // Also populate the pgvector column when ANN is provisioned.
                #[cfg(feature = "pgvector")]
                if self.ann_enabled {
                    let vec_literal: String = {
                        let parts: Vec<String> = emb.0.iter().map(|v| v.to_string()).collect();
                        format!("[{}]", parts.join(","))
                    };
                    sqlx::query(
                        "UPDATE nexus_vectors SET vec = $1::vector
                         WHERE index_name = $2 AND doc_id = $3 AND field_name = $4",
                    )
                    .bind(&vec_literal)
                    .bind(&self.config.name)
                    .bind(doc.id)
                    .bind(&field_schema.name)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| NexusError::Backend(e.to_string()))?;
                }
            }
        }

        sqlx::query(
            "DELETE FROM nexus_facets WHERE index_name = $1 AND doc_id = $2",
        )
        .bind(&self.config.name)
        .bind(doc.id)
        .execute(&mut *tx)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        for field_schema in self.config.schema.facet_fields() {
            if let Some(value) = doc.fields.get(&field_schema.name) {
                let label = match value {
                    nexus_core::document::FieldValue::Text(s) => s.clone(),
                    nexus_core::document::FieldValue::Integer(n) => n.to_string(),
                    nexus_core::document::FieldValue::Float(f) => format!("{f:.2}"),
                    nexus_core::document::FieldValue::Bool(b) => b.to_string(),
                    _ => continue,
                };
                sqlx::query(
                    "INSERT INTO nexus_facets (index_name, field_name, value, doc_id)
                     VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                )
                .bind(&self.config.name)
                .bind(&field_schema.name)
                .bind(&label)
                .bind(doc.id)
                .execute(&mut *tx)
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
            }
        }

        tx.commit()
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

        debug!(doc_id = %doc.id, "document indexed");
        Ok(doc.id)
    }

    /// Batch upsert using `COPY … FROM STDIN` for documents, postings and
    /// facets. 10–50x faster than sequential upserts for large ingestion jobs.
    ///
    /// Documents go through a temp table (so `to_tsvector` can be computed
    /// server-side); postings and facets are COPYed straight into their tables
    /// after the old rows for the batch are bulk-deleted.
    async fn upsert_many(&self, docs: Vec<Document>) -> Result<Vec<DocumentId>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }

        // For small batches fall through to the per-document path to avoid
        // the overhead of COPY setup.
        if docs.len() < 16 {
            let mut ids = Vec::with_capacity(docs.len());
            for doc in docs {
                ids.push(self.upsert(doc).await?);
            }
            return Ok(ids);
        }

        // ── Prepare rows (validate, compress, tokenize) ──────────────────────
        // Dedupe by id keeping the last occurrence, so the COPY streams never
        // violate primary keys within the batch.
        let mut order: Vec<Uuid> = Vec::with_capacity(docs.len());
        let mut by_id: HashMap<Uuid, (Document, Vec<u8>, String, i32)> =
            HashMap::with_capacity(docs.len());

        for mut doc in docs {
            doc.indexed_at = Some(chrono::Utc::now());
            self.config.schema.validate(&doc)?;
            let compressed = Self::compress(&doc)?;

            let mut tsv_text = String::new();
            let mut total_tokens = 0i32;
            for f in self.config.schema.text_fields() {
                if let Some(nexus_core::document::FieldValue::Text(t)) = doc.fields.get(&f.name) {
                    tsv_text.push(' ');
                    tsv_text.push_str(t);
                    total_tokens += self.tokenizer.tokenize(t).map(|v| v.len() as i32).unwrap_or(0);
                }
            }
            let id = doc.id;
            if by_id.insert(id, (doc, compressed, tsv_text, total_tokens)).is_none() {
                order.push(id);
            }
        }
        let ids = order.clone();

        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

        // ── Bulk-delete existing rows; FK cascades clear postings/facets/vectors ──
        sqlx::query(
            "DELETE FROM nexus_documents WHERE index_name = $1 AND doc_id = ANY($2)",
        )
        .bind(&self.config.name)
        .bind(&ids)
        .execute(&mut *tx)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        // ── COPY documents via temp table (tsvector computed server-side) ────
        sqlx::query(
            "CREATE TEMP TABLE _nx_copy_docs
               (doc_id UUID, body BYTEA, tsv_text TEXT, doc_len INTEGER)
             ON COMMIT DROP",
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        {
            let mut buf = String::new();
            for id in &order {
                let (_, compressed, tsv_text, doc_len) = &by_id[id];
                buf.push_str(&id.to_string());
                buf.push('\t');
                // bytea in COPY text format: \x<hex>, backslash escaped.
                buf.push_str("\\\\x");
                for b in compressed {
                    use std::fmt::Write;
                    let _ = write!(buf, "{b:02x}");
                }
                buf.push('\t');
                copy_escape(tsv_text, &mut buf);
                buf.push('\t');
                buf.push_str(&doc_len.to_string());
                buf.push('\n');
            }

            let mut copy = tx
                .copy_in_raw("COPY _nx_copy_docs (doc_id, body, tsv_text, doc_len) FROM STDIN")
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
            copy.send(buf.as_bytes())
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
            copy.finish()
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
        }

        sqlx::query(
            "INSERT INTO nexus_documents (index_name, doc_id, body, tsv, doc_len)
             SELECT $1, doc_id, body, to_tsvector('english', tsv_text), doc_len
             FROM _nx_copy_docs",
        )
        .bind(&self.config.name)
        .execute(&mut *tx)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        // ── COPY postings and facets directly ────────────────────────────────
        let mut postings_buf = String::new();
        let mut facets_buf = String::new();

        for id in &order {
            let (doc, _, _, _) = &by_id[id];

            for field_schema in self.config.schema.text_fields() {
                if let Some(nexus_core::document::FieldValue::Text(text)) =
                    doc.fields.get(&field_schema.name)
                {
                    let tokens = self.tokenizer.tokenize(text)?;
                    let mut tf_map: HashMap<String, (u32, Vec<i32>)> = HashMap::new();
                    for tok in &tokens {
                        let e = tf_map.entry(tok.text.clone()).or_default();
                        e.0 += 1;
                        e.1.push(tok.position as i32);
                    }
                    for (term, (tf, positions)) in tf_map {
                        copy_escape(&self.config.name, &mut postings_buf);
                        postings_buf.push('\t');
                        copy_escape(&term, &mut postings_buf);
                        postings_buf.push('\t');
                        postings_buf.push_str(&doc.id.to_string());
                        postings_buf.push('\t');
                        copy_escape(&field_schema.name, &mut postings_buf);
                        postings_buf.push('\t');
                        postings_buf.push_str(&tf.to_string());
                        postings_buf.push('\t');
                        postings_buf.push('{');
                        for (i, p) in positions.iter().enumerate() {
                            if i > 0 {
                                postings_buf.push(',');
                            }
                            postings_buf.push_str(&p.to_string());
                        }
                        postings_buf.push('}');
                        postings_buf.push('\n');
                    }
                }
            }

            for field_schema in self.config.schema.facet_fields() {
                if let Some(value) = doc.fields.get(&field_schema.name) {
                    let label = match value {
                        nexus_core::document::FieldValue::Text(s) => s.clone(),
                        nexus_core::document::FieldValue::Integer(n) => n.to_string(),
                        nexus_core::document::FieldValue::Float(f) => format!("{f:.2}"),
                        nexus_core::document::FieldValue::Bool(b) => b.to_string(),
                        _ => continue,
                    };
                    copy_escape(&self.config.name, &mut facets_buf);
                    facets_buf.push('\t');
                    copy_escape(&field_schema.name, &mut facets_buf);
                    facets_buf.push('\t');
                    copy_escape(&label, &mut facets_buf);
                    facets_buf.push('\t');
                    facets_buf.push_str(&doc.id.to_string());
                    facets_buf.push('\n');
                }
            }
        }

        if !postings_buf.is_empty() {
            let mut copy = tx
                .copy_in_raw(
                    "COPY nexus_postings (index_name, term, doc_id, field_name, tf, positions)
                     FROM STDIN",
                )
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
            copy.send(postings_buf.as_bytes())
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
            copy.finish()
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
        }

        if !facets_buf.is_empty() {
            let mut copy = tx
                .copy_in_raw(
                    "COPY nexus_facets (index_name, field_name, value, doc_id) FROM STDIN",
                )
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
            copy.send(facets_buf.as_bytes())
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
            copy.finish()
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
        }

        // ── Vectors: usually few rows; plain inserts inside the same tx ──────
        for id in &order {
            let (doc, _, _, _) = &by_id[id];
            for field_schema in self.config.schema.vector_fields() {
                if let Some(nexus_core::document::FieldValue::Vector(raw)) =
                    doc.fields.get(&field_schema.name)
                {
                    let emb = Embedding::new(raw.clone());
                    let dim = match field_schema.field_type {
                        nexus_core::schema::FieldType::Vector { dim } => dim,
                        _ => unreachable!(),
                    };
                    emb.check_dim(dim)?;
                    sqlx::query(
                        "INSERT INTO nexus_vectors (index_name, doc_id, field_name, embedding)
                         VALUES ($1, $2, $3, $4)",
                    )
                    .bind(&self.config.name)
                    .bind(doc.id)
                    .bind(&field_schema.name)
                    .bind(&emb.0)
                    .execute(&mut *tx)
                    .await
                    .map_err(|e| NexusError::Backend(e.to_string()))?;

                    #[cfg(feature = "pgvector")]
                    if self.ann_enabled {
                        let vec_literal: String = {
                            let parts: Vec<String> =
                                emb.0.iter().map(|v| v.to_string()).collect();
                            format!("[{}]", parts.join(","))
                        };
                        sqlx::query(
                            "UPDATE nexus_vectors SET vec = $1::vector
                             WHERE index_name = $2 AND doc_id = $3 AND field_name = $4",
                        )
                        .bind(&vec_literal)
                        .bind(&self.config.name)
                        .bind(doc.id)
                        .bind(&field_schema.name)
                        .execute(&mut *tx)
                        .await
                        .map_err(|e| NexusError::Backend(e.to_string()))?;
                    }
                }
            }
        }

        tx.commit()
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

        debug!(count = ids.len(), "batch indexed via COPY");
        Ok(ids)
    }

    async fn delete(&self, id: DocumentId) -> Result<()> {
        sqlx::query(
            "DELETE FROM nexus_documents WHERE index_name = $1 AND doc_id = $2",
        )
        .bind(&self.config.name)
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn delete_by_filter(&self, _filter: &FacetFilter) -> Result<u64> {
        warn!("delete_by_filter not yet fully implemented for PgIndex");
        Ok(0)
    }

    async fn get(&self, id: DocumentId) -> Result<Option<Document>> {
        let row: Option<(Vec<u8>,)> = sqlx::query_as(
            "SELECT body FROM nexus_documents WHERE index_name = $1 AND doc_id = $2",
        )
        .bind(&self.config.name)
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        match row {
            None => Ok(None),
            Some((bytes,)) => Ok(Some(Self::decompress(&bytes)?)),
        }
    }

    #[instrument(skip(self, query), fields(index = %self.config.name))]
    async fn search(&self, query: Query) -> Result<SearchResults> {
        let start = std::time::Instant::now();

        // ── 1. BM25 candidate retrieval ──────────────────────────────────────
        let mut bm25_hits: Vec<(Uuid, f32)> = Vec::new();

        if let Some(ref text) = query.text {
            let terms = self.tokenizer.terms(text)?;
            if !terms.is_empty() {
                let budget = typo_budget(terms.iter().map(|t| t.len()).max().unwrap_or(0));

                let candidates: Vec<(String,)> = sqlx::query_as(
                    "SELECT DISTINCT term FROM nexus_postings WHERE index_name = $1",
                )
                .bind(&self.config.name)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;

                let mut matched_terms: Vec<String> = Vec::new();
                for query_term in &terms {
                    for (cand,) in &candidates {
                        if query_term == cand {
                            matched_terms.push(cand.clone());
                        } else if budget > 0 && edit_distance(query_term, cand, budget).is_some() {
                            matched_terms.push(cand.clone());
                        }
                    }
                }
                matched_terms.dedup();

                if !matched_terms.is_empty() {
                    let (doc_count, avg_len) = self.fetch_stats_row().await?;
                    let scorer = Bm25Scorer {
                        avg_doc_len: avg_len,
                        doc_count: doc_count.max(1),
                        ..Default::default()
                    };
                    let mut scores: HashMap<Uuid, f32> = HashMap::new();

                    for term in &matched_terms {
                        let df: (i64,) = sqlx::query_as(
                            "SELECT COUNT(DISTINCT doc_id) FROM nexus_postings
                             WHERE index_name = $1 AND term = $2",
                        )
                        .bind(&self.config.name)
                        .bind(term)
                        .fetch_one(&self.pool)
                        .await
                        .map_err(|e| NexusError::Backend(e.to_string()))?;

                        let postings: Vec<(Uuid, i32, String, i32)> = sqlx::query_as(
                            "SELECT p.doc_id, p.tf, p.field_name, d.doc_len
                             FROM nexus_postings p
                             JOIN nexus_documents d
                               ON d.index_name = p.index_name AND d.doc_id = p.doc_id
                             WHERE p.index_name = $1 AND p.term = $2
                             LIMIT 10000",
                        )
                        .bind(&self.config.name)
                        .bind(term)
                        .fetch_all(&self.pool)
                        .await
                        .map_err(|e| NexusError::Backend(e.to_string()))?;

                        for (doc_id, tf, field_name, doc_len) in postings {
                            let boost = self
                                .config
                                .schema
                                .field(&field_name)
                                .map(|f| f.boost)
                                .unwrap_or(1.0);
                            let s = scorer.score(
                                tf as u32,
                                df.0 as u64,
                                doc_len.max(1) as u32,
                                boost,
                            );
                            *scores.entry(doc_id).or_default() += s;
                        }
                    }

                    let mut sorted: Vec<(Uuid, f32)> = scores.into_iter().collect();
                    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    bm25_hits = sorted;
                }
            }
        }

        // ── 2. Vector search ─────────────────────────────────────────────────
        let mut vector_hits: Vec<(Uuid, f32)> = Vec::new();

        if let Some(ref query_emb) = query.vector {
            #[cfg(feature = "pgvector")]
            {
                vector_hits = if self.ann_enabled {
                    self.vector_search_pgvector(query_emb, (query.limit + query.offset).max(1) * 10)
                        .await?
                } else {
                    self.vector_search_brute_force(query_emb).await?
                };
            }
            #[cfg(not(feature = "pgvector"))]
            {
                vector_hits = self.vector_search_brute_force(query_emb).await?;
            }
        }

        // ── 3. Fuse rankings ─────────────────────────────────────────────────
        let candidate_ids: Vec<(Uuid, f32)> = match (bm25_hits.is_empty(), vector_hits.is_empty()) {
            (false, true) => bm25_hits,
            (true, false) => vector_hits,
            (false, false) => Rrf::default().fuse(&bm25_hits, &vector_hits, query.hybrid_alpha),
            (true, true) => {
                let rows: Vec<(Uuid,)> = sqlx::query_as(
                    "SELECT doc_id FROM nexus_documents WHERE index_name = $1
                     ORDER BY indexed_at DESC LIMIT $2",
                )
                .bind(&self.config.name)
                .bind(query.limit as i64 + query.offset as i64)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
                rows.into_iter().map(|(id,)| (id, 1.0)).collect()
            }
        };

        // ── 4. Facet filter ──────────────────────────────────────────────────
        let filtered: Vec<(Uuid, f32)> = if let Some(ref filter) = query.filter {
            let mut kept = Vec::new();
            for (id, score) in candidate_ids {
                if let Ok(Some(doc)) = self.get(id).await {
                    if filter.matches(&doc.fields) {
                        kept.push((id, score));
                    }
                }
            }
            kept
        } else {
            candidate_ids
        };

        // ── 5. Min-score filter ──────────────────────────────────────────────
        let filtered: Vec<(Uuid, f32)> = if let Some(min) = query.min_score {
            filtered.into_iter().filter(|(_, s)| *s >= min).collect()
        } else {
            filtered
        };

        let total_hits = filtered.len() as u64;

        // ── 6. Pagination ────────────────────────────────────────────────────
        let page: Vec<(Uuid, f32)> = filtered
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect();

        // ── 7. Fetch documents + generate highlights ─────────────────────────
        let query_terms: Vec<String> = if let Some(ref text) = query.text {
            self.tokenizer.terms(text).unwrap_or_default()
        } else {
            vec![]
        };

        let mut hits = Vec::with_capacity(page.len());
        for (id, score) in page {
            if let Ok(Some(doc)) = self.get(id).await {
                let highlights = if !query.highlight_fields.is_empty() && !query_terms.is_empty() {
                    query
                        .highlight_fields
                        .iter()
                        .filter_map(|field| {
                            let text = doc.get_text(field)?;
                            self.highlighter.highlight_field(field, text, &query_terms)
                        })
                        .collect()
                } else {
                    vec![]
                };
                hits.push(SearchResult { id, score, highlights, document: doc });
            }
        }

        // ── 8. Facet distributions ───────────────────────────────────────────
        let mut facet_distributions = std::collections::HashMap::new();
        for facet_field in &query.facets {
            let rows: Vec<(String, i64)> = sqlx::query_as(
                "SELECT value, COUNT(*) FROM nexus_facets
                 WHERE index_name = $1 AND field_name = $2
                 GROUP BY value ORDER BY 2 DESC LIMIT 100",
            )
            .bind(&self.config.name)
            .bind(facet_field)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

            facet_distributions.insert(
                facet_field.clone(),
                nexus_core::facet::FacetDistribution {
                    field: facet_field.clone(),
                    buckets: rows
                        .into_iter()
                        .map(|(value, count)| nexus_core::facet::FacetBucket {
                            value,
                            count: count as u64,
                        })
                        .collect(),
                },
            );
        }

        Ok(SearchResults {
            hits,
            total_hits,
            processing_ms: start.elapsed().as_millis() as u64,
            facet_distributions,
            interpreted_query: query.text.clone(),
        })
    }

    async fn stats(&self) -> Result<IndexStats> {
        let (doc_count, avg_len) = self.fetch_stats_row().await?;
        let unique_terms: (Option<i64>,) = sqlx::query_as(
            "SELECT COUNT(DISTINCT term) FROM nexus_postings WHERE index_name = $1",
        )
        .bind(&self.config.name)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| NexusError::Backend(e.to_string()))?;

        Ok(IndexStats {
            document_count: doc_count,
            avg_doc_len: avg_len,
            unique_terms: unique_terms.0.unwrap_or(0) as u64,
            ..Default::default()
        })
    }

    async fn flush(&self) -> Result<()> {
        Ok(())
    }

    async fn drop_index(&self) -> Result<()> {
        sqlx::query("DELETE FROM nexus_indexes WHERE name = $1")
            .bind(&self.config.name)
            .execute(&self.pool)
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;
        Ok(())
    }
}
