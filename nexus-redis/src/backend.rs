use async_trait::async_trait;
use nexus_core::{
    document::{Document, DocumentId},
    error::{NexusError, Result},
    facet::FacetFilter,
    index::{Index, IndexConfig, IndexStats},
    query::{Query, SearchResult, SearchResults},
    score::{Bm25Scorer, typo_budget, edit_distance},
    tokenizer::{Tokenizer, TokenizerConfig},
};
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use std::collections::HashMap;
use tracing::{debug, instrument};
use uuid::Uuid;

fn doc_key(index: &str, id: Uuid) -> String {
    format!("nx:{index}:doc:{id}")
}

fn post_key(index: &str, term: &str) -> String {
    format!("nx:{index}:post:{term}")
}

fn scores_key(index: &str, term: &str) -> String {
    format!("nx:{index}:scores:{term}")
}

fn facet_key(index: &str, field: &str, value: &str) -> String {
    format!("nx:{index}:facet:{field}:{value}")
}

fn stats_key(index: &str) -> String {
    format!("nx:{index}:stats")
}

fn config_key(index: &str) -> String {
    format!("nx:{index}:config")
}

fn sync_channel(index: &str) -> String {
    format!("nx:{index}:sync")
}

/// Redis-backed Nexus index.
///
/// Cheap to clone — the `ConnectionManager` is an `Arc` internally.
pub struct RedisIndex {
    conn: ConnectionManager,
    config: IndexConfig,
    tokenizer: Tokenizer,
}

impl RedisIndex {
    pub async fn open(conn: ConnectionManager, config: IndexConfig) -> Result<Self> {
        let mut c = conn.clone();
        let config_json =
            serde_json::to_string(&config).map_err(NexusError::Serialization)?;
        let _: () = c
            .set(config_key(&config.name), &config_json)
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

        Ok(Self {
            conn,
            config,
            tokenizer: Tokenizer::new(TokenizerConfig::default()),
        })
    }

    fn compress(doc: &Document) -> Result<Vec<u8>> {
        let json = serde_json::to_vec(doc).map_err(NexusError::Serialization)?;
        Ok(lz4_flex::compress_prepend_size(&json))
    }

    fn decompress(bytes: &[u8]) -> Result<Document> {
        let json = lz4_flex::decompress_size_prepended(bytes)
            .map_err(|e| NexusError::Compression(e.to_string()))?;
        serde_json::from_slice(&json).map_err(NexusError::Serialization)
    }

    async fn get_stats(&self) -> Result<(u64, f64)> {
        let mut c = self.conn.clone();
        let doc_count: u64 = c
            .hget(stats_key(&self.config.name), "doc_count")
            .await
            .unwrap_or(0);
        let avg_len: f64 = {
            let total_tokens: u64 = c
                .hget(stats_key(&self.config.name), "total_tokens")
                .await
                .unwrap_or(0);
            if doc_count > 0 {
                total_tokens as f64 / doc_count as f64
            } else {
                100.0
            }
        };
        Ok((doc_count, avg_len))
    }

    /// Publish an invalidation event to the sync channel so other nodes can
    /// evict cached data for affected document IDs.
    async fn publish_sync(&self, msg: &str) {
        let mut c = self.conn.clone();
        let _: redis::RedisResult<()> = c.publish(sync_channel(&self.config.name), msg).await;
    }
}

#[async_trait]
impl Index for RedisIndex {
    fn config(&self) -> &IndexConfig {
        &self.config
    }

    #[instrument(skip(self, doc), fields(index = %self.config.name))]
    async fn upsert(&self, mut doc: Document) -> Result<DocumentId> {
        doc.indexed_at = Some(chrono::Utc::now());
        self.config.schema.validate(&doc)?;

        let compressed = Self::compress(&doc)?;
        let mut c = self.conn.clone();

        // Store compressed document body.
        let _: () = c
            .set(doc_key(&self.config.name, doc.id), compressed.as_slice())
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

        // Index text fields into inverted index (Redis hash + sorted set).
        let mut total_tokens = 0u64;
        for field_schema in self.config.schema.text_fields() {
            if let Some(nexus_core::document::FieldValue::Text(text)) =
                doc.fields.get(&field_schema.name)
            {
                let tokens = self.tokenizer.tokenize(text)?;
                let mut tf_map: HashMap<String, u32> = HashMap::new();
                for tok in &tokens {
                    *tf_map.entry(tok.text.clone()).or_default() += 1;
                }
                total_tokens += tokens.len() as u64;

                let doc_len = tokens.len() as u32;
                let (doc_count, avg_len) = self.get_stats().await?;
                let scorer = Bm25Scorer {
                    avg_doc_len: avg_len as f32,
                    doc_count: doc_count.max(1),
                    ..Default::default()
                };

                for (term, tf) in tf_map {
                    let boost = self
                        .config
                        .schema
                        .field(&field_schema.name)
                        .map(|f| f.boost)
                        .unwrap_or(1.0);
                    let score = scorer.score(tf, 1, doc_len, boost);

                    // Hash: term → {doc_id: tf}
                    let _: () = c
                        .hset(post_key(&self.config.name, &term), doc.id.to_string(), tf)
                        .await
                        .map_err(|e| NexusError::Backend(e.to_string()))?;

                    // Sorted set: term → {doc_id: bm25_score}
                    let _: () = c
                        .zadd(
                            scores_key(&self.config.name, &term),
                            doc.id.to_string(),
                            score as f64,
                        )
                        .await
                        .map_err(|e| NexusError::Backend(e.to_string()))?;
                }
            }
        }

        // Index facet fields into sets.
        for field_schema in self.config.schema.facet_fields() {
            if let Some(value) = doc.fields.get(&field_schema.name) {
                let label = match value {
                    nexus_core::document::FieldValue::Text(s) => s.clone(),
                    nexus_core::document::FieldValue::Integer(n) => n.to_string(),
                    nexus_core::document::FieldValue::Float(f) => format!("{f:.2}"),
                    nexus_core::document::FieldValue::Bool(b) => b.to_string(),
                    _ => continue,
                };
                let _: () = c
                    .sadd(
                        facet_key(&self.config.name, &field_schema.name, &label),
                        doc.id.to_string(),
                    )
                    .await
                    .map_err(|e| NexusError::Backend(e.to_string()))?;
            }
        }

        // Update stats.
        let _: () = c
            .hincr(stats_key(&self.config.name), "doc_count", 1u64)
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;
        let _: () = c
            .hincr(stats_key(&self.config.name), "total_tokens", total_tokens)
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;

        self.publish_sync(&format!("upsert:{}", doc.id)).await;
        debug!(doc_id = %doc.id, "redis document indexed");
        Ok(doc.id)
    }

    async fn upsert_many(&self, docs: Vec<Document>) -> Result<Vec<DocumentId>> {
        let mut ids = Vec::with_capacity(docs.len());
        for doc in docs {
            ids.push(self.upsert(doc).await?);
        }
        Ok(ids)
    }

    async fn delete(&self, id: DocumentId) -> Result<()> {
        let mut c = self.conn.clone();
        let _: () = c
            .del(doc_key(&self.config.name, id))
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;
        self.publish_sync(&format!("delete:{id}")).await;
        Ok(())
    }

    async fn delete_by_filter(&self, _filter: &FacetFilter) -> Result<u64> {
        Ok(0)
    }

    async fn get(&self, id: DocumentId) -> Result<Option<Document>> {
        let mut c = self.conn.clone();
        let bytes: Option<Vec<u8>> = c
            .get(doc_key(&self.config.name, id))
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;
        match bytes {
            None => Ok(None),
            Some(b) => Ok(Some(Self::decompress(&b)?)),
        }
    }

    #[instrument(skip(self, query), fields(index = %self.config.name))]
    async fn search(&self, query: Query) -> Result<SearchResults> {
        let start = std::time::Instant::now();
        let mut c = self.conn.clone();

        // ── 1. BM25 retrieval via sorted-set union ────────────────────────────
        let mut bm25_hits: Vec<(Uuid, f32)> = Vec::new();

        if let Some(ref text) = query.text {
            let terms = self.tokenizer.terms(text)?;
            if !terms.is_empty() {
                let budget = typo_budget(terms.iter().map(|t| t.len()).max().unwrap_or(0));

                // Gather all posting keys and find matching terms.
                let all_keys: Vec<String> = {
                    let pattern = format!("nx:{}:post:*", &self.config.name);
                    c.keys(&pattern)
                        .await
                        .map_err(|e| NexusError::Backend(e.to_string()))?
                };

                let prefix = format!("nx:{}:post:", &self.config.name);
                let mut matched_terms: Vec<String> = Vec::new();
                for key in &all_keys {
                    let cand = key.trim_start_matches(&prefix);
                    for qt in &terms {
                        if qt == cand {
                            matched_terms.push(cand.to_owned());
                        } else if budget > 0 && edit_distance(qt, cand, budget).is_some() {
                            matched_terms.push(cand.to_owned());
                        }
                    }
                }
                matched_terms.dedup();

                let mut scores_map: HashMap<Uuid, f32> = HashMap::new();
                for term in &matched_terms {
                    let entries: Vec<(String, f64)> = c
                        .zrange_withscores(scores_key(&self.config.name, term), 0, -1)
                        .await
                        .map_err(|e| NexusError::Backend(e.to_string()))?;
                    for (id_str, score) in entries {
                        if let Ok(id) = id_str.parse::<Uuid>() {
                            *scores_map.entry(id).or_default() += score as f32;
                        }
                    }
                }
                let mut sorted: Vec<(Uuid, f32)> = scores_map.into_iter().collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                bm25_hits = sorted;
            }
        }

        // ── 2. No vector search in Redis backend (use pgvector for ANN) ──────

        // ── 3. Facet filter ──────────────────────────────────────────────────
        let candidate_ids = bm25_hits;
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

        let filtered: Vec<(Uuid, f32)> = if let Some(min) = query.min_score {
            filtered.into_iter().filter(|(_, s)| *s >= min).collect()
        } else {
            filtered
        };

        let total_hits = filtered.len() as u64;
        let page: Vec<(Uuid, f32)> = filtered
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect();

        let mut hits = Vec::with_capacity(page.len());
        for (id, score) in page {
            if let Ok(Some(doc)) = self.get(id).await {
                hits.push(SearchResult {
                    id,
                    score,
                    highlights: vec![],
                    document: doc,
                });
            }
        }

        Ok(SearchResults {
            hits,
            total_hits,
            processing_ms: start.elapsed().as_millis() as u64,
            facet_distributions: HashMap::new(),
            interpreted_query: query.text.clone(),
        })
    }

    async fn stats(&self) -> Result<IndexStats> {
        let (doc_count, avg_len) = self.get_stats().await?;
        Ok(IndexStats {
            document_count: doc_count,
            avg_doc_len: avg_len as f32,
            ..Default::default()
        })
    }

    async fn flush(&self) -> Result<()> {
        Ok(())
    }

    async fn drop_index(&self) -> Result<()> {
        let mut c = self.conn.clone();
        let pattern = format!("nx:{}:*", &self.config.name);
        let keys: Vec<String> = c
            .keys(&pattern)
            .await
            .map_err(|e| NexusError::Backend(e.to_string()))?;
        if !keys.is_empty() {
            let _: () = c
                .del(keys)
                .await
                .map_err(|e| NexusError::Backend(e.to_string()))?;
        }
        Ok(())
    }
}
