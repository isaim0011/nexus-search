use crate::{document::{Document, DocumentId}, facet::{FacetDistribution, FacetFilter}, vector::Embedding};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A search query. Build via `QueryBuilder`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Free-text query string. Tokenized, typo-expanded, BM25 scored.
    pub text: Option<String>,
    /// Dense query vector for semantic (ANN) search. If both `text` and `vector`
    /// are set, results are fused via RRF.
    pub vector: Option<Embedding>,
    /// Pre-filter applied before scoring. Zero cost on un-indexed fields.
    pub filter: Option<FacetFilter>,
    /// Fields to return facet distributions for.
    pub facets: Vec<String>,
    /// Maximum number of results.
    pub limit: usize,
    /// Offset for pagination.
    pub offset: usize,
    /// Custom sort order. Overrides relevance ranking when set.
    pub sort: Vec<SortClause>,
    /// Fields to highlight match fragments in. Empty = no highlights.
    pub highlight_fields: Vec<String>,
    /// BM25 ↔ vector blend (0.0 = pure vector, 1.0 = pure BM25, 0.5 = equal).
    /// Ignored unless both `text` and `vector` are provided.
    pub hybrid_alpha: f32,
    /// Restrict query to specific fields (overrides schema default).
    pub search_fields: Vec<String>,
    /// Minimum score threshold — results below this are dropped.
    pub min_score: Option<f32>,
}

impl Default for Query {
    fn default() -> Self {
        Self {
            text: None,
            vector: None,
            filter: None,
            facets: Vec::new(),
            limit: 20,
            offset: 0,
            sort: Vec::new(),
            highlight_fields: Vec::new(),
            hybrid_alpha: 0.5,
            search_fields: Vec::new(),
            min_score: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortClause {
    pub field: String,
    pub order: SortOrder,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SortOrder {
    Asc,
    Desc,
}

/// A highlighted fragment of a matching field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Highlight {
    pub field: String,
    /// Original text with match spans wrapped in `<mark>…</mark>`.
    pub snippet: String,
    pub matched_tokens: Vec<String>,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: DocumentId,
    pub score: f32,
    /// The document itself, with only stored fields returned.
    pub document: Document,
    pub highlights: Vec<Highlight>,
}

/// The full response from a search call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub hits: Vec<SearchResult>,
    pub total_hits: u64,
    /// Time the query took on the backend, in milliseconds.
    pub processing_ms: u64,
    /// Facet distributions, one per requested facet field.
    pub facet_distributions: HashMap<String, FacetDistribution>,
    /// The query as interpreted (useful for debug / "did you mean?" UI).
    pub interpreted_query: Option<String>,
}

impl SearchResults {
    pub fn empty() -> Self {
        Self {
            hits: Vec::new(),
            total_hits: 0,
            processing_ms: 0,
            facet_distributions: HashMap::new(),
            interpreted_query: None,
        }
    }
}

/// Builder for a `Query`.
///
/// # Example
/// ```rust,no_run
/// use nexus_core::{QueryBuilder, facet::FacetFilter, document::FieldValue};
///
/// let q = QueryBuilder::new()
///     .text("async rust web framework")
///     .filter(FacetFilter::Gte { field: "stars".into(), value: 100.0 })
///     .facets(["language", "license"])
///     .limit(10)
///     .highlight(["title", "body"])
///     .build();
/// ```
#[derive(Default)]
pub struct QueryBuilder {
    inner: Query,
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            inner: Query::default(),
        }
    }

    pub fn text(mut self, q: impl Into<String>) -> Self {
        self.inner.text = Some(q.into());
        self
    }

    pub fn vector(mut self, v: impl Into<Embedding>) -> Self {
        self.inner.vector = Some(v.into());
        self
    }

    pub fn filter(mut self, f: FacetFilter) -> Self {
        self.inner.filter = Some(f);
        self
    }

    pub fn facets<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.facets = fields.into_iter().map(Into::into).collect();
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.inner.limit = n;
        self
    }

    pub fn offset(mut self, n: usize) -> Self {
        self.inner.offset = n;
        self
    }

    pub fn sort(mut self, field: impl Into<String>, order: SortOrder) -> Self {
        self.inner.sort.push(SortClause { field: field.into(), order });
        self
    }

    pub fn highlight<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.highlight_fields = fields.into_iter().map(Into::into).collect();
        self
    }

    pub fn hybrid_alpha(mut self, alpha: f32) -> Self {
        self.inner.hybrid_alpha = alpha.clamp(0.0, 1.0);
        self
    }

    pub fn search_fields<I, S>(mut self, fields: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.inner.search_fields = fields.into_iter().map(Into::into).collect();
        self
    }

    pub fn min_score(mut self, s: f32) -> Self {
        self.inner.min_score = Some(s);
        self
    }

    pub fn build(self) -> Query {
        self.inner
    }
}
