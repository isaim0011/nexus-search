use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique document identifier — always a UUID v4.
pub type DocumentId = Uuid;

/// A single field value. Nexus is typed at the schema level;
/// every field stored in a document must match its declared `FieldType`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldValue {
    Text(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Timestamp(DateTime<Utc>),
    /// A dense embedding vector. Dimension must match the schema declaration.
    Vector(Vec<f32>),
    /// Arbitrary nested JSON — stored as-is, not indexed for full-text.
    Json(serde_json::Value),
    /// Multi-valued field (e.g. tags, categories).
    Array(Vec<FieldValue>),
    Null,
}

impl From<&str> for FieldValue {
    fn from(s: &str) -> Self { FieldValue::Text(s.to_owned()) }
}
impl From<String> for FieldValue {
    fn from(s: String) -> Self { FieldValue::Text(s) }
}
impl From<i64> for FieldValue {
    fn from(n: i64) -> Self { FieldValue::Integer(n) }
}
impl From<f64> for FieldValue {
    fn from(f: f64) -> Self { FieldValue::Float(f) }
}
impl From<bool> for FieldValue {
    fn from(b: bool) -> Self { FieldValue::Bool(b) }
}
impl From<Vec<f32>> for FieldValue {
    fn from(v: Vec<f32>) -> Self { FieldValue::Vector(v) }
}

/// Named field on a document. Wraps the name + value together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub value: FieldValue,
}

impl Field {
    pub fn new(name: impl Into<String>, value: impl Into<FieldValue>) -> Self {
        Self { name: name.into(), value: value.into() }
    }
}

/// A document ready for indexing. Maps field names → values.
///
/// # Example
/// ```rust,no_run
/// use nexus_core::Document;
///
/// let doc = Document::builder()
///     .field("title", "Rust async programming")
///     .field("body", "Tokio makes async Rust ergonomic.")
///     .field("score", 4.9f64)
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Auto-assigned if not provided at build time.
    pub id: DocumentId,
    pub fields: HashMap<String, FieldValue>,
    /// Wall-clock time the document was indexed. Set by the engine, not the caller.
    pub indexed_at: Option<DateTime<Utc>>,
}

impl Document {
    pub fn builder() -> DocumentBuilder { DocumentBuilder::default() }

    pub fn get(&self, field: &str) -> Option<&FieldValue> {
        self.fields.get(field)
    }

    pub fn get_text(&self, field: &str) -> Option<&str> {
        match self.fields.get(field)? {
            FieldValue::Text(t) => Some(t.as_str()),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct DocumentBuilder {
    id: Option<DocumentId>,
    fields: HashMap<String, FieldValue>,
}

impl DocumentBuilder {
    pub fn id(mut self, id: DocumentId) -> Self {
        self.id = Some(id);
        self
    }

    pub fn field(mut self, name: impl Into<String>, value: impl Into<FieldValue>) -> Self {
        self.fields.insert(name.into(), value.into());
        self
    }

    pub fn build(self) -> Document {
        Document {
            id: self.id.unwrap_or_else(Uuid::new_v4),
            fields: self.fields,
            indexed_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_round_trip() {
        let doc = Document::builder()
            .field("title", "hello world")
            .field("score", 42i64)
            .build();

        assert_eq!(doc.get_text("title"), Some("hello world"));
        assert!(matches!(doc.get("score"), Some(FieldValue::Integer(42))));
        assert!(doc.id != Uuid::nil());
    }
}
