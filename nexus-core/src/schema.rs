use crate::error::{NexusError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Every field in the schema declares its type.
/// The engine enforces this at index time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldType {
    /// Full-text indexed. Tokenized, stemmed, BM25 scored.
    Text {
        /// Enable typo-tolerance on this field.
        typo_tolerance: bool,
        /// Store original value for highlight retrieval.
        stored: bool,
        /// Language for stemmer / stop-word list. `None` = language-agnostic.
        language: Option<crate::Language>,
    },
    /// Exact-match keyword. Not tokenized. Good for IDs, tags, slugs.
    Keyword { stored: bool },
    Integer { stored: bool, facet: bool },
    Float { stored: bool, facet: bool },
    Bool { stored: bool, facet: bool },
    Timestamp { stored: bool, sortable: bool },
    /// Dense embedding. Dimension must be stated at schema definition time.
    Vector { dim: usize },
    /// Pass-through JSON blob. Stored but not indexed.
    Json,
}

impl FieldType {
    /// Is this field eligible for BM25 full-text scoring?
    pub fn is_text(&self) -> bool {
        matches!(self, FieldType::Text { .. })
    }

    /// Does this field carry a vector that participates in ANN search?
    pub fn is_vector(&self) -> bool {
        matches!(self, FieldType::Vector { .. })
    }

    /// Is this field usable as a facet?
    pub fn is_facet(&self) -> bool {
        matches!(
            self,
            FieldType::Keyword { .. }
                | FieldType::Integer { facet: true, .. }
                | FieldType::Float { facet: true, .. }
                | FieldType::Bool { facet: true, .. }
        )
    }
}

/// A field declaration inside the schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSchema {
    pub name: String,
    pub field_type: FieldType,
    /// Boost this field's BM25 contribution. Default 1.0.
    pub boost: f32,
    /// Required fields must be present on every indexed document.
    pub required: bool,
}

impl FieldSchema {
    fn new(name: impl Into<String>, field_type: FieldType) -> Self {
        Self {
            name: name.into(),
            field_type,
            boost: 1.0,
            required: false,
        }
    }
}

/// Schema for an index. Immutable after index creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub(crate) fields: HashMap<String, FieldSchema>,
    /// The primary text field used for default query expansion.
    pub(crate) default_search_field: Option<String>,
}

impl Schema {
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    pub fn field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.get(name)
    }

    pub fn text_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields.values().filter(|f| f.field_type.is_text())
    }

    pub fn vector_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields.values().filter(|f| f.field_type.is_vector())
    }

    pub fn facet_fields(&self) -> impl Iterator<Item = &FieldSchema> {
        self.fields.values().filter(|f| f.field_type.is_facet())
    }

    /// Validate a document's fields against this schema.
    pub fn validate(&self, doc: &crate::Document) -> Result<()> {
        for (name, schema) in &self.fields {
            if schema.required && !doc.fields.contains_key(name) {
                return Err(NexusError::SchemaValidation(format!(
                    "Required field '{name}' missing"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct SchemaBuilder {
    fields: HashMap<String, FieldSchema>,
    default_search_field: Option<String>,
}

impl SchemaBuilder {
    /// Add a full-text field with typo tolerance enabled by default.
    pub fn text(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(
                name,
                FieldType::Text {
                    typo_tolerance: true,
                    stored: true,
                    language: None,
                },
            ),
        );
        self
    }

    pub fn text_lang(mut self, name: impl Into<String>, lang: crate::Language) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(
                name,
                FieldType::Text {
                    typo_tolerance: true,
                    stored: true,
                    language: Some(lang),
                },
            ),
        );
        self
    }

    pub fn keyword(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(name, FieldType::Keyword { stored: true }),
        );
        self
    }

    pub fn integer(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(name, FieldType::Integer { stored: true, facet: false }),
        );
        self
    }

    pub fn integer_facet(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(name, FieldType::Integer { stored: true, facet: true }),
        );
        self
    }

    pub fn float(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(name, FieldType::Float { stored: true, facet: false }),
        );
        self
    }

    pub fn bool_facet(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(name, FieldType::Bool { stored: true, facet: true }),
        );
        self
    }

    pub fn timestamp(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(name, FieldType::Timestamp { stored: true, sortable: true }),
        );
        self
    }

    pub fn vector(mut self, name: impl Into<String>, dim: usize) -> Self {
        let name = name.into();
        self.fields.insert(
            name.clone(),
            FieldSchema::new(name, FieldType::Vector { dim }),
        );
        self
    }

    /// Mark a field as the default field for un-qualified queries.
    pub fn default_search_field(mut self, name: impl Into<String>) -> Self {
        self.default_search_field = Some(name.into());
        self
    }

    pub fn boost(mut self, name: &str, boost: f32) -> Self {
        if let Some(f) = self.fields.get_mut(name) {
            f.boost = boost;
        }
        self
    }

    pub fn required(mut self, name: &str) -> Self {
        if let Some(f) = self.fields.get_mut(name) {
            f.required = true;
        }
        self
    }

    pub fn build(self) -> Schema {
        Schema {
            fields: self.fields,
            default_search_field: self.default_search_field,
        }
    }
}
