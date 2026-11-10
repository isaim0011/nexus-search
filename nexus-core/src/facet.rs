use crate::document::FieldValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single facet bucket: label + document count.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FacetBucket {
    pub value: String,
    pub count: u64,
}

/// Facet distribution for one field — the list of buckets, sorted by count desc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetDistribution {
    pub field: String,
    pub buckets: Vec<FacetBucket>,
}

impl FacetDistribution {
    /// Build a distribution from an iterator of (field_value) hits.
    pub fn from_values(field: impl Into<String>, values: &[&FieldValue]) -> Self {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for v in values {
            if let Some(label) = facet_label(v) {
                *counts.entry(label).or_default() += 1;
            }
        }
        let mut buckets: Vec<FacetBucket> = counts
            .into_iter()
            .map(|(value, count)| FacetBucket { value, count })
            .collect();
        buckets.sort_by(|a, b| b.count.cmp(&a.count).then(a.value.cmp(&b.value)));
        Self {
            field: field.into(),
            buckets,
        }
    }
}

/// A filter condition applied before scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum FacetFilter {
    Eq { field: String, value: FieldValue },
    NotEq { field: String, value: FieldValue },
    Lt { field: String, value: f64 },
    Lte { field: String, value: f64 },
    Gt { field: String, value: f64 },
    Gte { field: String, value: f64 },
    In { field: String, values: Vec<FieldValue> },
    And(Vec<FacetFilter>),
    Or(Vec<FacetFilter>),
    Not(Box<FacetFilter>),
}

impl FacetFilter {
    /// Evaluate this filter against a document's fields map.
    pub fn matches(&self, fields: &HashMap<String, FieldValue>) -> bool {
        match self {
            FacetFilter::Eq { field, value } => {
                fields.get(field).map_or(false, |v| v == value)
            }
            FacetFilter::NotEq { field, value } => {
                fields.get(field).map_or(true, |v| v != value)
            }
            FacetFilter::Lt { field, value } => {
                numeric(fields, field).map_or(false, |n| n < *value)
            }
            FacetFilter::Lte { field, value } => {
                numeric(fields, field).map_or(false, |n| n <= *value)
            }
            FacetFilter::Gt { field, value } => {
                numeric(fields, field).map_or(false, |n| n > *value)
            }
            FacetFilter::Gte { field, value } => {
                numeric(fields, field).map_or(false, |n| n >= *value)
            }
            FacetFilter::In { field, values } => {
                fields.get(field).map_or(false, |v| values.contains(v))
            }
            FacetFilter::And(filters) => filters.iter().all(|f| f.matches(fields)),
            FacetFilter::Or(filters) => filters.iter().any(|f| f.matches(fields)),
            FacetFilter::Not(inner) => !inner.matches(fields),
        }
    }
}

fn numeric(fields: &HashMap<String, FieldValue>, field: &str) -> Option<f64> {
    match fields.get(field)? {
        FieldValue::Integer(n) => Some(*n as f64),
        FieldValue::Float(f) => Some(*f),
        _ => None,
    }
}

fn facet_label(v: &FieldValue) -> Option<String> {
    match v {
        FieldValue::Text(s) => Some(s.clone()),
        FieldValue::Integer(n) => Some(n.to_string()),
        FieldValue::Float(f) => Some(format!("{f:.2}")),
        FieldValue::Bool(b) => Some(b.to_string()),
        FieldValue::Array(arr) => {
            // Multi-value fields: return the first element label for counting
            arr.first().and_then(|v| facet_label(v))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_eq() {
        let mut fields = HashMap::new();
        fields.insert("status".to_string(), FieldValue::Text("active".into()));
        let f = FacetFilter::Eq {
            field: "status".into(),
            value: FieldValue::Text("active".into()),
        };
        assert!(f.matches(&fields));
    }

    #[test]
    fn filter_range() {
        let mut fields = HashMap::new();
        fields.insert("price".to_string(), FieldValue::Float(29.99));
        let f = FacetFilter::And(vec![
            FacetFilter::Gte { field: "price".into(), value: 10.0 },
            FacetFilter::Lte { field: "price".into(), value: 50.0 },
        ]);
        assert!(f.matches(&fields));
    }

    #[test]
    fn facet_distribution() {
        let values: Vec<FieldValue> = vec![
            FieldValue::Text("rust".into()),
            FieldValue::Text("go".into()),
            FieldValue::Text("rust".into()),
            FieldValue::Text("rust".into()),
        ];
        let refs: Vec<&FieldValue> = values.iter().collect();
        let dist = FacetDistribution::from_values("lang", &refs);
        assert_eq!(dist.buckets[0].value, "rust");
        assert_eq!(dist.buckets[0].count, 3);
    }
}
