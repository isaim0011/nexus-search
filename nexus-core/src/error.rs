use thiserror::Error;

#[derive(Debug, Error)]
pub enum NexusError {
    #[error("backend error: {0}")]
    Backend(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("compression error: {0}")]
    Compression(String),

    #[error("schema validation: {0}")]
    SchemaValidation(String),

    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimMismatch { expected: usize, actual: usize },

    #[error("document not found: {0}")]
    NotFound(String),

    #[error("index not found: {0}")]
    IndexNotFound(String),

    #[error("invalid query: {0}")]
    InvalidQuery(String),
}

pub type Result<T> = std::result::Result<T, NexusError>;
