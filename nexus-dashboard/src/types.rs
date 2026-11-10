use nexus_core::Index;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Shared Axum state for the dashboard router.
///
/// Holds a type-erased `Index` so the dashboard works with Postgres, Redis,
/// or any other backend without generics leaking into the router.
#[derive(Clone)]
pub struct DashboardState {
    pub index: Arc<dyn Index>,
}

impl DashboardState {
    pub fn new(index: Arc<dyn Index>) -> Self {
        Self { index }
    }
}

/// Request body for `POST /indexes/:name/documents`.
#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub documents: Vec<nexus_core::Document>,
}

/// Request body for `GET /indexes/:name/search`.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub q: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub highlight: Vec<String>,
    #[serde(default)]
    pub facets: Vec<String>,
}

fn default_limit() -> usize {
    20
}

/// Generic JSON envelope for success responses.
#[derive(Debug, Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub data: T,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn ok(data: T) -> axum::Json<Self> {
        axum::Json(Self { data })
    }
}

/// Generic JSON error envelope.
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
}
