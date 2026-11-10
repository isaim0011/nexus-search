use crate::types::{ApiError, ApiResponse, DashboardState, SearchRequest, UpsertRequest};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use nexus_core::{query::QueryBuilder, DocumentId};
use serde_json::json;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// Build the Nexus dashboard router.
///
/// Routes:
/// - `GET  /indexes`                                    — list all indexes
/// - `GET  /indexes/:name/stats`                        — index stats
/// - `GET  /indexes/:name/search?q=&limit=&offset=`     — search
/// - `POST /indexes/:name/documents`                    — upsert documents
/// - `DELETE /indexes/:name/documents/:id`              — delete document
///
/// # Example
/// ```rust,no_run
/// use axum::Router;
/// use nexus_dashboard::{DashboardState, router};
/// use std::sync::Arc;
///
/// # async fn example() {
/// // let state = DashboardState::new(Arc::new(my_index));
/// // let app = Router::new().nest("/nexus", router(state));
/// # }
/// ```
pub fn router(state: DashboardState) -> Router {
    Router::new()
        .route("/indexes", get(list_indexes))
        .route("/indexes/:name/stats", get(index_stats))
        .route("/indexes/:name/search", get(search))
        .route("/indexes/:name/documents", post(upsert_documents))
        .route("/indexes/:name/documents/:id", delete(delete_document))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn list_indexes(State(state): State<DashboardState>) -> Response {
    let config = state.index.config();
    Json(json!({
        "indexes": [{
            "name": config.name,
            "schema": config.schema,
        }]
    }))
    .into_response()
}

async fn index_stats(
    State(state): State<DashboardState>,
    Path(_name): Path<String>,
) -> Response {
    match state.index.stats().await {
        Ok(stats) => ApiResponse::ok(stats).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn search(
    State(state): State<DashboardState>,
    Path(_name): Path<String>,
    Query(params): Query<SearchRequest>,
) -> Response {
    let mut qb = QueryBuilder::new()
        .limit(params.limit)
        .offset(params.offset);

    if let Some(q) = params.q {
        qb = qb.text(q);
    }
    if !params.highlight.is_empty() {
        qb = qb.highlight(params.highlight);
    }
    if !params.facets.is_empty() {
        qb = qb.facets(params.facets);
    }

    match state.index.search(qb.build()).await {
        Ok(results) => ApiResponse::ok(results).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn upsert_documents(
    State(state): State<DashboardState>,
    Path(_name): Path<String>,
    Json(body): Json<UpsertRequest>,
) -> Response {
    match state.index.upsert_many(body.documents).await {
        Ok(ids) => ApiResponse::ok(json!({ "indexed": ids.len(), "ids": ids })).into_response(),
        Err(e) => error_response(StatusCode::UNPROCESSABLE_ENTITY, &e.to_string()),
    }
}

async fn delete_document(
    State(state): State<DashboardState>,
    Path((_name, id)): Path<(String, DocumentId)>,
) -> Response {
    match state.index.delete(id).await {
        Ok(()) => ApiResponse::ok(json!({ "deleted": id })).into_response(),
        Err(e) => error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

fn error_response(status: StatusCode, msg: &str) -> Response {
    (status, Json(ApiError { error: msg.to_owned() })).into_response()
}
