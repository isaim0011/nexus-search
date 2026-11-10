//! HTTP dashboard and REST API for Nexus.
//!
//! Mount `router()` into an existing Axum application:
//!
//! ```rust,no_run
//! use axum::Router;
//! use nexus_dashboard::{DashboardState, router};
//! use nexus_core::{Index, IndexConfig, Schema};
//! use std::sync::Arc;
//!
//! # async fn example() {
//! // Build your index (nexus-postgres, nexus-redis, or any Index impl).
//! // let index: Arc<dyn Index> = Arc::new(my_index);
//! // let state = DashboardState::new(index);
//! // let app = Router::new().nest("/search", router(state));
//! # }
//! ```
pub mod router;
pub mod types;

pub use router::router;
pub use types::DashboardState;
