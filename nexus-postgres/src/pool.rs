use anyhow::Result;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions};
use std::str::FromStr;

pub type PgPool = sqlx::PgPool;

/// Configuration for the connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub database_url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    /// Connection acquire timeout in seconds.
    pub acquire_timeout_secs: u64,
    pub idle_timeout_secs: Option<u64>,
    pub max_lifetime_secs: Option<u64>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://localhost/nexus".into()),
            max_connections: 10,
            min_connections: 1,
            acquire_timeout_secs: 5,
            idle_timeout_secs: Some(600),
            max_lifetime_secs: Some(1800),
        }
    }
}

/// Build and validate a connection pool.
pub async fn connect(config: &PoolConfig) -> Result<PgPool> {
    let connect_opts = PgConnectOptions::from_str(&config.database_url)?
        .application_name("nexus");

    let pool = PgPoolOptions::new()
        .max_connections(config.max_connections)
        .min_connections(config.min_connections)
        .acquire_timeout(std::time::Duration::from_secs(config.acquire_timeout_secs))
        .idle_timeout(config.idle_timeout_secs.map(std::time::Duration::from_secs))
        .max_lifetime(config.max_lifetime_secs.map(std::time::Duration::from_secs))
        .connect_with(connect_opts)
        .await?;

    // Run migrations immediately so callers don't have to.
    crate::migrations::run(&pool).await?;

    Ok(pool)
}
