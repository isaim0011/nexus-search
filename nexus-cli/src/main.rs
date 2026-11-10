use anyhow::Result;
use clap::{Parser, Subcommand};

/// Nexus — search index management CLI
#[derive(Parser)]
#[command(
    name = "nexus",
    about = "Manage and monitor Nexus search indexes",
    version
)]
struct Cli {
    /// PostgreSQL connection URL (overrides DATABASE_URL env var)
    #[arg(short, long, env = "DATABASE_URL", global = true)]
    database_url: Option<String>,

    /// Log level: error | warn | info | debug | trace
    #[arg(short, long, default_value = "warn", global = true)]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show live stats for an index
    Stats {
        /// Index name
        index: String,
    },
    /// List all indexes
    List,
    /// Search an index
    Search {
        /// Index name
        index: String,
        /// Query text
        query: String,
        /// Maximum results to return
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Enable hybrid mode (requires --vector)
        #[arg(long)]
        hybrid: bool,
    },
    /// Reindex all documents in an index (useful after schema changes)
    Reindex {
        index: String,
    },
    /// Delete an index and all its data
    Drop {
        index: String,
        /// Skip confirmation prompt
        #[arg(short, long)]
        yes: bool,
    },
    /// Show failed / errored documents
    Errors {
        index: String,
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log_level.parse().unwrap_or_default()),
        )
        .init();

    let db_url = cli
        .database_url
        .unwrap_or_else(|| std::env::var("DATABASE_URL").unwrap_or_default());

    if db_url.is_empty() {
        anyhow::bail!("No database URL provided. Set DATABASE_URL or use --database-url.");
    }

    let pool_config = nexus_postgres::PoolConfig {
        database_url: db_url,
        ..Default::default()
    };
    let pool = nexus_postgres::pool::connect(&pool_config).await?;

    match cli.command {
        Commands::List => {
            let rows: Vec<(String, serde_json::Value)> =
                sqlx::query_as("SELECT name, stats FROM nexus_indexes ORDER BY name")
                    .fetch_all(&pool)
                    .await?;
            if rows.is_empty() {
                println!("No indexes found.");
            } else {
                println!("{:<30} {:>12} {:>12}", "INDEX", "DOCS", "TERMS");
                println!("{}", "-".repeat(56));
                for (name, stats) in rows {
                    let docs = stats["document_count"].as_u64().unwrap_or(0);
                    let terms = stats["unique_terms"].as_u64().unwrap_or(0);
                    println!("{:<30} {:>12} {:>12}", name, docs, terms);
                }
            }
        }

        Commands::Stats { index } => {
            let row: Option<(serde_json::Value, serde_json::Value)> = sqlx::query_as(
                "SELECT config, stats FROM nexus_indexes WHERE name = $1",
            )
            .bind(&index)
            .fetch_optional(&pool)
            .await?;

            match row {
                None => println!("Index '{}' not found.", index),
                Some((config, stats)) => {
                    println!("Index: {index}");
                    println!("Config:\n{}", serde_json::to_string_pretty(&config)?);
                    println!("Stats:\n{}", serde_json::to_string_pretty(&stats)?);

                    // Live counts
                    let (docs,): (i64,) =
                        sqlx::query_as("SELECT COUNT(*) FROM nexus_documents WHERE index_name = $1")
                            .bind(&index)
                            .fetch_one(&pool)
                            .await?;
                    let (terms,): (i64,) = sqlx::query_as(
                        "SELECT COUNT(DISTINCT term) FROM nexus_postings WHERE index_name = $1",
                    )
                    .bind(&index)
                    .fetch_one(&pool)
                    .await?;
                    println!("\nLive counts:");
                    println!("  Documents : {docs}");
                    println!("  Unique terms : {terms}");
                }
            }
        }

        Commands::Search {
            index,
            query,
            limit,
            ..
        } => {
            println!("Searching '{}' for: {}\n", index, query);
            // Basic tsvector search directly for CLI demo
            let rows: Vec<(uuid::Uuid, f64)> = sqlx::query_as(
                "SELECT doc_id, ts_rank(tsv, plainto_tsquery('english', $2)) AS rank
                 FROM nexus_documents
                 WHERE index_name = $1 AND tsv @@ plainto_tsquery('english', $2)
                 ORDER BY rank DESC
                 LIMIT $3",
            )
            .bind(&index)
            .bind(&query)
            .bind(limit as i64)
            .fetch_all(&pool)
            .await?;

            if rows.is_empty() {
                println!("No results.");
            } else {
                println!("{:<38} {:>8}", "DOC ID", "SCORE");
                println!("{}", "-".repeat(48));
                for (id, score) in rows {
                    println!("{:<38} {:>8.4}", id, score);
                }
            }
        }

        Commands::Drop { index, yes } => {
            if !yes {
                print!("Drop index '{index}'? This cannot be undone. [y/N] ");
                use std::io::BufRead;
                let mut line = String::new();
                std::io::stdin().lock().read_line(&mut line)?;
                if !line.trim().eq_ignore_ascii_case("y") {
                    println!("Aborted.");
                    return Ok(());
                }
            }
            sqlx::query("DELETE FROM nexus_indexes WHERE name = $1")
                .bind(&index)
                .execute(&pool)
                .await?;
            println!("Index '{index}' dropped.");
        }

        Commands::Reindex { index } => {
            println!("Reindex for '{index}' — not yet implemented.");
            println!("Coming in v0.2: stream all documents and re-build posting lists.");
        }

        Commands::Errors { index, limit: _ } => {
            println!("Error log for '{index}' — not yet implemented.");
            println!("Coming in v0.2: a nexus_errors table captures indexing failures.");
        }
    }

    Ok(())
}
