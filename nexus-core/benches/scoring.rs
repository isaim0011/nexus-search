use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nexus_core::{
    score::{Bm25Scorer, Rrf, edit_distance},
    tokenizer::{Tokenizer, TokenizerConfig},
    vector::{Embedding, FlatVectorIndex},
};
use uuid::Uuid;

// ── BM25 scoring ────────────────────────────────────────────────────────────

fn bench_bm25_score(c: &mut Criterion) {
    let scorer = Bm25Scorer {
        avg_doc_len: 150.0,
        doc_count: 100_000,
        ..Default::default()
    };
    c.bench_function("bm25_score_single", |b| {
        b.iter(|| black_box(scorer.score(black_box(3), black_box(500), black_box(120), black_box(1.0))))
    });
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

fn bench_tokenizer(c: &mut Criterion) {
    let tokenizer = Tokenizer::new(TokenizerConfig::default());
    let short = "Rust async runtime powered by Tokio";
    let long = "Rust is a systems programming language focused on three goals: \
                safety, speed, and concurrency. It accomplishes these goals \
                without a garbage collector, making it useful for a number of \
                use cases other languages aren't good at: embedding in other \
                languages, programs with specific space and time requirements, \
                and writing low-level code, like device drivers and operating systems.";

    let mut g = c.benchmark_group("tokenizer");
    g.bench_with_input(BenchmarkId::new("short", short.len()), &short, |b, text| {
        b.iter(|| black_box(tokenizer.tokenize(black_box(text)).unwrap()))
    });
    g.bench_with_input(BenchmarkId::new("long", long.len()), &long, |b, text| {
        b.iter(|| black_box(tokenizer.tokenize(black_box(text)).unwrap()))
    });
    g.finish();
}

// ── Edit distance / typo tolerance ──────────────────────────────────────────

fn bench_edit_distance(c: &mut Criterion) {
    let pairs = [
        ("programming", "progrming", 2),
        ("language", "languaje", 1),
        ("concurrent", "concurrent", 0),
    ];
    let mut g = c.benchmark_group("edit_distance");
    for (a, b, budget) in pairs {
        g.bench_with_input(BenchmarkId::new(a, b), &(a, b, budget), |bench, &(a, b, budget)| {
            bench.iter(|| black_box(edit_distance(black_box(a), black_box(b), black_box(budget))))
        });
    }
    g.finish();
}

// ── RRF fusion ───────────────────────────────────────────────────────────────

fn bench_rrf_fuse(c: &mut Criterion) {
    let n = 1000usize;
    let bm25: Vec<(Uuid, f32)> = (0..n).map(|i| (Uuid::new_v4(), 1.0 / (i + 1) as f32)).collect();
    let vec_hits: Vec<(Uuid, f32)> = (0..n).map(|i| (Uuid::new_v4(), 1.0 / (i + 1) as f32)).collect();
    let rrf = Rrf::default();

    let mut g = c.benchmark_group("rrf_fuse");
    g.throughput(Throughput::Elements(n as u64));
    g.bench_function("1k_per_list", |b| {
        b.iter(|| black_box(rrf.fuse(black_box(&bm25), black_box(&vec_hits), black_box(0.5))))
    });
    g.finish();
}

// ── Flat vector index (100k docs) ────────────────────────────────────────────

fn bench_flat_vector_index(c: &mut Criterion) {
    const N: usize = 10_000; // scaled down from 100k so bench completes quickly
    const DIM: usize = 128;

    let mut idx = FlatVectorIndex::default();
    for _ in 0..N {
        let raw: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.01).collect();
        idx.insert(Uuid::new_v4(), Embedding::new(raw));
    }
    let query = Embedding::new(vec![0.5f32; DIM]);

    let mut g = c.benchmark_group("flat_vector_index");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function(format!("search_{N}_docs_dim{DIM}"), |b| {
        b.iter(|| black_box(idx.search(black_box(&query), black_box(10))))
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_bm25_score,
    bench_tokenizer,
    bench_edit_distance,
    bench_rrf_fuse,
    bench_flat_vector_index
);
criterion_main!(benches);
