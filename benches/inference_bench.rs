//! Benchmark suite for the lite-llm inference runtime.
//!
//! Benchmarks cover:
//! - Token generation throughput (tokens/sec)
//! - Routing decision latency (microseconds)
//! - Matrix multiplication (GFLOPS for CPU and GPU paths)
//! - KV-cache append/lookup performance
//! - Model forward pass for different model sizes
//!
//! Run with: `cargo bench`

use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use lite_llm_inference::{
    InferenceConfig, InferenceEngine, KvCache, KvCacheConfig, Model,
};
use lite_llm_runtime::{
    DeterministicRouter, Router, RoutingConfig, RoutingSeed, TierConfig, TierId, TierSet,
};
use lite_llm_runtime::Placement;

// ---------------------------------------------------------------------------
// Token Generation Throughput
// ---------------------------------------------------------------------------

fn bench_token_generation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_generation");

    let config = InferenceConfig {
        model_size: "small".to_string(),
        max_length: 10,
        temperature: 0.7,
        top_k: 10,
        top_p: 0.9,
        seed: Some(42),
    };
    let engine = InferenceEngine::new(config);

    group.bench_function("generate_10_tokens", |b| {
        b.iter(|| {
            engine
                .generate_with_max_len(black_box("hello world"), black_box(10))
                .unwrap()
        })
    });

    // Measure tokens/sec for a longer generation
    group.bench_function("generate_50_tokens", |b| {
        b.iter(|| {
            engine
                .generate_with_max_len(black_box("the quick brown fox"), black_box(50))
                .unwrap()
        })
    });

    // Custom throughput measurement (tokens/sec)
    group.bench_function("throughput_tokens_per_sec", |b| {
        b.iter_custom(|iters| {
            let mut total_tokens = 0u64;
            let start = Instant::now();
            for _ in 0..iters {
                let result = engine
                    .generate_with_max_len(black_box("benchmark test"), black_box(20))
                    .unwrap();
                total_tokens += result.chars().count() as u64;
            }
            let elapsed = start.elapsed();
            let tokens_per_sec = total_tokens as f64 / elapsed.as_secs_f64();
            eprintln!(
                "  Token throughput: {:.1} tokens/sec",
                tokens_per_sec
            );
            elapsed
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Routing Decision Latency
// ---------------------------------------------------------------------------

fn bench_routing_decision_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing");

    let tiers = vec![
        TierConfig {
            id: TierId(1),
            groups: 4,
            experts_per_group: 4,
            placement: Placement::Hot,
        },
        TierConfig {
            id: TierId(10),
            groups: 4,
            experts_per_group: 4,
            placement: Placement::Warm,
        },
        TierConfig {
            id: TierId(100),
            groups: 2,
            experts_per_group: 2,
            placement: Placement::Cold,
        },
    ];

    let router = DeterministicRouter::new(RoutingSeed::new(42), tiers.clone());
    let tier_set = TierSet {
        tiers: vec![TierId(1), TierId(10)],
        cumulative: false,
    };
    let cfg = RoutingConfig {
        k_tier: 1,
        k_group: 2,
        k_expert: 2,
    };
    let token_state: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();

    group.bench_function("routing_single_decision", |b| {
        b.iter(|| {
            router
                .route(
                    black_box(&token_state),
                    black_box(0),
                    black_box(0),
                    black_box(&tier_set),
                    black_box(cfg),
                )
                .unwrap()
        })
    });

    // Measure in microseconds
    group.bench_function("routing_latency_microseconds", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for i in 0..iters {
                let _ = router
                    .route(
                        black_box(&token_state),
                        black_box(0),
                        black_box(i as u32),
                        black_box(&tier_set),
                        black_box(cfg),
                    )
                    .unwrap();
            }
            let elapsed = start.elapsed();
            let us_per_decision = elapsed.as_micros() as f64 / iters as f64;
            eprintln!(
                "  Routing latency: {:.2} µs/decision",
                us_per_decision
            );
            elapsed
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Matrix Multiplication (CPU path)
// ---------------------------------------------------------------------------

fn bench_matmul_cpu(c: &mut Criterion) {
    use lite_llm_inference::Tensor;

    let mut group = c.benchmark_group("matmul_cpu");

    // Small matrix: 64x64 @ 64x64
    let a_64 = Tensor::from_data(
        (0..64 * 64).map(|i| (i as f32) * 0.01).collect(),
        &[64, 64],
    );
    let b_64 = Tensor::from_data(
        (0..64 * 64).map(|i| (i as f32) * 0.009).collect(),
        &[64, 64],
    );

    group.bench_function("64x64", |b| {
        b.iter(|| black_box(&a_64).matmul(black_box(&b_64)))
    });

    // Medium matrix: 128x128 @ 128x128
    let a_128 = Tensor::from_data(
        (0..128 * 128).map(|i| (i as f32) * 0.01).collect(),
        &[128, 128],
    );
    let b_128 = Tensor::from_data(
        (0..128 * 128).map(|i| (i as f32) * 0.009).collect(),
        &[128, 128],
    );

    group.bench_function("128x128", |b| {
        b.iter(|| black_box(&a_128).matmul(black_box(&b_128)))
    });

    // Large matrix: 256x256 @ 256x256
    let a_256 = Tensor::from_data(
        (0..256 * 256).map(|i| (i as f32) * 0.01).collect(),
        &[256, 256],
    );
    let b_256 = Tensor::from_data(
        (0..256 * 256).map(|i| (i as f32) * 0.009).collect(),
        &[256, 256],
    );

    group.bench_function("256x256", |b| {
        b.iter(|| black_box(&a_256).matmul(black_box(&b_256)))
    });

    // Custom GFLOPS measurement
    group.bench_function("gflops_256x256", |b| {
        b.iter_custom(|iters| {
            let ops_per_iter = 2.0 * 256.0 * 256.0 * 256.0; // 2*M*N*K for matmul
            let start = Instant::now();
            for _ in 0..iters {
                let _ = black_box(&a_256).matmul(black_box(&b_256));
            }
            let elapsed = start.elapsed();
            let gflops = (ops_per_iter * iters as f64) / elapsed.as_secs_f64() / 1e9;
            eprintln!("  Matmul GFLOPS (256x256): {:.2}", gflops);
            elapsed
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// KV-Cache Append/Lookup
// ---------------------------------------------------------------------------

fn bench_kv_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");

    let config = KvCacheConfig {
        hot_token_limit: 128,
        warm_token_limit: 128,
        total_token_limit: 512,
        sliding_window_tokens: None,
    };

    group.bench_function("append_single_entry", |b| {
        let mut cache = KvCache::new(config.clone()).unwrap();
        let key = vec![0.5; 64];
        let value = vec![0.5; 64];

        b.iter(|| {
            cache
                .append(
                    black_box(1),
                    black_box(10),
                    black_box(0u16),
                    black_box(0u16),
                    black_box(0u64),
                    black_box(&key),
                    black_box(&value),
                )
                .unwrap()
        })
    });

    // Append many entries
    group.bench_function("append_100_entries", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut cache = KvCache::new(config.clone()).unwrap();
                let key = vec![0.5; 64];
                let value = vec![0.5; 64];
                for pos in 0..100 {
                    cache
                        .append(1, 10, 0, 0, pos as u64, &key, &value)
                        .unwrap();
                }
            }
            let elapsed = start.elapsed();
            eprintln!(
                "  Append 100 entries: {:.2} µs/iter",
                elapsed.as_micros() as f64 / iters as f64
            );
            elapsed
        })
    });

    // Slice/lookup benchmark
    group.bench_function("slice_50_entries", |b| {
        b.iter_custom(|iters| {
            let mut cache = KvCache::new(config.clone()).unwrap();
            let key = vec![0.5; 64];
            let value = vec![0.5; 64];
            for pos in 0..100 {
                cache.append(1, 10, 0, 0, pos as u64, &key, &value).unwrap();
            }

            let start = Instant::now();
            for _ in 0..iters {
                let _ = cache.slice(1, 10, 0, 0, 0, 50).unwrap();
            }
            let elapsed = start.elapsed();
            eprintln!(
                "  Slice 50 entries: {:.2} µs/iter",
                elapsed.as_micros() as f64 / iters as f64
            );
            elapsed
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Model Forward Pass
// ---------------------------------------------------------------------------

fn bench_model_forward_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_forward");

    // Small model: vocab=512, hidden=64, layers=2, heads=2
    let small_model = Model::new(512, 64, 2, 2);
    let small_input: Vec<u32> = (0..8).collect();

    group.bench_function("small_model_8_tokens", |b| {
        b.iter(|| small_model.forward(black_box(&small_input)))
    });

    // Medium model: vocab=1024, hidden=128, layers=4, heads=4
    let medium_model = Model::new(1024, 128, 4, 4);
    let medium_input: Vec<u32> = (0..16).collect();

    group.bench_function("medium_model_16_tokens", |b| {
        b.iter(|| medium_model.forward(black_box(&medium_input)))
    });

    // Large model: vocab=2048, hidden=256, layers=6, heads=8
    let large_model = Model::new(2048, 256, 6, 8);
    let large_input: Vec<u32> = (0..32).collect();

    group.bench_function("large_model_32_tokens", |b| {
        b.iter(|| large_model.forward(black_box(&large_input)))
    });

    // Parameterized benchmark across input lengths
    for input_len in [4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("small_model_input_len", input_len),
            &input_len,
            |b, &len| {
                let input: Vec<u32> = (0..len as u32).collect();
                b.iter(|| small_model.forward(black_box(&input)))
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion group and main
// ---------------------------------------------------------------------------

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
        .warm_up_time(Duration::from_secs(1));
    targets =
        bench_token_generation_throughput,
        bench_routing_decision_latency,
        bench_matmul_cpu,
        bench_kv_cache_operations,
        bench_model_forward_pass,
);

criterion_main!(benches);
