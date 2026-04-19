# lite-llm-inference

Inference runtime crate for Lite LLM (`SPEC-041` to `SPEC-050`).

## Overview
Implements deterministic inference primitives including TierSet selection, token routing, KV-cache management, and GPU-accelerated execution with modern transformer architectures.

This crate provides the complete inference stack: TierSet selection engine with budget solver, deterministic token routing and expert packing/dispatch, prefetch planning, KV-cache behavior with GPU support, streaming session runtime with replayable prefixes, cost-adaptive routing, Prometheus-compatible telemetry export, multi-tenant isolation controls, CUDA GPU backend with cuBLAS-accelerated tensors, and modern transformer layers (RoPE, RMSNorm, SwiGLU, GQA).

## Features

### Feature Flag: `default` (empty)
No optional features enabled by default. All CPU-based inference is available.

### Feature Flag: `cuda` (optional)
Enables `cudarc` with cuBLAS for CUDA-accelerated tensor operations. Requires an NVIDIA GPU and CUDA toolkit. Also requires a CUDA version feature (e.g., `cudarc/cuda-12060`).

## Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| serde | 1.0 | Serialization for checkpoints and config |
| rand | 0.8 | Random sampling and seeding |
| log | 0.4 | Logging for GPU init and runtime |
| cudarc | 0.12 (optional) | CUDA bindings for GPU tensor ops |
| tokio | 1 | Async runtime for GPU kernel launches |

## Key Modules
- `tierset_selection` — TierSet selection engine, budget solver, fixed/balanced/deep/max modes
- `pipeline` — deterministic inference pipeline with expert dispatch
- `prefetch` — prefetch planning and candidate scoring
- `kv_cache` — KV-cache management with hot/warm tiers and GPU entries
- `streaming` — streaming session runtime with replayable prefixes
- `cost_adaptive` — cost-adaptive routing with weighted scoring
- `telemetry` — in-memory telemetry collection and summary
- `tenant` — multi-tenant isolation with quota enforcement
- `gpu_backend` — CUDA tensor backend with `Tensor`, `GpuDeviceManager`, `CublasHandle`
- `modern_layers` — RoPE, RMSNorm, SwiGLU, GQA, `ModernFeedForward`
- `prometheus_exporter` — Prometheus-compatible metrics export
- `tokenizer` — character-level tokenizer
- `model` — transformer model definition (embedding, attention, feed-forward)
- `sampler` — generation sampler (greedy, temperature, top-k, top-p)
- `engine` — inference engine creation and configuration
- `checkpoint` — model checkpoint save/load
- `types` — shared type contracts (`ExpertKey`, `TierId`, `SessionId`, `TenantId`)
- `error` — inference error model

## Public API
### Core Types
- `Tensor` — unified CPU/GPU tensor with automatic device placement
- `GpuDeviceManager` — singleton managing CUDA devices, cuBLAS handles, memory tracking
- `CudaDeviceInfo` / `CudaInfo` — GPU device discovery and reporting
- `RotaryEmbedding` — RoPE with pre-computed cos/sin cache
- `RmsNorm` — root mean square normalization
- `ModernFeedForward` — complete FFN block with RMSNorm + SwiGLU
- `TierSetSelector` — TierSet selection with budget constraints
- `InferenceEngine` — main inference engine interface
- `DeterministicInferencePipeline` — deterministic token routing pipeline
- `HotExpertCache` — expert caching for inference
- `GpuKvCache` / `KvCache` — KV-cache interfaces
- `StreamingRuntime` / `StreamingSession` — streaming inference
- `InMemoryTelemetry` — telemetry collector
- `TenantIsolationEngine` — multi-tenant quota enforcement
- `MetricsRegistry` — Prometheus metrics registry
- `Counter` / `Gauge` / `Histogram` — Prometheus metric types

### Core Functions
- `cuda_info()` — GPU device discovery and reporting
- `swiglu()` — Swish-gated feed-forward activation
- `create_inference_engine()` / `create_inference_engine_default()` — engine factory
- `create_tokenizer()` — tokenizer factory
- `build_prefetch_plan()` — prefetch plan generation
- `select_cost_adaptive()` — cost-adaptive TierSet selection
- `save_checkpoint()` / `load_checkpoint()` — model checkpoint I/O
- `render_metrics()` — render metrics in Prometheus text format
- `map_telemetry_to_prometheus()` — convert telemetry events to Prometheus metrics

### Traits
- `InferencePipeline` — inference pipeline interface

## Quick Start
```rust
use lite_llm_inference::{
    create_inference_engine_default, TierSetSelector,
    TierSetSelectionRequest, SelectionMode, BudgetSpec,
    FixedModeTierSets, TierSet as InferenceTierSet,
    TierProfile, KvCacheConfig, Generator, GenerateOptions,
};

// Create inference engine
let engine = create_inference_engine_default(InferenceConfig {
    top_k: 40,
    top_p: 0.9,
    temperature: 0.7,
});

// Configure TierSet selection
let selector = TierSetSelector::new(
    1.0,
    FixedModeTierSets {
        fast: InferenceTierSet::new(vec![1], false),
        balanced: InferenceTierSet::new(vec![1, 2], false),
        deep: InferenceTierSet::new(vec![1, 2, 3], false),
        max: InferenceTierSet::new(vec![1, 2, 3, 4], false),
    },
    vec![TierProfile {
        id: 1, label: "hot".into(),
        capacity_value: 100, latency_cost_ms: 1.0,
        monetary_cost_units: 0.1, available: true,
    }],
).expect("selector should init");

// Generate text
let mut generator = Generator::new(engine);
let result = generator.generate("Hello world", GenerateOptions {
    max_tokens: 100,
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
    seed: Some(42),
});
```

## Running Tests
```bash
cargo fmt
cargo test
```

## Architecture
This crate implements the inference layer for the lite-llm platform. The GPU backend (`gpu_backend` module) provides CUDA-accelerated tensor operations via cudarc and cuBLAS. Modern transformer layers (`modern_layers` module) implement RoPE, RMSNorm, SwiGLU, and GQA — matching 2024-2026 LLM architecture designs. The Prometheus exporter (`prometheus_exporter` module) provides monitoring integration. It integrates with `lite-llm-training` for model evaluation during training and with `lite-llm` orchestrator for inference mode entrypoints.

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
