pub mod checkpoint;
pub mod cost_adaptive;
pub mod engine;
pub mod error;
pub mod gpu_backend;
pub mod kv_cache;
pub mod model;
pub mod modern_layers;
pub mod pipeline;
pub mod prefetch;
pub mod prometheus_exporter;
pub mod sampler;
pub mod streaming;
pub mod telemetry;
pub mod tenant;
pub mod tierset_selection;
pub mod tokenizer;
pub mod types;

pub use checkpoint::{load_checkpoint, save_checkpoint, ModelCheckpoint, TokenizerCheckpoint};
pub use cost_adaptive::{
    select_cost_adaptive, CostAdaptiveChoice, CostAdaptiveResult, CostVector, CostWeights,
    ExpertCostScore,
};
pub use engine::{
    create_inference_engine, create_inference_engine_default, InferenceConfig, InferenceEngine,
    InferenceResponse, InferenceSession,
};
pub use error::{InferenceError, InferenceResult};
pub use gpu_backend::{cuda_info, CudaDeviceInfo, CudaInfo, CudaDeviceManager, Device, Tensor};
pub use kv_cache::{GpuKvCache, KvCache, KvCacheConfig, KvEntry, KvTier};
#[cfg(feature = "cuda")]
pub use kv_cache::GpuKvEntry;
#[cfg(not(feature = "cuda"))]
pub use kv_cache::StubGpuKvEntry as GpuKvEntry;
pub use model::{
    create_model, Attention, FeedForward, LMHead, Linear, Model, ModelWeights, TransformerBlock,
};
pub use modern_layers::{
    swiglu, GroupedQueryConfig, ModernFeedForward, RmsNorm, RotaryEmbedding,
};
pub use pipeline::{
    DeterministicInferencePipeline, ExpertScore, InferencePipeline, PackedSegment, PackedToken,
    PipelineInput, PipelineOutput, PipelineStage,
};
pub use prefetch::{
    build_prefetch_plan, PrefetchAction, PrefetchCandidate, PrefetchPlan, PrefetchPolicy,
    PrefetchRequest,
};
pub use prometheus_exporter::{
    render_metrics, Counter, Gauge, Histogram, MetricsRegistry,
    map_telemetry_to_prometheus,
};
pub use sampler::{GenerateOptions, Generator, Sampler, SamplingMethod};
pub use streaming::{StreamStatus, StreamingRequest, StreamingRuntime, StreamingSession};
pub use telemetry::{
    InMemoryTelemetry, MetricKind, TelemetryCollector, TelemetryEvent, TelemetrySummary,
};
pub use tenant::{AdmissionDecision, TenantIsolationEngine, TenantQuota, TenantUsage};
pub use tierset_selection::{
    BudgetSpec, FixedModeTierSets, SelectionMode, TierProfile, TierSet, TierSetSelectionRequest,
    TierSetSelectionResult, TierSetSelector,
};
pub use tokenizer::{create_tokenizer, Tokenizer};
pub use types::{
    checksum_f32, fnv64_hex, seeded_hash_u64, ExpertKey, RoutingAssignment, SessionId, TenantId,
    TierId,
};
