pub mod cost_adaptive;
pub mod error;
pub mod kv_cache;
pub mod pipeline;
pub mod prefetch;
pub mod streaming;
pub mod telemetry;
pub mod tenant;
pub mod tierset_selection;
pub mod types;

pub use cost_adaptive::{
    select_cost_adaptive, CostAdaptiveChoice, CostAdaptiveResult, CostVector, CostWeights,
    ExpertCostScore,
};
pub use error::{InferenceError, InferenceResult};
pub use kv_cache::{KvCache, KvCacheConfig, KvEntry, KvTier};
pub use pipeline::{
    DeterministicInferencePipeline, ExpertScore, InferencePipeline, PackedSegment, PackedToken,
    PipelineInput, PipelineOutput, PipelineStage,
};
pub use prefetch::{
    build_prefetch_plan, PrefetchAction, PrefetchCandidate, PrefetchPlan, PrefetchPolicy,
    PrefetchRequest,
};
pub use streaming::{StreamStatus, StreamingRequest, StreamingRuntime, StreamingSession};
pub use telemetry::{
    InMemoryTelemetry, MetricKind, TelemetryCollector, TelemetryEvent, TelemetrySummary,
};
pub use tenant::{AdmissionDecision, TenantIsolationEngine, TenantQuota, TenantUsage};
pub use tierset_selection::{
    BudgetSpec, FixedModeTierSets, SelectionMode, TierProfile, TierSet, TierSetSelectionRequest,
    TierSetSelectionResult, TierSetSelector,
};
pub use types::{
    checksum_f32, fnv64_hex, seeded_hash_u64, ExpertKey, RoutingAssignment, SessionId, TenantId,
    TierId,
};
