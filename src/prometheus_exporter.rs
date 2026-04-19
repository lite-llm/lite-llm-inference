//! Prometheus-compatible metrics exporter for the lite-llm inference runtime.
//!
//! Provides `Counter`, `Gauge`, and `Histogram` metric types with a
//! `render_metrics()` function that produces Prometheus text exposition format
//! suitable for scraping by Prometheus or compatible monitoring systems.
//!
//! # Metric Types
//!
//! - **Counter**: Monotonically increasing value (e.g., total tokens processed)
//! - **Gauge**: Value that can go up or down (e.g., cache entries, cache hit ratio)
//! - **Histogram**: Bucketed distribution of values (e.g., latency seconds)
//!
//! # Exported Metrics
//!
//! | Metric Name                        | Type      | Description                              |
//! |------------------------------------|-----------|------------------------------------------|
//! | `lite_llm_inference_tokens_total`  | Counter   | Total number of tokens processed         |
//! | `lite_llm_inference_latency_seconds`| Histogram | Inference latency distribution           |
//! | `lite_llm_kv_cache_entries`        | Gauge     | Current KV cache entry count             |
//! | `lite_llm_routing_decisions_total` | Counter   | Total routing decisions made             |
//! | `lite_llm_cache_hit_ratio`        | Gauge     | KV cache hit ratio (0.0 - 1.0)           |

use std::collections::BTreeMap;
use std::fmt::Write;

// ---------------------------------------------------------------------------
// Counter
// ---------------------------------------------------------------------------

/// A monotonically increasing counter metric.
///
/// Counters can only increase — they represent cumulative totals
/// such as request counts, token counts, or error counts.
#[derive(Debug, Clone, Default)]
pub struct Counter {
    value: f64,
}

impl Counter {
    /// Create a new counter initialized to zero.
    pub fn new() -> Self {
        Self { value: 0.0 }
    }

    /// Create a counter with an initial value.
    pub fn with_value(value: f64) -> Self {
        Self {
            value: value.max(0.0),
        }
    }

    /// Increment the counter by the given amount.
    ///
    /// Negative increments are clamped to zero (counters cannot decrease).
    pub fn inc_by(&mut self, amount: f64) {
        if amount > 0.0 {
            self.value += amount;
        }
    }

    /// Increment the counter by 1.
    pub fn inc(&mut self) {
        self.inc_by(1.0);
    }

    /// Get the current counter value.
    pub fn get(&self) -> f64 {
        self.value
    }
}

// ---------------------------------------------------------------------------
// Gauge
// ---------------------------------------------------------------------------

/// A gauge metric that can increase or decrease.
///
/// Gauges represent point-in-time values such as queue depth,
/// cache entry counts, or cache hit ratios.
#[derive(Debug, Clone, Default)]
pub struct Gauge {
    value: f64,
}

impl Gauge {
    /// Create a new gauge initialized to zero.
    pub fn new() -> Self {
        Self { value: 0.0 }
    }

    /// Create a gauge with an initial value.
    pub fn with_value(value: f64) -> Self {
        Self { value }
    }

    /// Set the gauge to the given value.
    pub fn set(&mut self, value: f64) {
        self.value = value;
    }

    /// Increment the gauge by the given amount.
    pub fn inc_by(&mut self, amount: f64) {
        self.value += amount;
    }

    /// Decrement the gauge by the given amount.
    pub fn dec_by(&mut self, amount: f64) {
        self.value -= amount;
    }

    /// Get the current gauge value.
    pub fn get(&self) -> f64 {
        self.value
    }
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// Default histogram buckets for latency measurement (in seconds).
pub const DEFAULT_LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0,
];

/// A histogram metric that tracks value distributions using configurable buckets.
///
/// Histograms are ideal for latency measurements, providing both a count of
/// observations and a bucketed distribution for percentile calculations.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bucket boundaries (upper bounds, exclusive of +Inf).
    buckets: Vec<f64>,
    /// Count of observations that fall into each bucket.
    bucket_counts: Vec<u64>,
    /// Total number of observations.
    count: u64,
    /// Sum of all observed values.
    sum: f64,
}

impl Histogram {
    /// Create a new histogram with the given bucket boundaries.
    pub fn with_buckets(buckets: Vec<f64>) -> Self {
        let bucket_counts = vec![0u64; buckets.len()];
        Self {
            buckets,
            bucket_counts,
            count: 0,
            sum: 0.0,
        }
    }

    /// Create a histogram with default latency buckets.
    pub fn new_latency_seconds() -> Self {
        Self::with_buckets(DEFAULT_LATENCY_BUCKETS.to_vec())
    }

    /// Observe a new value, updating all applicable bucket counts.
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;

        for (i, &bucket_bound) in self.buckets.iter().enumerate() {
            if value <= bucket_bound {
                self.bucket_counts[i] += 1;
            }
        }
    }

    /// Get the total number of observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get the sum of all observed values.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get the bucket boundaries.
    pub fn buckets(&self) -> &[f64] {
        &self.buckets
    }

    /// Get the cumulative count for each bucket.
    pub fn bucket_counts(&self) -> &[u64] {
        &self.bucket_counts
    }

    /// Estimate the percentile value from the observed data.
    ///
    /// Returns `None` if no observations have been made.
    pub fn estimate_percentile(&self, percentile: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }

        let target_rank = (percentile / 100.0) * self.count as f64;
        let mut cumulative = 0u64;

        for (i, &bucket_bound) in self.buckets.iter().enumerate() {
            cumulative += self.bucket_counts[i];
            if cumulative as f64 >= target_rank {
                return Some(bucket_bound);
            }
        }

        // Value exceeds all buckets — return the last bucket boundary.
        self.buckets.last().copied()
    }
}

// ---------------------------------------------------------------------------
// MetricsRegistry
// ---------------------------------------------------------------------------

/// Internal registry holding all metric types for export.
#[derive(Debug, Clone, Default)]
pub struct MetricsRegistry {
    pub counters: BTreeMap<String, Counter>,
    pub gauges: BTreeMap<String, Gauge>,
    pub histograms: BTreeMap<String, Histogram>,
}

// ---------------------------------------------------------------------------
// PrometheusRenderer
// ---------------------------------------------------------------------------

/// Renders metrics in Prometheus text exposition format.
///
/// The output follows the format specified by Prometheus:
/// - `# HELP` lines describe the metric
/// - `# TYPE` lines declare the metric type
/// - Metric lines in `metric_name{labels} value` format
pub struct PrometheusRenderer;

impl PrometheusRenderer {
    /// Render the full metrics registry in Prometheus text exposition format.
    pub fn render(registry: &MetricsRegistry) -> String {
        let mut output = String::new();

        // Render counters
        for (name, counter) in &registry.counters {
            let prom_name = counter_prom_name(name);
            writeln!(output, "# HELP {} {}", prom_name, counter_help(name)).unwrap();
            writeln!(output, "# TYPE {} counter", prom_name).unwrap();
            writeln!(output, "{} {}", prom_name, format_f64(counter.get())).unwrap();
        }

        // Render gauges
        for (name, gauge) in &registry.gauges {
            let prom_name = gauge_prom_name(name);
            writeln!(output, "# HELP {} {}", prom_name, gauge_help(name)).unwrap();
            writeln!(output, "# TYPE {} gauge", prom_name).unwrap();
            writeln!(output, "{} {}", prom_name, format_f64(gauge.get())).unwrap();
        }

        // Render histograms
        for (name, histogram) in &registry.histograms {
            let prom_name = histogram_prom_name(name);
            writeln!(output, "# HELP {} {}", prom_name, histogram_help(name)).unwrap();
            writeln!(output, "# TYPE {} histogram", prom_name).unwrap();

            let mut cumulative = 0u64;
            for (i, &bucket_bound) in histogram.buckets().iter().enumerate() {
                cumulative += histogram.bucket_counts()[i];
                writeln!(
                    output,
                    "{}_bucket{{le=\"{}\"}} {}",
                    prom_name,
                    format_bucket_le(bucket_bound),
                    cumulative
                )
                .unwrap();
            }

            // +Inf bucket
            writeln!(
                output,
                "{}_bucket{{le=\"+Inf\"}} {}",
                prom_name, histogram.count()
            )
            .unwrap();

            writeln!(output, "{}_sum {}", prom_name, format_f64(histogram.sum())).unwrap();
            writeln!(output, "{}_count {}", prom_name, histogram.count()).unwrap();
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Helper functions for Prometheus format
// ---------------------------------------------------------------------------

/// Convert an internal metric name to a Prometheus-compatible name.
fn to_prometheus_name(name: &str) -> String {
    name.replace(' ', "_")
        .replace('-', "_")
        .replace('.', "_")
        .to_lowercase()
}

/// Build the full Prometheus name for a counter metric.
fn counter_prom_name(name: &str) -> String {
    format!("lite_llm_inference_{}", to_prometheus_name(name))
}

/// Build the full Prometheus name for a gauge metric.
fn gauge_prom_name(name: &str) -> String {
    format!("lite_llm_{}", to_prometheus_name(name))
}

/// Build the full Prometheus name for a histogram metric.
fn histogram_prom_name(name: &str) -> String {
    format!("lite_llm_inference_{}", to_prometheus_name(name))
}

/// Format an f64 for Prometheus output (avoid scientific notation for small values).
fn format_f64(value: f64) -> String {
    if value == value.floor() && value.abs() < 1e15 {
        format!("{:.1}", value)
    } else {
        format!("{}", value)
    }
}

/// Format a bucket boundary for the `le` label.
fn format_bucket_le(value: f64) -> String {
    if value == value.floor() && value.abs() < 1e10 {
        format!("{:.1}", value)
    } else {
        format!("{}", value)
    }
}

fn counter_help(name: &str) -> String {
    match name {
        "tokens_total" => "Total number of tokens processed".to_string(),
        "routing_decisions_total" => "Total number of routing decisions made".to_string(),
        _ => format!("Counter metric: {}", name),
    }
}

fn gauge_help(name: &str) -> String {
    match name {
        "kv_cache_entries" => "Current number of KV cache entries".to_string(),
        "cache_hit_ratio" => "KV cache hit ratio (0.0 to 1.0)".to_string(),
        _ => format!("Gauge metric: {}", name),
    }
}

fn histogram_help(name: &str) -> String {
    match name {
        "latency_seconds" => "Inference latency in seconds".to_string(),
        _ => format!("Histogram metric: {}", name),
    }
}

// ---------------------------------------------------------------------------
// Telemetry-to-Prometheus mapping
// ---------------------------------------------------------------------------

/// Maps telemetry events from the `InMemoryTelemetry` collector to Prometheus metrics.
///
/// This function consumes a summary of recorded events and populates the
/// `MetricsRegistry` with equivalent Prometheus-compatible metrics.
pub fn map_telemetry_to_prometheus(
    registry: &mut MetricsRegistry,
    total_tokens: u64,
    total_routing_decisions: u64,
    kv_cache_entry_count: u64,
    cache_hit_ratio: f64,
    latency_values_ms: &[f64],
) {
    // lite_llm_inference_tokens_total (counter)
    let tokens_counter = registry
        .counters
        .entry("tokens_total".to_owned())
        .or_insert_with(Counter::new);
    tokens_counter.inc_by(total_tokens as f64);

    // lite_llm_routing_decisions_total (counter)
    let routing_counter = registry
        .counters
        .entry("routing_decisions_total".to_owned())
        .or_insert_with(Counter::new);
    routing_counter.inc_by(total_routing_decisions as f64);

    // lite_llm_kv_cache_entries (gauge)
    let cache_entries_gauge = registry
        .gauges
        .entry("kv_cache_entries".to_owned())
        .or_insert_with(Gauge::new);
    cache_entries_gauge.set(kv_cache_entry_count as f64);

    // lite_llm_cache_hit_ratio (gauge)
    let hit_ratio_gauge = registry
        .gauges
        .entry("cache_hit_ratio".to_owned())
        .or_insert_with(Gauge::new);
    hit_ratio_gauge.set(cache_hit_ratio);

    // lite_llm_inference_latency_seconds (histogram)
    let latency_histogram = registry
        .histograms
        .entry("latency_seconds".to_owned())
        .or_insert_with(Histogram::new_latency_seconds);

    for &latency_ms in latency_values_ms {
        // Convert milliseconds to seconds for Prometheus standard
        latency_histogram.observe(latency_ms / 1000.0);
    }
}

// ---------------------------------------------------------------------------
// Public API: render_metrics()
// ---------------------------------------------------------------------------

/// Render all metrics in Prometheus text exposition format.
///
/// This is the primary entry point for the Prometheus metrics endpoint.
/// The returned string is ready to be served on `/metrics` with
/// `Content-Type: text/plain; version=0.0.4; charset=utf-8`.
///
/// # Example
///
/// ```rust
/// use lite_llm_inference::prometheus_exporter::{
///     MetricsRegistry, Counter, Gauge, Histogram, render_metrics,
/// };
///
/// let mut registry = MetricsRegistry::default();
///
/// // Add some metrics
/// let mut tokens = Counter::new();
/// tokens.inc_by(150.0);
/// registry.counters.insert("tokens_total".to_owned(), tokens);
///
/// let mut cache_entries = Gauge::new();
/// cache_entries.set(42.0);
/// registry.gauges.insert("kv_cache_entries".to_owned(), cache_entries);
///
/// let mut latency = Histogram::new_latency_seconds();
/// latency.observe(0.05);
/// registry.histograms.insert("latency_seconds".to_owned(), latency);
///
/// let output = render_metrics(&registry);
/// assert!(output.contains("lite_llm_inference_tokens_total 150.0"));
/// assert!(output.contains("lite_llm_kv_cache_entries 42.0"));
/// ```
pub fn render_metrics(registry: &MetricsRegistry) -> String {
    PrometheusRenderer::render(registry)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_starts_at_zero() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0.0);
    }

    #[test]
    fn counter_only_increases() {
        let mut counter = Counter::new();
        counter.inc_by(10.0);
        counter.inc_by(5.0);
        assert_eq!(counter.get(), 15.0);
    }

    #[test]
    fn counter_ignores_negative_increments() {
        let mut counter = Counter::new();
        counter.inc_by(10.0);
        counter.inc_by(-5.0);
        assert_eq!(counter.get(), 10.0);
    }

    #[test]
    fn counter_inc_by_one() {
        let mut counter = Counter::new();
        counter.inc();
        counter.inc();
        assert_eq!(counter.get(), 2.0);
    }

    #[test]
    fn gauge_can_increase_and_decrease() {
        let mut gauge = Gauge::new();
        gauge.set(100.0);
        gauge.inc_by(10.0);
        gauge.dec_by(25.0);
        assert_eq!(gauge.get(), 85.0);
    }

    #[test]
    fn histogram_tracks_bucket_counts() {
        let mut histogram = Histogram::with_buckets(vec![1.0, 5.0, 10.0]);

        histogram.observe(0.5);  // bucket 0 (<=1.0)
        histogram.observe(3.0);  // bucket 1 (<=5.0)
        histogram.observe(7.0);  // bucket 2 (<=10.0)
        histogram.observe(15.0); // exceeds all buckets

        assert_eq!(histogram.count(), 4);
        assert_eq!(histogram.sum(), 25.5);

        let counts = histogram.bucket_counts();
        assert_eq!(counts[0], 1); // <=1.0
        assert_eq!(counts[1], 2); // <=5.0 (0.5 + 3.0)
        assert_eq!(counts[2], 3); // <=10.0 (0.5 + 3.0 + 7.0)
    }

    #[test]
    fn histogram_percentile_estimation() {
        let mut histogram = Histogram::with_buckets(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        for i in 1..=10 {
            histogram.observe(i as f64);
        }

        let p50 = histogram.estimate_percentile(50.0);
        let p95 = histogram.estimate_percentile(95.0);

        assert!(p50.is_some());
        assert!(p95.is_some());
        assert!(p50.unwrap() <= p95.unwrap());
    }

    #[test]
    fn histogram_percentile_returns_none_when_empty() {
        let histogram = Histogram::with_buckets(vec![1.0, 5.0, 10.0]);
        assert!(histogram.estimate_percentile(50.0).is_none());
    }

    #[test]
    fn render_metrics_produces_counter_lines() {
        let mut registry = MetricsRegistry::default();
        let mut counter = Counter::new();
        counter.inc_by(42.0);
        registry.counters.insert("tokens_total".to_owned(), counter);

        let output = render_metrics(&registry);

        assert!(output.contains("# HELP lite_llm_inference_tokens_total"));
        assert!(output.contains("# TYPE lite_llm_inference_tokens_total counter"));
        assert!(output.contains("lite_llm_inference_tokens_total 42.0"));
    }

    #[test]
    fn render_metrics_produces_gauge_lines() {
        let mut registry = MetricsRegistry::default();
        let mut gauge = Gauge::new();
        gauge.set(100.0);
        registry
            .gauges
            .insert("kv_cache_entries".to_owned(), gauge);

        let output = render_metrics(&registry);

        assert!(output.contains("# HELP lite_llm_kv_cache_entries"));
        assert!(output.contains("# TYPE lite_llm_kv_cache_entries gauge"));
        assert!(output.contains("lite_llm_kv_cache_entries 100.0"));
    }

    #[test]
    fn render_metrics_produces_histogram_lines() {
        let mut registry = MetricsRegistry::default();
        let mut histogram = Histogram::with_buckets(vec![1.0, 5.0, 10.0]);
        histogram.observe(3.0);
        histogram.observe(7.0);
        registry
            .histograms
            .insert("latency_seconds".to_owned(), histogram);

        let output = render_metrics(&registry);

        assert!(output.contains("# HELP lite_llm_inference_latency_seconds"));
        assert!(output.contains("# TYPE lite_llm_inference_latency_seconds histogram"));
        assert!(output.contains("_bucket{le=\"1.0\"}"));
        assert!(output.contains("_bucket{le=\"5.0\"}"));
        assert!(output.contains("_bucket{le=\"10.0\"}"));
        assert!(output.contains("_bucket{le=\"+Inf\"} 2"));
        assert!(output.contains("_sum 10"));
        assert!(output.contains("_count 2"));
    }

    #[test]
    fn render_metrics_empty_registry_produces_empty_output() {
        let registry = MetricsRegistry::default();
        let output = render_metrics(&registry);
        assert!(output.is_empty());
    }

    #[test]
    fn render_metrics_multiple_counters() {
        let mut registry = MetricsRegistry::default();

        let mut tokens = Counter::new();
        tokens.inc_by(100.0);
        registry.counters.insert("tokens_total".to_owned(), tokens);

        let mut routing = Counter::new();
        routing.inc_by(50.0);
        registry
            .counters
            .insert("routing_decisions_total".to_owned(), routing);

        let output = render_metrics(&registry);

        assert!(output.contains("lite_llm_inference_tokens_total 100.0"));
        assert!(output.contains("lite_llm_inference_routing_decisions_total 50.0"));
    }

    #[test]
    fn render_metrics_cache_hit_ratio_gauge() {
        let mut registry = MetricsRegistry::default();
        let mut ratio = Gauge::new();
        ratio.set(0.85);
        registry
            .gauges
            .insert("cache_hit_ratio".to_owned(), ratio);

        let output = render_metrics(&registry);

        assert!(output.contains("# TYPE lite_llm_cache_hit_ratio gauge"));
        assert!(output.contains("lite_llm_cache_hit_ratio 0.85"));
    }

    #[test]
    fn map_telemetry_populates_all_metric_types() {
        let mut registry = MetricsRegistry::default();

        let latency_values = vec![10.0, 25.0, 50.0, 100.0, 150.0];
        map_telemetry_to_prometheus(
            &mut registry,
            500,    // total_tokens
            120,    // total_routing_decisions
            42,     // kv_cache_entry_count
            0.75,   // cache_hit_ratio
            &latency_values,
        );

        // Verify counters
        assert!(registry.counters.contains_key("tokens_total"));
        assert!(registry.counters.contains_key("routing_decisions_total"));
        assert_eq!(registry.counters["tokens_total"].get(), 500.0);
        assert_eq!(registry.counters["routing_decisions_total"].get(), 120.0);

        // Verify gauges
        assert!(registry.gauges.contains_key("kv_cache_entries"));
        assert!(registry.gauges.contains_key("cache_hit_ratio"));
        assert_eq!(registry.gauges["kv_cache_entries"].get(), 42.0);
        assert!((registry.gauges["cache_hit_ratio"].get() - 0.75).abs() < 0.001);

        // Verify histogram
        assert!(registry.histograms.contains_key("latency_seconds"));
        let hist = &registry.histograms["latency_seconds"];
        assert_eq!(hist.count(), 5);
        assert!((hist.sum() - 0.335).abs() < 0.001); // (10+25+50+100+150)/1000 = 0.335
    }

    #[test]
    fn render_full_prometheus_output_format() {
        let mut registry = MetricsRegistry::default();

        let mut tokens = Counter::new();
        tokens.inc_by(1000.0);
        registry.counters.insert("tokens_total".to_owned(), tokens);

        let mut routing = Counter::new();
        routing.inc_by(200.0);
        registry
            .counters
            .insert("routing_decisions_total".to_owned(), routing);

        let mut cache_entries = Gauge::new();
        cache_entries.set(256.0);
        registry
            .gauges
            .insert("kv_cache_entries".to_owned(), cache_entries);

        let mut hit_ratio = Gauge::new();
        hit_ratio.set(0.92);
        registry
            .gauges
            .insert("cache_hit_ratio".to_owned(), hit_ratio);

        let mut latency = Histogram::new_latency_seconds();
        latency.observe(0.025); // 25ms
        latency.observe(0.050); // 50ms
        latency.observe(0.100); // 100ms
        registry
            .histograms
            .insert("latency_seconds".to_owned(), latency);

        let output = render_metrics(&registry);

        // Verify all metric types are present with correct names
        assert!(output.contains("lite_llm_inference_tokens_total 1000.0"));
        assert!(output.contains("lite_llm_inference_routing_decisions_total 200.0"));
        assert!(output.contains("lite_llm_kv_cache_entries 256.0"));
        assert!(output.contains("lite_llm_cache_hit_ratio 0.92"));
        assert!(output.contains("lite_llm_inference_latency_seconds"));

        // Verify histogram bucket format
        assert!(output.contains("_bucket{le=\"0.001\"}"));
        assert!(output.contains("_bucket{le=\"0.025\"}"));
        assert!(output.contains("_bucket{le=\"+Inf\"} 3"));
        assert!(output.contains("_sum 0.175"));
        assert!(output.contains("_count 3"));
    }

    #[test]
    fn counter_with_value() {
        let counter = Counter::with_value(42.0);
        assert_eq!(counter.get(), 42.0);
    }

    #[test]
    fn counter_with_negative_value_clamps_to_zero() {
        let counter = Counter::with_value(-10.0);
        assert_eq!(counter.get(), 0.0);
    }

    #[test]
    fn gauge_with_value() {
        let gauge = Gauge::with_value(-5.0);
        assert_eq!(gauge.get(), -5.0);
    }

    #[test]
    fn histogram_default_latency_buckets() {
        let histogram = Histogram::new_latency_seconds();
        assert_eq!(histogram.buckets().len(), DEFAULT_LATENCY_BUCKETS.len());
        assert_eq!(histogram.buckets()[0], 0.001);
        assert_eq!(histogram.buckets()[histogram.buckets().len() - 1], 10.0);
    }
}
