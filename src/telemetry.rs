use std::collections::BTreeMap;

use crate::types::{SessionId, TenantId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetricKind {
    Latency,
    Resource,
    Routing,
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryEvent {
    pub trace_id: String,
    pub tenant_id: TenantId,
    pub session_id: SessionId,
    pub step: u64,
    pub kind: MetricKind,
    pub name: String,
    pub value: f64,
    pub tags: BTreeMap<String, String>,
}

pub trait TelemetryCollector {
    fn record(&mut self, event: TelemetryEvent);
}

#[derive(Debug, Clone, PartialEq)]
pub struct TelemetrySummary {
    pub total_events: usize,
    pub avg_by_name: BTreeMap<String, f64>,
    pub counts_by_tenant: BTreeMap<TenantId, usize>,
    pub p95_latency_ms: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryTelemetry {
    events: Vec<TelemetryEvent>,
}

impl TelemetryCollector for InMemoryTelemetry {
    fn record(&mut self, event: TelemetryEvent) {
        self.events.push(event);
    }
}

impl InMemoryTelemetry {
    pub fn record_sampled(&mut self, event: TelemetryEvent, sample_every_n: usize) {
        if sample_every_n <= 1 {
            self.events.push(event);
            return;
        }

        if (event.step as usize) % sample_every_n == 0 {
            self.events.push(event);
        }
    }

    pub fn events(&self) -> &[TelemetryEvent] {
        &self.events
    }

    pub fn summarize(&self) -> TelemetrySummary {
        let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
        let mut counts_by_tenant: BTreeMap<TenantId, usize> = BTreeMap::new();
        let mut latency_values = Vec::new();

        for event in &self.events {
            let entry = sums.entry(event.name.clone()).or_insert((0.0, 0));
            entry.0 += event.value;
            entry.1 += 1;

            *counts_by_tenant.entry(event.tenant_id).or_insert(0) += 1;

            if event.kind == MetricKind::Latency {
                latency_values.push(event.value);
            }
        }

        let avg_by_name = sums
            .into_iter()
            .map(|(name, (sum, count))| (name, if count == 0 { 0.0 } else { sum / count as f64 }))
            .collect::<BTreeMap<String, f64>>();

        latency_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95_latency_ms = if latency_values.is_empty() {
            None
        } else {
            let idx = ((latency_values.len() as f64 - 1.0) * 0.95).round() as usize;
            latency_values.get(idx).copied()
        };

        TelemetrySummary {
            total_events: self.events.len(),
            avg_by_name,
            counts_by_tenant,
            p95_latency_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{InMemoryTelemetry, MetricKind, TelemetryCollector, TelemetryEvent};

    fn event(step: u64, tenant: u64, name: &str, kind: MetricKind, value: f64) -> TelemetryEvent {
        TelemetryEvent {
            trace_id: format!("trace-{step}"),
            tenant_id: tenant,
            session_id: 10,
            step,
            kind,
            name: name.to_owned(),
            value,
            tags: BTreeMap::new(),
        }
    }

    #[test]
    fn sampled_recording_is_deterministic() {
        let mut telemetry = InMemoryTelemetry::default();
        for step in 0..10 {
            telemetry.record_sampled(
                event(step, 1, "token_latency", MetricKind::Latency, step as f64),
                3,
            );
        }

        let kept = telemetry
            .events()
            .iter()
            .map(|e| e.step)
            .collect::<Vec<u64>>();
        assert_eq!(kept, vec![0, 3, 6, 9]);
    }

    #[test]
    fn summary_aggregates_by_name_and_tenant() {
        let mut telemetry = InMemoryTelemetry::default();
        telemetry.record(event(0, 1, "token_latency", MetricKind::Latency, 10.0));
        telemetry.record(event(1, 1, "token_latency", MetricKind::Latency, 20.0));
        telemetry.record(event(2, 2, "cache_hit", MetricKind::Routing, 1.0));

        let summary = telemetry.summarize();
        assert_eq!(summary.total_events, 3);
        assert_eq!(summary.counts_by_tenant.get(&1), Some(&2));
        assert_eq!(summary.counts_by_tenant.get(&2), Some(&1));
        assert_eq!(
            summary.avg_by_name.get("token_latency").copied(),
            Some(15.0)
        );
        assert!(summary.p95_latency_ms.is_some());
    }

    #[test]
    fn telemetry_collector_trait_records_events() {
        let mut telemetry = InMemoryTelemetry::default();
        telemetry.record(event(0, 1, "dispatch_bytes", MetricKind::Resource, 1024.0));
        assert_eq!(telemetry.events().len(), 1);
    }
}
