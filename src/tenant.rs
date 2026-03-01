use std::collections::{BTreeMap, BTreeSet};

use crate::error::{InferenceError, InferenceResult};
use crate::types::{TenantId, TierId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantQuota {
    pub max_hot_bytes: u64,
    pub max_bandwidth_bytes_per_step: u64,
    pub max_concurrent_tokens: u32,
    pub weight: u32,
    pub allowed_tiers: BTreeSet<TierId>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TenantUsage {
    pub hot_bytes: u64,
    pub bandwidth_bytes_this_step: u64,
    pub concurrent_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionDecision {
    Allowed,
    Throttled(&'static str),
}

#[derive(Debug, Clone, Default)]
pub struct TenantIsolationEngine {
    quotas: BTreeMap<TenantId, TenantQuota>,
    usage: BTreeMap<TenantId, TenantUsage>,
}

impl TenantIsolationEngine {
    pub fn register_tenant(
        &mut self,
        tenant_id: TenantId,
        quota: TenantQuota,
    ) -> InferenceResult<()> {
        if quota.max_hot_bytes == 0
            || quota.max_bandwidth_bytes_per_step == 0
            || quota.max_concurrent_tokens == 0
            || quota.weight == 0
        {
            return Err(InferenceError::InvalidConfig(
                "tenant quota values must be greater than zero",
            ));
        }
        if quota.allowed_tiers.is_empty() {
            return Err(InferenceError::InvalidConfig(
                "tenant must be allowed at least one tier",
            ));
        }

        self.quotas.insert(tenant_id, quota);
        self.usage.entry(tenant_id).or_default();
        Ok(())
    }

    pub fn authorize_tier(&self, tenant_id: TenantId, tier: TierId) -> InferenceResult<bool> {
        let quota = self
            .quotas
            .get(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant".to_owned()))?;
        Ok(quota.allowed_tiers.contains(&tier))
    }

    pub fn try_admit_tokens(
        &mut self,
        tenant_id: TenantId,
        new_tokens: u32,
    ) -> InferenceResult<AdmissionDecision> {
        let quota = self
            .quotas
            .get(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant".to_owned()))?;
        let usage = self
            .usage
            .get_mut(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant usage".to_owned()))?;

        let projected = usage.concurrent_tokens.saturating_add(new_tokens);
        if projected > quota.max_concurrent_tokens {
            return Ok(AdmissionDecision::Throttled(
                "concurrent token quota exceeded",
            ));
        }

        usage.concurrent_tokens = projected;
        Ok(AdmissionDecision::Allowed)
    }

    pub fn release_tokens(
        &mut self,
        tenant_id: TenantId,
        completed_tokens: u32,
    ) -> InferenceResult<()> {
        let usage = self
            .usage
            .get_mut(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant usage".to_owned()))?;
        usage.concurrent_tokens = usage.concurrent_tokens.saturating_sub(completed_tokens);
        Ok(())
    }

    pub fn record_hot_bytes(
        &mut self,
        tenant_id: TenantId,
        bytes: u64,
    ) -> InferenceResult<AdmissionDecision> {
        let quota = self
            .quotas
            .get(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant".to_owned()))?;
        let usage = self
            .usage
            .get_mut(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant usage".to_owned()))?;

        usage.hot_bytes = bytes;
        if usage.hot_bytes > quota.max_hot_bytes {
            Ok(AdmissionDecision::Throttled("hot memory quota exceeded"))
        } else {
            Ok(AdmissionDecision::Allowed)
        }
    }

    pub fn record_bandwidth(
        &mut self,
        tenant_id: TenantId,
        bytes: u64,
    ) -> InferenceResult<AdmissionDecision> {
        let quota = self
            .quotas
            .get(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant".to_owned()))?;
        let usage = self
            .usage
            .get_mut(&tenant_id)
            .ok_or_else(|| InferenceError::TenantViolation("unknown tenant usage".to_owned()))?;

        usage.bandwidth_bytes_this_step = usage.bandwidth_bytes_this_step.saturating_add(bytes);
        if usage.bandwidth_bytes_this_step > quota.max_bandwidth_bytes_per_step {
            Ok(AdmissionDecision::Throttled("bandwidth quota exceeded"))
        } else {
            Ok(AdmissionDecision::Allowed)
        }
    }

    pub fn reset_step_bandwidth(&mut self) {
        for usage in self.usage.values_mut() {
            usage.bandwidth_bytes_this_step = 0;
        }
    }

    pub fn usage(&self, tenant_id: TenantId) -> Option<TenantUsage> {
        self.usage.get(&tenant_id).copied()
    }

    pub fn weighted_fair_schedule(
        &self,
        requested_tokens: &BTreeMap<TenantId, u32>,
    ) -> InferenceResult<Vec<TenantId>> {
        let mut remaining = BTreeMap::<TenantId, u32>::new();
        for (tenant, tokens) in requested_tokens {
            if *tokens == 0 {
                continue;
            }
            if !self.quotas.contains_key(tenant) {
                return Err(InferenceError::TenantViolation(format!(
                    "unknown tenant in schedule request: {tenant}"
                )));
            }
            remaining.insert(*tenant, *tokens);
        }

        let mut schedule = Vec::new();

        loop {
            let mut progressed = false;
            for (tenant_id, quota) in &self.quotas {
                let Some(left) = remaining.get_mut(tenant_id) else {
                    continue;
                };

                if *left == 0 {
                    continue;
                }

                let grant = (*left).min(quota.weight);
                for _ in 0..grant {
                    schedule.push(*tenant_id);
                }
                *left -= grant;
                progressed = progressed || grant > 0;
            }

            if !progressed {
                break;
            }

            remaining.retain(|_, left| *left > 0);
            if remaining.is_empty() {
                break;
            }
        }

        Ok(schedule)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::{AdmissionDecision, TenantIsolationEngine, TenantQuota};

    fn engine() -> TenantIsolationEngine {
        let mut engine = TenantIsolationEngine::default();
        engine
            .register_tenant(
                1,
                TenantQuota {
                    max_hot_bytes: 100,
                    max_bandwidth_bytes_per_step: 100,
                    max_concurrent_tokens: 4,
                    weight: 3,
                    allowed_tiers: BTreeSet::from([1, 2]),
                },
            )
            .expect("tenant should register");
        engine
            .register_tenant(
                2,
                TenantQuota {
                    max_hot_bytes: 100,
                    max_bandwidth_bytes_per_step: 100,
                    max_concurrent_tokens: 4,
                    weight: 1,
                    allowed_tiers: BTreeSet::from([1]),
                },
            )
            .expect("tenant should register");
        engine
    }

    #[test]
    fn tier_authorization_enforces_isolation() {
        let engine = engine();
        assert_eq!(engine.authorize_tier(1, 2).expect("auth should work"), true);
        assert_eq!(
            engine.authorize_tier(2, 2).expect("auth should work"),
            false
        );
    }

    #[test]
    fn admission_rejects_quota_excess() {
        let mut engine = engine();

        assert_eq!(
            engine
                .try_admit_tokens(1, 2)
                .expect("admission should work"),
            AdmissionDecision::Allowed
        );
        assert_eq!(
            engine
                .try_admit_tokens(1, 3)
                .expect("admission should work"),
            AdmissionDecision::Throttled("concurrent token quota exceeded")
        );
    }

    #[test]
    fn weighted_scheduler_is_fair_and_deterministic() {
        let engine = engine();
        let requests = BTreeMap::from([(1_u64, 6_u32), (2_u64, 6_u32)]);

        let schedule = engine
            .weighted_fair_schedule(&requests)
            .expect("schedule should compute");

        let count_t1 = schedule.iter().filter(|tenant| **tenant == 1).count();
        let count_t2 = schedule.iter().filter(|tenant| **tenant == 2).count();

        assert!(count_t1 >= count_t2);
        assert_eq!(count_t1 + count_t2, 12);

        let schedule_again = engine
            .weighted_fair_schedule(&requests)
            .expect("schedule should compute");
        assert_eq!(schedule, schedule_again);
    }

    #[test]
    fn tenant_usage_state_is_isolated() {
        let mut engine = engine();
        let _ = engine.record_hot_bytes(1, 90).expect("record should work");
        let _ = engine.record_hot_bytes(2, 10).expect("record should work");

        assert_ne!(
            engine.usage(1).expect("usage should exist"),
            engine.usage(2).expect("usage should exist")
        );
    }
}
