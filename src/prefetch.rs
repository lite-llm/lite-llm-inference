use std::collections::BTreeSet;

use crate::error::{InferenceError, InferenceResult};
use crate::types::{seeded_hash_u64, ExpertKey, TierId};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefetchPolicy {
    pub lookahead_tokens: u32,
    pub max_bytes_inflight: u64,
    pub aggressive_latency_budget_ms: f32,
    pub deterministic_seed: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrefetchCandidate {
    pub expert: ExpertKey,
    pub predicted_probability: f32,
    pub bytes: u64,
    pub tier_latency_ms: f32,
    pub resident_hot: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefetchRequest {
    pub allowed_tiers: Vec<TierId>,
    pub latency_budget_ms: f32,
    pub candidates: Vec<PrefetchCandidate>,
    pub cancel: Vec<ExpertKey>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrefetchAction {
    Fetch {
        expert: ExpertKey,
        priority: f32,
        bytes: u64,
    },
    Cancel {
        expert: ExpertKey,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrefetchPlan {
    pub actions: Vec<PrefetchAction>,
    pub scheduled_bytes: u64,
    pub expected_hit_gain: f32,
    pub wasted_bytes_upper_bound: u64,
}

pub fn build_prefetch_plan(
    request: &PrefetchRequest,
    policy: PrefetchPolicy,
) -> InferenceResult<PrefetchPlan> {
    if policy.lookahead_tokens == 0 {
        return Err(InferenceError::InvalidConfig(
            "lookahead_tokens must be greater than zero",
        ));
    }
    if policy.max_bytes_inflight == 0 {
        return Err(InferenceError::InvalidConfig(
            "max_bytes_inflight must be greater than zero",
        ));
    }

    let allowed: BTreeSet<TierId> = request.allowed_tiers.iter().copied().collect();
    let canceled: BTreeSet<ExpertKey> = request.cancel.iter().copied().collect();

    let mut scored = request
        .candidates
        .iter()
        .copied()
        .filter(|candidate| allowed.contains(&candidate.expert.tier))
        .filter(|candidate| !candidate.resident_hot)
        .filter(|candidate| !canceled.contains(&candidate.expert))
        .map(|candidate| {
            let urgency = if request.latency_budget_ms <= policy.aggressive_latency_budget_ms {
                1.5_f32
            } else {
                1.0_f32
            };

            let lookahead_weight = policy.lookahead_tokens as f32;
            let latency_term = 1.0 / candidate.tier_latency_ms.max(0.001);
            let size_penalty = (candidate.bytes as f32).sqrt().max(1.0);
            let priority = urgency
                * lookahead_weight
                * candidate.predicted_probability.max(0.0)
                * latency_term
                / size_penalty;

            (candidate, priority)
        })
        .collect::<Vec<(PrefetchCandidate, f32)>>();

    scored.sort_by(|(a, p_a), (b, p_b)| {
        p_b.partial_cmp(p_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let tie_a = seeded_hash_u64(
                    policy.deterministic_seed,
                    &format!("{}:{}:{}", a.expert.tier, a.expert.group, a.expert.expert),
                );
                let tie_b = seeded_hash_u64(
                    policy.deterministic_seed,
                    &format!("{}:{}:{}", b.expert.tier, b.expert.group, b.expert.expert),
                );
                tie_a.cmp(&tie_b)
            })
            .then(a.expert.cmp(&b.expert))
    });

    let mut actions = request
        .cancel
        .iter()
        .copied()
        .map(|expert| PrefetchAction::Cancel { expert })
        .collect::<Vec<PrefetchAction>>();

    let mut scheduled_bytes = 0_u64;
    let mut expected_hit_gain = 0.0_f32;
    let mut wasted_bytes_upper_bound = 0_u64;

    for (candidate, priority) in scored {
        if scheduled_bytes.saturating_add(candidate.bytes) > policy.max_bytes_inflight {
            wasted_bytes_upper_bound = wasted_bytes_upper_bound.saturating_add(candidate.bytes);
            continue;
        }

        scheduled_bytes = scheduled_bytes.saturating_add(candidate.bytes);
        expected_hit_gain += candidate.predicted_probability.max(0.0);
        actions.push(PrefetchAction::Fetch {
            expert: candidate.expert,
            priority,
            bytes: candidate.bytes,
        });
    }

    Ok(PrefetchPlan {
        actions,
        scheduled_bytes,
        expected_hit_gain,
        wasted_bytes_upper_bound,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_prefetch_plan, PrefetchCandidate, PrefetchPolicy, PrefetchRequest};
    use crate::types::ExpertKey;

    #[test]
    fn prefetch_plan_respects_allowed_tiers_and_capacity() {
        let plan = build_prefetch_plan(
            &PrefetchRequest {
                allowed_tiers: vec![1],
                latency_budget_ms: 20.0,
                candidates: vec![
                    PrefetchCandidate {
                        expert: ExpertKey::new(1, 0, 0),
                        predicted_probability: 0.8,
                        bytes: 100,
                        tier_latency_ms: 1.0,
                        resident_hot: false,
                    },
                    PrefetchCandidate {
                        expert: ExpertKey::new(2, 0, 0),
                        predicted_probability: 0.9,
                        bytes: 100,
                        tier_latency_ms: 1.0,
                        resident_hot: false,
                    },
                ],
                cancel: vec![],
            },
            PrefetchPolicy {
                lookahead_tokens: 2,
                max_bytes_inflight: 120,
                aggressive_latency_budget_ms: 10.0,
                deterministic_seed: 1,
            },
        )
        .expect("plan should build");

        assert_eq!(plan.scheduled_bytes, 100);
        assert_eq!(plan.actions.len(), 1);
    }

    #[test]
    fn canceled_experts_are_emitted_as_cancel_actions() {
        let expert = ExpertKey::new(1, 1, 1);
        let plan = build_prefetch_plan(
            &PrefetchRequest {
                allowed_tiers: vec![1],
                latency_budget_ms: 5.0,
                candidates: vec![PrefetchCandidate {
                    expert,
                    predicted_probability: 0.9,
                    bytes: 100,
                    tier_latency_ms: 1.0,
                    resident_hot: false,
                }],
                cancel: vec![expert],
            },
            PrefetchPolicy {
                lookahead_tokens: 2,
                max_bytes_inflight: 1000,
                aggressive_latency_budget_ms: 10.0,
                deterministic_seed: 1,
            },
        )
        .expect("plan should build");

        assert_eq!(plan.actions.len(), 1);
    }

    #[test]
    fn plan_is_deterministic() {
        let request = PrefetchRequest {
            allowed_tiers: vec![1, 2],
            latency_budget_ms: 8.0,
            candidates: vec![
                PrefetchCandidate {
                    expert: ExpertKey::new(1, 0, 1),
                    predicted_probability: 0.4,
                    bytes: 80,
                    tier_latency_ms: 2.0,
                    resident_hot: false,
                },
                PrefetchCandidate {
                    expert: ExpertKey::new(1, 0, 0),
                    predicted_probability: 0.4,
                    bytes: 80,
                    tier_latency_ms: 2.0,
                    resident_hot: false,
                },
            ],
            cancel: vec![],
        };

        let policy = PrefetchPolicy {
            lookahead_tokens: 3,
            max_bytes_inflight: 300,
            aggressive_latency_budget_ms: 10.0,
            deterministic_seed: 42,
        };

        let a = build_prefetch_plan(&request, policy).expect("plan should build");
        let b = build_prefetch_plan(&request, policy).expect("plan should build");
        assert_eq!(a, b);
    }
}
