use crate::error::{InferenceError, InferenceResult};
use crate::types::{seeded_hash_u64, ExpertKey};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostVector {
    pub latency: f32,
    pub memory: f32,
    pub energy: f32,
}

impl CostVector {
    pub fn add(self, other: CostVector) -> Self {
        Self {
            latency: self.latency + other.latency,
            memory: self.memory + other.memory,
            energy: self.energy + other.energy,
        }
    }

    pub fn within(self, budget: CostVector) -> bool {
        self.latency <= budget.latency + 1e-6
            && self.memory <= budget.memory + 1e-6
            && self.energy <= budget.energy + 1e-6
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostWeights {
    pub latency: f32,
    pub memory: f32,
    pub energy: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertCostScore {
    pub expert: ExpertKey,
    pub base_score: f32,
    pub cost: CostVector,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostAdaptiveChoice {
    pub expert: ExpertKey,
    pub adjusted_score: f32,
    pub cost: CostVector,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CostAdaptiveResult {
    pub selected: Vec<CostAdaptiveChoice>,
    pub total_cost: CostVector,
}

pub fn select_cost_adaptive(
    candidates: &[ExpertCostScore],
    top_k: usize,
    weights: CostWeights,
    budget: CostVector,
    seed: u64,
) -> InferenceResult<CostAdaptiveResult> {
    if top_k == 0 {
        return Err(InferenceError::InvalidConfig(
            "top_k must be greater than zero",
        ));
    }
    if candidates.is_empty() {
        return Err(InferenceError::InvalidInput("candidates must not be empty"));
    }

    let normalized = normalize_costs(candidates);
    let mut scored = normalized
        .into_iter()
        .map(|(candidate, normalized_cost)| {
            let penalty = weights.latency * normalized_cost.latency
                + weights.memory * normalized_cost.memory
                + weights.energy * normalized_cost.energy;

            CostAdaptiveChoice {
                expert: candidate.expert,
                adjusted_score: candidate.base_score - penalty,
                cost: candidate.cost,
            }
        })
        .collect::<Vec<CostAdaptiveChoice>>();

    scored.sort_by(|a, b| {
        b.adjusted_score
            .partial_cmp(&a.adjusted_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                let tie_a = seeded_hash_u64(seed, &a.expert.encode());
                let tie_b = seeded_hash_u64(seed, &b.expert.encode());
                tie_a.cmp(&tie_b)
            })
            .then(a.expert.cmp(&b.expert))
    });

    let mut selected = scored
        .iter()
        .copied()
        .take(top_k.min(scored.len()))
        .collect::<Vec<_>>();
    let mut total_cost = sum_cost(&selected);

    if total_cost.within(budget) {
        return Ok(CostAdaptiveResult {
            selected,
            total_cost,
        });
    }

    let mut cursor = top_k.min(scored.len());

    while !total_cost.within(budget) && !selected.is_empty() {
        selected.sort_by(|a, b| {
            a.adjusted_score
                .partial_cmp(&b.adjusted_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.expert.cmp(&b.expert))
        });

        selected.remove(0);

        while cursor < scored.len() {
            let candidate = scored[cursor];
            cursor += 1;
            if !selected
                .iter()
                .any(|entry| entry.expert == candidate.expert)
            {
                selected.push(candidate);
                break;
            }
        }

        total_cost = sum_cost(&selected);
    }

    if selected.is_empty() {
        let cheapest = scored
            .iter()
            .copied()
            .min_by(|a, b| {
                let cost_a = a.cost.latency + a.cost.memory + a.cost.energy;
                let cost_b = b.cost.latency + b.cost.memory + b.cost.energy;
                cost_a
                    .partial_cmp(&cost_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.expert.cmp(&b.expert))
            })
            .ok_or(InferenceError::BudgetUnsatisfied(
                "unable to find fallback expert under budget",
            ))?;

        return Ok(CostAdaptiveResult {
            selected: vec![cheapest],
            total_cost: cheapest.cost,
        });
    }

    Ok(CostAdaptiveResult {
        selected,
        total_cost,
    })
}

fn normalize_costs(candidates: &[ExpertCostScore]) -> Vec<(ExpertCostScore, CostVector)> {
    let max_latency = candidates
        .iter()
        .fold(0.0_f32, |acc, c| {
            if c.cost.latency > acc {
                c.cost.latency
            } else {
                acc
            }
        })
        .max(1e-6);
    let max_memory = candidates
        .iter()
        .fold(0.0_f32, |acc, c| {
            if c.cost.memory > acc {
                c.cost.memory
            } else {
                acc
            }
        })
        .max(1e-6);
    let max_energy = candidates
        .iter()
        .fold(0.0_f32, |acc, c| {
            if c.cost.energy > acc {
                c.cost.energy
            } else {
                acc
            }
        })
        .max(1e-6);

    candidates
        .iter()
        .copied()
        .map(|candidate| {
            (
                candidate,
                CostVector {
                    latency: candidate.cost.latency / max_latency,
                    memory: candidate.cost.memory / max_memory,
                    energy: candidate.cost.energy / max_energy,
                },
            )
        })
        .collect()
}

fn sum_cost(values: &[CostAdaptiveChoice]) -> CostVector {
    values.iter().fold(
        CostVector {
            latency: 0.0,
            memory: 0.0,
            energy: 0.0,
        },
        |acc, value| acc.add(value.cost),
    )
}

#[cfg(test)]
mod tests {
    use super::{select_cost_adaptive, CostVector, CostWeights, ExpertCostScore};
    use crate::types::ExpertKey;

    fn candidates() -> Vec<ExpertCostScore> {
        vec![
            ExpertCostScore {
                expert: ExpertKey::new(1, 0, 0),
                base_score: 0.8,
                cost: CostVector {
                    latency: 1.0,
                    memory: 1.0,
                    energy: 1.0,
                },
            },
            ExpertCostScore {
                expert: ExpertKey::new(1, 0, 1),
                base_score: 0.9,
                cost: CostVector {
                    latency: 8.0,
                    memory: 8.0,
                    energy: 8.0,
                },
            },
            ExpertCostScore {
                expert: ExpertKey::new(1, 0, 2),
                base_score: 0.7,
                cost: CostVector {
                    latency: 2.0,
                    memory: 2.0,
                    energy: 2.0,
                },
            },
        ]
    }

    #[test]
    fn high_cost_experts_are_penalized() {
        let result = select_cost_adaptive(
            &candidates(),
            1,
            CostWeights {
                latency: 1.0,
                memory: 1.0,
                energy: 1.0,
            },
            CostVector {
                latency: 100.0,
                memory: 100.0,
                energy: 100.0,
            },
            42,
        )
        .expect("selection should succeed");

        assert_eq!(result.selected.len(), 1);
        assert_ne!(result.selected[0].expert, ExpertKey::new(1, 0, 1));
    }

    #[test]
    fn budget_pruning_keeps_cost_under_limit() {
        let result = select_cost_adaptive(
            &candidates(),
            2,
            CostWeights {
                latency: 0.2,
                memory: 0.2,
                energy: 0.2,
            },
            CostVector {
                latency: 3.0,
                memory: 3.0,
                energy: 3.0,
            },
            7,
        )
        .expect("selection should succeed");

        assert!(result.total_cost.latency <= 3.0 + 1e-6);
        assert!(result.total_cost.memory <= 3.0 + 1e-6);
        assert!(result.total_cost.energy <= 3.0 + 1e-6);
    }

    #[test]
    fn selection_is_deterministic() {
        let args = (
            candidates(),
            2,
            CostWeights {
                latency: 0.4,
                memory: 0.4,
                energy: 0.4,
            },
            CostVector {
                latency: 6.0,
                memory: 6.0,
                energy: 6.0,
            },
            11,
        );

        let a = select_cost_adaptive(&args.0, args.1, args.2, args.3, args.4)
            .expect("selection should succeed");
        let b = select_cost_adaptive(&args.0, args.1, args.2, args.3, args.4)
            .expect("selection should succeed");

        assert_eq!(a, b);
    }
}
