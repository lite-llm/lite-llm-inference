use std::collections::{BTreeMap, BTreeSet};

use crate::error::{InferenceError, InferenceResult};
use crate::types::TierId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierSet {
    pub tiers: Vec<TierId>,
    pub cumulative: bool,
}

impl TierSet {
    pub fn new(mut tiers: Vec<TierId>, cumulative: bool) -> Self {
        tiers.sort_unstable();
        tiers.dedup();
        Self { tiers, cumulative }
    }

    pub fn empty() -> Self {
        Self {
            tiers: Vec::new(),
            cumulative: false,
        }
    }

    pub fn contains(&self, tier: TierId) -> bool {
        if self.cumulative {
            self.tiers
                .iter()
                .max()
                .map(|max| tier <= *max)
                .unwrap_or(false)
        } else {
            self.tiers.binary_search(&tier).is_ok()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    Fast,
    Balanced,
    Deep,
    Max,
    BudgetBased,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TierProfile {
    pub id: TierId,
    pub label: String,
    pub capacity_value: u64,
    pub latency_cost_ms: f32,
    pub monetary_cost_units: f32,
    pub available: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BudgetSpec {
    pub latency_budget_ms: Option<f32>,
    pub cost_budget_units: Option<f32>,
}

impl BudgetSpec {
    pub fn is_satisfied(self, latency_ms: f32, cost_units: f32) -> bool {
        let latency_ok = self
            .latency_budget_ms
            .map(|budget| latency_ms <= budget + 1e-6)
            .unwrap_or(true);
        let cost_ok = self
            .cost_budget_units
            .map(|budget| cost_units <= budget + 1e-6)
            .unwrap_or(true);
        latency_ok && cost_ok
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TierSetSelectionRequest {
    pub mode: SelectionMode,
    pub explicit_tiers: Option<Vec<TierId>>,
    pub include_tiers: Vec<TierId>,
    pub exclude_tiers: Vec<TierId>,
    pub budget: BudgetSpec,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TierSetSelectionResult {
    pub selected: TierSet,
    pub estimated_latency_ms: f32,
    pub estimated_cost_units: f32,
    pub estimated_capacity_value: u64,
    pub budget_satisfied: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedModeTierSets {
    pub fast: TierSet,
    pub balanced: TierSet,
    pub deep: TierSet,
    pub max: TierSet,
}

#[derive(Debug, Clone)]
pub struct TierSetSelector {
    pub base_latency_ms: f32,
    pub fixed: FixedModeTierSets,
    pub tiers: BTreeMap<TierId, TierProfile>,
}

impl TierSetSelector {
    pub fn new(
        base_latency_ms: f32,
        fixed: FixedModeTierSets,
        tiers: Vec<TierProfile>,
    ) -> InferenceResult<Self> {
        if base_latency_ms < 0.0 {
            return Err(InferenceError::InvalidConfig(
                "base_latency_ms must be non-negative",
            ));
        }

        let mut map = BTreeMap::new();
        for profile in tiers {
            if profile.label.trim().is_empty() {
                return Err(InferenceError::InvalidConfig(
                    "tier label must not be empty",
                ));
            }
            map.insert(profile.id, profile);
        }

        if map.is_empty() {
            return Err(InferenceError::InvalidConfig(
                "selector requires at least one tier profile",
            ));
        }

        Ok(Self {
            base_latency_ms,
            fixed,
            tiers: map,
        })
    }

    pub fn select(
        &self,
        request: &TierSetSelectionRequest,
    ) -> InferenceResult<TierSetSelectionResult> {
        let exclude: BTreeSet<TierId> = request.exclude_tiers.iter().copied().collect();
        let include: BTreeSet<TierId> = request
            .include_tiers
            .iter()
            .copied()
            .filter(|tier| !exclude.contains(tier))
            .collect();

        let available: BTreeSet<TierId> = self
            .tiers
            .values()
            .filter(|profile| profile.available)
            .map(|profile| profile.id)
            .filter(|tier| !exclude.contains(tier))
            .collect();

        if available.is_empty() {
            return Err(InferenceError::InvalidState(
                "no available tiers after exclusions",
            ));
        }

        let mut selected = if let Some(explicit) = &request.explicit_tiers {
            let explicit_set: BTreeSet<TierId> = explicit.iter().copied().collect();
            self.filter_supported_tiers(&explicit_set, &available)
        } else {
            match request.mode {
                SelectionMode::Fast => self.filter_supported_tiers(
                    &self.fixed.fast.tiers.iter().copied().collect(),
                    &available,
                ),
                SelectionMode::Balanced => self.filter_supported_tiers(
                    &self.fixed.balanced.tiers.iter().copied().collect(),
                    &available,
                ),
                SelectionMode::Deep => self.filter_supported_tiers(
                    &self.fixed.deep.tiers.iter().copied().collect(),
                    &available,
                ),
                SelectionMode::Max => self.filter_supported_tiers(
                    &self.fixed.max.tiers.iter().copied().collect(),
                    &available,
                ),
                SelectionMode::BudgetBased => {
                    let solved = self.solve_budget(&available, &include, request.budget)?;
                    solved.unwrap_or_default()
                }
            }
        };

        for forced in include {
            if available.contains(&forced) {
                selected.insert(forced);
            }
        }

        selected = selected
            .into_iter()
            .filter(|tier| available.contains(tier))
            .collect();

        if selected.is_empty() {
            if let Some(fallback) = self.lowest_latency_tier(&available) {
                selected.insert(fallback);
            }
        }

        if selected.is_empty() {
            return Err(InferenceError::InvalidState(
                "unable to select any valid tier",
            ));
        }

        let (latency, cost, capacity) = self.estimate_set(&selected)?;
        let budget_satisfied = request.budget.is_satisfied(latency, cost);
        let reason = if budget_satisfied {
            "selected under constraints".to_owned()
        } else {
            "fallback selected; budget may be exceeded".to_owned()
        };

        Ok(TierSetSelectionResult {
            selected: TierSet::new(selected.into_iter().collect(), false),
            estimated_latency_ms: latency,
            estimated_cost_units: cost,
            estimated_capacity_value: capacity,
            budget_satisfied,
            reason,
        })
    }

    fn solve_budget(
        &self,
        available: &BTreeSet<TierId>,
        forced_include: &BTreeSet<TierId>,
        budget: BudgetSpec,
    ) -> InferenceResult<Option<BTreeSet<TierId>>> {
        let forced = self.filter_supported_tiers(forced_include, available);
        let optional: Vec<TierId> = available
            .iter()
            .copied()
            .filter(|tier| !forced.contains(tier))
            .collect();

        if optional.len() > 20 {
            return Err(InferenceError::InvalidConfig(
                "budget solver currently supports at most 20 optional tiers",
            ));
        }

        let mut best: Option<(BTreeSet<TierId>, u64, f32, f32)> = None;
        let combos = 1_usize << optional.len();

        for mask in 0..combos {
            let mut candidate = forced.clone();
            for (idx, tier) in optional.iter().enumerate() {
                if ((mask >> idx) & 1) == 1 {
                    candidate.insert(*tier);
                }
            }

            if candidate.is_empty() {
                continue;
            }

            let (latency, cost, capacity) = self.estimate_set(&candidate)?;
            if !budget.is_satisfied(latency, cost) {
                continue;
            }

            let replace = match &best {
                None => true,
                Some((best_set, best_capacity, best_latency, best_cost)) => {
                    if capacity > *best_capacity {
                        true
                    } else if capacity < *best_capacity {
                        false
                    } else if latency + 1e-6 < *best_latency {
                        true
                    } else if *best_latency + 1e-6 < latency {
                        false
                    } else if cost + 1e-6 < *best_cost {
                        true
                    } else if *best_cost + 1e-6 < cost {
                        false
                    } else {
                        candidate.iter().copied().collect::<Vec<TierId>>()
                            < best_set.iter().copied().collect::<Vec<TierId>>()
                    }
                }
            };

            if replace {
                best = Some((candidate, capacity, latency, cost));
            }
        }

        Ok(best.map(|(set, _, _, _)| set))
    }

    fn estimate_set(&self, tiers: &BTreeSet<TierId>) -> InferenceResult<(f32, f32, u64)> {
        let mut latency = self.base_latency_ms;
        let mut cost = 0.0_f32;
        let mut capacity = 0_u64;

        for tier in tiers {
            let profile = self
                .tiers
                .get(tier)
                .ok_or(InferenceError::InvalidInput("unknown tier in selection"))?;
            latency += profile.latency_cost_ms.max(0.0);
            cost += profile.monetary_cost_units.max(0.0);
            capacity = capacity.saturating_add(profile.capacity_value);
        }

        Ok((latency, cost, capacity))
    }

    fn filter_supported_tiers(
        &self,
        requested: &BTreeSet<TierId>,
        available: &BTreeSet<TierId>,
    ) -> BTreeSet<TierId> {
        requested
            .iter()
            .copied()
            .filter(|tier| self.tiers.contains_key(tier) && available.contains(tier))
            .collect()
    }

    fn lowest_latency_tier(&self, available: &BTreeSet<TierId>) -> Option<TierId> {
        let mut candidates: Vec<&TierProfile> = self
            .tiers
            .values()
            .filter(|profile| available.contains(&profile.id))
            .collect();

        candidates.sort_by(|a, b| {
            a.latency_cost_ms
                .partial_cmp(&b.latency_cost_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });

        candidates.first().map(|profile| profile.id)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BudgetSpec, FixedModeTierSets, SelectionMode, TierProfile, TierSet,
        TierSetSelectionRequest, TierSetSelector,
    };

    fn selector() -> TierSetSelector {
        TierSetSelector::new(
            2.0,
            FixedModeTierSets {
                fast: TierSet::new(vec![1], false),
                balanced: TierSet::new(vec![1, 2], false),
                deep: TierSet::new(vec![1, 2, 3], false),
                max: TierSet::new(vec![1, 2, 3, 4], false),
            },
            vec![
                TierProfile {
                    id: 1,
                    label: "hot".to_owned(),
                    capacity_value: 100,
                    latency_cost_ms: 1.0,
                    monetary_cost_units: 0.2,
                    available: true,
                },
                TierProfile {
                    id: 2,
                    label: "warm".to_owned(),
                    capacity_value: 200,
                    latency_cost_ms: 2.0,
                    monetary_cost_units: 0.6,
                    available: true,
                },
                TierProfile {
                    id: 3,
                    label: "cold".to_owned(),
                    capacity_value: 400,
                    latency_cost_ms: 4.0,
                    monetary_cost_units: 1.5,
                    available: true,
                },
                TierProfile {
                    id: 4,
                    label: "archive".to_owned(),
                    capacity_value: 600,
                    latency_cost_ms: 7.0,
                    monetary_cost_units: 2.4,
                    available: false,
                },
            ],
        )
        .expect("selector should build")
    }

    #[test]
    fn fixed_fast_mode_selects_hot() {
        let selector = selector();
        let result = selector
            .select(&TierSetSelectionRequest {
                mode: SelectionMode::Fast,
                explicit_tiers: None,
                include_tiers: vec![],
                exclude_tiers: vec![],
                budget: BudgetSpec::default(),
            })
            .expect("selection should succeed");

        assert_eq!(result.selected.tiers, vec![1]);
    }

    #[test]
    fn budget_solver_maximizes_capacity_under_budget() {
        let selector = selector();
        let result = selector
            .select(&TierSetSelectionRequest {
                mode: SelectionMode::BudgetBased,
                explicit_tiers: None,
                include_tiers: vec![],
                exclude_tiers: vec![],
                budget: BudgetSpec {
                    latency_budget_ms: Some(6.0),
                    cost_budget_units: Some(1.0),
                },
            })
            .expect("selection should succeed");

        assert_eq!(result.selected.tiers, vec![1, 2]);
        assert!(result.budget_satisfied);
    }

    #[test]
    fn include_and_exclude_overrides_are_applied() {
        let selector = selector();
        let result = selector
            .select(&TierSetSelectionRequest {
                mode: SelectionMode::Balanced,
                explicit_tiers: None,
                include_tiers: vec![3],
                exclude_tiers: vec![2],
                budget: BudgetSpec::default(),
            })
            .expect("selection should succeed");

        assert_eq!(result.selected.tiers, vec![1, 3]);
    }

    #[test]
    fn tight_budget_falls_back_to_minimal_tier() {
        let selector = selector();
        let result = selector
            .select(&TierSetSelectionRequest {
                mode: SelectionMode::BudgetBased,
                explicit_tiers: None,
                include_tiers: vec![],
                exclude_tiers: vec![],
                budget: BudgetSpec {
                    latency_budget_ms: Some(0.1),
                    cost_budget_units: Some(0.05),
                },
            })
            .expect("selection should succeed");

        assert_eq!(result.selected.tiers, vec![1]);
        assert!(!result.budget_satisfied);
    }

    #[test]
    fn budget_selection_is_deterministic() {
        let selector = selector();
        let req = TierSetSelectionRequest {
            mode: SelectionMode::BudgetBased,
            explicit_tiers: None,
            include_tiers: vec![2],
            exclude_tiers: vec![],
            budget: BudgetSpec {
                latency_budget_ms: Some(9.0),
                cost_budget_units: Some(2.0),
            },
        };

        let a = selector.select(&req).expect("selection should succeed");
        let b = selector.select(&req).expect("selection should succeed");

        assert_eq!(a, b);
    }
}
