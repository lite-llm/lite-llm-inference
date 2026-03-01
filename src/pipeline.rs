use std::collections::{BTreeMap, BTreeSet};

use crate::error::{InferenceError, InferenceResult};
use crate::types::{fnv64_hex, seeded_hash_u64, ExpertKey, RoutingAssignment};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineStage {
    DenseCompute,
    Routing,
    Selection,
    Pack,
    Dispatch,
    ExpertCompute,
    Unpack,
    Combine,
    Residual,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertScore {
    pub expert: ExpertKey,
    pub destination_rank: u32,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineInput {
    pub seed: u64,
    pub rank: u32,
    pub world_size: u32,
    pub top_k: usize,
    pub tokens: Vec<Vec<f32>>,
    pub expert_scores: Vec<Vec<ExpertScore>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedToken {
    pub token_index: u32,
    pub source_rank: u32,
    pub destination_rank: u32,
    pub expert: ExpertKey,
    pub weight: f32,
    pub payload: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedSegment {
    pub destination_rank: u32,
    pub records: Vec<PackedToken>,
    pub checksum_hex: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineOutput {
    pub assignments: Vec<RoutingAssignment>,
    pub segments: Vec<PackedSegment>,
    pub combined_tokens: Vec<Vec<f32>>,
    pub stage_latency_ms: BTreeMap<PipelineStage, f32>,
    pub total_latency_ms: f32,
    pub throughput_tokens_per_ms: f32,
    pub imbalance_ratio: f32,
}

pub trait InferencePipeline {
    fn run(&self, input: &PipelineInput) -> InferenceResult<PipelineOutput>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeterministicInferencePipeline {
    pub enable_compression: bool,
}

impl Default for DeterministicInferencePipeline {
    fn default() -> Self {
        Self {
            enable_compression: false,
        }
    }
}

impl InferencePipeline for DeterministicInferencePipeline {
    fn run(&self, input: &PipelineInput) -> InferenceResult<PipelineOutput> {
        validate_input(input)?;

        let assignments = route_assignments(input)?;
        let segments = pack_tokens(
            &assignments,
            &input.tokens,
            input.rank,
            self.enable_compression,
            input.seed,
        )?;
        let combined_tokens = run_expert_path(&segments, input.tokens.len())?;

        let stage_latency_ms = estimate_stage_latencies(
            input.tokens.len(),
            input.tokens.first().map(|row| row.len()).unwrap_or(0),
            assignments.len(),
            &segments,
        );
        let total_latency_ms = stage_latency_ms.values().sum::<f32>();
        let throughput_tokens_per_ms = if total_latency_ms <= f32::EPSILON {
            0.0
        } else {
            input.tokens.len() as f32 / total_latency_ms
        };

        Ok(PipelineOutput {
            assignments,
            segments: segments.clone(),
            combined_tokens,
            stage_latency_ms,
            total_latency_ms,
            throughput_tokens_per_ms,
            imbalance_ratio: dispatch_imbalance_ratio(&segments),
        })
    }
}

fn validate_input(input: &PipelineInput) -> InferenceResult<()> {
    if input.world_size == 0 {
        return Err(InferenceError::InvalidConfig(
            "world_size must be greater than zero",
        ));
    }
    if input.top_k == 0 {
        return Err(InferenceError::InvalidConfig(
            "top_k must be greater than zero",
        ));
    }
    if input.tokens.len() != input.expert_scores.len() {
        return Err(InferenceError::InvalidInput(
            "tokens and expert_scores must have equal length",
        ));
    }

    let expected_dim = input.tokens.first().map(|row| row.len()).unwrap_or(0);
    for token in &input.tokens {
        if token.len() != expected_dim {
            return Err(InferenceError::InvalidInput(
                "all token vectors must have the same hidden dimension",
            ));
        }
    }

    for token_scores in &input.expert_scores {
        if token_scores.is_empty() {
            return Err(InferenceError::InvalidInput(
                "each token must have at least one expert score",
            ));
        }

        for score in token_scores {
            if score.destination_rank >= input.world_size {
                return Err(InferenceError::InvalidInput(
                    "destination_rank exceeds world_size",
                ));
            }
        }
    }

    Ok(())
}

fn route_assignments(input: &PipelineInput) -> InferenceResult<Vec<RoutingAssignment>> {
    let mut assignments = Vec::new();

    for (token_index, scores) in input.expert_scores.iter().enumerate() {
        let selected = deterministic_top_k(scores, input.top_k, input.seed, token_index as u32);
        let weights = normalize_positive_weights(&selected);

        for (score, weight) in selected.into_iter().zip(weights) {
            assignments.push(RoutingAssignment {
                token_index: token_index as u32,
                expert: score.expert,
                score: score.score,
                weight,
                destination_rank: score.destination_rank,
            });
        }
    }

    Ok(assignments)
}

fn deterministic_top_k(
    scores: &[ExpertScore],
    top_k: usize,
    seed: u64,
    token_index: u32,
) -> Vec<ExpertScore> {
    let mut indexed: Vec<(ExpertScore, u64)> = scores
        .iter()
        .copied()
        .map(|score| {
            let payload = format!(
                "{}|{}|{}|{}",
                token_index, score.expert.tier, score.expert.group, score.expert.expert
            );
            let tie = seeded_hash_u64(seed, &payload);
            (score, tie)
        })
        .collect();

    indexed.sort_by(|(a, tie_a), (b, tie_b)| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(tie_a.cmp(tie_b))
            .then(a.destination_rank.cmp(&b.destination_rank))
            .then(a.expert.cmp(&b.expert))
    });

    indexed
        .into_iter()
        .take(top_k.min(scores.len()))
        .map(|(score, _)| score)
        .collect()
}

fn normalize_positive_weights(scores: &[ExpertScore]) -> Vec<f32> {
    let mut raw = scores
        .iter()
        .map(|score| score.score.max(0.0))
        .collect::<Vec<f32>>();
    let sum = raw.iter().sum::<f32>();

    if sum <= f32::EPSILON {
        let uniform = 1.0 / raw.len().max(1) as f32;
        raw.fill(uniform);
        return raw;
    }

    for value in &mut raw {
        *value /= sum;
    }

    raw
}

fn pack_tokens(
    assignments: &[RoutingAssignment],
    tokens: &[Vec<f32>],
    source_rank: u32,
    compress: bool,
    seed: u64,
) -> InferenceResult<Vec<PackedSegment>> {
    let mut by_destination: BTreeMap<u32, Vec<PackedToken>> = BTreeMap::new();

    for assignment in assignments {
        let payload = tokens
            .get(assignment.token_index as usize)
            .ok_or(InferenceError::InvalidInput(
                "assignment token index out of bounds",
            ))?
            .clone();

        let payload = if compress {
            compress_payload(&payload)
        } else {
            payload
        };

        by_destination
            .entry(assignment.destination_rank)
            .or_default()
            .push(PackedToken {
                token_index: assignment.token_index,
                source_rank,
                destination_rank: assignment.destination_rank,
                expert: assignment.expert,
                weight: assignment.weight,
                payload,
            });
    }

    let mut segments = Vec::new();

    for (destination_rank, mut records) in by_destination {
        records.sort_by(|a, b| {
            a.token_index
                .cmp(&b.token_index)
                .then_with(|| {
                    let key_a = seeded_hash_u64(
                        seed,
                        &format!(
                            "{}|{}|{}|{}",
                            a.token_index, a.expert.tier, a.expert.group, a.expert.expert
                        ),
                    );
                    let key_b = seeded_hash_u64(
                        seed,
                        &format!(
                            "{}|{}|{}|{}",
                            b.token_index, b.expert.tier, b.expert.group, b.expert.expert
                        ),
                    );
                    key_a.cmp(&key_b)
                })
                .then(a.expert.cmp(&b.expert))
        });

        let checksum_hex = packed_segment_checksum(destination_rank, &records);
        segments.push(PackedSegment {
            destination_rank,
            records,
            checksum_hex,
        });
    }

    Ok(segments)
}

fn packed_segment_checksum(destination_rank: u32, records: &[PackedToken]) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&destination_rank.to_le_bytes());

    for record in records {
        bytes.extend_from_slice(&record.token_index.to_le_bytes());
        bytes.extend_from_slice(&record.source_rank.to_le_bytes());
        bytes.extend_from_slice(&record.destination_rank.to_le_bytes());
        bytes.extend_from_slice(&record.expert.tier.to_le_bytes());
        bytes.extend_from_slice(&record.expert.group.to_le_bytes());
        bytes.extend_from_slice(&record.expert.expert.to_le_bytes());
        bytes.extend_from_slice(&record.weight.to_bits().to_le_bytes());
        for value in &record.payload {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }

    fnv64_hex(&bytes)
}

fn compress_payload(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| (value * 1024.0).round() / 1024.0)
        .collect()
}

fn run_expert_path(
    segments: &[PackedSegment],
    token_count: usize,
) -> InferenceResult<Vec<Vec<f32>>> {
    let hidden_dim = segments
        .iter()
        .flat_map(|segment| segment.records.iter())
        .next()
        .map(|record| record.payload.len())
        .unwrap_or(0);

    let mut outputs = vec![vec![0.0_f32; hidden_dim]; token_count];
    let mut visited = BTreeSet::new();

    for segment in segments {
        for record in &segment.records {
            visited.insert(record.token_index as usize);
            let scale = expert_scale(record.expert) * record.weight;
            for (idx, value) in record.payload.iter().enumerate() {
                outputs[record.token_index as usize][idx] += value * scale;
            }
        }
    }

    for token in visited {
        if token >= token_count {
            return Err(InferenceError::InvalidState(
                "token index in packed segment exceeds token_count",
            ));
        }
    }

    Ok(outputs)
}

fn expert_scale(expert: ExpertKey) -> f32 {
    let composite = (expert.tier as u64)
        .wrapping_mul(131)
        .wrapping_add((expert.group as u64).wrapping_mul(17))
        .wrapping_add(expert.expert as u64);

    0.75 + (composite % 250) as f32 / 1000.0
}

fn dispatch_imbalance_ratio(segments: &[PackedSegment]) -> f32 {
    if segments.is_empty() {
        return 1.0;
    }

    let counts: Vec<f32> = segments
        .iter()
        .map(|segment| segment.records.len() as f32)
        .collect();
    let mean = counts.iter().sum::<f32>() / counts.len() as f32;
    let max = counts
        .iter()
        .copied()
        .fold(0.0_f32, |acc, value| if value > acc { value } else { acc });

    if mean <= f32::EPSILON {
        1.0
    } else {
        max / mean
    }
}

fn estimate_stage_latencies(
    token_count: usize,
    hidden_dim: usize,
    assignment_count: usize,
    segments: &[PackedSegment],
) -> BTreeMap<PipelineStage, f32> {
    let token_count = token_count as f32;
    let hidden_dim = hidden_dim as f32;
    let assignment_count = assignment_count as f32;
    let transfer_count = segments
        .iter()
        .map(|segment| segment.records.len() as f32)
        .sum::<f32>();

    let mut map = BTreeMap::new();
    map.insert(
        PipelineStage::DenseCompute,
        token_count * hidden_dim * 0.0010 + 0.20,
    );
    map.insert(PipelineStage::Routing, assignment_count * 0.0030 + 0.05);
    map.insert(PipelineStage::Selection, assignment_count * 0.0015 + 0.03);
    map.insert(PipelineStage::Pack, transfer_count * 0.0018 + 0.02);
    map.insert(
        PipelineStage::Dispatch,
        transfer_count * 0.0022 + segments.len() as f32 * 0.02,
    );
    map.insert(
        PipelineStage::ExpertCompute,
        assignment_count * hidden_dim.max(1.0) * 0.0007 + 0.12,
    );
    map.insert(PipelineStage::Unpack, transfer_count * 0.0013 + 0.02);
    map.insert(
        PipelineStage::Combine,
        token_count * hidden_dim.max(1.0) * 0.0004 + 0.02,
    );
    map.insert(PipelineStage::Residual, token_count * 0.0006 + 0.01);
    map
}

#[cfg(test)]
mod tests {
    use super::{
        DeterministicInferencePipeline, ExpertScore, InferencePipeline, PipelineInput,
        PipelineStage,
    };
    use crate::types::ExpertKey;

    fn sample_input() -> PipelineInput {
        let tokens = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
        ];

        let scores = vec![
            vec![
                ExpertScore {
                    expert: ExpertKey::new(1, 0, 0),
                    destination_rank: 0,
                    score: 0.9,
                },
                ExpertScore {
                    expert: ExpertKey::new(1, 0, 1),
                    destination_rank: 1,
                    score: 0.9,
                },
                ExpertScore {
                    expert: ExpertKey::new(2, 0, 0),
                    destination_rank: 1,
                    score: 0.4,
                },
            ],
            vec![
                ExpertScore {
                    expert: ExpertKey::new(1, 0, 0),
                    destination_rank: 0,
                    score: 0.7,
                },
                ExpertScore {
                    expert: ExpertKey::new(2, 0, 1),
                    destination_rank: 1,
                    score: 0.6,
                },
                ExpertScore {
                    expert: ExpertKey::new(3, 1, 0),
                    destination_rank: 1,
                    score: 0.4,
                },
            ],
            vec![
                ExpertScore {
                    expert: ExpertKey::new(1, 1, 0),
                    destination_rank: 0,
                    score: 0.8,
                },
                ExpertScore {
                    expert: ExpertKey::new(2, 1, 0),
                    destination_rank: 1,
                    score: 0.6,
                },
                ExpertScore {
                    expert: ExpertKey::new(3, 1, 1),
                    destination_rank: 1,
                    score: 0.5,
                },
            ],
            vec![
                ExpertScore {
                    expert: ExpertKey::new(1, 1, 1),
                    destination_rank: 0,
                    score: 0.85,
                },
                ExpertScore {
                    expert: ExpertKey::new(2, 1, 1),
                    destination_rank: 1,
                    score: 0.65,
                },
                ExpertScore {
                    expert: ExpertKey::new(3, 0, 0),
                    destination_rank: 1,
                    score: 0.55,
                },
            ],
        ];

        PipelineInput {
            seed: 42,
            rank: 0,
            world_size: 2,
            top_k: 2,
            tokens,
            expert_scores: scores,
        }
    }

    #[test]
    fn pipeline_output_is_deterministic() {
        let pipeline = DeterministicInferencePipeline::default();
        let input = sample_input();

        let a = pipeline.run(&input).expect("pipeline should run");
        let b = pipeline.run(&input).expect("pipeline should run");

        assert_eq!(a, b);
    }

    #[test]
    fn stable_tie_breaking_is_seeded() {
        let pipeline = DeterministicInferencePipeline::default();
        let mut input_a = sample_input();
        input_a.seed = 10;

        let mut input_b = input_a.clone();
        input_b.seed = 11;

        let out_a = pipeline.run(&input_a).expect("pipeline should run");
        let out_b = pipeline.run(&input_b).expect("pipeline should run");

        assert_ne!(out_a.assignments, out_b.assignments);
    }

    #[test]
    fn stage_timings_include_all_pipeline_stages() {
        let pipeline = DeterministicInferencePipeline::default();
        let output = pipeline.run(&sample_input()).expect("pipeline should run");

        assert!(output
            .stage_latency_ms
            .contains_key(&PipelineStage::DenseCompute));
        assert!(output
            .stage_latency_ms
            .contains_key(&PipelineStage::Dispatch));
        assert!(output
            .stage_latency_ms
            .contains_key(&PipelineStage::Residual));
        assert!(output.total_latency_ms > 0.0);
    }

    #[test]
    fn latency_regression_guardrail() {
        let pipeline = DeterministicInferencePipeline::default();
        let output = pipeline.run(&sample_input()).expect("pipeline should run");

        assert!(output.total_latency_ms <= 1.50);
    }

    #[test]
    fn throughput_regression_guardrail() {
        let pipeline = DeterministicInferencePipeline::default();
        let output = pipeline.run(&sample_input()).expect("pipeline should run");

        assert!(output.throughput_tokens_per_ms >= 2.0);
    }
}
