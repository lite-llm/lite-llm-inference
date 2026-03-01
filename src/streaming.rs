use std::collections::BTreeMap;

use crate::error::{InferenceError, InferenceResult};
use crate::kv_cache::{KvCache, KvCacheConfig};
use crate::types::{seeded_hash_u64, SessionId, TenantId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamStatus {
    Running,
    Backpressured,
    Completed,
    TimedOut,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingRequest {
    pub tenant_id: TenantId,
    pub session_id: SessionId,
    pub prompt_tokens: Vec<u32>,
    pub seed: u64,
    pub max_new_tokens: u32,
    pub backpressure_limit: u32,
    pub heartbeat_timeout_steps: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingSession {
    pub tenant_id: TenantId,
    pub session_id: SessionId,
    pub seed: u64,
    pub prompt_tokens: Vec<u32>,
    pub emitted_tokens: Vec<u32>,
    pub max_new_tokens: u32,
    pub backpressure_limit: u32,
    pub heartbeat_timeout_steps: u64,
    pub buffered_unacked: u32,
    pub step_index: u64,
    pub last_heartbeat_step: u64,
    pub status: StreamStatus,
}

#[derive(Debug, Clone)]
pub struct StreamingRuntime {
    sessions: BTreeMap<(TenantId, SessionId), StreamingSession>,
    kv_cache: KvCache,
}

impl StreamingRuntime {
    pub fn new(kv_config: KvCacheConfig) -> InferenceResult<Self> {
        Ok(Self {
            sessions: BTreeMap::new(),
            kv_cache: KvCache::new(kv_config)?,
        })
    }

    pub fn start_session(&mut self, request: StreamingRequest) -> InferenceResult<()> {
        if request.max_new_tokens == 0 {
            return Err(InferenceError::InvalidConfig(
                "max_new_tokens must be greater than zero",
            ));
        }
        if request.backpressure_limit == 0 {
            return Err(InferenceError::InvalidConfig(
                "backpressure_limit must be greater than zero",
            ));
        }

        let key = (request.tenant_id, request.session_id);
        if self.sessions.contains_key(&key) {
            return Err(InferenceError::InvalidState("session already exists"));
        }

        let session = StreamingSession {
            tenant_id: request.tenant_id,
            session_id: request.session_id,
            seed: request.seed,
            prompt_tokens: request.prompt_tokens,
            emitted_tokens: Vec::new(),
            max_new_tokens: request.max_new_tokens,
            backpressure_limit: request.backpressure_limit,
            heartbeat_timeout_steps: request.heartbeat_timeout_steps,
            buffered_unacked: 0,
            step_index: 0,
            last_heartbeat_step: 0,
            status: StreamStatus::Running,
        };

        self.sessions.insert(key, session);
        Ok(())
    }

    pub fn generate_next(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
    ) -> InferenceResult<Option<u32>> {
        let key = (tenant_id, session_id);
        let session = self
            .sessions
            .get_mut(&key)
            .ok_or_else(|| InferenceError::TenantViolation("session not found".to_owned()))?;

        match session.status {
            StreamStatus::Cancelled | StreamStatus::Completed | StreamStatus::TimedOut => {
                return Ok(None)
            }
            StreamStatus::Backpressured => {
                if session.buffered_unacked >= session.backpressure_limit {
                    return Ok(None);
                }
                session.status = StreamStatus::Running;
            }
            StreamStatus::Running => {}
        }

        if session.emitted_tokens.len() as u32 >= session.max_new_tokens {
            session.status = StreamStatus::Completed;
            return Ok(None);
        }

        if session.buffered_unacked >= session.backpressure_limit {
            session.status = StreamStatus::Backpressured;
            return Ok(None);
        }

        let token = next_token(
            session.seed,
            session.step_index,
            session.prompt_tokens.last().copied(),
            session.emitted_tokens.last().copied(),
        );

        self.kv_cache.append(
            tenant_id,
            session_id,
            0,
            0,
            session.step_index,
            &[token as f32],
            &[token as f32],
        )?;

        session.emitted_tokens.push(token);
        session.buffered_unacked = session.buffered_unacked.saturating_add(1);
        session.step_index = session.step_index.saturating_add(1);

        if session.emitted_tokens.len() as u32 >= session.max_new_tokens {
            session.status = StreamStatus::Completed;
        }

        Ok(Some(token))
    }

    pub fn ack_tokens(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
        ack_count: u32,
    ) -> InferenceResult<()> {
        let session = self
            .sessions
            .get_mut(&(tenant_id, session_id))
            .ok_or_else(|| InferenceError::TenantViolation("session not found".to_owned()))?;

        session.buffered_unacked = session.buffered_unacked.saturating_sub(ack_count);
        if session.status == StreamStatus::Backpressured
            && session.buffered_unacked < session.backpressure_limit
        {
            session.status = StreamStatus::Running;
        }
        Ok(())
    }

    pub fn heartbeat(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
        current_step: u64,
    ) -> InferenceResult<()> {
        let session = self
            .sessions
            .get_mut(&(tenant_id, session_id))
            .ok_or_else(|| InferenceError::TenantViolation("session not found".to_owned()))?;

        session.last_heartbeat_step = current_step;
        Ok(())
    }

    pub fn check_timeouts(&mut self, current_step: u64) {
        for session in self.sessions.values_mut() {
            if matches!(
                session.status,
                StreamStatus::Completed | StreamStatus::Cancelled | StreamStatus::TimedOut
            ) {
                continue;
            }
            if current_step.saturating_sub(session.last_heartbeat_step)
                > session.heartbeat_timeout_steps
            {
                session.status = StreamStatus::TimedOut;
            }
        }
    }

    pub fn append_client_input(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
        new_tokens: &[u32],
    ) -> InferenceResult<()> {
        let session = self
            .sessions
            .get_mut(&(tenant_id, session_id))
            .ok_or_else(|| InferenceError::TenantViolation("session not found".to_owned()))?;

        session.prompt_tokens.extend_from_slice(new_tokens);
        Ok(())
    }

    pub fn replay_prefix(
        &self,
        tenant_id: TenantId,
        session_id: SessionId,
        count: usize,
    ) -> InferenceResult<Vec<u32>> {
        let session = self
            .sessions
            .get(&(tenant_id, session_id))
            .ok_or_else(|| InferenceError::TenantViolation("session not found".to_owned()))?;

        Ok(session.emitted_tokens.iter().copied().take(count).collect())
    }

    pub fn status(&self, tenant_id: TenantId, session_id: SessionId) -> Option<StreamStatus> {
        self.sessions
            .get(&(tenant_id, session_id))
            .map(|session| session.status)
    }

    pub fn cancel(&mut self, tenant_id: TenantId, session_id: SessionId) {
        if let Some(session) = self.sessions.get_mut(&(tenant_id, session_id)) {
            session.status = StreamStatus::Cancelled;
            self.kv_cache.reset_session(tenant_id, session_id);
        }
    }

    pub fn kv_cache(&self) -> &KvCache {
        &self.kv_cache
    }
}

fn next_token(
    seed: u64,
    step_index: u64,
    prompt_last: Option<u32>,
    emitted_last: Option<u32>,
) -> u32 {
    let source = format!(
        "{}|{}|{}|{}",
        step_index,
        prompt_last.unwrap_or(0),
        emitted_last.unwrap_or(0),
        seed
    );
    let hash = seeded_hash_u64(seed, &source);
    1 + (hash % 32_000) as u32
}

#[cfg(test)]
mod tests {
    use super::{StreamStatus, StreamingRequest, StreamingRuntime};
    use crate::kv_cache::KvCacheConfig;

    fn runtime() -> StreamingRuntime {
        StreamingRuntime::new(KvCacheConfig {
            hot_token_limit: 4,
            warm_token_limit: 4,
            total_token_limit: 16,
            sliding_window_tokens: None,
        })
        .expect("runtime should initialize")
    }

    #[test]
    fn token_generation_is_deterministic() {
        let mut rt_a = runtime();
        let mut rt_b = runtime();

        let req = StreamingRequest {
            tenant_id: 1,
            session_id: 99,
            prompt_tokens: vec![101, 102],
            seed: 42,
            max_new_tokens: 4,
            backpressure_limit: 10,
            heartbeat_timeout_steps: 20,
        };

        rt_a.start_session(req.clone())
            .expect("session should start");
        rt_b.start_session(req).expect("session should start");

        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();

        for _ in 0..4 {
            seq_a.push(
                rt_a.generate_next(1, 99)
                    .expect("generation should succeed"),
            );
            seq_b.push(
                rt_b.generate_next(1, 99)
                    .expect("generation should succeed"),
            );
        }

        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn backpressure_blocks_until_ack() {
        let mut rt = runtime();
        rt.start_session(StreamingRequest {
            tenant_id: 1,
            session_id: 1,
            prompt_tokens: vec![],
            seed: 1,
            max_new_tokens: 5,
            backpressure_limit: 1,
            heartbeat_timeout_steps: 10,
        })
        .expect("session should start");

        assert!(rt
            .generate_next(1, 1)
            .expect("generation should succeed")
            .is_some());
        assert!(rt
            .generate_next(1, 1)
            .expect("generation should succeed")
            .is_none());
        assert_eq!(rt.status(1, 1), Some(StreamStatus::Backpressured));

        rt.ack_tokens(1, 1, 1).expect("ack should succeed");
        assert!(rt
            .generate_next(1, 1)
            .expect("generation should succeed")
            .is_some());
    }

    #[test]
    fn timeout_transitions_session_state() {
        let mut rt = runtime();
        rt.start_session(StreamingRequest {
            tenant_id: 1,
            session_id: 2,
            prompt_tokens: vec![],
            seed: 1,
            max_new_tokens: 5,
            backpressure_limit: 3,
            heartbeat_timeout_steps: 2,
        })
        .expect("session should start");

        rt.heartbeat(1, 2, 1).expect("heartbeat should succeed");
        rt.check_timeouts(4);
        assert_eq!(rt.status(1, 2), Some(StreamStatus::TimedOut));
    }

    #[test]
    fn replay_prefix_is_idempotent() {
        let mut rt = runtime();
        rt.start_session(StreamingRequest {
            tenant_id: 1,
            session_id: 3,
            prompt_tokens: vec![55],
            seed: 9,
            max_new_tokens: 3,
            backpressure_limit: 10,
            heartbeat_timeout_steps: 20,
        })
        .expect("session should start");

        for _ in 0..3 {
            let _ = rt.generate_next(1, 3).expect("generation should succeed");
        }

        let prefix_a = rt
            .replay_prefix(1, 3, 2)
            .expect("prefix should be available");
        let prefix_b = rt
            .replay_prefix(1, 3, 2)
            .expect("prefix should be available");
        assert_eq!(prefix_a, prefix_b);
    }
}
