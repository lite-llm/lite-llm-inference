use std::collections::BTreeMap;

use crate::error::{InferenceError, InferenceResult};
use crate::types::{SessionId, TenantId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvTier {
    Hot,
    Warm,
    Cold,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvEntry {
    pub layer: u16,
    pub head: u16,
    pub position: u64,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub tier: KvTier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub hot_token_limit: usize,
    pub warm_token_limit: usize,
    pub total_token_limit: usize,
    pub sliding_window_tokens: Option<usize>,
}

impl KvCacheConfig {
    pub fn validate(self) -> InferenceResult<Self> {
        if self.hot_token_limit == 0 {
            return Err(InferenceError::InvalidConfig(
                "hot_token_limit must be greater than zero",
            ));
        }
        if self.total_token_limit == 0 || self.total_token_limit < self.hot_token_limit {
            return Err(InferenceError::InvalidConfig(
                "total_token_limit must be >= hot_token_limit",
            ));
        }

        if self.hot_token_limit + self.warm_token_limit > self.total_token_limit {
            return Err(InferenceError::InvalidConfig(
                "hot + warm limits must not exceed total_token_limit",
            ));
        }

        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SessionCache {
    records: Vec<KvEntry>,
}

impl SessionCache {
    fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvCache {
    config: KvCacheConfig,
    sessions: BTreeMap<(TenantId, SessionId), SessionCache>,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> InferenceResult<Self> {
        Ok(Self {
            config: config.validate()?,
            sessions: BTreeMap::new(),
        })
    }

    pub fn append(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
        layer: u16,
        head: u16,
        position: u64,
        key: &[f32],
        value: &[f32],
    ) -> InferenceResult<()> {
        if key.len() != value.len() {
            return Err(InferenceError::InvalidInput(
                "kv key/value lengths must be equal",
            ));
        }

        let session = self
            .sessions
            .entry((tenant_id, session_id))
            .or_insert_with(SessionCache::new);

        session.records.push(KvEntry {
            layer,
            head,
            position,
            key: key.to_vec(),
            value: value.to_vec(),
            tier: KvTier::Hot,
        });

        self.rebalance_session(tenant_id, session_id)?;
        Ok(())
    }

    pub fn slice(
        &self,
        tenant_id: TenantId,
        session_id: SessionId,
        layer: u16,
        head: u16,
        from_position: u64,
        to_position_exclusive: u64,
    ) -> InferenceResult<Vec<KvEntry>> {
        if from_position > to_position_exclusive {
            return Err(InferenceError::InvalidInput(
                "from_position must be <= to_position_exclusive",
            ));
        }

        let session = self.sessions.get(&(tenant_id, session_id)).ok_or_else(|| {
            InferenceError::TenantViolation("session not found for tenant".to_owned())
        })?;

        let mut out = session
            .records
            .iter()
            .filter(|entry| entry.layer == layer)
            .filter(|entry| entry.head == head)
            .filter(|entry| {
                entry.position >= from_position && entry.position < to_position_exclusive
            })
            .cloned()
            .collect::<Vec<KvEntry>>();

        out.sort_by(|a, b| a.position.cmp(&b.position));
        Ok(out)
    }

    pub fn reset_session(&mut self, tenant_id: TenantId, session_id: SessionId) {
        self.sessions.remove(&(tenant_id, session_id));
    }

    pub fn session_len(&self, tenant_id: TenantId, session_id: SessionId) -> usize {
        self.sessions
            .get(&(tenant_id, session_id))
            .map(|session| session.records.len())
            .unwrap_or(0)
    }

    pub fn tier_counts(&self, tenant_id: TenantId, session_id: SessionId) -> (usize, usize, usize) {
        if let Some(session) = self.sessions.get(&(tenant_id, session_id)) {
            let hot = session
                .records
                .iter()
                .filter(|entry| entry.tier == KvTier::Hot)
                .count();
            let warm = session
                .records
                .iter()
                .filter(|entry| entry.tier == KvTier::Warm)
                .count();
            let cold = session
                .records
                .iter()
                .filter(|entry| entry.tier == KvTier::Cold)
                .count();
            (hot, warm, cold)
        } else {
            (0, 0, 0)
        }
    }

    fn rebalance_session(
        &mut self,
        tenant_id: TenantId,
        session_id: SessionId,
    ) -> InferenceResult<()> {
        let session =
            self.sessions
                .get_mut(&(tenant_id, session_id))
                .ok_or(InferenceError::InvalidState(
                    "session missing during rebalance",
                ))?;

        session.records.sort_by(|a, b| {
            a.position
                .cmp(&b.position)
                .then(a.layer.cmp(&b.layer))
                .then(a.head.cmp(&b.head))
        });

        if let Some(window) = self.config.sliding_window_tokens {
            if window == 0 {
                return Err(InferenceError::InvalidConfig(
                    "sliding_window_tokens must be greater than zero when set",
                ));
            }

            if session.records.len() > window {
                let drop_count = session.records.len() - window;
                session.records.drain(0..drop_count);
            }
        }

        if session.records.len() > self.config.total_token_limit {
            let drop_count = session.records.len() - self.config.total_token_limit;
            session.records.drain(0..drop_count);
        }

        let total = session.records.len();
        for (idx, entry) in session.records.iter_mut().enumerate() {
            let reverse_idx = total - idx;
            entry.tier = if reverse_idx <= self.config.hot_token_limit {
                KvTier::Hot
            } else if reverse_idx <= self.config.hot_token_limit + self.config.warm_token_limit {
                KvTier::Warm
            } else {
                KvTier::Cold
            };
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{KvCache, KvCacheConfig, KvTier};

    fn cache() -> KvCache {
        KvCache::new(KvCacheConfig {
            hot_token_limit: 2,
            warm_token_limit: 2,
            total_token_limit: 5,
            sliding_window_tokens: None,
        })
        .expect("cache should initialize")
    }

    #[test]
    fn append_and_slice_preserve_order() {
        let mut cache = cache();
        for pos in 0..4 {
            cache
                .append(1, 10, 0, 0, pos, &[pos as f32], &[pos as f32 + 1.0])
                .expect("append should succeed");
        }

        let slice = cache
            .slice(1, 10, 0, 0, 0, 4)
            .expect("slice should succeed");
        let positions = slice
            .iter()
            .map(|entry| entry.position)
            .collect::<Vec<u64>>();
        assert_eq!(positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn deterministic_tier_assignment_after_rebalance() {
        let mut cache = cache();
        for pos in 0..5 {
            cache
                .append(1, 10, 0, 0, pos, &[0.0], &[0.0])
                .expect("append should succeed");
        }

        let (hot, warm, cold) = cache.tier_counts(1, 10);
        assert_eq!((hot, warm, cold), (2, 2, 1));

        let entries = cache
            .slice(1, 10, 0, 0, 0, 10)
            .expect("slice should succeed");
        assert_eq!(entries.first().map(|entry| entry.tier), Some(KvTier::Cold));
        assert_eq!(entries.last().map(|entry| entry.tier), Some(KvTier::Hot));
    }

    #[test]
    fn total_limit_evicts_oldest_entries() {
        let mut cache = cache();
        for pos in 0..8 {
            cache
                .append(1, 10, 0, 0, pos, &[0.0], &[0.0])
                .expect("append should succeed");
        }

        let entries = cache
            .slice(1, 10, 0, 0, 0, 100)
            .expect("slice should succeed");
        let positions = entries
            .iter()
            .map(|entry| entry.position)
            .collect::<Vec<u64>>();
        assert_eq!(positions, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn tenant_isolation_keeps_sessions_separate() {
        let mut cache = cache();
        cache
            .append(1, 10, 0, 0, 0, &[1.0], &[2.0])
            .expect("append should succeed");
        cache
            .append(2, 10, 0, 0, 0, &[3.0], &[4.0])
            .expect("append should succeed");

        let a = cache
            .slice(1, 10, 0, 0, 0, 1)
            .expect("slice should succeed");
        let b = cache
            .slice(2, 10, 0, 0, 0, 1)
            .expect("slice should succeed");

        assert_ne!(a[0].key, b[0].key);

        cache.reset_session(1, 10);
        assert_eq!(cache.session_len(1, 10), 0);
        assert_eq!(cache.session_len(2, 10), 1);
    }
}
