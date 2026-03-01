pub type TierId = u16;
pub type TenantId = u64;
pub type SessionId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertKey {
    pub tier: TierId,
    pub group: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub const fn new(tier: TierId, group: u32, expert: u32) -> Self {
        Self {
            tier,
            group,
            expert,
        }
    }

    pub fn encode(self) -> String {
        format!("{}:{}:{}", self.tier, self.group, self.expert)
    }

    pub fn parse(value: &str) -> Option<Self> {
        let parts: Vec<&str> = value.split(':').collect();
        if parts.len() != 3 {
            return None;
        }

        Some(Self {
            tier: parts[0].parse().ok()?,
            group: parts[1].parse().ok()?,
            expert: parts[2].parse().ok()?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutingAssignment {
    pub token_index: u32,
    pub expert: ExpertKey,
    pub score: f32,
    pub weight: f32,
    pub destination_rank: u32,
}

pub fn fnv64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub fn seeded_hash_u64(seed: u64, payload: &str) -> u64 {
    let bytes = format!("{seed}|{payload}").into_bytes();
    let hash_hex = fnv64_hex(&bytes);
    u64::from_str_radix(&hash_hex, 16).unwrap_or(0)
}

pub fn checksum_f32(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    fnv64_hex(&bytes)
}

#[cfg(test)]
mod tests {
    use super::{checksum_f32, seeded_hash_u64, ExpertKey};

    #[test]
    fn expert_key_roundtrip() {
        let key = ExpertKey::new(3, 2, 9);
        assert_eq!(ExpertKey::parse(&key.encode()), Some(key));
    }

    #[test]
    fn seeded_hash_is_deterministic() {
        let a = seeded_hash_u64(42, "x");
        let b = seeded_hash_u64(42, "x");
        let c = seeded_hash_u64(43, "x");

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn f32_checksum_is_deterministic() {
        let values = vec![0.1, 0.2, -0.3];
        assert_eq!(checksum_f32(&values), checksum_f32(&values));
    }
}
