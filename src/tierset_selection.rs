#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TierId(pub u16);

#[derive(Debug, Clone, Default)]
pub struct TierSet {
    pub tiers: Vec<TierId>,
    pub cumulative: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionMode {
    Fast,
    Balanced,
    Deep,
    Max,
}

#[derive(Debug, Clone)]
pub struct TierSetSelector {
    pub fast: TierSet,
    pub balanced: TierSet,
    pub deep: TierSet,
    pub max: TierSet,
}

impl TierSetSelector {
    pub fn select(&self, mode: SelectionMode) -> TierSet {
        match mode {
            SelectionMode::Fast => self.fast.clone(),
            SelectionMode::Balanced => self.balanced.clone(),
            SelectionMode::Deep => self.deep.clone(),
            SelectionMode::Max => self.max.clone(),
        }
    }
}
