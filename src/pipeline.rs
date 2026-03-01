#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

pub trait InferencePipeline {
    fn run_step(&self, stage: PipelineStage) -> Result<(), &'static str>;
}

#[derive(Debug, Default)]
pub struct NoopPipeline;

impl InferencePipeline for NoopPipeline {
    fn run_step(&self, _stage: PipelineStage) -> Result<(), &'static str> {
        Ok(())
    }
}

