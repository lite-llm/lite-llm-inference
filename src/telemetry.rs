#[derive(Debug, Clone)]
pub struct TelemetryEvent {
    pub name: String,
    pub value: f64,
    pub step: u64,
}

pub trait TelemetryCollector {
    fn record(&mut self, event: TelemetryEvent);
}

#[derive(Debug, Default)]
pub struct InMemoryTelemetry {
    events: Vec<TelemetryEvent>,
}

impl TelemetryCollector for InMemoryTelemetry {
    fn record(&mut self, event: TelemetryEvent) {
        self.events.push(event);
    }
}

