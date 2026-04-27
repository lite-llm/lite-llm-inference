use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    InvalidConfig(&'static str),
    InvalidInput(&'static str),
    InvalidState(&'static str),
    ParseError(&'static str),
    BudgetUnsatisfied(&'static str),
    TenantViolation(String),
    Throttled(String),
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    IoError(String),
}

pub type InferenceResult<T> = Result<T, InferenceError>;

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Self::InvalidState(msg) => write!(f, "invalid state: {msg}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::BudgetUnsatisfied(msg) => write!(f, "budget unsatisfied: {msg}"),
            Self::TenantViolation(msg) => write!(f, "tenant isolation violation: {msg}"),
            Self::Throttled(msg) => write!(f, "request throttled: {msg}"),
            Self::ChecksumMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "checksum mismatch for {path}: expected {expected}, got {actual}"
            ),
            Self::IoError(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl Error for InferenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<std::io::Error> for InferenceError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value.to_string())
    }
}
