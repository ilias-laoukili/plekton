// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
use thiserror::Error;

/// All errors that can be produced by `plekton` operations.
#[derive(Debug, Error)]
pub enum SynthGraphError {
    /// A parameter value was out of range, NaN, wrong size, or otherwise invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Graph generation failed due to a runtime condition (e.g. distribution initialisation).
    #[error("generation failed: {0}")]
    GenerationFailed(String),

    /// Node feature vectors were inconsistent (missing on some nodes, wrong dimension, etc.).
    #[error("feature error: {0}")]
    Feature(String),

    /// JSON serialisation or deserialisation failed.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// File I/O failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
