// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! A measurement that was not taken, and why.
//!
//! Every metric in this crate returns `Option<f64>`, and a bare `None` cannot
//! say which of several reasons produced it: too large to compute, mathematically
//! undefined for this graph, or carried over from an earlier version of the graph
//! and never recomputed. A `None` beside a [`SkippedMetric`] can.
//!
//! The type is here rather than in each consumer because three crates were
//! already writing the same two strings into their own copy of it.

use serde::{Deserialize, Serialize};

/// A metric that is `None`, and the reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedMetric {
    /// Field name that is `null` as a result.
    pub metric: String,
    /// Why it could not be computed.
    pub reason: String,
}

impl SkippedMetric {
    /// Names a metric that was skipped, with its reason.
    #[must_use]
    pub fn new(metric: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            metric: metric.into(),
            reason: reason.into(),
        }
    }

    /// A metric whose graph is larger than the threshold its algorithm can afford.
    #[must_use]
    pub fn too_large(metric: impl Into<String>, n_nodes: usize, threshold: usize) -> Self {
        let metric = metric.into();
        Self {
            reason: format!(
                "{n_nodes} nodes exceeds the {threshold} threshold for {metric}; triangle \
                 counting is quadratic in a node's degree and would dominate the run"
            ),
            metric,
        }
    }

    /// A metric the graph gives no basis for — no edges, no labels, no variance.
    #[must_use]
    pub fn undefined(metric: impl Into<String>, because: impl Into<String>) -> Self {
        Self::new(metric, format!("undefined here: {}", because.into()))
    }

    /// A metric carried over from before an operation that invalidated it.
    ///
    /// **A value not recomputed is not a value.** A figure copied across a
    /// perturbation looks exactly like a fresh one, and comparing a perturbed
    /// graph to its original then returns an agreement that was copied rather
    /// than measured.
    #[must_use]
    pub fn stale(metric: impl Into<String>, after: impl Into<String>) -> Self {
        Self::new(
            metric,
            format!(
                "not recomputed after {}; the previous value would look fresh and would not be",
                after.into()
            ),
        )
    }
}
