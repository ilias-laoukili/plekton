// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! A graph that forms while the capture is still running.
//!
//! # A live view is NOT an artifact, and nothing here may look like one
//!
//! Every artifact in this repository is byte-reproducible: the same input gives the same bytes,
//! which is what makes signing one mean something. **A stream is not.** What a live view shows
//! depends on when somebody looked at it, so it is not signable, not evidence, and carries no
//! `schema` field — a test fails if one appears.
//!
//! Proof of a live session comes from a **fixed window replayed from a recorded stream**: same
//! input, same output, deterministic again. That is why the window below is *tumbling*.
//!
//! # Tumbling, not sliding, and the first reason is the one that decides it
//!
//! 1. **Replay determinism.** With disjoint windows, replaying a recording reproduces exactly
//!    what was seen. With a sliding window the emitted graph depends on *when the observer
//!    looked*, so a replay does not reproduce it — which breaks the only path by which a live
//!    session can become proof.
//! 2. **Comparability means independence.** Consecutive sliding readings share most of their
//!    data and are autocorrelated; a viewer watching a smooth curve reads a trend into the same
//!    data recounted.
//! 3. **One thing to explain.** A tumbling window and the alternative below compute the *same*
//!    numbers.
//!
//! Fixed tumbling windows turn a live event stream into comparable snapshots
//! rather than one unbounded cumulative graph.
//!
//! # Why a cumulative view of BLE grows without bound in an empty room
//!
//! On one real capture, **723 of 783 identities rotated by design**. The same phones mint new
//! addresses forever, so a cumulative node count rises with nothing changing in the room. The
//! window is the remedy, and the rest has to be said out loud: on a live view **the node count
//! measures address rotation as much as population**, and the window decides which dominates.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    graph::{build_bounded, SkippedMetric, StructuredGraph},
    model::{AdapterError, Entity, Extraction, Limits, Relation},
};

/// Largest number of relations one window will hold.
///
/// The `complement.rs` pattern: **project, refuse with the right remedy, then allocate.** The
/// remedy is *narrow the window* and never *raise the ceiling* — raising it makes the next
/// window bigger, which is the opposite of what a budget is for. This repository carries the
/// record of a 120 GB projection taking out a 26 GB machine.
pub const MAX_WINDOW_RELATIONS: usize = 500_000;

/// Below this many entities, whole-graph structure is not interpretable.
///
/// **A resolution floor, pre-registered here rather than chosen when a number disappoints.** It
/// joins `min_calibrating_host_hours` and the generator-fidelity `corpus_floor`. Transitivity
/// over five nodes and assortativity over four edges are arithmetic, not measurements, and a
/// live view redrawing every few seconds is exactly where a reader would watch them jump and
/// read meaning into it.
pub const MIN_WINDOW_ENTITIES: usize = 30;

/// Below this many relations, the same.
pub const MIN_WINDOW_RELATIONS: usize = 30;

/// Why a live window could not be produced.
#[derive(Debug)]
pub enum LiveError {
    /// One window projected past [`MAX_WINDOW_RELATIONS`].
    WindowBudget {
        /// Relations the window had reached.
        projected: usize,
        /// The ceiling.
        budget: usize,
        /// The window that was too wide, in seconds.
        window_s: f64,
    },
    /// The graph could not be built from a window that fit.
    Build(AdapterError),
}

impl std::fmt::Display for LiveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WindowBudget {
                projected,
                budget,
                window_s,
            } => write!(
                formatter,
                "a {window_s}s window reached {projected} relation(s), above the {budget} \
                 ceiling. NARROW THE WINDOW rather than raising the ceiling: raising it makes \
                 the next window bigger, which is the opposite of what the budget is for."
            ),
            Self::Build(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for LiveError {}

/// One observation's contribution, with when the source said it happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timed {
    /// Nanoseconds on whatever wall clock the source stated. Never a clock this process read.
    pub at_nanos: i64,
    /// Entities this observation introduced.
    pub entities: Vec<Entity>,
    /// Relations this observation supports.
    pub relations: Vec<Relation>,
}

/// A window that closed, and everything a reader needs to compare it with the next one.
///
/// **`window_seconds` sits beside every number in this object on purpose.** A degree, a
/// component count or a homophily without its window is a number confused with elapsed time,
/// and putting the window in a reservation somewhere else in the document is how that happens.
#[derive(Debug, Clone, Serialize)]
pub struct Closed {
    /// Always `true`, and there is deliberately **no `schema` field** beside it. A live view is
    /// not an artifact and must not be mistaken for one.
    pub live: bool,
    /// The window's width, in seconds.
    pub window_seconds: f64,
    /// Which window this is, counted from the first observation. Windows are disjoint, so this
    /// is a sample index and successive ones are independent.
    pub window_index: u64,
    /// The window's inclusive start, on the source's clock, in nanoseconds.
    pub window_start_nanos: i64,
    /// The graph this window supports.
    pub graph: StructuredGraph,
    /// Observations that reached this window and were not used, counted by reason.
    ///
    /// **Never silent.** If decoding outruns the graph, something gives; whatever gives, what is
    /// dropped is counted and emitted. Observations vanishing under load is a defect this
    /// repository has now recorded three times.
    pub dropped: BTreeMap<String, usize>,
}

/// A tumbling window over a stream of observations.
#[derive(Debug, Clone)]
pub struct LiveWindow {
    seconds: f64,
    window_nanos: i64,
    top_hubs: usize,
    limits: Limits,
    start: Option<i64>,
    index: u64,
    pending: Extraction,
    dropped: BTreeMap<String, usize>,
}

impl LiveWindow {
    /// A window `seconds` wide.
    ///
    /// # Panics
    /// Never. A non-finite or non-positive width is clamped to one second, because a live view
    /// that refuses to start is less useful than one that tells you its width.
    #[must_use]
    pub fn new(seconds: f64, top_hubs: usize, limits: Limits) -> Self {
        let seconds = if seconds.is_finite() && seconds > 0.0 {
            seconds
        } else {
            1.0
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a window in seconds is far inside i64 nanoseconds, and the truncation is \
                      sub-nanosecond"
        )]
        let window_nanos = (seconds * 1e9) as i64;
        Self {
            seconds,
            window_nanos: window_nanos.max(1),
            top_hubs,
            limits,
            start: None,
            index: 0,
            pending: Extraction::default(),
            dropped: BTreeMap::new(),
        }
    }

    /// Record that one observation was not used, and why.
    fn drop_one(&mut self, reason: &str) {
        *self.dropped.entry(reason.to_owned()).or_insert(0) += 1;
    }

    /// Record something the **caller** could not turn into an observation at all.
    ///
    /// A window can only count what reaches it, and a producer that filters before calling
    /// [`Self::observe`] makes its losses invisible to the account that exists to hold them —
    /// which is exactly what `radio-live` did with relations carrying no readable timestamp. It
    /// lands in the same `dropped` map as the window's own drops, because a reader wants one
    /// account of what did not make it, not two.
    pub fn drop_unusable(&mut self, reason: &str) {
        self.drop_one(reason);
    }

    /// Fold one observation in, closing the current window first if this one is past it.
    ///
    /// # Errors
    /// [`LiveError::WindowBudget`] when one window exceeds [`MAX_WINDOW_RELATIONS`], and
    /// [`LiveError::Build`] if the graph cannot be built from a window that fit.
    pub fn observe(&mut self, timed: Timed) -> Result<Option<Closed>, LiveError> {
        let Some(start) = self.start else {
            self.start = Some(timed.at_nanos);
            self.absorb(timed)?;
            return Ok(None);
        };

        if timed.at_nanos < start {
            // Earlier than the window that is already open. Counted rather than folded in: a
            // late arrival would land in a window it did not happen in, and silently.
            self.drop_one("an observation older than the open window");
            return Ok(None);
        }

        if timed.at_nanos < start.saturating_add(self.window_nanos) {
            return self.absorb(timed).map(|()| None);
        }

        // Past the window. Close it, then open the one this observation belongs to — skipping
        // any empty windows between, since a window with nothing in it is not a sample.
        let closed = self.close_current(start)?;
        self.start = Some(timed.at_nanos);
        self.absorb(timed)?;
        Ok(Some(closed))
    }

    /// Add to the open window, refusing before allocating past the budget.
    fn absorb(&mut self, timed: Timed) -> Result<(), LiveError> {
        // PROJECT FIRST. Nothing below this line grows the window past the ceiling.
        let projected = self.pending.relations.len() + timed.relations.len();
        if projected > MAX_WINDOW_RELATIONS {
            return Err(LiveError::WindowBudget {
                projected,
                budget: MAX_WINDOW_RELATIONS,
                window_s: self.seconds,
            });
        }
        self.pending.entities.extend(timed.entities);
        self.pending.relations.extend(timed.relations);
        self.pending.records_parsed += 1;
        Ok(())
    }

    /// Close the open window and hand back its graph.
    fn close_current(&mut self, start: i64) -> Result<Closed, LiveError> {
        let extraction = std::mem::take(&mut self.pending);
        let entities = extraction.entities.len();
        let relations = extraction.relations.len();
        let mut graph =
            build_bounded(extraction, self.top_hubs, self.limits).map_err(LiveError::Build)?;

        // **Below the floor, whole-graph structure is withheld with its reason** rather than
        // computed over a handful of nodes and rendered as if it meant something.
        if entities < MIN_WINDOW_ENTITIES || relations < MIN_WINDOW_RELATIONS {
            graph.metrics.transitivity = None;
            graph.metrics.degree_assortativity = None;
            graph.metrics.clustering_coefficient = None;
            graph.metrics.metrics_skipped.push(SkippedMetric::new(
                "transitivity, degree_assortativity, clustering_coefficient",
                format!(
                    "this window holds {entities} entit(ies) and {relations} relation(s), \
                     below the {MIN_WINDOW_ENTITIES}/{MIN_WINDOW_RELATIONS} floor at which \
                     whole-graph structure is interpretable. Widen the window rather than \
                     reading these"
                ),
            ));
        }

        let closed = Closed {
            live: true,
            window_seconds: self.seconds,
            window_index: self.index,
            window_start_nanos: start,
            graph,
            dropped: std::mem::take(&mut self.dropped),
        };
        self.index += 1;
        Ok(closed)
    }

    /// Close whatever is open, at the end of a stream.
    ///
    /// # Errors
    /// As [`Self::observe`].
    pub fn finish(&mut self) -> Result<Option<Closed>, LiveError> {
        let Some(start) = self.start.take() else {
            return Ok(None);
        };
        if self.pending.entities.is_empty() && self.pending.relations.is_empty() {
            return Ok(None);
        }
        self.close_current(start).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::{Closed, LiveError, LiveWindow, Timed, MAX_WINDOW_RELATIONS, MIN_WINDOW_ENTITIES};
    use crate::model::{Entity, Limits, Relation};

    fn timed(at_seconds: f64, pairs: &[(&str, &str)]) -> Timed {
        Timed {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "test timestamps are small integers of seconds"
            )]
            at_nanos: (at_seconds * 1e9) as i64,
            entities: pairs
                .iter()
                .flat_map(|(a, b)| [Entity::new(*a, "host"), Entity::new(*b, "host")])
                .collect(),
            relations: pairs
                .iter()
                .map(|(a, b)| Relation::new(*a, *b, "talks_to"))
                .collect(),
        }
    }

    /// A stream wide enough to fill several windows and clear the resolution floor.
    fn stream() -> Vec<Timed> {
        (0..120)
            .map(|step| {
                let left = format!("host:{}", step % 40);
                let right = format!("host:{}", (step * 7) % 40);
                let at = f64::from(step) * 0.25;
                timed(at, &[(&left, &right)])
            })
            .collect()
    }

    fn replay(stream: &[Timed], seconds: f64) -> Vec<Closed> {
        let mut window = LiveWindow::new(seconds, 5, Limits::default());
        let mut out = Vec::new();
        for observation in stream {
            if let Ok(Some(closed)) = window.observe(observation.clone()) {
                out.push(closed);
            }
        }
        if let Ok(Some(closed)) = window.finish() {
            out.push(closed);
        }
        out
    }

    // ── The property the whole design rests on ───────────────────────────────

    /// **A recorded stream replayed twice gives byte-identical windows.**
    ///
    /// This is why the window tumbles rather than slides. With a sliding window the emitted
    /// graph depends on when the observer looked, so a replay would not reproduce what was
    /// seen — and a live session could never become proof. If this fails, the design is wrong
    /// rather than the test.
    #[test]
    fn replaying_a_recorded_stream_reproduces_it_byte_for_byte() -> Result<(), serde_json::Error> {
        let recorded = stream();
        let first = serde_json::to_string(&replay(&recorded, 5.0))?;
        let second = serde_json::to_string(&replay(&recorded, 5.0))?;
        assert_eq!(first, second);
        Ok(())
    }

    /// **Windows are disjoint, so two readings are independent samples.**
    #[test]
    fn windows_do_not_overlap() {
        let closed = replay(&stream(), 5.0);
        assert!(closed.len() >= 2, "the stream should fill several windows");
        for pair in closed.windows(2) {
            let [earlier, later] = pair else { continue };
            assert!(
                later.window_start_nanos >= earlier.window_start_nanos + 5_000_000_000,
                "a later window starts at or after the previous one ended",
            );
            assert_eq!(later.window_index, earlier.window_index + 1);
        }
    }

    // ── A live view is not an artifact ───────────────────────────────────────

    /// **No `schema` field, and `live: true` on every window.**
    #[test]
    fn nothing_in_a_live_window_looks_like_an_artifact() -> Result<(), serde_json::Error> {
        for closed in replay(&stream(), 5.0) {
            let rendered = serde_json::to_value(&closed)?;
            assert_eq!(
                rendered.get("live").and_then(serde_json::Value::as_bool),
                Some(true)
            );
            assert!(
                rendered.get("schema").is_none(),
                "a stream depends on when somebody looked; a schema field invites signing one",
            );
        }
        Ok(())
    }

    /// **The window travels beside every number, not in a reservation elsewhere.**
    #[test]
    fn every_window_carries_its_own_width() -> Result<(), serde_json::Error> {
        for closed in replay(&stream(), 2.5) {
            let rendered = serde_json::to_value(&closed)?;
            assert_eq!(
                rendered
                    .get("window_seconds")
                    .and_then(serde_json::Value::as_f64),
                Some(2.5),
                "a degree without its window is a number confused with elapsed time",
            );
        }
        Ok(())
    }

    // ── The resolution floor ─────────────────────────────────────────────────

    /// **Below the floor, whole-graph structure is withheld with its reason.**
    #[test]
    fn a_window_too_small_to_measure_says_so_rather_than_answering() {
        // A quarter-second window holds one observation.
        let closed = replay(&stream(), 0.25);
        let Some(first) = closed.first() else {
            panic!("the stream should have closed at least one window")
        };
        assert!(first.graph.metrics.n_entities < MIN_WINDOW_ENTITIES);
        assert_eq!(first.graph.metrics.transitivity, None);
        assert_eq!(first.graph.metrics.degree_assortativity, None);
        assert!(
            first
                .graph
                .metrics
                .metrics_skipped
                .iter()
                .any(|skipped| skipped.reason.contains("below the")),
            "a transitivity over three nodes is arithmetic, not a measurement",
        );
    }

    /// And above it, they are answered.
    #[test]
    fn a_window_wide_enough_reports_whole_graph_structure() {
        let closed = replay(&stream(), 30.0);
        let Some(first) = closed.first() else {
            panic!("a thirty-second window should close")
        };
        assert!(first.graph.metrics.n_entities >= MIN_WINDOW_ENTITIES);
        assert!(
            first.graph.metrics.degree_assortativity.is_some()
                || first
                    .graph
                    .metrics
                    .metrics_skipped
                    .iter()
                    .any(|s| s.metric.contains("degree_assortativity")),
            "either a number or a stated reason, never a bare null",
        );
    }

    // ── Loss is counted, never silent ────────────────────────────────────────

    /// **An observation older than the open window is counted, not folded in.**
    #[test]
    fn a_late_arrival_is_counted_rather_than_landing_in_the_wrong_window() {
        let mut window = LiveWindow::new(5.0, 5, Limits::default());
        let _ = window.observe(timed(100.0, &[("a", "b")]));
        // Arrives after, but happened before the open window began.
        let _ = window.observe(timed(1.0, &[("c", "d")]));
        let Ok(Some(closed)) = window.finish() else {
            panic!("the open window should close")
        };
        assert_eq!(
            closed.dropped.values().sum::<usize>(),
            1,
            "observations vanishing under load is a defect this repository has recorded three \
             times; the count is what stops a fourth",
        );
        assert!(closed
            .dropped
            .keys()
            .any(|reason| reason.contains("older than the open window")));
    }

    // ── The budget refuses the run rather than trimming it ───────────────────

    #[test]
    fn a_window_past_the_budget_refuses_and_says_narrow_the_window() {
        let mut window = LiveWindow::new(3600.0, 5, Limits::default());
        let flood = Timed {
            at_nanos: 0,
            entities: Vec::new(),
            relations: (0..=MAX_WINDOW_RELATIONS)
                .map(|index| Relation::new("a", format!("b{index}"), "talks_to"))
                .collect(),
        };
        let Err(error) = window.observe(flood) else {
            panic!("a window past the ceiling must refuse")
        };
        let text = error.to_string();
        assert!(text.contains("NARROW THE WINDOW"), "{text}");
        assert!(
            !text.to_lowercase().contains("raise the budget")
                && !text.to_lowercase().contains("increase the"),
            "raising the ceiling makes the next window bigger: {text}",
        );
        assert!(matches!(error, LiveError::WindowBudget { .. }));
    }

    /// An empty stream closes nothing rather than emitting an empty window.
    #[test]
    fn a_stream_with_no_observations_emits_no_window() {
        let mut window = LiveWindow::new(5.0, 5, Limits::default());
        assert!(matches!(window.finish(), Ok(None)));
    }
}
