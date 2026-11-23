// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Cross-domain edges from timestamps, and the two ways this refuses to draw one.
//!
//! # The reservation this implements against
//!
//! `docs/internal/cross-domain-graphs.md` names the co-occurrence join and refuses it:
//! *"never, on co-occurrence alone — this is the seductive one and it is fabrication."* The
//! reservation was raised, the choice was reaffirmed, and this implements it as asked: **the
//! edge is emitted every time, with no blocking threshold.** The note is amended in the same
//! batch, because normative prose contradicting shipped code leaves nobody able to say which
//! governs.
//!
//! What the edge means is therefore carried by its own `kind` and by [`co_occurrence`]'s
//! output, not by a filter:
//!
//! * `co_occurs_with` — two things from different domains were observed inside one window. That
//!   is **all** it says. In a busy window everything co-occurs with everything.
//! * `same_identifier` — two domains minted the same value. Stronger, and still not identity:
//!   two sources agreeing on `192.0.2.1` is one address, not necessarily one physical entity.
//!
//! Both are emitted. Distinguishing them is data modelling; suppressing one would be a filter
//! wearing a schema.
//!
//! # Two refusals, and neither is a threshold
//!
//! **An incomparable time base.** Five frontiers write four things into `at`: two wall-clock
//! formats, seconds since a recording began, microseconds since a sniffer booted. Joining them
//! as one scale would place a burst 12 seconds into a recording beside a packet from 1970. The
//! artifact now states its base and this refuses the run-relative ones by name — they are not
//! convertible, because the wall clock the run started at is not in the file.
//!
//! **An unqualified id.** The core treats equal ids as the same thing, so a bare `0x0201` from
//! CAN and from Zigbee would already have merged before this ran. See [`crate::namespace`].
//!
//! # The budget, which is the `complement.rs` pattern and not a new one
//!
//! A windowed join is quadratic in window density, and this repository carries the record of a
//! 120 GB projection taking out a 26 GB machine. So: **project the edge count first, refuse
//! above budget with the right remedy, then allocate once.** The remedy is *narrow the window*
//! and never *raise the cap* — raising it makes the next run bigger, which is the opposite of
//! what a budget is for.

use std::collections::BTreeMap;

use crate::{
    model::{Extraction, Relation},
    namespace::parse,
};

/// The attribute an adapter puts the source's own timestamp in.
pub const AT: &str = "at";
/// The attribute an adapter puts the contract's time base in.
pub const TIME_BASE: &str = "time_base";

/// How many cross-domain edges a join may produce before it refuses.
///
/// Deliberately not a tuning knob with a nice default: it is a ceiling, and the action when it
/// trips is to narrow the window.
pub const MAX_EDGES: usize = 5_000_000;

/// How many examples a report keeps, matching [`crate::graph`]'s cap on the same kind of list.
const MAX_EXAMPLES: usize = 10;

/// What a join emitted, and what it never looked at.
///
/// # Why two counts of the same edges
///
/// [`co_occurrence`] emits one edge per pair of **observations**, not per pair of entities: two
/// devices heard seventy-five times in the same second become seventy-five edges. That is a real
/// answer to a real question — *how many coincidences* — and de-duplicating would be a filter, so
/// the multiplicity stays. What was missing is that the output never said which question it had
/// answered, and `282 344 co-occurrence edges` reads as 282 344 pairs to anyone deriving a
/// density from it. So both numbers are reported and neither has to be inferred from the other.
///
/// # Why a count of what was not observed
///
/// `observations` reads the **destination** endpoint of every relation and nothing else. For the
/// probe adapter's star that is right — a spoke's `dst` *is* the identity and the `src` is the
/// probe — but for a host-to-host relation it means an entity that only ever initiates is
/// invisible to every cross-domain join. That scope is kept, and it is now **counted**, which is
/// the treatment [`crate::graph::build_bounded`] already gives a relation it has to drop.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JoinReport {
    /// Edges emitted — one per pair of observations.
    pub edges: usize,
    /// Distinct entity pairs those edges cover.
    ///
    /// Never larger than `edges`, and bounded above by the number of inter-domain pairs the
    /// observations can form at all. Comparing an emitted count against that ceiling is what
    /// found the defect this field exists to answer.
    pub distinct_pairs: usize,
    /// Entities that appear only as a relation's source, and so never became an observation.
    ///
    /// Not an error and not a drop: these entities are in the graph, they are simply outside
    /// what a join keyed on destinations can reach. A caller that sees a large number here is
    /// looking at a host-to-host medium, and half its domain did not participate.
    pub sources_never_observed: usize,
    /// Up to [`MAX_EXAMPLES`] of them, so the number can be acted on rather than only read.
    pub source_examples: Vec<String>,
}

/// Why a join refused.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinError {
    /// A domain's timestamps are counted from the start of its own run.
    IncomparableTimeBase {
        /// Which domain.
        domain: String,
        /// What its base is.
        base: String,
    },
    /// A domain's artifact does not say what its timestamps are counted from.
    ///
    /// **Kept apart from [`Self::IncomparableTimeBase`]**, which the first form folded it into.
    /// A run-relative base is a stated fact with a known consequence; an unstated one is an
    /// artifact written before the field existed, or by something that does not declare it, and
    /// the two license opposite actions — join the others, versus re-run the probe.
    UnstatedTimeBase {
        /// Which domain.
        domain: String,
    },
    /// An id carried no domain, so a cross-domain join cannot tell what it belongs to — and the
    /// core would already have merged it with anything else of the same value.
    UnqualifiedId {
        /// The id.
        id: String,
    },
    /// The projection exceeded the budget. Nothing was allocated.
    EdgeBudget {
        /// How many edges the window would produce.
        projected: usize,
        /// The ceiling.
        budget: usize,
        /// The window that produced it, in seconds.
        window_s: f64,
    },
    /// A timestamp did not parse on the base its contract declared.
    Unparseable {
        /// What was written.
        at: String,
        /// What it was read as.
        base: String,
    },
}

impl std::fmt::Display for JoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::IncomparableTimeBase {
                ref domain,
                ref base,
            } => write!(
                f,
                "`{domain}` stamps its observations on a `{base}` base, which is counted from \
                 the start of its own run. Nothing converts it to a wall clock: the artifact \
                 does not carry the time that run began. Join the wall-clock domains, or \
                 produce an artifact for `{domain}` that states one."
            ),
            Self::UnstatedTimeBase { ref domain } => write!(
                f,
                "`{domain}`'s artifact does not say what its timestamps are counted from. That \
                 is not the same as a base this cannot use: it is a file written before the \
                 contract carried one, or by something that does not declare it. Re-run the \
                 probe against a build that states its `time_base`."
            ),
            Self::UnqualifiedId { ref id } => write!(
                f,
                "`{id}` carries no domain. A cross-domain join cannot tell what it belongs to, \
                 and the graph core treats equal ids as the same thing — so an unqualified id \
                 has already merged with anything sharing its value. Namespace it as \
                 `<domain>:<kind>:<value>`."
            ),
            Self::EdgeBudget {
                projected,
                budget,
                window_s,
            } => write!(
                f,
                "a {window_s}s window projects {projected} cross-domain edges, above the \
                 {budget} budget, and nothing has been allocated. NARROW THE WINDOW: a windowed \
                 join is quadratic in window density, so halving it does far more than halve \
                 this. Raising the ceiling makes the next run bigger, which is the opposite of \
                 what the budget is for."
            ),
            Self::Unparseable { ref at, ref base } => write!(
                f,
                "`{at}` does not parse as `{base}`, which is the base its contract declares. \
                 The contract and the artifact disagree, and this will not guess which is right."
            ),
        }
    }
}

impl std::error::Error for JoinError {}

/// One observation, reduced to what a join needs.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Observed {
    /// Nanoseconds since the Unix epoch.
    at_nanos: i64,
    /// Which domain minted the id.
    domain: String,
    /// The entity.
    id: String,
    /// The identity's own value, for `same_identifier`.
    value: String,
}

/// Read one timestamp on the base its contract declared.
fn instant(at: &str, base: &str) -> Result<i64, JoinError> {
    let unparseable = || JoinError::Unparseable {
        at: at.to_owned(),
        base: base.to_owned(),
    };
    match base {
        "rfc3339" => chrono::DateTime::parse_from_rfc3339(at)
            .ok()
            .and_then(|stamp| stamp.timestamp_nanos_opt())
            .ok_or_else(unparseable),
        "unix_seconds" => {
            let seconds: f64 = at.parse().map_err(|_| unparseable())?;
            if !seconds.is_finite() {
                return Err(unparseable());
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a wall clock in seconds is far inside i64 nanoseconds, and the \
                          truncation is sub-nanosecond"
            )]
            Ok((seconds * 1e9) as i64)
        }
        _ => Err(JoinError::IncomparableTimeBase {
            domain: String::new(),
            base: base.to_owned(),
        }),
    }
}

/// What [`observations`] produced, and what it passed over.
struct Reduced {
    /// The observations, sorted by instant then id.
    rows: Vec<Observed>,
    /// Ids that appeared only as a source endpoint.
    sources_never_observed: Vec<String>,
}

/// Reduce an extraction to the observations a join can use, refusing what it cannot.
///
/// Only the **destination** endpoint becomes an observation. The sources are collected rather
/// than ignored: whatever is left over after the destinations are known is an entity this join
/// cannot reach, and it is returned to be counted.
fn observations(extraction: &Extraction) -> Result<Reduced, JoinError> {
    let mut out = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    for relation in &extraction.relations {
        let (Some(at), Some(base)) = (
            relation.attrs.get(AT).and_then(serde_json::Value::as_str),
            relation
                .attrs
                .get(TIME_BASE)
                .and_then(serde_json::Value::as_str),
        ) else {
            continue;
        };
        sources.push(relation.src.clone());
        let Some(qualified) = parse(&relation.dst) else {
            return Err(JoinError::UnqualifiedId {
                id: relation.dst.clone(),
            });
        };
        match base {
            "rfc3339" | "unix_seconds" => {}
            "unstated" => {
                return Err(JoinError::UnstatedTimeBase {
                    domain: qualified.domain.to_owned(),
                })
            }
            other => {
                return Err(JoinError::IncomparableTimeBase {
                    domain: qualified.domain.to_owned(),
                    base: other.to_owned(),
                })
            }
        }
        out.push(Observed {
            at_nanos: instant(at, base)?,
            domain: qualified.domain.to_owned(),
            id: relation.dst.clone(),
            value: qualified.value.to_owned(),
        });
    }
    out.sort_by(|a, b| a.at_nanos.cmp(&b.at_nanos).then_with(|| a.id.cmp(&b.id)));

    // A source that also appears as some destination *is* observed, through that other relation.
    // Only what is left over is unreachable, which is why this is computed once at the end
    // rather than per relation.
    let observed: std::collections::BTreeSet<&str> =
        out.iter().map(|row| row.id.as_str()).collect();
    let mut unreachable: Vec<String> = sources
        .into_iter()
        .filter(|source| !observed.contains(source.as_str()))
        .collect();
    unreachable.sort_unstable();
    unreachable.dedup();

    Ok(Reduced {
        rows: out,
        sources_never_observed: unreachable,
    })
}

/// Count the cross-domain pairs a sliding window would produce, allocating nothing.
///
/// Two pointers over the sorted observations. Per window the count is
/// `(total² − Σ per-domain²) / 2`, which is the number of pairs that are not same-domain — kept
/// as a running tally rather than by enumerating, because enumerating is the thing being
/// budgeted for.
fn project(sorted: &[Observed], window_nanos: i64) -> usize {
    let mut projected = 0usize;
    let mut left = 0usize;
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut total = 0usize;

    for right in 0..sorted.len() {
        // **Evict first, then count.** The first form did the reverse and over-counted by
        // whatever was still in the window from beyond its own boundary — a projection larger
        // than the allocation, which is a budget guarding a number nobody produces.
        // `the_projection_is_exactly_what_gets_allocated` is what said so.
        while sorted[right].at_nanos.saturating_sub(sorted[left].at_nanos) > window_nanos {
            if let Some(count) = counts.get_mut(sorted[left].domain.as_str()) {
                *count = count.saturating_sub(1);
            }
            total = total.saturating_sub(1);
            left += 1;
        }

        let same_domain = counts
            .get(sorted[right].domain.as_str())
            .copied()
            .unwrap_or(0);
        projected = projected.saturating_add(total.saturating_sub(same_domain));
        *counts.entry(sorted[right].domain.as_str()).or_insert(0) += 1;
        total += 1;
    }
    projected
}

/// Emit a `co_occurs_with` edge for every cross-domain pair inside `window_s`, plus a
/// `same_identifier` edge wherever two domains minted the same value.
///
/// **No threshold blocks an edge.** The budget refuses the whole run rather than trimming it:
/// a join that silently dropped the edges past a ceiling would report a sparse graph as a
/// complete one.
///
/// # Errors
///
/// [`JoinError::IncomparableTimeBase`] for a run-relative base, [`JoinError::UnqualifiedId`] for
/// an id with no domain, [`JoinError::Unparseable`] when a timestamp does not match its declared
/// base, and [`JoinError::EdgeBudget`] when the projection exceeds `budget` — in which case
/// nothing has been allocated.
pub fn co_occurrence(
    extraction: &Extraction,
    window_s: f64,
    budget: usize,
) -> Result<(Vec<Relation>, JoinReport), JoinError> {
    let reduced = observations(extraction)?;
    let unreachable = reduced.sources_never_observed;
    let sorted = reduced.rows;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a join window is seconds, far inside i64 nanoseconds"
    )]
    let window_nanos = (window_s.max(0.0) * 1e9) as i64;

    // PROJECT FIRST. Nothing below this line allocates per edge.
    let projected = project(&sorted, window_nanos);
    if projected > budget {
        return Err(JoinError::EdgeBudget {
            projected,
            budget,
            window_s,
        });
    }

    // Then allocate once.
    let mut edges = Vec::with_capacity(projected);
    // Borrowed from `sorted`, so a pair costs a set node and no string. The set is bounded by the
    // edge count, which the budget above has already refused if it is too large.
    let mut pairs: std::collections::BTreeSet<(&str, &str)> = std::collections::BTreeSet::new();
    let mut left = 0usize;
    for right in 0..sorted.len() {
        while sorted[right].at_nanos.saturating_sub(sorted[left].at_nanos) > window_nanos {
            left += 1;
        }
        for other in &sorted[left..right] {
            if other.domain == sorted[right].domain {
                continue;
            }
            let kind = if other.value == sorted[right].value {
                "same_identifier"
            } else {
                "co_occurs_with"
            };
            // Unordered: this graph is undirected, so A-B and B-A are one pair.
            let pair = if other.id <= sorted[right].id {
                (other.id.as_str(), sorted[right].id.as_str())
            } else {
                (sorted[right].id.as_str(), other.id.as_str())
            };
            pairs.insert(pair);
            let mut edge = Relation::new(&other.id, &sorted[right].id, kind);
            let gap_ns = sorted[right].at_nanos.saturating_sub(other.at_nanos);
            edge.attrs.insert(
                "gap_s".to_owned(),
                serde_json::json!(
                    f64::from(i32::try_from(gap_ns / 1_000_000).unwrap_or(i32::MAX)) / 1000.0
                ),
            );
            edges.push(edge);
        }
    }

    let report = JoinReport {
        edges: edges.len(),
        distinct_pairs: pairs.len(),
        sources_never_observed: unreachable.len(),
        source_examples: unreachable.into_iter().take(MAX_EXAMPLES).collect(),
    };
    Ok((edges, report))
}

/// What a co-occurrence count looks like when the timestamps carry no relationship.
///
/// # What this is for, and what it deliberately does not do
///
/// `co_occurs_with` says two things from different domains were seen inside one window. In a
/// busy window everything co-occurs with everything, so the count on its own is a statement
/// about **density** as much as about anything happening together. This measures the density
/// half by destroying the relationship and counting again.
///
/// **It blocks no edge.** Nothing here filters, and `--calibrate` is off the default path. It
/// reports a number beside the count, and what to make of the two is the reader's.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Calibration {
    /// Edges the real timestamps produced.
    pub observed: usize,
    /// How many shifted rounds were run.
    pub rounds: usize,
    /// Mean edges per round under the null.
    pub mean_under_chance: f64,
    /// The most any single round produced.
    pub max_under_chance: usize,
}

impl Calibration {
    /// Observed over mean-under-chance, or `None` when chance produced nothing.
    ///
    /// `None` rather than infinity, and rather than a large number: a ratio over a zero
    /// denominator is not a big ratio, it is a question the run cannot answer — which is
    /// `collapsed-negatives.md`'s subject exactly.
    #[must_use]
    pub fn lift(&self) -> Option<f64> {
        (self.mean_under_chance > 0.0).then(|| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "an edge count above 2^53 has already failed the budget"
            )]
            let observed = self.observed as f64;
            observed / self.mean_under_chance
        })
    }
}

/// Count co-occurrences again with the relationship between domains destroyed.
///
/// **Circular shift, and deterministic.** Every domain but the first has its timestamps rotated
/// forward by an offset and wrapped back into the observed span, `rounds` times at evenly spaced
/// offsets. Rotating preserves each domain's own rate and burstiness — which is the point, since
/// a null model that also flattened the rate would be answering a different question — and
/// destroys only the alignment between domains.
///
/// Evenly spaced rather than random so two runs over one input agree. A null model whose number
/// moves between runs is a number nobody can quote.
///
/// # Errors
///
/// As [`co_occurrence`]. The budget is checked on the real timestamps first, so a calibration
/// never allocates more than the join it describes.
pub fn calibrate(
    extraction: &Extraction,
    window_s: f64,
    budget: usize,
    rounds: usize,
) -> Result<Calibration, JoinError> {
    let observed = co_occurrence(extraction, window_s, budget)?.0.len();
    let sorted = observations(extraction)?.rows;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a join window is seconds, far inside i64 nanoseconds"
    )]
    let window_nanos = (window_s.max(0.0) * 1e9) as i64;

    let (Some(first), Some(last)) = (sorted.first(), sorted.last()) else {
        return Ok(Calibration {
            observed,
            rounds: 0,
            mean_under_chance: 0.0,
            max_under_chance: 0,
        });
    };
    let span = last.at_nanos.saturating_sub(first.at_nanos);
    if span <= 0 || rounds == 0 {
        // Every observation shares one instant, so there is nothing to rotate: shifting a
        // single point in time by anything lands on the same point. Reported as zero rounds
        // rather than as a mean of zero, which would read as "chance produces nothing".
        return Ok(Calibration {
            observed,
            rounds: 0,
            mean_under_chance: 0.0,
            max_under_chance: 0,
        });
    }

    let anchor_domain = first.domain.clone();
    let mut total = 0usize;
    let mut max = 0usize;
    for round in 1..=rounds {
        let offset = span
            .saturating_mul(i64::try_from(round).unwrap_or(1))
            .checked_div(i64::try_from(rounds).unwrap_or(1))
            .unwrap_or(0);
        let mut shifted: Vec<Observed> = sorted
            .iter()
            .map(|observation| {
                let mut moved = observation.clone();
                if moved.domain != anchor_domain {
                    let from_start = moved.at_nanos.saturating_sub(first.at_nanos);
                    let rotated = (from_start.saturating_add(offset)) % span;
                    moved.at_nanos = first.at_nanos.saturating_add(rotated);
                }
                moved
            })
            .collect();
        shifted.sort_by(|a, b| a.at_nanos.cmp(&b.at_nanos).then_with(|| a.id.cmp(&b.id)));
        let count = project(&shifted, window_nanos);
        total = total.saturating_add(count);
        max = max.max(count);
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "counts here are bounded by the budget, far below 2^53"
    )]
    let mean = total as f64 / rounds as f64;
    Ok(Calibration {
        observed,
        rounds,
        mean_under_chance: mean,
        max_under_chance: max,
    })
}

#[cfg(test)]
mod tests {
    use super::{co_occurrence, JoinError, MAX_EDGES};
    use crate::model::{Extraction, Relation};

    /// One `observed` edge, as the probe adapter writes it.
    fn observed(domain: &str, kind: &str, value: &str, at: &str, base: &str) -> Relation {
        let id = crate::namespace::qualify(domain, kind, value).unwrap_or_default();
        let probe = crate::namespace::qualify(domain, "probe", domain).unwrap_or_default();
        let mut relation = Relation::new(probe, id, "observed");
        relation.attrs.insert("at".to_owned(), at.into());
        relation.attrs.insert("time_base".to_owned(), base.into());
        relation
    }

    fn extraction(relations: Vec<Relation>) -> Extraction {
        Extraction {
            relations,
            ..Extraction::default()
        }
    }

    #[test]
    fn two_domains_inside_the_window_get_an_edge_and_two_of_one_domain_do_not() {
        let input = extraction(vec![
            observed("ip", "host", "192.0.2.1", "2026-08-22T10:00:00Z", "rfc3339"),
            observed("ip", "host", "192.0.2.9", "2026-08-22T10:00:01Z", "rfc3339"),
            observed(
                "lorawan",
                "dev_eui",
                "0102030405060708",
                "2026-08-22T10:00:02Z",
                "rfc3339",
            ),
        ]);
        let Ok((edges, _report)) = co_occurrence(&input, 60.0, MAX_EDGES) else {
            panic!("a wall-clock join must succeed")
        };
        assert_eq!(
            edges.len(),
            2,
            "the LoRaWAN device pairs with both hosts, and the hosts do not pair with each other"
        );
        assert!(edges.iter().all(|e| e.kind == "co_occurs_with"));
    }

    /// The edge is emitted every time. No threshold blocks one.
    #[test]
    fn nothing_is_filtered_by_how_close_in_time_it_was() {
        let far = extraction(vec![
            observed("ip", "host", "192.0.2.1", "2026-08-22T10:00:00Z", "rfc3339"),
            observed(
                "can",
                "arbitration_id",
                "0x0201",
                "2026-08-22T10:00:59Z",
                "rfc3339",
            ),
        ]);
        let Ok((edges, _report)) = co_occurrence(&far, 60.0, MAX_EDGES) else {
            panic!("must succeed")
        };
        assert_eq!(
            edges.len(),
            1,
            "59 seconds apart in a 60s window is still an edge"
        );
    }

    /// The same value from two domains is a stronger claim, and still not identity.
    #[test]
    fn two_domains_minting_one_value_get_a_different_kind() {
        let input = extraction(vec![
            observed("ip", "host", "192.0.2.1", "2026-08-22T10:00:00Z", "rfc3339"),
            observed(
                "probe",
                "host",
                "192.0.2.1",
                "2026-08-22T10:00:01Z",
                "rfc3339",
            ),
        ]);
        let Ok((edges, _report)) = co_occurrence(&input, 60.0, MAX_EDGES) else {
            panic!("must succeed")
        };
        assert_eq!(edges.len(), 1);
        assert_eq!(
            edges[0].kind, "same_identifier",
            "one address is one address; it is not one device, and the kind is what says so",
        );
    }

    /// **The refusal that matters most**, because the alternative is silent nonsense.
    #[test]
    fn a_run_relative_time_base_is_refused_by_name() {
        let mixed = extraction(vec![
            observed(
                "lorawan",
                "dev_eui",
                "0102030405060708",
                "2026-08-22T10:00:00Z",
                "rfc3339",
            ),
            observed(
                "radio",
                "ble_address",
                "00:00:5E:00:53:01",
                "555017945",
                "since_run_start",
            ),
        ]);
        let Err(error) = co_occurrence(&mixed, 60.0, MAX_EDGES) else {
            panic!("microseconds since a sniffer booted are not a wall clock")
        };
        assert!(
            matches!(error, JoinError::IncomparableTimeBase { ref domain, .. } if domain == "radio")
        );
        let text = error.to_string();
        assert!(text.contains("does not carry the time that run began"));
    }

    /// Both wall-clock formats are comparable with each other.
    #[test]
    fn a_candump_stamp_and_an_rfc3339_stamp_join() {
        let input = extraction(vec![
            observed(
                "lorawan",
                "dev_eui",
                "0102030405060708",
                "2026-08-22T10:00:00Z",
                "rfc3339",
            ),
            observed(
                "can",
                "arbitration_id",
                "0x0C9",
                "1787392800.5",
                "unix_seconds",
            ),
        ]);
        // 2026-08-22T10:00:00Z is 1787392800 exactly, so these are half a second apart.
        let Ok((edges, _report)) = co_occurrence(&input, 5.0, MAX_EDGES) else {
            panic!("two wall clocks are comparable whatever their rendering")
        };
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn an_unqualified_id_is_refused_rather_than_joined() {
        let mut relation = Relation::new("probe", "192.0.2.1", "observed");
        relation
            .attrs
            .insert("at".to_owned(), "2026-08-22T10:00:00Z".into());
        relation
            .attrs
            .insert("time_base".to_owned(), "rfc3339".into());
        let Err(error) = co_occurrence(&extraction(vec![relation]), 60.0, MAX_EDGES) else {
            panic!("a bare id has already merged with anything sharing its value")
        };
        assert!(matches!(error, JoinError::UnqualifiedId { .. }));
    }

    /// **The budget: projected, refused, nothing allocated — and the remedy is the right one.**
    #[test]
    fn a_dense_window_is_refused_with_narrow_the_window_and_never_raise_the_cap() {
        let mut relations = Vec::new();
        for index in 0..200u32 {
            let domain = if index % 2 == 0 { "ip" } else { "can" };
            relations.push(observed(
                domain,
                "thing",
                &index.to_string(),
                "2026-08-22T10:00:00Z",
                "rfc3339",
            ));
        }
        // 100 × 100 = 10 000 cross-domain pairs in one instant.
        let Err(error) = co_occurrence(&extraction(relations), 60.0, 1_000) else {
            panic!("10 000 edges over a 1 000 budget must refuse")
        };
        let JoinError::EdgeBudget { projected, .. } = error else {
            panic!("the budget is what refused")
        };
        assert_eq!(projected, 10_000);

        let text = error.to_string();
        assert!(text.contains("NARROW THE WINDOW"), "{text}");
        assert!(text.contains("nothing has been allocated"), "{text}");
        assert!(
            !text.contains("raise the budget") && !text.contains("increase"),
            "the remedy must never be to raise the ceiling: {text}",
        );
    }

    /// The projection must equal what is actually produced, or the budget guards nothing.
    #[test]
    fn the_projection_is_exactly_what_gets_allocated() {
        let mut relations = Vec::new();
        for index in 0..40u32 {
            let domain = match index % 3 {
                0 => "ip",
                1 => "can",
                _ => "lorawan",
            };
            relations.push(observed(
                domain,
                "thing",
                &index.to_string(),
                &format!("2026-08-22T10:00:{:02}Z", index % 60),
                "rfc3339",
            ));
        }
        let input = extraction(relations);
        for window in [0.0, 1.0, 5.0, 3600.0] {
            let Ok((edges, _report)) = co_occurrence(&input, window, MAX_EDGES) else {
                panic!("window {window}: a large budget produces the edges")
            };
            // A budget one below the count is what makes the projection observable. A budget of
            // zero would NOT refuse an empty projection, and that is right rather than a gap:
            // nothing does not exceed nothing.
            let Some(just_under) = edges.len().checked_sub(1) else {
                assert_eq!(edges.len(), 0, "window {window}");
                continue;
            };
            let Err(JoinError::EdgeBudget { projected, .. }) =
                co_occurrence(&input, window, just_under)
            else {
                panic!("window {window}: a budget below the count must refuse and report it")
            };
            assert_eq!(
                projected,
                edges.len(),
                "window {window}: a projection that does not equal the allocation guards nothing",
            );
        }
    }

    /// Entry 63: the edge count reads as a pair count, and on real data it is not one.
    #[test]
    fn the_distinct_pair_count_is_reported_beside_the_edge_count() {
        // Two entities, one per domain, heard five times each inside one window. Every one of
        // the twenty-five cross pairs is an edge; there is exactly one pair of entities.
        let mut relations = Vec::new();
        for second in 0..5 {
            let at = format!("2026-08-22T10:00:{second:02}Z");
            relations.push(observed("ip", "host", "192.0.2.1", &at, "rfc3339"));
            relations.push(observed(
                "radio",
                "ble_address",
                "00:00:5E:00:53:01",
                &at,
                "rfc3339",
            ));
        }
        let Ok((edges, report)) = co_occurrence(&extraction(relations), 60.0, MAX_EDGES) else {
            panic!("a wall-clock join must succeed")
        };
        assert_eq!(
            report.edges,
            edges.len(),
            "the report must count what it returned"
        );
        assert!(
            report.edges > 1,
            "the fixture is meant to produce multiplicity, and produced {}",
            report.edges
        );
        assert_eq!(
            report.distinct_pairs, 1,
            "one entity per domain is ONE pair however many times they co-occurred; reporting \
             only {} edges lets a reader derive a density that is wrong by the multiplicity",
            report.edges
        );
    }

    /// `collapsed-negatives.md`, "How to catch it": read an emitted count against the ceiling its
    /// own inputs imply. That check is what found entry 63 in the first place.
    #[test]
    fn the_distinct_pair_count_never_exceeds_the_ceiling_its_own_inputs_imply() {
        let mut relations = Vec::new();
        for index in 0..4u32 {
            let at = format!("2026-08-22T10:00:{index:02}Z");
            relations.push(observed(
                "ip",
                "host",
                &format!("192.0.2.{index}"),
                &at,
                "rfc3339",
            ));
            relations.push(observed(
                "can",
                "arbitration_id",
                &format!("0x{index:03}"),
                &at,
                "rfc3339",
            ));
        }
        let Ok((_edges, report)) = co_occurrence(&extraction(relations), 60.0, MAX_EDGES) else {
            panic!("a wall-clock join must succeed")
        };
        let ceiling = 4 * 4;
        assert!(
            report.distinct_pairs <= ceiling,
            "{} distinct pairs claimed from 4 entities against 4, whose ceiling is {ceiling}",
            report.distinct_pairs,
        );
        assert!(
            report.distinct_pairs > 0,
            "eight observations across two domains cover at least one pair"
        );
    }

    /// Entry 64: the join reads destinations, and an entity that only ever initiates is invisible
    /// to it. The scope is kept; being silent about it is not.
    #[test]
    fn an_entity_that_only_ever_initiates_is_counted_rather_than_silently_unjoinable() {
        // A peer-to-peer relation: both endpoints are entities, not a collector-to-subject spoke.
        // real entities, and only one of them can ever become an observation.
        let initiator = crate::namespace::qualify("ip", "host", "192.0.2.7").unwrap_or_default();
        let responder = crate::namespace::qualify("ip", "host", "192.0.2.8").unwrap_or_default();
        let mut talks = Relation::new(&initiator, &responder, "talks_to");
        talks
            .attrs
            .insert("at".to_owned(), "2026-08-22T10:00:00Z".into());
        talks.attrs.insert("time_base".to_owned(), "rfc3339".into());

        let input = extraction(vec![
            talks,
            observed(
                "can",
                "arbitration_id",
                "0x0C9",
                "2026-08-22T10:00:01Z",
                "rfc3339",
            ),
        ]);
        let Ok((_edges, report)) = co_occurrence(&input, 60.0, MAX_EDGES) else {
            panic!("a wall-clock join must succeed")
        };
        assert!(
            report.sources_never_observed >= 1,
            "192.0.2.7 initiates and never answers, so no join can reach it — and nothing said so"
        );
        assert!(
            report.source_examples.iter().any(|id| id == &initiator),
            "the count must be actionable: {:?} names none of the entities it is about",
            report.source_examples,
        );
        assert!(
            !report.source_examples.iter().any(|id| id == &responder),
            "the responder DID become an observation and must not be reported as unreachable"
        );
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::{calibrate, co_occurrence, MAX_EDGES};
    use crate::model::{Extraction, Relation};

    fn observed(domain: &str, value: &str, at_second: u32) -> Relation {
        let id = crate::namespace::qualify(domain, "thing", value).unwrap_or_default();
        let probe = crate::namespace::qualify(domain, "probe", domain).unwrap_or_default();
        let mut relation = Relation::new(probe, id, "observed");
        relation.attrs.insert(
            "at".to_owned(),
            format!("2026-08-22T10:{:02}:{:02}Z", at_second / 60, at_second % 60).into(),
        );
        relation
            .attrs
            .insert("time_base".to_owned(), "rfc3339".into());
        relation
    }

    /// Two domains that fire together get far more edges than chance; the same two spread out
    /// over the same span get roughly what chance gives.
    ///
    /// This is the whole claim `--calibrate` exists to let a reader make, and it is asserted as
    /// a comparison between two inputs rather than against a magic number.
    #[test]
    fn alignment_shows_up_as_lift_and_the_absence_of_it_does_not() {
        // Aligned: each `ip` observation has a `can` observation in the same second.
        let mut aligned = Vec::new();
        for second in (0..600).step_by(60) {
            aligned.push(observed("ip", &second.to_string(), second));
            aligned.push(observed("can", &second.to_string(), second));
        }
        // Spread: the same counts, but `can` sits between the `ip` observations.
        let mut spread = Vec::new();
        for second in (0..600).step_by(60) {
            spread.push(observed("ip", &second.to_string(), second));
            spread.push(observed("can", &second.to_string(), second + 30));
        }

        let of = |relations: Vec<Relation>| {
            calibrate(
                &Extraction {
                    relations,
                    ..Extraction::default()
                },
                5.0,
                MAX_EDGES,
                8,
            )
        };
        let (Ok(tight), Ok(loose)) = (of(aligned), of(spread)) else {
            panic!("both calibrations must run")
        };
        assert!(
            tight.observed > loose.observed,
            "aligned domains co-occur more inside a 5s window: {} vs {}",
            tight.observed,
            loose.observed,
        );
        assert!(
            tight.lift().unwrap_or(0.0) > loose.lift().unwrap_or(f64::MAX),
            "and the lift over chance separates them: {:?} vs {:?}",
            tight.lift(),
            loose.lift(),
        );
    }

    /// It blocks nothing. The join's own count is unchanged by calibrating.
    #[test]
    fn calibrating_removes_no_edge() {
        let relations: Vec<Relation> = (0..20u32)
            .map(|index| {
                observed(
                    if index % 2 == 0 { "ip" } else { "can" },
                    &index.to_string(),
                    index,
                )
            })
            .collect();
        let input = Extraction {
            relations,
            ..Extraction::default()
        };
        let Ok((edges, _report)) = co_occurrence(&input, 10.0, MAX_EDGES) else {
            panic!("must join")
        };
        let Ok(calibration) = calibrate(&input, 10.0, MAX_EDGES, 5) else {
            panic!("must calibrate")
        };
        assert_eq!(
            calibration.observed,
            edges.len(),
            "calibration reports the same count the join produces; it is a number beside the \
             edges and never a filter on them",
        );
    }

    /// A ratio over a zero denominator is a question, not a big number.
    #[test]
    fn no_lift_is_reported_when_chance_produced_nothing() {
        // One instant shared by everything: there is nothing to rotate.
        let relations: Vec<Relation> = (0..6u32)
            .map(|index| {
                observed(
                    if index % 2 == 0 { "ip" } else { "can" },
                    &index.to_string(),
                    0,
                )
            })
            .collect();
        let Ok(calibration) = calibrate(
            &Extraction {
                relations,
                ..Extraction::default()
            },
            5.0,
            MAX_EDGES,
            4,
        ) else {
            panic!("must calibrate")
        };
        assert_eq!(calibration.rounds, 0, "a single instant cannot be rotated");
        assert_eq!(calibration.lift(), None, "and no ratio is invented over it");
    }
}
