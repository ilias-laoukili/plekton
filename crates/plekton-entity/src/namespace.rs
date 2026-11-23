// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Identity namespacing: `<domain>:<type>:<value>`.
//!
//! # The failure this exists to prevent
//!
//! The core treats two entities sharing an id as **the same thing**. That is the right rule and
//! it is why the graph is heterogeneous for free — `host`/`domain` and `widget`/`sprocket` yield
//! byte-identical metrics because `kind` is an open string the core never reads.
//!
//! It becomes a hazard the moment two domains are put in one graph. A CAN arbitration id and a
//! Zigbee short address are both small integers, `0x0201` is a plausible value of each, and
//! nothing in the core would stop them merging into one node. **The merge would be silent** —
//! no error, no diagnostic, just a hub that is two unrelated things and a centrality score that
//! is the sum of two populations.
//!
//! Prefixing is not the only fix and it is the cheap one: it makes the collision impossible to
//! express rather than unlikely to occur.
//!
//! # Why the value may contain colons and the first two fields may not
//!
//! An IPv6 address is full of colons and so is a MAC. So a qualified id splits on the **first
//! two** separators and the rest is the value, verbatim — `ip:host:fe80::1` parses as
//! `("ip", "host", "fe80::1")`. The domain and the kind are refused at construction if they
//! contain one, which is where the ambiguity would have to be introduced.
//!
//! # What is NOT namespaced yet, and why that is stated rather than fixed here
//!
//! Requiring every adapter to declare its domain prevents silent cross-source merges.
//! in a shipped surface that people already consume, so it is scheduled rather than done in
//! passing. The hazard is contained in the meantime by the place it actually lives:
//! [`crate::join`] **refuses** to join un-namespaced ids instead of merging them quietly.

/// The separator. Not configurable: a separator a caller chooses is a separator two callers
/// choose differently.
pub const SEPARATOR: char = ':';

/// A parsed identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qualified<'a> {
    /// Which domain minted this identity — `ip`, `rf`, `can`, `ble`, `lorawan`.
    pub domain: &'a str,
    /// What kind of thing it is, in that domain's own vocabulary.
    pub kind: &'a str,
    /// The value, verbatim, colons and all.
    pub value: &'a str,
}

/// Why a domain or kind was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualifyError {
    /// A domain or kind was empty, which would make the id ambiguous to parse.
    Empty {
        /// Which field.
        field: &'static str,
    },
    /// A domain or kind contained the separator.
    ContainsSeparator {
        /// Which field.
        field: &'static str,
        /// What it was.
        value: String,
    },
}

impl std::fmt::Display for QualifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Empty { field } => {
                write!(f, "the {field} of an identity may not be empty")
            }
            Self::ContainsSeparator { field, ref value } => write!(
                f,
                "the {field} of an identity may not contain `{SEPARATOR}` (got `{value}`); the \
                 VALUE may, which is why only the first two separators are structural"
            ),
        }
    }
}

impl std::error::Error for QualifyError {}

/// Build `<domain>:<kind>:<value>`.
///
/// # Errors
///
/// [`QualifyError`] when the domain or the kind is empty or contains the separator. The value is
/// never refused — an identity a source minted is not this function's to judge.
pub fn qualify(domain: &str, kind: &str, value: &str) -> Result<String, QualifyError> {
    for (field, part) in [("domain", domain), ("kind", kind)] {
        if part.is_empty() {
            return Err(QualifyError::Empty { field });
        }
        if part.contains(SEPARATOR) {
            return Err(QualifyError::ContainsSeparator {
                field,
                value: part.to_owned(),
            });
        }
    }
    Ok(format!("{domain}{SEPARATOR}{kind}{SEPARATOR}{value}"))
}

/// Read an id back, or `None` when it is not qualified.
///
/// A bare `192.0.2.1` yields `None`. So does `ip:host` — two fields is a domain and a value with
/// no kind, or a kind and a value with no domain, and guessing which would be inventing one.
#[must_use]
pub fn parse(id: &str) -> Option<Qualified<'_>> {
    let mut parts = id.splitn(3, SEPARATOR);
    let domain = parts.next()?;
    let kind = parts.next()?;
    let value = parts.next()?;
    if domain.is_empty() || kind.is_empty() || value.is_empty() {
        return None;
    }
    Some(Qualified {
        domain,
        kind,
        value,
    })
}

/// [`qualify`] for a domain and kind that are literals in this crate.
///
/// Every adapter's domain and kind are `&'static str` written in its own source, so the two ways
/// [`qualify`] can fail — empty, or carrying the separator — are programming errors rather than
/// input errors, and threading a `Result` through every `Entity::new` call site would be handling
/// a case the caller has already made impossible.
///
/// The guard against that assumption going stale is not the return type: it is
/// `every_adapter_emits_only_qualified_ids`, which reads what the adapters actually emit.
///
/// # Panics
/// In debug builds only, if `domain` or `kind` is empty or contains [`SEPARATOR`]. A release
/// build formats regardless, because a malformed id that reaches the join is refused there.
#[must_use]
#[track_caller]
pub fn qualify_owned(domain: &'static str, kind: &'static str, value: &str) -> String {
    debug_assert!(
        !domain.is_empty() && !domain.contains(SEPARATOR),
        "`{domain}` is not usable as a domain",
    );
    debug_assert!(
        !kind.is_empty() && !kind.contains(SEPARATOR),
        "`{kind}` is not usable as a kind",
    );
    format!("{domain}{SEPARATOR}{kind}{SEPARATOR}{value}")
}

/// Whether an id carries a domain and a kind.
#[must_use]
pub fn is_qualified(id: &str) -> bool {
    parse(id).is_some()
}

#[cfg(test)]
mod tests {
    use super::{is_qualified, parse, qualify, Qualified, QualifyError};

    #[test]
    fn a_value_may_contain_separators_and_the_first_two_fields_may_not() {
        let id = qualify("ip", "host", "fe80::1").unwrap_or_default();
        assert_eq!(id, "ip:host:fe80::1");
        assert_eq!(
            parse(&id),
            Some(Qualified {
                domain: "ip",
                kind: "host",
                value: "fe80::1",
            }),
            "an IPv6 address is full of separators and is still one value",
        );

        let mac = qualify("ble", "address", "00:00:5E:00:53:01").unwrap_or_default();
        assert_eq!(parse(&mac).map(|q| q.value), Some("00:00:5E:00:53:01"));
    }

    #[test]
    fn a_domain_or_kind_carrying_a_separator_is_refused() {
        assert!(matches!(
            qualify("ip:v6", "host", "::1"),
            Err(QualifyError::ContainsSeparator {
                field: "domain",
                ..
            })
        ));
        assert!(matches!(
            qualify("ip", "ho:st", "::1"),
            Err(QualifyError::ContainsSeparator { field: "kind", .. })
        ));
        assert!(matches!(
            qualify("", "host", "::1"),
            Err(QualifyError::Empty { field: "domain" })
        ));
    }

    /// The collision the module exists for, made unrepresentable.
    #[test]
    fn a_can_arbitration_id_and_a_zigbee_short_address_cannot_collide() {
        // Both are small integers and `0x0201` is a plausible value of each. Bare, they are one
        // node and the merge is silent.
        let can = qualify("can", "arbitration_id", "0x0201").unwrap_or_default();
        let zigbee = qualify("ieee802154", "short_address", "0x0201").unwrap_or_default();
        assert_ne!(can, zigbee);
        assert_eq!(parse(&can).map(|q| q.domain), Some("can"));
        assert_eq!(parse(&zigbee).map(|q| q.domain), Some("ieee802154"));
    }

    #[test]
    fn a_bare_id_is_not_qualified_and_neither_is_a_two_field_one() {
        assert!(!is_qualified("192.0.2.1"));
        assert!(
            !is_qualified("ip:host"),
            "two fields cannot say which is missing"
        );
        assert!(!is_qualified(""));
        assert!(!is_qualified("ip::1"), "an empty kind is not a kind");
        assert!(is_qualified("ip:host:192.0.2.1"));
    }
}
