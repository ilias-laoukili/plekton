// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Adamic–Adar scores must be the same bits in every process.
//!
//! The score of an edge is a floating-point sum over its common neighbours, and
//! floating-point addition is not associative: the same terms added in another
//! order can round differently. The neighbourhoods were hash sets whose hasher is
//! seeded at random, so the order, and with it the last bits of the scores,
//! changed from one process to the next. This test computes the scores, has a
//! fresh copy of this test binary compute them again in a child process, and
//! compares the bits.

use std::process::Command;

use plekton_metrics::similarity::adamic_adar_scores;

/// Name of the test below: the child process runs it alone.
const TEST_NAME: &str = "adamic_adar_scores_are_the_same_bits_in_two_processes";
/// Set in the child process, which prints its scores instead of comparing them.
const CHILD: &str = "PLEKTON_ADAMIC_ADAR_CHILD";
/// Marks the child's scores among the test harness's own output.
const MARKER: &str = "adamic-adar-bits ";
const N_NODES: usize = 300;

/// `SplitMix64`'s output function: a well-mixed 64-bit hash of `x`.
const fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A fixed random graph, each pair kept with probability 1/8: about 5 600 edges,
/// most with several common neighbours of varied degrees, so that a change of
/// order shows in the sums.
fn edges() -> Vec<(usize, usize)> {
    let mut pair = 0_u64;
    let mut edges = Vec::new();
    for u in 0..N_NODES {
        for v in u + 1..N_NODES {
            pair += 1;
            if mix(pair).is_multiple_of(8) {
                edges.push((u, v));
            }
        }
    }
    edges
}

#[test]
fn adamic_adar_scores_are_the_same_bits_in_two_processes() -> Result<(), Box<dyn std::error::Error>>
{
    let edges = edges();
    let here: Vec<String> = adamic_adar_scores(N_NODES, &edges, true)
        .iter()
        .map(|score| format!("{:016x}", score.to_bits()))
        .collect();

    if std::env::var_os(CHILD).is_some() {
        println!("{MARKER}{}", here.join(" "));
        return Ok(());
    }

    let output = Command::new(std::env::current_exe()?)
        .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()?;
    assert!(
        output.status.success(),
        "the child process failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    // The harness may print on the same line before the marker.
    let there: Vec<&str> = stdout
        .lines()
        .find_map(|line| line.split_once(MARKER).map(|(_, bits)| bits))
        .ok_or("the child process printed no scores")?
        .split(' ')
        .collect();

    assert_eq!(there.len(), here.len(), "each process scores every edge");
    let differing = here.iter().zip(&there).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} Adamic–Adar scores differ between two processes",
        here.len()
    );
    Ok(())
}
