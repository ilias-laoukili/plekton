// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ilias Laoukili
//! Connected-component counting via iterative BFS.

/// Count the number of connected components in the graph using iterative BFS.
///
/// An isolated node (degree 0) forms its own component. An empty graph
/// (`n_nodes = 0`) has 0 components. A graph with nodes but no edges has
/// `n_nodes` components.
///
/// This implementation uses a `Vec<bool>` visited flag array rather than a
/// `HashSet<usize>`, which is more cache-friendly for dense node-ID spaces.
///
/// # Arguments
/// * `n_nodes` – Total number of nodes; node IDs are `0..n_nodes`.
/// * `edges`   – Canonical edge list; each pair satisfies `source < target`.
///
/// # Examples
/// ```
/// use plekton_metrics::components::count_components;
///
/// // Triangle: 3 nodes, 3 edges → 1 component
/// assert_eq!(count_components(3, &[(0, 1), (1, 2), (0, 2)]), 1);
///
/// // Two isolated nodes: 2 components
/// assert_eq!(count_components(2, &[]), 2);
/// ```
#[must_use]
pub fn count_components(n_nodes: usize, edges: &[(usize, usize)]) -> usize {
    if n_nodes == 0 {
        return 0;
    }

    // Build an undirected adjacency list from the edge set.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for &(source, target) in edges {
        adjacency[source].push(target);
        adjacency[target].push(source);
    }

    let mut visited = vec![false; n_nodes];
    let mut component_count: usize = 0;

    for start_node in 0..n_nodes {
        if visited[start_node] {
            continue;
        }
        // Flood-fill from `start_node` to mark all reachable nodes.
        component_count += 1;
        visited[start_node] = true;
        let mut bfs_queue: Vec<usize> = vec![start_node];
        let mut queue_head: usize = 0;

        while queue_head < bfs_queue.len() {
            let current_node = bfs_queue[queue_head];
            queue_head += 1;
            for &neighbour in &adjacency[current_node] {
                if !visited[neighbour] {
                    visited[neighbour] = true;
                    bfs_queue.push(neighbour);
                }
            }
        }
    }

    component_count
}
