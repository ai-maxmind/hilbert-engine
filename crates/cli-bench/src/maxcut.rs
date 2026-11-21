//! Generator and cut evaluator for Max-Cut problems mapped to the Ising model.
//!
//! # Mathematical Mapping
//!
//! Given an undirected graph $G = (V, E)$ with $|V| = N$ vertices and edge weights $w_{uv} = 1$:
//! A cut partitions $V$ into two disjoint subsets $S$ and $\bar{S}$, represented by spin variables
//! $s_u \in \{-1, +1\}$ where $s_u = +1 \iff u \in S$.
//!
//! The cut value (number of edges crossing the partition) is:
//!
//! $$\text{Cut}(\mathbf{s}) = \sum_{(u, v) \in E} \frac{1 - s_u s_v}{2} = \frac{|E|}{2} - \frac{1}{2} \sum_{(u,v) \in E} s_u s_v$$
//!
//! In standard Ising minimization format:
//!
//! $$H(\mathbf{s}) = - \frac{1}{2} \sum_{u=0}^{N-1} \sum_{v=0}^{N-1} J_{uv} s_u s_v - \sum_{u=0}^{N-1} h_u s_u$$
//!
//! Setting $J_{uv} = J_{vu} = -1.0$ for all $(u, v) \in E$ and $h_u = 0$ yields:
//!
//! $$H(\mathbf{s}) = \sum_{(u,v) \in E} s_u s_v$$
//!
//! Therefore:
//!
//! $$\text{Cut}(\mathbf{s}) = \frac{|E| - H(\mathbf{s})}{2}$$
//!
//! Minimizing the Ising Hamiltonian $H(\mathbf{s})$ directly maximizes the cut value $\text{Cut}(\mathbf{s})$.

use backend_physics::{IsingCsr, IsingError};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Represents an unweighted or weighted undirected graph instance.
#[derive(Debug, Clone)]
pub struct MaxCutInstance {
    pub num_vertices: usize,
    pub edges: Vec<(usize, usize)>,
    pub ising: IsingCsr,
}

impl MaxCutInstance {
    /// Generate an Erdős–Rényi random graph $G(N, p)$ with deterministic PRNG seeding.
    pub fn generate_erdos_renyi(
        num_vertices: usize,
        edge_probability: f64,
        seed: u64,
    ) -> Result<Self, IsingError> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut edges = Vec::new();
        let mut ising_edges = Vec::new();

        for u in 0..num_vertices {
            for v in (u + 1)..num_vertices {
                let r: f64 = rng.gen();
                if r < edge_probability {
                    edges.push((u, v));
                    // J_uv = -1.0 in Ising model
                    ising_edges.push((u, v, -1.0f32));
                }
            }
        }

        let ising = IsingCsr::from_undirected_edges(num_vertices, &ising_edges, None)?;

        Ok(Self {
            num_vertices,
            edges,
            ising,
        })
    }

    /// Number of edges in the graph $|E|$.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Graph density: $2 |E| / (N (N - 1))$.
    pub fn density(&self) -> f64 {
        if self.num_vertices <= 1 {
            return 0.0;
        }
        let max_edges = (self.num_vertices * (self.num_vertices - 1)) / 2;
        self.edges.len() as f64 / max_edges as f64
    }

    /// Calculate cut value directly from edge partition:
    ///
    /// $$\text{Cut}(\mathbf{s}) = \sum_{(u,v) \in E} \frac{1 - s_u s_v}{2}$$
    pub fn calculate_cut_direct(&self, spins: &[i8]) -> usize {
        assert_eq!(spins.len(), self.num_vertices);
        let mut cut = 0;
        for &(u, v) in &self.edges {
            if spins[u] != spins[v] {
                cut += 1;
            }
        }
        cut
    }

    /// Calculate cut value from Ising Hamiltonian energy:
    ///
    /// $$\text{Cut}(E) = \frac{|E| - E_{\text{Ising}}}{2}$$
    pub fn calculate_cut_from_energy(&self, energy: f64) -> f64 {
        (self.num_edges() as f64 - energy) / 2.0
    }
}
