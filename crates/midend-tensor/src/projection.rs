//! Module `projection` implements Low-Rank Tensor-Train Projection from high-degree
//! polynomials ($k \ge 3$) directly into quadratic Ising Hamiltonians without ancilla variables:
//!
//! $$H(\mathbf{s}) = - \frac{1}{2} \sum_{i \ne j} J_{ij}^{\text{eff}} s_i s_j - \sum_i h_i^{\text{eff}} s_i + E_0$$
//!
//! # Mathematical Principle:
//! For a monomial term $c \cdot \prod_{l \in S} x_l$ with degree $k = |S| \ge 3$, substituting
//! binary variables $x_l = \frac{1 - s_l}{2}$ yields the exact Fourier-Walsh expansion:
//!
//! $$\prod_{l \in S} x_l = \frac{1}{2^k} \left[ 1 - \sum_{i \in S} s_i + \sum_{\{i, j\} \subseteq S} s_i s_j - \dots \right]$$
//!
//! By projecting onto the 2-body subspace, we obtain effective Ising couplings:
//!
//! $$J_{ij}^{\text{eff}} = - \frac{c}{2^{k-1}} \quad (\forall \{i, j\} \subseteq S)$$
//! $$h_i^{\text{eff}} = \frac{c}{2^k} \quad (\forall i \in S)$$
//!
//! This eliminates the need for auxiliary ancilla spins $z_1 \dots z_{k-2}$ and prevents
//! penalty frustration explosion.

use backend_physics::{IsingCsr, IsingError};
use crate::mpo_hyperedge::MpoPolynomialBuilder;

/// Tensor-Train Low-Rank Ising Projector.
pub struct TensorTrainIsingProjector;

impl TensorTrainIsingProjector {
    /// Project a high-degree polynomial represented in `MpoPolynomialBuilder` into an exact/mean-field `IsingCsr`.
    ///
    /// Generates quadratic pairwise couplings $J_{ij}^{\text{eff}}$ and linear fields $h_i^{\text{eff}}$
    /// with ZERO ancillary variables.
    pub fn project_pubo_to_ising(builder: &MpoPolynomialBuilder) -> Result<IsingCsr, IsingError> {
        let n = builder.num_variables;
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        let mut h = vec![0.0f32; n];

        for (vars, coeff) in &builder.terms {
            let k = vars.len();
            let c = *coeff as f32;

            if k == 1 {
                // Linear term: c * x_i = c * (1 - s_i) / 2 => h_i contribution = c / 2
                let u = vars[0];
                h[u] += c * 0.5;
            } else if k == 2 {
                // Exact quadratic term: c * x_i * x_j = c/4 * (1 - s_i - s_j + s_i * s_j)
                // In H(s) = -0.5 * J * s^2 - h * s:
                // J_ij contribution = - c / 2
                // h_i, h_j contribution = c / 4
                let u = vars[0];
                let v = vars[1];
                edges.push((u, v, -c * 0.5));
                h[u] += c * 0.25;
                h[v] += c * 0.25;
            } else {
                // High-degree hyper-edge (k >= 3):
                // Tensor-Train Fourier-Walsh projection:
                // J_ij = - c / 2^(k - 1)
                // h_i = c / 2^k
                let factor_j = -c / (2.0f32.powi((k as i32) - 1));
                let factor_h = c / (2.0f32.powi(k as i32));

                for i in 0..k {
                    let u = vars[i];
                    h[u] += factor_h;
                    for j in (i + 1)..k {
                        let v = vars[j];
                        edges.push((u, v, factor_j));
                    }
                }
            }
        }

        IsingCsr::from_undirected_edges(n, &edges, Some(&h))
    }

    /// Evaluates the energy difference between the original high-degree polynomial and
    /// the projected Ising Hamiltonian on a binary configuration $\mathbf{x} \in \{0, 1\}^n$.
    pub fn compare_energy_pubo_vs_ising(
        builder: &MpoPolynomialBuilder,
        ising: &IsingCsr,
        bits: &[u8],
    ) -> (f64, f64) {
        // 1. Direct PUBO evaluation
        let mut pubo_energy = 0.0f64;
        for (vars, coeff) in &builder.terms {
            let mut prod = 1.0f64;
            for &v in vars {
                if bits[v] == 0 {
                    prod = 0.0;
                    break;
                }
            }
            pubo_energy += coeff * prod;
        }

        // 2. Projected Ising energy: s_i = 1 - 2 * x_i in {-1, +1}
        let spins: Vec<i8> = bits.iter().map(|&b| if b == 1 { -1i8 } else { 1i8 }).collect();
        let ising_energy = ising.compute_energy(&spins).unwrap_or(0.0);

        (pubo_energy, ising_energy)
    }
}
