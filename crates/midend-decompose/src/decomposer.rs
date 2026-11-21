//! Interaction graph decomposition, Dynamic Range Scaling, and CSR compilation.
//!
//! # Mathematical Mapping: QUBO to Ising
//!
//! Given a quadratic binary minimization problem $Q(\mathbf{x})$:
//!
//! $$Q(\mathbf{x}) = c_0 + \sum_{i=0}^{N-1} q_i x_i + \sum_{0 \le i < j < N} q_{ij} x_i x_j$$
//!
//! Under the linear isomorphism $x_i = \frac{1 - s_i}{2}$ where $s_i \in \{-1, +1\}$:
//!
//! $$x_i x_j = \frac{1 - s_i - s_j + s_i s_j}{4}$$
//!
//! Expanding and matching terms with the standard Ising Hamiltonian:
//!
//! $$H(\mathbf{s}) = - \frac{1}{2} \sum_{i \ne j} J_{ij} s_i s_j - \sum_i h_i s_i + \Delta E$$
//!
//! We obtain exact closed-form expressions:
//!
//! $$J_{ij} = J_{ji} = - \frac{q_{ij}}{4} \quad (\forall i \ne j)$$
//! $$h_i = \frac{q_i}{2} + \sum_{j \ne i} \frac{q_{ij}}{4}$$
//! $$\Delta E = c_0 + \sum_i \frac{q_i}{2} + \sum_{i < j} \frac{q_{ij}}{4}$$
//!
//! # Dynamic Range Scaling
//!
//! To prevent numerical instability or floating-point precision loss during integration:
//!
//! $$S = \max \left( \max_{i < j} |J_{ij}|, \max_i |h_i| \right)$$
//! $$J_{\text{scaled}} = \frac{J}{S}, \quad \mathbf{h}_{\text{scaled}} = \frac{\mathbf{h}}{S}$$
//! $$H_{\text{true}}(\mathbf{s}) = S \cdot H_{\text{scaled}}(\mathbf{s}) + \Delta E$$

use std::collections::BTreeMap;
use backend_physics::{IsingCsr, IsingError};
use midend_egraph::pubo::{PuboError, PuboPolynomial};
use petgraph::algo::connected_components;
use petgraph::graph::UnGraph;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecomposeError {
    #[error("PUBO polynomial is not quadratic (degree={degree} > 2)")]
    NotQuadratic { degree: usize },

    #[error("Ising model creation error: {0}")]
    Ising(#[from] IsingError),

    #[error("Pubo evaluation error: {0}")]
    Pubo(#[from] PuboError),

    #[error("GPU solver error: {0}")]
    Gpu(#[from] backend_physics::gpu::GpuError),

    #[error("dSB solver error: {0}")]
    Dsb(#[from] backend_physics::dsb::DsbError),

    #[error("Multilevel error: {0}")]
    Multilevel(String),
}

/// Compilation product holding the CSR Ising model and topology metrics.
#[derive(Debug, Clone)]
pub struct DecomposedIsing {
    /// Compressed Sparse Row Ising model ready for GPU or CPU solver.
    pub ising_csr: IsingCsr,
    /// Dynamic Range Scale factor $S$ applied to $J$ and $\mathbf{h}$.
    pub scale_factor: f64,
    /// Constant energy offset $\Delta E$ to map Ising energy back to original PUBO/QUBO.
    pub energy_offset: f64,
    /// Total number of variables in the expanded system ($N_{\text{orig}} + N_{\text{ancilla}}$).
    pub num_variables: usize,
    /// Number of original problem variables.
    pub num_original_vars: usize,
    /// Number of auxiliary ancilla variables introduced.
    pub num_ancillas: usize,
    /// Total number of interaction edges in the graph.
    pub num_edges: usize,
    /// Number of connected components in the interaction graph.
    pub connected_components: usize,
    /// Maximum vertex degree in the interaction graph.
    pub max_degree: usize,
    /// Average vertex degree.
    pub avg_degree: f64,
}

impl DecomposedIsing {
    /// Decode a solution of spins $s_i \in \{-1, +1\}$ back to binary variables $x_i \in \{0, 1\}$.
    ///
    /// Returns `(original_binary_variables, unscaled_true_energy)`.
    pub fn decode_spins(&self, spins: &[i8]) -> Result<(Vec<u8>, f64), DecomposeError> {
        assert_eq!(spins.len(), self.num_variables);

        let binary_vars: Vec<u8> = spins
            .iter()
            .map(|&s| if s == -1 { 1u8 } else { 0u8 })
            .collect();

        let ising_scaled_energy = self.ising_csr.compute_energy(spins)?;
        let true_energy = self.scale_factor * ising_scaled_energy + self.energy_offset;

        let original_vars = binary_vars[0..self.num_original_vars].to_vec();
        Ok((original_vars, true_energy))
    }
}

/// Decompose and compile a quadratic polynomial into `IsingCsr` with graph analysis and scaling.
pub fn decompose_to_ising(
    qubo: &PuboPolynomial,
    num_original_vars: usize,
) -> Result<DecomposedIsing, DecomposeError> {
    if !qubo.is_quadratic() {
        return Err(DecomposeError::NotQuadratic {
            degree: qubo.degree(),
        });
    }

    let n = qubo.num_variables;
    let num_ancillas = n.saturating_sub(num_original_vars);

    // 1. Extract linear, quadratic, and constant terms
    let mut c0: f64 = 0.0;
    let mut q_linear = vec![0.0f64; n];
    let mut q_matrix: BTreeMap<(usize, usize), f64> = BTreeMap::new();

    for (vars, &coeff) in &qubo.terms {
        if coeff.abs() <= 1e-12 {
            continue;
        }
        match vars.len() {
            0 => c0 += coeff,
            1 => q_linear[vars[0]] += coeff,
            2 => {
                let u = vars[0];
                let v = vars[1];
                let key = if u < v { (u, v) } else { (v, u) };
                *q_matrix.entry(key).or_insert(0.0) += coeff;
            }
            _ => unreachable!("is_quadratic returned true"),
        }
    }

    // 2. Map to raw Ising J and h
    // J_ij = -q_ij / 2
    // h_i = q_i / 2 + sum_{j != i} q_ij / 4
    let mut h_raw = vec![0.0f64; n];
    for i in 0..n {
        h_raw[i] = q_linear[i] / 2.0;
    }

    let mut j_map: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut delta_e = c0;
    for i in 0..n {
        delta_e += q_linear[i] / 2.0;
    }

    for (&(u, v), &q_uv) in &q_matrix {
        if q_uv.abs() <= 1e-12 {
            continue;
        }
        let j_val = -q_uv / 4.0;
        j_map.insert((u, v), j_val);
        j_map.insert((v, u), j_val);

        h_raw[u] += q_uv / 4.0;
        h_raw[v] += q_uv / 4.0;
        delta_e += q_uv / 4.0;
    }

    // 3. Interaction Graph Topology Analysis with Petgraph
    let mut graph = UnGraph::<usize, f32>::new_undirected();
    let node_indices: Vec<_> = (0..n).map(|i| graph.add_node(i)).collect();

    for (&(u, v), &j_val) in &j_map {
        if u < v && j_val.abs() > 1e-7 {
            graph.add_edge(node_indices[u], node_indices[v], j_val as f32);
        }
    }

    let connected_comps = connected_components(&graph);
    let mut max_deg = 0;
    let mut total_deg = 0;
    for node in graph.node_indices() {
        let deg = graph.neighbors(node).count();
        if deg > max_deg {
            max_deg = deg;
        }
        total_deg += deg;
    }
    let avg_deg = if n > 0 { total_deg as f64 / n as f64 } else { 0.0 };
    let num_edges = graph.edge_count();

    // 4. Dynamic Range Scaling
    let mut max_val: f64 = 0.0;
    for &h_i in &h_raw {
        if h_i.abs() > max_val {
            max_val = h_i.abs();
        }
    }
    for &j_val in j_map.values() {
        if j_val.abs() > max_val {
            max_val = j_val.abs();
        }
    }

    let scale_factor = if max_val > 1e-12 { max_val } else { 1.0 };

    // 5. Build Compressed Sparse Row (CSR) representation
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let mut h_scaled = Vec::with_capacity(n);

    row_ptr.push(0);

    for i in 0..n {
        h_scaled.push((h_raw[i] / scale_factor) as f32);

        // Find all non-zero entries in row i
        let mut neighbors = Vec::new();
        for j in 0..n {
            if i != j {
                if let Some(&val) = j_map.get(&(i, j)) {
                    if val.abs() > 1e-7 {
                        neighbors.push((j as u32, (val / scale_factor) as f32));
                    }
                }
            }
        }
        neighbors.sort_by_key(|&(col, _)| col);

        for (col, val) in neighbors {
            col_indices.push(col);
            values.push(val);
        }
        row_ptr.push(col_indices.len() as u32);
    }

    let ising_csr = IsingCsr::new(n, row_ptr, col_indices, values, h_scaled)?;

    Ok(DecomposedIsing {
        ising_csr,
        scale_factor,
        energy_offset: delta_e,
        num_variables: n,
        num_original_vars,
        num_ancillas,
        num_edges,
        connected_components: connected_comps,
        max_degree: max_deg,
        avg_degree: avg_deg,
    })
}
