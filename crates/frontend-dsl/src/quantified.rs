//! Module `quantified` implements DSL builder for 2-player zero-sum adversarial games
//! and Quantified Boolean/Ising Optimization:
//!
//! $$\min_{\mathbf{x} \in \{-1, +1\}^{N_x}} \max_{\mathbf{y} \in \{-1, +1\}^{N_y}} H(\mathbf{x}, \mathbf{y})$$
//!
//! where:
//!
//! $$H(\mathbf{x}, \mathbf{y}) = - \frac{1}{2} \mathbf{x}^T J_{xx} \mathbf{x} - \frac{1}{2} \mathbf{y}^T J_{yy} \mathbf{y} - \mathbf{x}^T K_{xy} \mathbf{y} - \mathbf{h}_x^T \mathbf{x} - \mathbf{h}_y^T \mathbf{y}$$

use backend_physics::{IsingCsr, IsingError};

/// Builder for two-player adversarial minimax optimization problems.
#[derive(Debug, Clone)]
pub struct QuantifiedProblemBuilder {
    pub num_x: usize,
    pub num_y: usize,
    pub j_xx: Vec<(usize, usize, f32)>,
    pub j_yy: Vec<(usize, usize, f32)>,
    pub k_xy: Vec<(usize, usize, f32)>,
    pub h_x: Vec<f32>,
    pub h_y: Vec<f32>,
}

impl QuantifiedProblemBuilder {
    /// Create a new 2-player adversarial problem with `num_x` Minimizer variables and `num_y` Maximizer variables.
    pub fn new(num_x: usize, num_y: usize) -> Self {
        Self {
            num_x,
            num_y,
            j_xx: Vec::new(),
            j_yy: Vec::new(),
            k_xy: Vec::new(),
            h_x: vec![0.0f32; num_x],
            h_y: vec![0.0f32; num_y],
        }
    }

    /// Add internal coupling between Minimizer variables x_u and x_v.
    pub fn add_x_coupling(&mut self, u: usize, v: usize, weight: f32) {
        assert!(u < self.num_x && v < self.num_x && u != v);
        self.j_xx.push((u, v, weight));
    }

    /// Add internal coupling between Maximizer variables y_u and y_v.
    pub fn add_y_coupling(&mut self, u: usize, v: usize, weight: f32) {
        assert!(u < self.num_y && v < self.num_y && u != v);
        self.j_yy.push((u, v, weight));
    }

    /// Add adversarial cross-coupling between Minimizer x_u and Maximizer y_v.
    ///
    /// Payoff term is - K_{xy, uv} * x_u * y_v.
    /// If weight > 0: x_u and y_v prefer aligning for lower energy (Minimizer wants x_u = y_v, Maximizer wants opposite).
    pub fn add_adversarial_interaction(&mut self, x_var: usize, y_var: usize, weight: f32) {
        assert!(x_var < self.num_x && y_var < self.num_y);
        self.k_xy.push((x_var, y_var, weight));
    }

    /// Add linear bias for Minimizer variable x_u.
    pub fn add_x_bias(&mut self, u: usize, bias: f32) {
        assert!(u < self.num_x);
        self.h_x[u] += bias;
    }

    /// Add linear bias for Maximizer variable y_v.
    pub fn add_y_bias(&mut self, v: usize, bias: f32) {
        assert!(v < self.num_y);
        self.h_y[v] += bias;
    }

    /// Total number of variables in the adversarial system ($N_x + N_y$).
    #[inline]
    pub fn total_variables(&self) -> usize {
        self.num_x + self.num_y
    }

    /// Compile into a unified `IsingCsr` representation of size (N_x + N_y)
    /// where indices [0..N_x) correspond to x, and [N_x..N_x + N_y) correspond to y.
    pub fn build_unified_ising(&self) -> Result<(IsingCsr, usize), IsingError> {
        let total_n = self.num_x + self.num_y;
        let mut edges = Vec::new();

        // 1. Internal J_xx
        for &(u, v, w) in &self.j_xx {
            edges.push((u, v, w));
        }

        // 2. Internal J_yy (offset by num_x)
        for &(u, v, w) in &self.j_yy {
            edges.push((self.num_x + u, self.num_x + v, w));
        }

        // 3. Cross-terms K_xy
        for &(x_var, y_var, w) in &self.k_xy {
            edges.push((x_var, self.num_x + y_var, w));
        }

        // 4. Combined linear field
        let mut h = Vec::with_capacity(total_n);
        h.extend_from_slice(&self.h_x);
        h.extend_from_slice(&self.h_y);

        let ising = IsingCsr::from_undirected_edges(total_n, &edges, Some(&h))?;
        Ok((ising, self.num_x))
    }
}
