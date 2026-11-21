//! Module `mpo_hyperedge` implements exact representation of high-degree monomials
//! and Polynomial Unconstrained Binary Optimization (PUBO) hyper-edges as Matrix Product Operators.
//!
//! # Exact Canonical Representation:
//! Any monomial $T(\mathbf{x}) = c \cdot \prod_{j \in S} x_j$ of arbitrary degree $k = |S|$
//! over $N$ binary variables is represented *exactly* as a Tensor-Train with bond dimension $\chi = 1$:
//!
//! - For $m \in S$:
//!   $$A^{(m)}[0] = [0], \quad A^{(m)}[1] = [1] \quad (\text{or } [c] \text{ for the last index in } S)$$
//! - For $m \notin S$:
//!   $$A^{(m)}[0] = [1], \quad A^{(m)}[1] = [1]$$

use crate::tensor_train::{TensorTrain, TtCore, TtError};

/// Builder for constructing Matrix Product Operators for high-degree polynomial expressions.
#[derive(Debug, Clone)]
pub struct MpoPolynomialBuilder {
    pub num_variables: usize,
    /// List of monomials: (indices, coefficient)
    pub terms: Vec<(Vec<usize>, f64)>,
}

impl MpoPolynomialBuilder {
    /// Create a new polynomial builder over `num_variables` variables.
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            terms: Vec::new(),
        }
    }

    /// Add a high-degree monomial term $c \cdot \prod_{j \in \text{vars}} x_j$.
    pub fn add_monomial(&mut self, vars: &[usize], coeff: f64) {
        if coeff.abs() < 1e-15 || vars.is_empty() {
            return;
        }
        let mut sorted_vars = vars.to_vec();
        sorted_vars.sort_unstable();
        sorted_vars.dedup();
        for &v in &sorted_vars {
            assert!(v < self.num_variables, "Variable index {} out of bounds {}", v, self.num_variables);
        }
        self.terms.push((sorted_vars, coeff));
    }

    /// Build an exact rank-1 Tensor-Train for a single monomial term $c \cdot \prod_{j \in S} x_j$.
    pub fn build_single_monomial_tt(num_variables: usize, vars: &[usize], coeff: f64) -> Result<TensorTrain, TtError> {
        let mut sorted_vars = vars.to_vec();
        sorted_vars.sort_unstable();
        sorted_vars.dedup();

        let last_var = *sorted_vars.last().expect("Vars must not be empty");
        let mut cores = Vec::with_capacity(num_variables);

        for m in 0..num_variables {
            let mut core = TtCore::zeros(1, 1);
            if sorted_vars.contains(&m) {
                // If variable m is part of the monomial:
                // x_m = 0 => 0
                // x_m = 1 => 1 (or coeff if it's the last variable in S)
                let val = if m == last_var { coeff } else { 1.0 };
                core.set(0, 0, 0, 0.0);
                core.set(0, 1, 0, val);
            } else {
                // Not in monomial: pass-through identity for both 0 and 1
                core.set(0, 0, 0, 1.0);
                core.set(0, 1, 0, 1.0);
            }
            cores.push(core);
        }

        TensorTrain::new(cores)
    }

    /// Build a unified Tensor-Train for the entire polynomial:
    ///
    /// $$P(\mathbf{x}) = \sum_{p=1}^M c_p \prod_{j \in S_p} x_j$$
    ///
    /// The resulting MPO has bond dimension $\chi \le M + 1$.
    pub fn build_polynomial_mpo(&self) -> Result<TensorTrain, TtError> {
        let n = self.num_variables;
        let m_terms = self.terms.len();

        if m_terms == 0 {
            // Return zero polynomial TT
            let mut cores = Vec::with_capacity(n);
            for _ in 0..n {
                let core = TtCore::zeros(1, 1);
                cores.push(core);
            }
            return TensorTrain::new(cores);
        }

        if m_terms == 1 {
            let (vars, coeff) = &self.terms[0];
            return Self::build_single_monomial_tt(n, vars, *coeff);
        }

        // Direct-sum MPO state machine:
        // Bond states at site m: [term_0_active, term_1_active, ..., term_{M-1}_active]
        // Left boundary rank r_0 = 1, right boundary rank r_N = 1.
        // Intermediate ranks r_m = M.
        let mut cores = Vec::with_capacity(n);

        for m in 0..n {
            let r_in = if m == 0 { 1 } else { m_terms };
            let r_out = if m == n - 1 { 1 } else { m_terms };

            let mut core = TtCore::zeros(r_in, r_out);

            for (p, (vars, coeff)) in self.terms.iter().enumerate() {
                let in_term = vars.contains(&m);
                let is_last_var = vars.last().copied() == Some(m);

                for x in 0..2 {
                    let factor = if in_term {
                        if x == 1 {
                            if is_last_var { *coeff } else { 1.0 }
                        } else {
                            0.0
                        }
                    } else {
                        1.0 // not part of term, pass-through
                    };

                    if factor.abs() > 1e-15 {
                        let i = if m == 0 { 0 } else { p };
                        let j = if m == n - 1 { 0 } else { p };
                        let cur = core.get(i, x, j);
                        core.set(i, x, j, cur + factor);
                    }
                }
            }

            cores.push(core);
        }

        TensorTrain::new(cores)
    }
}
