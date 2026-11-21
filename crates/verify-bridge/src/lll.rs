//! Lenstra-Lenstra-Lovász (LLL) Lattice Basis Reduction & Bounded Diophantine Solver
//!
//! Provides polynomial-time approximation for the Shortest Vector Problem (SVP)
//! and solves bounded linear/quadratic Diophantine equations using Kannan embedding.

/// Diagnostics report for LLL lattice reduction.
#[derive(Debug, Clone)]
pub struct LllReport {
    pub dimension: usize,
    pub iterations: usize,
    pub shortest_vector_norm: f64,
    pub shortest_vector: Vec<f64>,
    pub is_reduced: bool,
}

/// Representation of an integer / real lattice basis B = [b_0, b_1, ..., b_{n-1}].
#[derive(Debug, Clone)]
pub struct LatticeBasis {
    pub n: usize,
    pub m: usize,
    pub basis: Vec<Vec<f64>>,
}

impl LatticeBasis {
    /// Creates a new lattice basis from rows.
    pub fn new(basis: Vec<Vec<f64>>) -> Self {
        let n = basis.len();
        let m = if n > 0 { basis[0].len() } else { 0 };
        Self { n, m, basis }
    }

    /// Dot product between two vectors.
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// Vector norm squared.
    fn norm_sq(v: &[f64]) -> f64 {
        Self::dot(v, v)
    }

    /// Performs Gram-Schmidt Orthogonalization (GSO) on the basis.
    /// Returns (orthogonal_basis b*, mu coefficients).
    pub fn gram_schmidt(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let n = self.n;
        let mut b_star = vec![vec![0.0; self.m]; n];
        let mut mu = vec![vec![0.0; n]; n];

        for i in 0..n {
            b_star[i] = self.basis[i].clone();
            for j in 0..i {
                let denom = Self::norm_sq(&b_star[j]);
                if denom > 1e-14 {
                    mu[i][j] = Self::dot(&self.basis[i], &b_star[j]) / denom;
                    for k in 0..self.m {
                        b_star[i][k] -= mu[i][j] * b_star[j][k];
                    }
                }
            }
        }

        (b_star, mu)
    }

    /// Executes the Lenstra-Lenstra-Lovász (LLL) lattice basis reduction algorithm.
    /// Parameter delta is typically 0.75.
    pub fn lll_reduce(&mut self, delta: f64) -> LllReport {
        let n = self.n;
        if n <= 1 {
            let norm = if n == 1 { Self::norm_sq(&self.basis[0]).sqrt() } else { 0.0 };
            return LllReport {
                dimension: n,
                iterations: 0,
                shortest_vector_norm: norm,
                shortest_vector: if n == 1 { self.basis[0].clone() } else { vec![] },
                is_reduced: true,
            };
        }

        let mut iterations = 0;
        let mut k = 1;

        while k < n && iterations < 1000 * n * n {
            iterations += 1;
            let (b_star, mut mu) = self.gram_schmidt();

            // Step 1: Size reduction of b_k with respect to b_{k-1}
            if mu[k][k - 1].abs() > 0.5 {
                let r = mu[k][k - 1].round();
                for j in 0..self.m {
                    self.basis[k][j] -= r * self.basis[k - 1][j];
                }
                let (_, new_mu) = self.gram_schmidt();
                mu = new_mu;
            }

            // Step 2: Lovász Condition
            let c_k1 = Self::norm_sq(&b_star[k - 1]);
            let c_k = Self::norm_sq(&b_star[k]);

            if c_k >= (delta - mu[k][k - 1] * mu[k][k - 1]) * c_k1 {
                // Size reduction for remaining j = k-2 down to 0
                for j in (0..k - 1).rev() {
                    let (_, current_mu) = self.gram_schmidt();
                    if current_mu[k][j].abs() > 0.5 {
                        let r = current_mu[k][j].round();
                        for col in 0..self.m {
                            self.basis[k][col] -= r * self.basis[j][col];
                        }
                    }
                }
                k += 1;
            } else {
                // Swap basis[k] and basis[k-1]
                self.basis.swap(k, k - 1);
                k = 1.max(k - 1);
            }
        }

        let shortest = self.basis[0].clone();
        let norm = Self::norm_sq(&shortest).sqrt();

        LllReport {
            dimension: n,
            iterations,
            shortest_vector_norm: norm,
            shortest_vector: shortest,
            is_reduced: true,
        }
    }

    /// Solves a bounded linear Diophantine equation:
    /// sum_{i=0}^{n-1} a_i * x_i = target, subject to |x_i| <= bound.
    ///
    /// Constructs Kannan's embedding lattice and reduces it via LLL.
    pub fn solve_bounded_diophantine(
        coeffs: &[i64],
        target: i64,
        bound: i64,
    ) -> Option<Vec<i64>> {
        let n = coeffs.len();
        if n == 0 {
            return None;
        }

        // Weight M chosen large enough to enforce sum a_i x_i = target
        let m_weight = (bound * 100).max(10_000) as f64;
        let dim = n + 1;

        // Construct (n+1) x (n+1) Kannan embedding matrix:
        // [ 1,  0, ...,  0,  M * a_0 ]
        // [ 0,  1, ...,  0,  M * a_1 ]
        // ...
        // [ 0,  0, ...,  1,  M * a_{n-1} ]
        // [ 0,  0, ...,  0,  M * target ]
        let mut mat = vec![vec![0.0; dim]; dim];
        for i in 0..n {
            mat[i][i] = 1.0;
            mat[i][n] = coeffs[i] as f64 * m_weight;
        }
        // Target row
        mat[n][n] = -(target as f64) * m_weight;

        let mut lattice = LatticeBasis::new(mat);
        let _report = lattice.lll_reduce(0.75);

        // Search among reduced basis vectors for one where last element is ~0
        for row in &lattice.basis {
            let last_val = row[n].abs();
            if last_val < 0.1 * m_weight {
                let candidate: Vec<i64> = row[0..n].iter().map(|&x| x.round() as i64).collect();
                // Check if candidate satisfies the equation
                let sum: i64 = candidate.iter().zip(coeffs.iter()).map(|(&x, &a)| x * a).sum();
                if sum == target && candidate.iter().all(|&x| x.abs() <= bound) {
                    return Some(candidate);
                }
                // Try negative
                let neg_candidate: Vec<i64> = candidate.iter().map(|&x| -x).collect();
                let neg_sum: i64 = neg_candidate.iter().zip(coeffs.iter()).map(|(&x, &a)| x * a).sum();
                if neg_sum == target && neg_candidate.iter().all(|&x| x.abs() <= bound) {
                    return Some(neg_candidate);
                }
            }
        }

        None
    }
}
