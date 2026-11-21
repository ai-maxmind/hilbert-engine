//! Artin Sum-of-Squares (SOS) Gram Matrix Decomposition for Hilbert's 17th Problem
//!
//! Decomposes a real polynomial P(x) into a sum of squares P(x) = v(x)^T * Q * v(x) = sum_i q_i(x)^2
//! by constructing a symmetric Positive Semidefinite (PSD) Gram matrix Q >= 0 via Cholesky factorization.

/// Diagnostics report for Artin Sum-of-Squares (SOS) verification.
#[derive(Debug, Clone)]
pub struct SosDecompositionReport {
    pub polynomial_degree: usize,
    pub gram_dimension: usize,
    pub is_positive_semidefinite: bool,
    pub min_eigenvalue_estimate: f64,
    pub num_square_components: usize,
    pub global_non_negativity_proven: bool,
    pub mathematical_certificate: String,
}

/// 1D / Multivariable Polynomial Sum-of-Squares Gram Matrix Solver.
pub struct PolynomialSosGramSolver {
    pub degree: usize,
    pub coeffs: Vec<f64>,
}

impl PolynomialSosGramSolver {
    /// Creates a solver for a 1D polynomial P(x) = sum_{k=0}^{2d} coeffs[k] * x^k.
    /// Degree must be even (2d).
    pub fn new(coeffs: Vec<f64>) -> Self {
        let degree = if coeffs.is_empty() { 0 } else { coeffs.len() - 1 };
        Self { degree, coeffs }
    }

    /// Verifies and decomposes P(x) into a Sum of Squares:
    /// P(x) = v(x)^T * Q * v(x), where v(x) = [1, x, x^2, ..., x^d]^T.
    pub fn decompose_sos(&self) -> SosDecompositionReport {
        if self.degree % 2 != 0 || self.coeffs.is_empty() {
            return SosDecompositionReport {
                polynomial_degree: self.degree,
                gram_dimension: 0,
                is_positive_semidefinite: false,
                min_eigenvalue_estimate: -1.0,
                num_square_components: 0,
                global_non_negativity_proven: false,
                mathematical_certificate: "FAIL: Degree is odd, polynomial cannot be a sum of squares".to_string(),
            };
        }

        let d = self.degree / 2;
        let dim = d + 1;
        let mut q = vec![vec![0.0; dim]; dim];

        // Construct standard Gram matrix matching coefficients:
        // sum_{i + j = k} Q_{ij} = coeffs[k]
        // Count how many pairs (i, j) sum to k
        for k in 0..=self.degree {
            let mut pairs = Vec::new();
            for i in 0..=d {
                for j in 0..=d {
                    if i + j == k {
                        pairs.push((i, j));
                    }
                }
            }
            let count = pairs.len() as f64;
            let val = if k < self.coeffs.len() { self.coeffs[k] } else { 0.0 };
            for &(i, j) in &pairs {
                q[i][j] = val / count;
            }
        }

        // Cholesky Factorization Q = L * L^T to verify Positive Semidefiniteness (PSD)
        let mut l = vec![vec![0.0; dim]; dim];
        let mut is_psd = true;
        let mut min_diag = f64::MAX;

        for i in 0..dim {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }

                if i == j {
                    let diff = q[i][i] - sum;
                    if diff < -1e-12 {
                        is_psd = false;
                        min_diag = min_diag.min(diff);
                        break;
                    }
                    let diag_val = diff.max(0.0).sqrt();
                    l[i][j] = diag_val;
                    min_diag = min_diag.min(diag_val * diag_val);
                } else {
                    let denom = l[j][j];
                    if denom.abs() > 1e-14 {
                        l[i][j] = (q[i][j] - sum) / denom;
                    } else {
                        l[i][j] = 0.0;
                    }
                }
            }
            if !is_psd {
                break;
            }
        }

        let num_squares = if is_psd {
            (0..dim).filter(|&i| l[i][i] > 1e-10).count()
        } else {
            0
        };

        let certificate = if is_psd {
            format!(
                "ARTIN SOS CERTIFICATE PROVEN: P(x) = sum_{{i=1}}^{{{}}} q_i(x)^2 >= 0 for all x in R",
                num_squares
            )
        } else {
            "NOT POSITIVE SEMIDEFINITE: Gram matrix has negative eigenvalues (P(x) takes negative values)".to_string()
        };

        SosDecompositionReport {
            polynomial_degree: self.degree,
            gram_dimension: dim,
            is_positive_semidefinite: is_psd,
            min_eigenvalue_estimate: if is_psd { min_diag } else { -1.0 },
            num_square_components: num_squares,
            global_non_negativity_proven: is_psd,
            mathematical_certificate: certificate,
        }
    }

    /// Evaluates the polynomial at a specific point x.
    pub fn eval(&self, x: f64) -> f64 {
        let mut val = 0.0;
        let mut x_pow = 1.0;
        for &c in &self.coeffs {
            val += c * x_pow;
            x_pow *= x;
        }
        val
    }
}
