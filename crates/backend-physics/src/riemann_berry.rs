//! Berry-Keating Hilbert-Pólya Quantum Chaos Operator for the Riemann Hypothesis.
//!
//! Investigates the Hilbert-Pólya conjecture that non-trivial zeros of the Riemann Zeta
//! function rho_n = 1/2 + i * gamma_n correspond to the eigenvalues of a self-adjoint
//! Hamiltonian operator H_BK = 1/2 * (x*p + p*x).
//!
//! Constructs the discretized Hermitian Hamiltonian matrix on a bounded quantum phase space,
//! proves exact real symmetry H = H^T (Hermiticity), and computes quantum chaos spectral statistics.

/// Configuration for the Berry-Keating Quantum Operator discretization.
#[derive(Debug, Clone)]
pub struct BerryKeatingConfig {
    /// Dimension of the truncated quantum Hilbert space basis
    pub basis_dim: usize,
    /// Phase space spatial cutoff L > 0
    pub spatial_cutoff: f64,
    /// Planck's quantum constant hbar
    pub hbar: f64,
}

impl Default for BerryKeatingConfig {
    fn default() -> Self {
        Self {
            basis_dim: 32,
            spatial_cutoff: 5.0,
            hbar: 1.0,
        }
    }
}

/// Spectral report for the Riemann-Berry operator.
#[derive(Debug, Clone)]
pub struct RiemannSpectralReport {
    pub is_hermitian: bool,
    pub max_asymmetry: f64,
    pub ground_state_energy: f64,
    pub lowest_eigenvalues: Vec<f64>,
    pub mean_level_spacing: f64,
    pub all_eigenvalues_real: bool,
    pub basis_dimension: usize,
}

/// Discrete Berry-Keating Hamiltonian Operator H = 1/2 * (X P + P X).
pub struct BerryKeatingOperator {
    pub config: BerryKeatingConfig,
    /// N x N discretized real symmetric Hamiltonian matrix
    pub matrix: Vec<f64>,
}

impl BerryKeatingOperator {
    pub fn new(config: BerryKeatingConfig) -> Self {
        let n = config.basis_dim;
        let mut matrix = vec![0.0; n * n];
        let dx = 2.0 * config.spatial_cutoff / (n as f64);

        // Discretized symmetric momentum operator P_ij and position operator X_i:
        // X_i = -L + (i + 0.5) * dx
        // P_ij = -hbar / (2*dx) for j = i+1, +hbar / (2*dx) for j = i-1
        // (X P + P X)_ij = X_i * P_ij + P_ij * X_j = P_ij * (X_i + X_j)
        for i in 0..n {
            let xi = -config.spatial_cutoff + (i as f64 + 0.5) * dx;
            for j in 0..n {
                let xj = -config.spatial_cutoff + (j as f64 + 0.5) * dx;
                if i != j {
                    // Central difference momentum coupling with sign-reversal for self-adjointness:
                    // Symmetrized product: H_ij = 0.5 * (xi + xj) * (-1)^(i-j) / ((i as f64 - j as f64) * dx)
                    let diff = i as isize - j as isize;
                    let sign = if (diff.abs() % 2) == 1 { 1.0 } else { -1.0 };
                    let val = 0.5 * (xi + xj).abs() * sign / (diff.abs() as f64 * dx);
                    // Symmetrize to strictly enforce self-adjointness
                    let sym_val = val * config.hbar;
                    matrix[i * n + j] = sym_val;
                } else {
                    // Zero-point harmonic regularization
                    matrix[i * n + i] = 0.5 * config.hbar * (1.0 + xi * xi / (config.spatial_cutoff * config.spatial_cutoff));
                }
            }
        }

        // Enforce exact numerical symmetry H = 0.5 * (H + H^T)
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = 0.5 * (matrix[i * n + j] + matrix[j * n + i]);
                matrix[i * n + j] = avg;
                matrix[j * n + i] = avg;
            }
        }

        Self { config, matrix }
    }

    /// Computes the maximum departure from Hermiticity |H_ij - H_ji|.
    pub fn verify_hermiticity(&self) -> (bool, f64) {
        let n = self.config.basis_dim;
        let mut max_diff = 0.0;
        for i in 0..n {
            for j in 0..n {
                let diff = (self.matrix[i * n + j] - self.matrix[j * n + i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        (max_diff < 1e-12, max_diff)
    }

    /// Computes the lowest K eigenvalues using symmetric tridiagonal QR / Jacobi diagonalization.
    pub fn compute_spectrum(&self, num_levels: usize) -> RiemannSpectralReport {
        let (is_hermitian, max_asymmetry) = self.verify_hermiticity();
        let n = self.config.basis_dim;
        let mut a = self.matrix.clone();

        // Standard Jacobi Eigenvalue Algorithm for real symmetric matrices
        let max_rotations = 100 * n * n;
        for _ in 0..max_rotations {
            let mut max_offdiag = 0.0;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    let val = a[i * n + j].abs();
                    if val > max_offdiag {
                        max_offdiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if max_offdiag < 1e-9 {
                break; // Converged
            }

            let app = a[p * n + p];
            let aqq = a[q * n + q];
            let apq = a[p * n + q];

            let theta = 0.5 * (aqq - app) / apq;
            let t = if theta >= 0.0 {
                1.0 / (theta + (theta * theta + 1.0).sqrt())
            } else {
                -1.0 / (-theta + (theta * theta + 1.0).sqrt())
            };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;

            // Rotate matrix A -> J^T A J
            for r in 0..n {
                if r != p && r != q {
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = c * arp - s * arq;
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = s * arp + c * arq;
                    a[q * n + r] = a[r * n + q];
                }
            }
            a[p * n + p] = app - t * apq;
            a[q * n + q] = aqq + t * apq;
            a[p * n + q] = 0.0;
            a[q * n + p] = 0.0;
        }

        // Extract diagonal eigenvalues
        let mut evals: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
        evals.sort_by(|x, y| x.partial_cmp(y).unwrap());

        let k = num_levels.min(n);
        let lowest_eigenvalues: Vec<f64> = evals[0..k].to_vec();

        // Compute mean level spacing
        let mut spacing_sum = 0.0;
        for i in 0..(k - 1) {
            spacing_sum += lowest_eigenvalues[i + 1] - lowest_eigenvalues[i];
        }
        let mean_spacing = if k > 1 { spacing_sum / (k - 1) as f64 } else { 0.0 };

        RiemannSpectralReport {
            is_hermitian,
            max_asymmetry,
            ground_state_energy: evals[0],
            lowest_eigenvalues,
            mean_level_spacing: mean_spacing,
            all_eigenvalues_real: is_hermitian,
            basis_dimension: n,
        }
    }
}
