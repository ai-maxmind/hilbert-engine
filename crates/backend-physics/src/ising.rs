//! Module `ising` provides the Compressed Sparse Row (CSR) representation
//! of the Ising Hamiltonian:
//!
//! $$H(\mathbf{s}) = - \frac{1}{2} \mathbf{s}^T J \mathbf{s} - \mathbf{h}^T \mathbf{s} = - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N J_{ij} s_i s_j - \sum_{i=1}^N h_i s_i$$
//!
//! where $s_i \in \{-1, +1\}$, $J_{ij} = J_{ji}$ is symmetric with $J_{ii} = 0$,
//! and $\mathbf{h} \in \mathbb{R}^N$ is the linear external magnetic field vector.

use thiserror::Error;

/// Errors arising during Ising model construction and validation.
#[derive(Debug, Error, PartialEq)]
pub enum IsingError {
    #[error("Number of spins ({num_spins}) exceeds maximum supported size")]
    TooManySpins { num_spins: usize },

    #[error("Invalid CSR row pointer length: expected {expected}, got {actual}")]
    InvalidRowPtrLen { expected: usize, actual: usize },

    #[error("CSR row pointer must start with 0, got {0}")]
    InvalidRowPtrStart(u32),

    #[error("Mismatch between row_ptr[num_spins] ({last_ptr}) and col_indices len ({indices_len})")]
    RowPtrIndicesMismatch { last_ptr: usize, indices_len: usize },

    #[error("Mismatch between col_indices len ({indices_len}) and values len ({values_len})")]
    IndicesValuesMismatch { indices_len: usize, values_len: usize },

    #[error("Mismatch between h vector len ({h_len}) and num_spins ({num_spins})")]
    InvalidHLen { h_len: usize, num_spins: usize },

    #[error("Column index {col} out of bounds for num_spins {num_spins}")]
    ColIndexOutOfBounds { col: u32, num_spins: usize },

    #[error("Non-zero diagonal self-coupling at spin {spin}: value {val}")]
    NonZeroDiagonal { spin: usize, val: f32 },

    #[error("Spin state length ({actual}) does not match model num_spins ({expected})")]
    InvalidSpinLen { expected: usize, actual: usize },

    #[error("Asymmetric coupling detected between spin {i} and {j}: J[{i},{j}]={val_ij}, J[{j},{i}]={val_ji}")]
    AsymmetricCoupling { i: usize, j: usize, val_ij: f32, val_ji: f32 },
}

/// Compressed Sparse Row (CSR) representation of an Ising coupling matrix $J$ and field $\mathbf{h}$.
///
/// Memory layout is 100% compatible with GPU buffer uploads (`bytemuck::cast_slice`).
#[derive(Debug, Clone, PartialEq)]
pub struct IsingCsr {
    /// Total number of spin variables $N$.
    pub num_spins: usize,
    /// CSR row offsets of size $N + 1$. `row_ptr[i]` is the starting index in `col_indices`/`values`.
    pub row_ptr: Vec<u32>,
    /// Column indices of non-zero coupling entries of size `nnz`.
    pub col_indices: Vec<u32>,
    /// Coupling coefficients $J_{ij}$ of size `nnz`.
    pub values: Vec<f32>,
    /// External linear magnetic fields $h_i$ of size $N$.
    pub h: Vec<f32>,
}

impl IsingCsr {
    /// Construct a verified `IsingCsr` instance with strict structural validation.
    pub fn new(
        num_spins: usize,
        row_ptr: Vec<u32>,
        col_indices: Vec<u32>,
        values: Vec<f32>,
        h: Vec<f32>,
    ) -> Result<Self, IsingError> {
        // 1. Check row pointer size
        if row_ptr.len() != num_spins + 1 {
            return Err(IsingError::InvalidRowPtrLen {
                expected: num_spins + 1,
                actual: row_ptr.len(),
            });
        }

        // 2. First row offset must be 0
        if row_ptr[0] != 0 {
            return Err(IsingError::InvalidRowPtrStart(row_ptr[0]));
        }

        let nnz = col_indices.len();

        // 3. Last row pointer must match nnz
        if row_ptr[num_spins] as usize != nnz {
            return Err(IsingError::RowPtrIndicesMismatch {
                last_ptr: row_ptr[num_spins] as usize,
                indices_len: nnz,
            });
        }

        // 4. Values and indices must have identical length
        if values.len() != nnz {
            return Err(IsingError::IndicesValuesMismatch {
                indices_len: nnz,
                values_len: values.len(),
            });
        }

        // 5. h length must equal num_spins
        if h.len() != num_spins {
            return Err(IsingError::InvalidHLen {
                h_len: h.len(),
                num_spins,
            });
        }

        // 6. Validate column bounds and ensure zero diagonal
        for i in 0..num_spins {
            let start = row_ptr[i] as usize;
            let end = row_ptr[i + 1] as usize;
            for k in start..end {
                let col = col_indices[k];
                if col as usize >= num_spins {
                    return Err(IsingError::ColIndexOutOfBounds { col, num_spins });
                }
                if col as usize == i && values[k].abs() > 1e-7 {
                    return Err(IsingError::NonZeroDiagonal {
                        spin: i,
                        val: values[k],
                    });
                }
            }
        }

        Ok(Self {
            num_spins,
            row_ptr,
            col_indices,
            values,
            h,
        })
    }

    /// Number of non-zero coupling entries in $J$.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Compute the Ising Hamiltonian energy for a given spin configuration $\mathbf{s} \in \{-1, +1\}^N$:
    ///
    /// $$E(\mathbf{s}) = - \frac{1}{2} \sum_{i=0}^{N-1} s_i \left( \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{values}[k] \cdot s_{\text{col\_indices}[k]} \right) - \sum_{i=0}^{N-1} h_i s_i$$
    pub fn compute_energy(&self, spins: &[i8]) -> Result<f64, IsingError> {
        if spins.len() != self.num_spins {
            return Err(IsingError::InvalidSpinLen {
                expected: self.num_spins,
                actual: spins.len(),
            });
        }

        let mut quadratic_sum: f64 = 0.0;
        let mut linear_sum: f64 = 0.0;

        for i in 0..self.num_spins {
            let si = spins[i] as f64;
            let start = self.row_ptr[i] as usize;
            let end = self.row_ptr[i + 1] as usize;

            let mut local_field: f64 = 0.0;
            for k in start..end {
                let j = self.col_indices[k] as usize;
                let sj = spins[j] as f64;
                local_field += (self.values[k] as f64) * sj;
            }

            quadratic_sum += si * local_field;
            linear_sum += (self.h[i] as f64) * si;
        }

        // Energy = -0.5 * s^T J s - h^T s
        Ok(-0.5 * quadratic_sum - linear_sum)
    }

    /// Sparse Matrix-Vector multiplication (SpMV):
    ///
    /// $$\text{out}[i] = \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{values}[k] \cdot x[\text{col\_indices}[k]]$$
    pub fn spmv(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.num_spins);
        debug_assert_eq!(out.len(), self.num_spins);

        for i in 0..self.num_spins {
            let start = self.row_ptr[i] as usize;
            let end = self.row_ptr[i + 1] as usize;
            let mut sum: f32 = 0.0;
            for k in start..end {
                let col = self.col_indices[k] as usize;
                sum += self.values[k] * x[col];
            }
            out[i] = sum;
        }
    }

    /// Calculate Goto's theoretical coupling scale constant $c_0$:
    ///
    /// $$\sigma_J = \sqrt{\frac{\sum_{k=0}^{\text{nnz}-1} \text{values}[k]^2}{N(N-1)}}$$
    /// $$c_0 = \frac{0.5}{\sigma_J \sqrt{N}} = \frac{0.5 \sqrt{N-1}}{\sqrt{\sum_{k=0}^{\text{nnz}-1} \text{values}[k]^2}}$$
    ///
    /// If $N \le 1$ or $\sum \text{values}^2 = 0$, defaults to $1.0$.
    pub fn compute_c0(&self) -> f32 {
        if self.num_spins <= 1 || self.values.is_empty() {
            return 1.0;
        }

        let sum_sq: f64 = self.values.iter().map(|&v| (v as f64) * (v as f64)).sum();
        if sum_sq <= 1e-12 {
            return 1.0;
        }

        let n = self.num_spins as f64;
        let c0 = 0.5 * (n - 1.0).sqrt() / sum_sq.sqrt();
        c0 as f32
    }

    /// Compute the saddle-point equilibrium violation (Nash exploitability) for Minimax problems.
    ///
    /// For joint spin state $\mathbf{s} = (\mathbf{x}, \mathbf{y})$ with $N_x$ Minimizer spins:
    /// - For $i < N_x$ (Minimizer): wants $\Delta H_i = 2 s_i F_i \ge 0$. Profitable deviation if $\Delta H_i < 0$.
    /// - For $k \ge N_x$ (Maximizer): wants $\Delta H_k = 2 s_k F_k \le 0$. Profitable deviation if $\Delta H_k > 0$.
    ///
    /// Returns `(total_violation_magnitude, num_profitable_deviations)`.
    pub fn compute_saddle_violation(&self, num_x: usize, spins: &[i8]) -> (f64, usize) {
        let n = self.num_spins;
        debug_assert_eq!(spins.len(), n);
        debug_assert!(num_x <= n);

        let mut total_violation = 0.0f64;
        let mut num_deviations = 0usize;

        for i in 0..n {
            let start = self.row_ptr[i] as usize;
            let end = self.row_ptr[i + 1] as usize;

            let mut field = self.h[i] as f64;
            for k in start..end {
                let j = self.col_indices[k] as usize;
                field += (self.values[k] as f64) * (spins[j] as f64);
            }

            let delta_h = 2.0 * (spins[i] as f64) * field;

            if i < num_x {
                if delta_h < -1e-6 {
                    total_violation += -delta_h;
                    num_deviations += 1;
                }
            } else {
                if delta_h > 1e-6 {
                    total_violation += delta_h;
                    num_deviations += 1;
                }
            }
        }

        (total_violation, num_deviations)
    }

    /// Construct `IsingCsr` from a list of undirected edges $(u, v, w)$ where $u \ne v$.
    ///
    /// Since the Ising interaction is symmetric ($J_{uv} = J_{vu} = w$), each edge $(u, v, w)$
    /// adds both $J_{uv} = w$ and $J_{vu} = w$.
    pub fn from_undirected_edges(
        num_spins: usize,
        edges: &[(usize, usize, f32)],
        h: Option<&[f32]>,
    ) -> Result<Self, IsingError> {
        let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_spins];

        for &(u, v, w) in edges {
            if u >= num_spins {
                return Err(IsingError::ColIndexOutOfBounds {
                    col: u as u32,
                    num_spins,
                });
            }
            if v >= num_spins {
                return Err(IsingError::ColIndexOutOfBounds {
                    col: v as u32,
                    num_spins,
                });
            }
            if u == v {
                if w.abs() > 1e-7 {
                    return Err(IsingError::NonZeroDiagonal { spin: u, val: w });
                }
                continue;
            }
            adj[u].push((v as u32, w));
            adj[v].push((u as u32, w));
        }

        let mut row_ptr: Vec<u32> = Vec::with_capacity(num_spins + 1);
        let mut col_indices: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();

        row_ptr.push(0);

        for row in 0..num_spins {
            let mut neighbors = std::mem::take(&mut adj[row]);
            neighbors.sort_by_key(|&(col, _)| col);

            // Merge duplicate entries if any
            let current_offset = *row_ptr.last().unwrap();
            for (col, val) in neighbors {
                if let Some(&last_col) = col_indices.last() {
                    if last_col == col && col_indices.len() > current_offset as usize {
                        let last_idx = values.len() - 1;
                        values[last_idx] += val;
                        continue;
                    }
                }
                col_indices.push(col);
                values.push(val);
            }
            row_ptr.push(col_indices.len() as u32);
        }

        let h_vec = match h {
            Some(slice) => {
                if slice.len() != num_spins {
                    return Err(IsingError::InvalidHLen {
                        h_len: slice.len(),
                        num_spins,
                    });
                }
                slice.to_vec()
            }
            None => vec![0.0f32; num_spins],
        };

        Self::new(num_spins, row_ptr, col_indices, values, h_vec)
    }

    /// Construct `IsingCsr` from dense $N \times N$ matrix and optional linear fields $h$.
    pub fn from_dense(matrix: &[Vec<f32>], h: Option<&[f32]>) -> Result<Self, IsingError> {
        let n = matrix.len();
        let mut row_ptr: Vec<u32> = Vec::with_capacity(n + 1);
        let mut col_indices: Vec<u32> = Vec::new();
        let mut values: Vec<f32> = Vec::new();

        row_ptr.push(0);

        for (i, row) in matrix.iter().enumerate() {
            if row.len() != n {
                return Err(IsingError::InvalidRowPtrLen {
                    expected: n,
                    actual: row.len(),
                });
            }
            for (j, &val) in row.iter().enumerate() {
                if i == j {
                    if val.abs() > 1e-7 {
                        return Err(IsingError::NonZeroDiagonal { spin: i, val });
                    }
                    continue;
                }
                // Verify symmetry
                let val_ji = matrix[j][i];
                if (val - val_ji).abs() > 1e-5 {
                    return Err(IsingError::AsymmetricCoupling {
                        i,
                        j,
                        val_ij: val,
                        val_ji,
                    });
                }

                if val.abs() > 1e-7 {
                    col_indices.push(j as u32);
                    values.push(val);
                }
            }
            row_ptr.push(col_indices.len() as u32);
        }

        let h_vec = match h {
            Some(slice) => {
                if slice.len() != n {
                    return Err(IsingError::InvalidHLen {
                        h_len: slice.len(),
                        num_spins: n,
                    });
                }
                slice.to_vec()
            }
            None => vec![0.0f32; n],
        };

        Self::new(n, row_ptr, col_indices, values, h_vec)
    }
}
