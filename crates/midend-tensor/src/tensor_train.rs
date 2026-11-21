//! Module `tensor_train` provides the Tensor-Train (TT) / Matrix Product State (MPS)
//! representation for high-degree binary polynomials and hyper-edge tensors.
//!
//! A rank-$k$ tensor $\mathcal{T}(x_1, x_2, \dots, x_k)$ is factored into a chain of $k$ 3-way TT-cores:
//!
//! $$\mathcal{T}(x_1, \dots, x_k) = A^{(1)}[x_1] \cdot A^{(2)}[x_2] \cdots A^{(k)}[x_k]$$
//!
//! where each $A^{(m)}[x_m]$ is a matrix of size $r_{m-1} \times r_m$, with boundary conditions
//! $r_0 = 1$ and $r_k = 1$, and $x_m \in \{0, 1\}$.

use thiserror::Error;

/// Errors arising during Tensor-Train operations.
#[derive(Debug, Error, PartialEq)]
pub enum TtError {
    #[error("Dimension mismatch: core {index} left rank {left_rank} does not match previous right rank {prev_rank}")]
    RankMismatch {
        index: usize,
        left_rank: usize,
        prev_rank: usize,
    },

    #[error("Empty Tensor-Train has no cores")]
    EmptyTensorTrain,

    #[error("Evaluation input length {actual} does not match number of cores {expected}")]
    InputLengthMismatch { expected: usize, actual: usize },

    #[error("Invalid binary input bit {val} at position {index}: must be 0 or 1")]
    InvalidBit { index: usize, val: u8 },

    #[error("Singular value decomposition failed or numerical instability")]
    SvdFailed,
}

/// A single 3-way TT-core tensor $A^{(m)} \in \mathbb{R}^{r_{\text{in}} \times 2 \times r_{\text{out}}}$.
#[derive(Debug, Clone, PartialEq)]
pub struct TtCore {
    /// Left bond dimension $r_{\text{in}}$.
    pub r_in: usize,
    /// Physical dimension (always 2 for binary variables $x \in \{0, 1\}$).
    pub physical_dim: usize,
    /// Right bond dimension $r_{\text{out}}$.
    pub r_out: usize,
    /// Flattened matrix storage: `data[x * (r_in * r_out) + i * r_out + j]`
    /// where $x \in \{0, 1\}$, $i \in [0, r_{\text{in}})$, $j \in [0, r_{\text{out}})$.
    pub data: Vec<f64>,
}

impl TtCore {
    /// Create a new TT-core of shape $(r_{\text{in}}, 2, r_{\text{out}})$ initialized with values.
    pub fn new(r_in: usize, r_out: usize, data: Vec<f64>) -> Result<Self, TtError> {
        let expected_len = r_in * 2 * r_out;
        if data.len() != expected_len {
            panic!("Data length {} does not match core shape ({} x 2 x {}) = {}",
                data.len(), r_in, r_out, expected_len);
        }
        Ok(Self {
            r_in,
            physical_dim: 2,
            r_out,
            data,
        })
    }

    /// Create a zero-initialized TT-core of shape $(r_{\text{in}}, 2, r_{\text{out}})$.
    pub fn zeros(r_in: usize, r_out: usize) -> Self {
        Self {
            r_in,
            physical_dim: 2,
            r_out,
            data: vec![0.0; r_in * 2 * r_out],
        }
    }

    /// Create an identity-passing TT-core (copies bond state for both $x=0$ and $x=1$).
    pub fn identity(r: usize) -> Self {
        let mut core = Self::zeros(r, r);
        for x in 0..2 {
            for i in 0..r {
                core.set(i, x, i, 1.0);
            }
        }
        core
    }

    /// Access element $A[i, x, j]$.
    #[inline]
    pub fn get(&self, i: usize, x: usize, j: usize) -> f64 {
        debug_assert!(i < self.r_in && x < 2 && j < self.r_out);
        self.data[x * (self.r_in * self.r_out) + i * self.r_out + j]
    }

    /// Set element $A[i, x, j] = \text{val}$.
    #[inline]
    pub fn set(&mut self, i: usize, x: usize, j: usize, val: f64) {
        debug_assert!(i < self.r_in && x < 2 && j < self.r_out);
        self.data[x * (self.r_in * self.r_out) + i * self.r_out + j] = val;
    }

    /// Multiply row vector $v \in \mathbb{R}^{r_{\text{in}}}$ by matrix $A[x] \in \mathbb{R}^{r_{\text{in}} \times r_{\text{out}}}$.
    /// Output is $v' \in \mathbb{R}^{r_{\text{out}}}$.
    pub fn vector_mat_mul(&self, v: &[f64], x: usize, out: &mut [f64]) {
        debug_assert_eq!(v.len(), self.r_in);
        debug_assert_eq!(out.len(), self.r_out);
        debug_assert!(x < 2);

        out.fill(0.0);
        let offset = x * (self.r_in * self.r_out);

        for i in 0..self.r_in {
            let vi = v[i];
            if vi.abs() < 1e-15 {
                continue;
            }
            let row_offset = offset + i * self.r_out;
            for j in 0..self.r_out {
                out[j] += vi * self.data[row_offset + j];
            }
        }
    }
}

/// A complete Tensor-Train chain representing a rank-$k$ tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorTrain {
    /// Chain of TT-cores from site $1$ to site $k$.
    pub cores: Vec<TtCore>,
}

impl TensorTrain {
    /// Construct and validate a Tensor-Train chain from a list of cores.
    pub fn new(cores: Vec<TtCore>) -> Result<Self, TtError> {
        if cores.is_empty() {
            return Err(TtError::EmptyTensorTrain);
        }

        // Validate boundary ranks: r_0 = 1 and r_k = 1
        if cores[0].r_in != 1 {
            return Err(TtError::RankMismatch {
                index: 0,
                left_rank: cores[0].r_in,
                prev_rank: 1,
            });
        }

        for i in 1..cores.len() {
            if cores[i].r_in != cores[i - 1].r_out {
                return Err(TtError::RankMismatch {
                    index: i,
                    left_rank: cores[i].r_in,
                    prev_rank: cores[i - 1].r_out,
                });
            }
        }

        let last_idx = cores.len() - 1;
        if cores[last_idx].r_out != 1 {
            return Err(TtError::RankMismatch {
                index: last_idx + 1,
                left_rank: 1,
                prev_rank: cores[last_idx].r_out,
            });
        }

        Ok(Self { cores })
    }

    /// Number of variables / sites $k$ in the Tensor-Train.
    #[inline]
    pub fn num_sites(&self) -> usize {
        self.cores.len()
    }

    /// Vector of bond dimensions $(r_0, r_1, \dots, r_k)$.
    pub fn bond_dimensions(&self) -> Vec<usize> {
        let mut ranks = Vec::with_capacity(self.cores.len() + 1);
        ranks.push(self.cores[0].r_in);
        for core in &self.cores {
            ranks.push(core.r_out);
        }
        ranks
    }

    /// Maximum bond dimension $\chi = \max_m r_m$.
    pub fn max_bond_dimension(&self) -> usize {
        self.cores.iter().map(|c| c.r_out.max(c.r_in)).max().unwrap_or(1)
    }

    /// Evaluate the Tensor-Train at a given binary bitstring $\mathbf{x} \in \{0, 1\}^k$.
    ///
    /// Computes $A^{(1)}[x_1] \cdot A^{(2)}[x_2] \cdots A^{(k)}[x_k]$ in $O(k \cdot \chi^2)$ time.
    pub fn eval(&self, bits: &[u8]) -> Result<f64, TtError> {
        if bits.len() != self.cores.len() {
            return Err(TtError::InputLengthMismatch {
                expected: self.cores.len(),
                actual: bits.len(),
            });
        }

        // Initialize v = [1.0] of size r_0 = 1
        let mut v = vec![1.0f64];
        let mut next_v = Vec::new();

        for (m, &b) in bits.iter().enumerate() {
            if b > 1 {
                return Err(TtError::InvalidBit { index: m, val: b });
            }
            let core = &self.cores[m];
            next_v.resize(core.r_out, 0.0);
            core.vector_mat_mul(&v, b as usize, &mut next_v);
            std::mem::swap(&mut v, &mut next_v);
        }

        Ok(v[0])
    }

    /// Compute the partition sum $\sum_{\mathbf{x} \in \{0, 1\}^k} \mathcal{T}(\mathbf{x})$ in $O(k \cdot \chi^2)$ time.
    pub fn sum_all(&self) -> f64 {
        let mut v = vec![1.0f64];
        let mut next_v = Vec::new();

        for core in &self.cores {
            next_v.resize(core.r_out, 0.0);
            next_v.fill(0.0);
            let mut tmp = vec![0.0f64; core.r_out];

            // Sum over x in {0, 1}
            for x in 0..2 {
                core.vector_mat_mul(&v, x, &mut tmp);
                for j in 0..core.r_out {
                    next_v[j] += tmp[j];
                }
            }
            std::mem::swap(&mut v, &mut next_v);
        }

        v[0]
    }
}
