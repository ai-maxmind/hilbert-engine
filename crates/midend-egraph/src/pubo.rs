//! Canonical representation of Polynomial Unconstrained Binary Optimization (PUBO).
//!
//! # Mathematical Representation:
//!
//! A multilinear polynomial in $N$ binary variables $x_i \in \{0, 1\}$:
//!
//! $$P(\mathbf{x}) = c_0 + \sum_{i} c_i x_i + \sum_{i < j} c_{ij} x_i x_j + \sum_{i < j < k} c_{ijk} x_i x_j x_k + \dots$$
//!
//! Since binary variables satisfy $x_i^2 = x_i$, any product containing duplicate
//! variables collapses to square-free multilinear form: $x_i^p \equiv x_i$ for all $p \ge 1$.

use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PuboError {
    #[error("Variable index {index} out of bounds for num_variables {num_vars}")]
    VariableOutOfBounds { index: usize, num_vars: usize },

    #[error("Polynomial is not quadratic: maximum degree is {degree}")]
    NotQuadratic { degree: usize },

    #[error("Assignment length {actual} does not match variable count {expected}")]
    InvalidAssignmentLen { expected: usize, actual: usize },
}

/// A canonical multilinear polynomial in binary variables $x_i \in \{0, 1\}$.
#[derive(Debug, Clone, PartialEq)]
pub struct PuboPolynomial {
    /// Total number of variables in the polynomial.
    pub num_variables: usize,
    /// Mapping from sorted, deduplicated variable indices to monomial coefficients.
    /// An empty key `vec![]` represents the constant term $c_0$.
    pub terms: BTreeMap<Vec<usize>, f64>,
}

impl PuboPolynomial {
    /// Create a new empty polynomial with a designated number of variables.
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            terms: BTreeMap::new(),
        }
    }

    /// Add a monomial term $c \cdot \prod_{i \in \text{vars}} x_i$.
    ///
    /// Applies binary idempotence ($x_i^2 = x_i$) by sorting and deduplicating `vars`.
    pub fn add_term(&mut self, vars: &[usize], coeff: f64) -> Result<(), PuboError> {
        if coeff.abs() < 1e-12 {
            return Ok(());
        }

        let mut sorted_vars = vars.to_vec();
        sorted_vars.sort_unstable();
        sorted_vars.dedup(); // x_i^2 = x_i idempotence

        for &v in &sorted_vars {
            if v >= self.num_variables {
                return Err(PuboError::VariableOutOfBounds {
                    index: v,
                    num_vars: self.num_variables,
                });
            }
        }

        let entry = self.terms.entry(sorted_vars).or_insert(0.0);
        *entry += coeff;
        if entry.abs() < 1e-12 {
            // Clean up near-zero terms
            // Note: entry will be removed on vacuum or kept as negligible
        }
        Ok(())
    }

    /// Highest degree among all non-zero monomials in the polynomial.
    pub fn degree(&self) -> usize {
        self.terms
            .iter()
            .filter(|(_, &coeff)| coeff.abs() > 1e-9)
            .map(|(vars, _)| vars.len())
            .max()
            .unwrap_or(0)
    }

    /// Check if the polynomial is quadratic (degree <= 2).
    pub fn is_quadratic(&self) -> bool {
        self.degree() <= 2
    }

    /// Evaluate the polynomial at a specific binary assignment $\mathbf{x} \in \{0, 1\}^N$.
    pub fn evaluate(&self, assignment: &[u8]) -> Result<f64, PuboError> {
        if assignment.len() != self.num_variables {
            return Err(PuboError::InvalidAssignmentLen {
                expected: self.num_variables,
                actual: assignment.len(),
            });
        }

        let mut total = 0.0;
        for (vars, &coeff) in &self.terms {
            if coeff.abs() <= 1e-12 {
                continue;
            }
            if vars.is_empty() {
                total += coeff;
                continue;
            }
            let mut prod = 1.0;
            for &v in vars {
                if assignment[v] == 0 {
                    prod = 0.0;
                    break;
                }
            }
            total += coeff * prod;
        }

        Ok(total)
    }

    /// Compute the exact global minimum by brute-force evaluation over all $2^N$ assignments.
    ///
    /// Intended for testing and verification on $N \le 22$.
    pub fn brute_force_min(&self) -> (f64, Vec<u8>) {
        assert!(
            self.num_variables <= 24,
            "Brute force only feasible for N <= 24"
        );
        let total_states = 1usize << self.num_variables;
        let mut min_val = f64::INFINITY;
        let mut best_assignment = vec![0u8; self.num_variables];

        let mut current = vec![0u8; self.num_variables];
        for state in 0..total_states {
            for i in 0..self.num_variables {
                current[i] = ((state >> i) & 1) as u8;
            }
            let val = self.evaluate(&current).unwrap();
            if val < min_val {
                min_val = val;
                best_assignment.copy_from_slice(&current);
            }
        }

        (min_val, best_assignment)
    }
}
