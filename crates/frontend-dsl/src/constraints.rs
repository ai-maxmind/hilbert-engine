//! Formal definitions and polynomial expansions of hard combinatorial constraints.
//!
//! # Mathematical Penalty Functions
//!
//! Every constraint $C(\mathbf{x})$ is mapped to an energy penalty $P(\mathbf{x})$ such that:
//!
//! $$P(\mathbf{x}) = 0 \iff C(\mathbf{x}) \text{ is satisfied}$$
//! $$P(\mathbf{x}) \ge 1 \iff C(\mathbf{x}) \text{ is violated}$$

use midend_egraph::pubo::{PuboError, PuboPolynomial};

/// Literal in a Boolean clause: (variable_index, is_positive).
/// If `is_positive` is true, literal is $x_i$.
/// If `is_positive` is false, literal is $\neg x_i$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Literal(pub usize, pub bool);

/// 3-SAT Clause: $\ell_1 \lor \ell_2 \lor \ell_3$.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause3Sat {
    pub l1: Literal,
    pub l2: Literal,
    pub l3: Literal,
}

/// Hard constraint variants expressible in the Frontend DSL.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Exactly one of the designated variables must equal 1: $\sum_{i \in S} x_i = 1$.
    ExactOne {
        vars: Vec<usize>,
        penalty_weight: f64,
    },
    /// Logical AND gate: $\text{out} = x \land y$.
    And {
        x: usize,
        y: usize,
        out: usize,
        penalty_weight: f64,
    },
    /// Logical OR clause: $\bigvee_{i} x_i = 1$.
    Or {
        vars: Vec<usize>,
        penalty_weight: f64,
    },
    /// Logical XOR gate: $\text{out} = x \oplus y$.
    Xor {
        x: usize,
        y: usize,
        out: usize,
        penalty_weight: f64,
    },
    /// 3-SAT Clause: $\ell_1 \lor \ell_2 \lor \ell_3 = 1$.
    Clause3Sat {
        clause: Clause3Sat,
        penalty_weight: f64,
    },
}

impl Constraint {
    /// Expand the constraint into monomial penalty terms and add them to a target `PuboPolynomial`.
    pub fn apply_to_pubo(&self, pubo: &mut PuboPolynomial) -> Result<(), PuboError> {
        match self {
            Constraint::ExactOne { vars, penalty_weight } => {
                let m = *penalty_weight;
                // P = M * (sum x_i - 1)^2 = M * (1 - sum x_i + 2 sum_{i < j} x_i x_j)
                pubo.add_term(&[], m)?;
                for &v in vars {
                    pubo.add_term(&[v], -m)?;
                }
                let len = vars.len();
                for i in 0..len {
                    for j in (i + 1)..len {
                        pubo.add_term(&[vars[i], vars[j]], 2.0 * m)?;
                    }
                }
            }

            Constraint::And { x, y, out, penalty_weight } => {
                let m = *penalty_weight;
                // P = M * (x*y - 2*x*out - 2*y*out + 3*out)
                pubo.add_term(&[*x, *y], m)?;
                pubo.add_term(&[*x, *out], -2.0 * m)?;
                pubo.add_term(&[*y, *out], -2.0 * m)?;
                pubo.add_term(&[*out], 3.0 * m)?;
            }

            Constraint::Or { vars, penalty_weight } => {
                let m = *penalty_weight;
                // For 2 variables: (1 - x)(1 - y) = 1 - x - y + x*y
                if vars.len() == 2 {
                    let (x, y) = (vars[0], vars[1]);
                    pubo.add_term(&[], m)?;
                    pubo.add_term(&[x], -m)?;
                    pubo.add_term(&[y], -m)?;
                    pubo.add_term(&[x, y], m)?;
                } else if vars.len() == 3 {
                    let (x, y, z) = (vars[0], vars[1], vars[2]);
                    // (1 - x)(1 - y)(1 - z) = 1 - x - y - z + xy + yz + xz - xyz
                    pubo.add_term(&[], m)?;
                    pubo.add_term(&[x], -m)?;
                    pubo.add_term(&[y], -m)?;
                    pubo.add_term(&[z], -m)?;
                    pubo.add_term(&[x, y], m)?;
                    pubo.add_term(&[y, z], m)?;
                    pubo.add_term(&[x, z], m)?;
                    pubo.add_term(&[x, y, z], -m)?;
                } else {
                    // Fallback for general set via inclusion-exclusion expansion
                    expand_zero_product(vars, m, pubo)?;
                }
            }

            Constraint::Xor { x, y, out, penalty_weight } => {
                let m = *penalty_weight;
                // P = M * (x + y - 2xy - 2x*out - 2y*out + 4xy*out + out)
                pubo.add_term(&[*x], m)?;
                pubo.add_term(&[*y], m)?;
                pubo.add_term(&[*x, *y], -2.0 * m)?;
                pubo.add_term(&[*x, *out], -2.0 * m)?;
                pubo.add_term(&[*y, *out], -2.0 * m)?;
                pubo.add_term(&[*x, *y, *out], 4.0 * m)?;
                pubo.add_term(&[*out], m)?;
            }

            Constraint::Clause3Sat { clause, penalty_weight } => {
                let m = *penalty_weight;
                // Violation for literal l_k: v_k = if positive { 1 - x } else { x }
                // Penalty = M * v_1 * v_2 * v_3
                // Each v_k is represented as a polynomial: a_k + b_k * x_k
                let (a1, b1, x1) = if clause.l1.1 { (1.0, -1.0, clause.l1.0) } else { (0.0, 1.0, clause.l1.0) };
                let (a2, b2, x2) = if clause.l2.1 { (1.0, -1.0, clause.l2.0) } else { (0.0, 1.0, clause.l2.0) };
                let (a3, b3, x3) = if clause.l3.1 { (1.0, -1.0, clause.l3.0) } else { (0.0, 1.0, clause.l3.0) };

                // Multiply: (a1 + b1*x1) * (a2 + b2*x2) * (a3 + b3*x3)
                // Degree 0
                let c0 = a1 * a2 * a3 * m;
                if c0.abs() > 1e-12 { pubo.add_term(&[], c0)?; }

                // Degree 1
                let c_x1 = b1 * a2 * a3 * m;
                if c_x1.abs() > 1e-12 { pubo.add_term(&[x1], c_x1)?; }
                let c_x2 = a1 * b2 * a3 * m;
                if c_x2.abs() > 1e-12 { pubo.add_term(&[x2], c_x2)?; }
                let c_x3 = a1 * a2 * b3 * m;
                if c_x3.abs() > 1e-12 { pubo.add_term(&[x3], c_x3)?; }

                // Degree 2
                let c_x12 = b1 * b2 * a3 * m;
                if c_x12.abs() > 1e-12 { pubo.add_term(&[x1, x2], c_x12)?; }
                let c_x13 = b1 * a2 * b3 * m;
                if c_x13.abs() > 1e-12 { pubo.add_term(&[x1, x3], c_x13)?; }
                let c_x23 = a1 * b2 * b3 * m;
                if c_x23.abs() > 1e-12 { pubo.add_term(&[x2, x3], c_x23)?; }

                // Degree 3
                let c_x123 = b1 * b2 * b3 * m;
                if c_x123.abs() > 1e-12 { pubo.add_term(&[x1, x2, x3], c_x123)?; }
            }
        }
        Ok(())
    }
}

/// Helper expanding product prod_{i in vars} (1 - x_i) using inclusion-exclusion.
fn expand_zero_product(vars: &[usize], m: f64, pubo: &mut PuboPolynomial) -> Result<(), PuboError> {
    let n = vars.len();
    let total_subsets = 1usize << n;
    for mask in 0..total_subsets {
        let mut subset = Vec::new();
        for i in 0..n {
            if (mask & (1 << i)) != 0 {
                subset.push(vars[i]);
            }
        }
        let sign = if subset.len() % 2 == 0 { 1.0 } else { -1.0 };
        pubo.add_term(&subset, sign * m)?;
    }
    Ok(())
}
