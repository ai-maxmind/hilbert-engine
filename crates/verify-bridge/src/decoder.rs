//! Solution decoding and constraint residual validation.

use frontend_dsl::constraints::Constraint;

/// Detailed verification report for a candidate solution against all constraints.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// True if all constraints are satisfied (0 violations).
    pub is_feasible: bool,
    /// Total number of violated constraints.
    pub violation_count: usize,
    /// Per-constraint satisfaction flag: `true` if satisfied, `false` if violated.
    pub constraint_status: Vec<bool>,
    /// Per-constraint residual values.
    pub residuals: Vec<f64>,
}

/// Evaluates whether a binary assignment satisfies a specific constraint.
pub fn evaluate_constraint(constraint: &Constraint, assignment: &[u8]) -> (bool, f64) {
    match constraint {
        Constraint::ExactOne { vars, .. } => {
            let sum: usize = vars.iter().map(|&v| assignment[v] as usize).sum();
            let satisfied = sum == 1;
            let residual = (sum as f64 - 1.0).abs();
            (satisfied, residual)
        }

        Constraint::And { x, y, out, .. } => {
            let expected = assignment[*x] & assignment[*y];
            let actual = assignment[*out];
            let satisfied = expected == actual;
            let residual = (expected as f64 - actual as f64).abs();
            (satisfied, residual)
        }

        Constraint::Or { vars, .. } => {
            let any_true = vars.iter().any(|&v| assignment[v] == 1);
            let residual = if any_true { 0.0 } else { 1.0 };
            (any_true, residual)
        }

        Constraint::Xor { x, y, out, .. } => {
            let expected = assignment[*x] ^ assignment[*y];
            let actual = assignment[*out];
            let satisfied = expected == actual;
            let residual = (expected as f64 - actual as f64).abs();
            (satisfied, residual)
        }

        Constraint::Clause3Sat { clause, .. } => {
            let s1 = if clause.l1.1 { assignment[clause.l1.0] == 1 } else { assignment[clause.l1.0] == 0 };
            let s2 = if clause.l2.1 { assignment[clause.l2.0] == 1 } else { assignment[clause.l2.0] == 0 };
            let s3 = if clause.l3.1 { assignment[clause.l3.0] == 1 } else { assignment[clause.l3.0] == 0 };
            let satisfied = s1 || s2 || s3;
            let residual = if satisfied { 0.0 } else { 1.0 };
            (satisfied, residual)
        }
    }
}

/// Verify a binary assignment against a slice of constraints.
pub fn verify_constraints(constraints: &[Constraint], assignment: &[u8]) -> VerificationReport {
    let mut constraint_status = Vec::with_capacity(constraints.len());
    let mut residuals = Vec::with_capacity(constraints.len());
    let mut violation_count = 0;

    for c in constraints {
        let (sat, res) = evaluate_constraint(c, assignment);
        if !sat {
            violation_count += 1;
        }
        constraint_status.push(sat);
        residuals.push(res);
    }

    VerificationReport {
        is_feasible: violation_count == 0,
        violation_count,
        constraint_status,
        residuals,
    }
}
