//! Module `alm` implements the closed-loop Augmented Lagrangian Method (ALM)
//! with adaptive, localized penalty weight adjustment.
//!
//! # Mathematical Foundations of Closed-Loop ALM
//!
//! Optimization problems with hard constraints $g_k(\mathbf{x}) = 0$ ($k = 1, \dots, K$)
//! and objective $f_0(\mathbf{x})$:
//!
//! $$\min_{\mathbf{x} \in \{0, 1\}^N} f_0(\mathbf{x}) \quad \text{subject to } g_k(\mathbf{x}) = 0$$
//!
//! Instead of guessing a static global penalty $M$ (which causes numerical distortion
//! if too high, or constraint violations if too low), the **Augmented Lagrangian Loop**
//! dynamically adjusts penalty weights $\mu_k$ per constraint:
//!
//! $$\mathcal{L}^{(\ell)}(\mathbf{x}) = f_0(\mathbf{x}) + \sum_{k=1}^K \mu_k^{(\ell)} P_k(\mathbf{x})$$
//!
//! ## Multiplier Update Rule:
//!
//! After solving on GPU and obtaining candidate $\mathbf{x}^{(\ell)}$:
//!
//! $$\mu_k^{(\ell+1)} = \begin{cases} \gamma \cdot \mu_k^{(\ell)} & \text{if constraint } k \text{ is violated} \\ \mu_k^{(\ell)} & \text{if constraint } k \text{ is satisfied} \end{cases}$$
//!
//! where growth factor $\gamma > 1$ (typically $2.0$).
//! The loop terminates immediately when **0 constraints are violated** ($\sum_k r_k = 0$).

use backend_physics::{solve_dsb_gpu_with_context, DsbConfig, GpuContext};
use frontend_dsl::constraints::Constraint;
use frontend_dsl::ProblemBuilder;
use midend_decompose::decompose_to_ising;
use midend_egraph::rosenberg::quadratize_rosenberg;
use thiserror::Error;

use crate::decoder::verify_constraints;

#[derive(Debug, Error)]
pub enum AlmError {
    #[error("DSL build error: {0}")]
    Dsl(#[from] frontend_dsl::builder::DslError),

    #[error("E-Graph quadratization error: {0}")]
    EGraph(#[from] midend_egraph::pubo::PuboError),

    #[error("Decomposition error: {0}")]
    Decompose(#[from] midend_decompose::DecomposeError),

    #[error("GPU execution error: {0}")]
    Gpu(#[from] backend_physics::GpuError),
}

/// Configuration parameters for the Augmented Lagrangian Loop.
#[derive(Debug, Clone)]
pub struct AlmConfig {
    /// Maximum outer closed-loop iterations (default: 8).
    pub max_outer_iterations: usize,
    /// Initial penalty weight $\mu_0$ for each constraint (default: 1.0).
    pub initial_penalty: f64,
    /// Multiplicative penalty growth rate $\gamma$ for violated constraints (default: 2.0).
    pub penalty_growth_factor: f64,
    /// Number of parallel simulation replicas on GPU (default: 64).
    pub gpu_replicas: usize,
    /// Number of dSB integration time steps (default: 600).
    pub gpu_steps: usize,
    /// Base seed for deterministic PRNG.
    pub seed: u64,
}

impl Default for AlmConfig {
    fn default() -> Self {
        Self {
            max_outer_iterations: 8,
            initial_penalty: 1.0,
            penalty_growth_factor: 2.0,
            gpu_replicas: 64,
            gpu_steps: 600,
            seed: 42,
        }
    }
}

/// Result of running the Augmented Lagrangian optimization loop.
#[derive(Debug, Clone)]
pub struct AlmResult {
    /// Best binary solution found $\mathbf{x}^* \in \{0, 1\}^N$.
    pub best_solution: Vec<u8>,
    /// Number of violated constraints in best solution (0 if feasible).
    pub best_violations: usize,
    /// True if the best solution is completely feasible (0 violations).
    pub is_feasible: bool,
    /// Number of outer loop iterations executed before convergence.
    pub iterations_executed: usize,
    /// Final adaptive penalty weights per constraint.
    pub final_penalties: Vec<f64>,
    /// History of violation counts per outer iteration.
    pub violation_history: Vec<usize>,
}

/// Problem definition for the ALM solver loop.
#[derive(Debug, Clone)]
pub struct AlmProblem {
    pub num_variables: usize,
    pub objectives: Vec<(Vec<usize>, f64)>,
    pub constraints: Vec<Constraint>,
}

impl AlmProblem {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            objectives: Vec::new(),
            constraints: Vec::new(),
        }
    }

    pub fn add_objective(&mut self, vars: &[usize], weight: f64) {
        self.objectives.push((vars.to_vec(), weight));
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }
}

/// Execute the closed-loop Augmented Lagrangian optimization loop on GPU.
pub fn run_augmented_lagrangian(
    problem: &AlmProblem,
    config: &AlmConfig,
    ctx: &GpuContext,
) -> Result<AlmResult, AlmError> {
    let k = problem.constraints.len();
    let mut penalties = vec![config.initial_penalty; k];

    let mut best_solution = vec![0u8; problem.num_variables];
    let mut best_violations = usize::MAX;
    let mut violation_history = Vec::with_capacity(config.max_outer_iterations);
    let mut iterations_executed = 0;

    for iter in 0..config.max_outer_iterations {
        iterations_executed += 1;

        // 1. Build problem with current localized penalty multipliers
        let mut builder = ProblemBuilder::new();
        builder.add_variables(problem.num_variables);

        for (vars, weight) in &problem.objectives {
            builder.add_cost_polynomial(vars, *weight);
        }

        let mut pubo = builder.build()?;
        for (idx, c) in problem.constraints.iter().enumerate() {
            let mut c_weighted = c.clone();
            match &mut c_weighted {
                Constraint::ExactOne { penalty_weight, .. } => *penalty_weight = penalties[idx],
                Constraint::And { penalty_weight, .. } => *penalty_weight = penalties[idx],
                Constraint::Or { penalty_weight, .. } => *penalty_weight = penalties[idx],
                Constraint::Xor { penalty_weight, .. } => *penalty_weight = penalties[idx],
                Constraint::Clause3Sat { penalty_weight, .. } => *penalty_weight = penalties[idx],
            }
            c_weighted.apply_to_pubo(&mut pubo)?;
        }

        // 2. Midend E-Graph: Quadratize with Ancilla Reuse
        let quad_res = quadratize_rosenberg(&pubo)?;

        // 3. Midend Decomposer: Dynamic Range Scaling & CSR Compilation
        let decomposed = decompose_to_ising(&quad_res.qubo, problem.num_variables)?;

        // 4. Backend Physics: Solve on GPU
        let dsb_config = DsbConfig {
            num_steps: config.gpu_steps,
            dt: 0.5,
            a0: 1.0,
            c0: None,
            num_replicas: config.gpu_replicas,
            seed: config.seed.wrapping_add(iter as u64 * 10007),
            ..Default::default()
        };

        let dsb_res = solve_dsb_gpu_with_context(ctx, &decomposed.ising_csr, &dsb_config)?;

        // 5. Decode solution
        let (cand_x, _energy) = decomposed.decode_spins(&dsb_res.best_spins)?;

        // 6. Verify constraints
        let report = verify_constraints(&problem.constraints, &cand_x);
        violation_history.push(report.violation_count);

        if report.violation_count < best_violations {
            best_violations = report.violation_count;
            best_solution.copy_from_slice(&cand_x);
        }

        // 7. Termination check: 0 violations => 100% Feasible!
        if report.is_feasible {
            break;
        }

        // 8. Adaptive Penalty Update: Increase penalty only for violated constraints
        for idx in 0..k {
            if !report.constraint_status[idx] {
                penalties[idx] *= config.penalty_growth_factor;
            }
        }
    }

    Ok(AlmResult {
        best_solution,
        best_violations,
        is_feasible: best_violations == 0,
        iterations_executed,
        final_penalties: penalties,
        violation_history,
    })
}
