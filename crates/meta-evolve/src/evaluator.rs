//! Benchmark evaluation of candidate compiler profiles using GPU and verification bridge.

use std::time::Instant;
use backend_physics::{solve_dsb_gpu_with_context, DsbConfig, GpuContext};
use frontend_dsl::constraints::Constraint;
use frontend_dsl::ProblemBuilder;
use midend_decompose::decompose_to_ising;
use midend_egraph::rosenberg::quadratize_rosenberg;
use verify_bridge::decoder::verify_constraints;

use crate::profile::SpecializedSolverProfile;

/// A self-contained combinatorial benchmark instance used to evaluate compiler profiles.
#[derive(Debug, Clone)]
pub struct EvalBenchmark {
    pub name: String,
    pub num_variables: usize,
    pub objectives: Vec<(Vec<usize>, f64)>,
    pub constraints: Vec<Constraint>,
}

impl EvalBenchmark {
    pub fn new(name: &str, num_variables: usize) -> Self {
        Self {
            name: name.to_string(),
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

/// Outcome of evaluating a profile across a benchmark suite.
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    pub fitness: f64,
    pub feasible_rate: f64,
    pub total_violations: usize,
    pub avg_time_ms: f64,
}

/// Evaluate a candidate profile on GPU, filtering through `verify-bridge`.
pub fn evaluate_profile(
    profile: &SpecializedSolverProfile,
    benchmarks: &[EvalBenchmark],
    ctx: &GpuContext,
    base_seed: u64,
) -> EvaluationMetrics {
    let mut total_violations = 0;
    let mut total_feasible = 0;
    let mut total_energy = 0.0;
    let mut total_time_ms = 0.0;

    for (prob_idx, prob) in benchmarks.iter().enumerate() {
        let t_start = Instant::now();

        // 1. Build problem with scaled constraint penalties
        let mut builder = ProblemBuilder::new();
        builder.add_variables(prob.num_variables);

        for (vars, weight) in &prob.objectives {
            builder.add_cost_polynomial(vars, *weight);
        }

        let mut pubo = match builder.build() {
            Ok(p) => p,
            Err(_) => return EvaluationMetrics { fitness: -1e9, feasible_rate: 0.0, total_violations: 999, avg_time_ms: 0.0 },
        };

        for c in &prob.constraints {
            let mut c_scaled = c.clone();
            match &mut c_scaled {
                Constraint::ExactOne { penalty_weight, .. } => *penalty_weight *= profile.penalty_scale,
                Constraint::And { penalty_weight, .. } => *penalty_weight *= profile.penalty_scale,
                Constraint::Or { penalty_weight, .. } => *penalty_weight *= profile.penalty_scale,
                Constraint::Xor { penalty_weight, .. } => *penalty_weight *= profile.penalty_scale,
                Constraint::Clause3Sat { penalty_weight, .. } => *penalty_weight *= profile.penalty_scale,
            }
            if c_scaled.apply_to_pubo(&mut pubo).is_err() {
                return EvaluationMetrics { fitness: -1e9, feasible_rate: 0.0, total_violations: 999, avg_time_ms: 0.0 };
            }
        }

        // 2. Midend E-Graph: Quadratize
        let quad_res = match quadratize_rosenberg(&pubo) {
            Ok(q) => q,
            Err(_) => return EvaluationMetrics { fitness: -1e9, feasible_rate: 0.0, total_violations: 999, avg_time_ms: 0.0 },
        };

        // 3. Midend Decomposer: Scale & CSR
        let decomposed = match decompose_to_ising(&quad_res.qubo, prob.num_variables) {
            Ok(d) => d,
            Err(_) => return EvaluationMetrics { fitness: -1e9, feasible_rate: 0.0, total_violations: 999, avg_time_ms: 0.0 },
        };

        // 4. Backend Physics GPU Execution
        let c0_base = decomposed.ising_csr.compute_c0();
        let c0_eff = c0_base * profile.c0_scale;

        let dsb_config = DsbConfig {
            num_steps: profile.recommended_steps,
            dt: profile.dt,
            a0: profile.a0,
            c0: Some(c0_eff),
            num_replicas: profile.recommended_replicas,
            seed: base_seed.wrapping_add(prob_idx as u64 * 31337),
            ..Default::default()
        };

        let dsb_res = match solve_dsb_gpu_with_context(ctx, &decomposed.ising_csr, &dsb_config) {
            Ok(r) => r,
            Err(_) => return EvaluationMetrics { fitness: -1e9, feasible_rate: 0.0, total_violations: 999, avg_time_ms: 0.0 },
        };

        let dur = t_start.elapsed();
        total_time_ms += dur.as_secs_f64() * 1000.0;

        // 5. Decode and Filter via verify-bridge
        let (cand_x, true_energy) = match decomposed.decode_spins(&dsb_res.best_spins) {
            Ok(p) => p,
            Err(_) => return EvaluationMetrics { fitness: -1e9, feasible_rate: 0.0, total_violations: 999, avg_time_ms: 0.0 },
        };

        let report = verify_constraints(&prob.constraints, &cand_x);
        total_violations += report.violation_count;
        if report.is_feasible {
            total_feasible += 1;
        }
        total_energy += true_energy;
    }

    let b_len = benchmarks.len().max(1) as f64;
    let feasible_rate = (total_feasible as f64 / b_len) * 100.0;
    let avg_time = total_time_ms / b_len;

    // Lexicographic fitness: Heavy penalty for constraint violations, then energy, then time
    let fitness = -((total_violations as f64 * 5000.0) + total_energy + (avg_time * 0.005));

    EvaluationMetrics {
        fitness,
        feasible_rate,
        total_violations,
        avg_time_ms: avg_time,
    }
}
