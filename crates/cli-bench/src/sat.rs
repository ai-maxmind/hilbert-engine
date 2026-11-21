//! End-to-end 3-SAT generator, pipeline compiler, and verifier.
//!
//! # 3-SAT Formulation:
//!
//! A Boolean Satisfiability formula with $N = 20$ variables and $C = 80$ clauses:
//!
//! $$\Phi = \bigwedge_{m=1}^{80} (\ell_{m, 1} \lor \ell_{m, 2} \lor \ell_{m, 3})$$
//!
//! Each clause is mapped to a degree-3 penalty monomial:
//!
//! $$P_m(\mathbf{x}) = v_{m, 1} \cdot v_{m, 2} \cdot v_{m, 3}$$
//!
//! where $v_{m, k} = (1 - x_i)$ if literal $\ell_{m, k} = x_i$, and $v_{m, k} = x_i$ if $\ell_{m, k} = \neg x_i$.

use backend_physics::{DsbConfig, DsbResult, GpuContext};
use frontend_dsl::constraints::Clause3Sat;
use frontend_dsl::ProblemBuilder;
use midend_decompose::{decompose_to_ising, DecomposedIsing};
use midend_egraph::rosenberg::quadratize_rosenberg;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// A generated 3-SAT instance with planted satisfiability guarantee.
pub struct SatInstance {
    pub num_vars: usize,
    pub num_clauses: usize,
    pub planted_solution: Vec<u8>,
    pub clauses: Vec<Clause3Sat>,
    pub builder: ProblemBuilder,
}

impl SatInstance {
    /// Generate a 3-SAT problem instance guaranteed to be satisfiable.
    pub fn generate_planted(num_vars: usize, num_clauses: usize, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut builder = ProblemBuilder::new();
        builder.add_variables(num_vars);

        // 1. Plant a secret ground truth assignment x*
        let planted: Vec<u8> = (0..num_vars).map(|_| rng.gen_range(0..=1)).collect();

        // 2. Generate random 3-SAT clauses compatible with x*
        let mut clauses = Vec::with_capacity(num_clauses);
        let all_vars: Vec<usize> = (0..num_vars).collect();

        for _ in 0..num_clauses {
            // Pick 3 distinct variables
            let chosen: Vec<usize> = all_vars.choose_multiple(&mut rng, 3).cloned().collect();
            let mut pol = [rng.gen_bool(0.5), rng.gen_bool(0.5), rng.gen_bool(0.5)];

            // Ensure clause is satisfied by planted solution
            let sat0 = if pol[0] { planted[chosen[0]] == 1 } else { planted[chosen[0]] == 0 };
            let sat1 = if pol[1] { planted[chosen[1]] == 1 } else { planted[chosen[1]] == 0 };
            let sat2 = if pol[2] { planted[chosen[2]] == 1 } else { planted[chosen[2]] == 0 };

            if !sat0 && !sat1 && !sat2 {
                // Invert one literal so at least one is satisfied
                let idx_to_flip = rng.gen_range(0..3);
                pol[idx_to_flip] = !pol[idx_to_flip];
            }

            builder.add_clause_3sat(
                (chosen[0], pol[0]),
                (chosen[1], pol[1]),
                (chosen[2], pol[2]),
                1.0,
            );

            clauses.push(Clause3Sat {
                l1: frontend_dsl::constraints::Literal(chosen[0], pol[0]),
                l2: frontend_dsl::constraints::Literal(chosen[1], pol[1]),
                l3: frontend_dsl::constraints::Literal(chosen[2], pol[2]),
            });
        }

        Self {
            num_vars,
            num_clauses,
            planted_solution: planted,
            clauses,
            builder,
        }
    }

    /// Verify how many clauses are satisfied by a given binary variable assignment.
    pub fn count_satisfied_clauses(&self, assignment: &[u8]) -> usize {
        let mut count = 0;
        for c in &self.clauses {
            let s1 = if c.l1.1 { assignment[c.l1.0] == 1 } else { assignment[c.l1.0] == 0 };
            let s2 = if c.l2.1 { assignment[c.l2.0] == 1 } else { assignment[c.l2.0] == 0 };
            let s3 = if c.l3.1 { assignment[c.l3.0] == 1 } else { assignment[c.l3.0] == 0 };
            if s1 || s2 || s3 {
                count += 1;
            }
        }
        count
    }
}

/// Compilation pipeline output holding intermediate artifacts.
pub struct SatPipeline {
    pub instance: SatInstance,
    pub decomposed: DecomposedIsing,
}

/// Run the entire compilation pipeline from DSL to GPU-ready CSR Ising model.
pub fn compile_3sat_pipeline(num_vars: usize, num_clauses: usize, seed: u64) -> SatPipeline {
    // 1. Frontend DSL Construction
    let instance = SatInstance::generate_planted(num_vars, num_clauses, seed);
    let pubo = instance.builder.clone().build().expect("DSL build failed");

    // 2. Midend E-Graph: Rosenberg Quadratization with Ancilla Reuse
    let quad_res = quadratize_rosenberg(&pubo).expect("Quadratization failed");

    // 3. Midend Decomposer: Petgraph Analysis, Dynamic Range Scaling, Ising CSR
    let decomposed = decompose_to_ising(&quad_res.qubo, num_vars).expect("Decomposition failed");

    SatPipeline {
        instance,
        decomposed,
    }
}

/// Execute end-to-end 3-SAT solution on GPU and verify satisfaction.
pub fn solve_3sat_end_to_end_gpu(
    pipeline: &SatPipeline,
    ctx: &GpuContext,
    replicas: usize,
    steps: usize,
    seed: u64,
) -> (DsbResult, Vec<u8>, usize) {
    let config = DsbConfig {
        num_steps: steps,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: replicas,
        seed,
        ..Default::default()
    };

    let dsb_result = backend_physics::solve_dsb_gpu_with_context(
        ctx,
        &pipeline.decomposed.ising_csr,
        &config,
    )
    .expect("GPU dSB solver failed");

    let (decoded_x, _true_energy) = pipeline
        .decomposed
        .decode_spins(&dsb_result.best_spins)
        .expect("Decoding failed");

    let satisfied = pipeline.instance.count_satisfied_clauses(&decoded_x);
    (dsb_result, decoded_x, satisfied)
}
