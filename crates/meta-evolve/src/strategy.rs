//! Evolutionary Strategy for automated compiler hyperparameter synthesis.
//!
//! # Mathematical Optimization Formulation:
//!
//! $$\max_{\boldsymbol{\theta} \in \Omega} \mathcal{F}(\boldsymbol{\theta})$$
//!
//! where $\boldsymbol{\theta} = (\Delta t, a_0, c_0^{\text{scale}}, M^{\text{scale}}) \in \Omega$
//! and $\mathcal{F}(\boldsymbol{\theta})$ is evaluated by compiling benchmarks and running
//! parallel instances on the GPU.

use backend_physics::GpuContext;
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::evaluator::{evaluate_profile, EvalBenchmark};
use crate::profile::SpecializedSolverProfile;

/// Configuration for the evolutionary synthesis engine.
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Population size per generation (default: 8).
    pub population_size: usize,
    /// Number of evolutionary generations (default: 4).
    pub num_generations: usize,
    /// Number of elite individuals preserved unchanged (default: 2).
    pub elite_count: usize,
    /// Probability of mutating an offspring parameter (default: 0.35).
    pub mutation_rate: f32,
    /// PRNG seed for deterministic simulation testing.
    pub seed: u64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 8,
            num_generations: 4,
            elite_count: 2,
            mutation_rate: 0.35,
            seed: 42,
        }
    }
}

/// Run Evolutionary Strategy to synthesize the optimal `SpecializedSolverProfile`.
pub fn evolve_solver_profile(
    benchmarks: &[EvalBenchmark],
    config: &EvolutionConfig,
    ctx: &GpuContext,
) -> SpecializedSolverProfile {
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    // 1. Initialize random population within bounded parameter space Omega
    let mut population: Vec<SpecializedSolverProfile> = Vec::with_capacity(config.population_size);

    // Seed population with default profile
    population.push(SpecializedSolverProfile::default());

    while population.len() < config.population_size {
        let dt = rng.gen_range(0.3f32..0.8f32);
        let a0 = rng.gen_range(0.8f32..1.5f32);
        let c0_scale = rng.gen_range(0.5f32..2.0f32);
        let penalty_scale = rng.gen_range(1.5f64..4.5f64);

        population.push(SpecializedSolverProfile {
            dt,
            a0,
            c0_scale,
            penalty_scale,
            recommended_steps: 500,
            recommended_replicas: 32,
            fitness: -1e9,
            feasible_rate: 0.0,
            generation: 0,
        });
    }

    // 2. Initial Evaluation (Generation 0)
    for ind in &mut population {
        let metrics = evaluate_profile(ind, benchmarks, ctx, config.seed);
        ind.fitness = metrics.fitness;
        ind.feasible_rate = metrics.feasible_rate;
    }

    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    let mut best_profile = population[0].clone();

    // 3. Generational Evolutionary Loop
    for gen in 1..=config.num_generations {
        let mut new_population = Vec::with_capacity(config.population_size);

        // Elitism: Retain top individuals verbatim
        for elite in population.iter().take(config.elite_count) {
            let mut el = elite.clone();
            el.generation = gen;
            new_population.push(el);
        }

        // Generate offspring via Arithmetic Crossover and Gaussian Mutation
        while new_population.len() < config.population_size {
            let p1_idx = rng.gen_range(0..config.elite_count.max(1));
            let p2_idx = rng.gen_range(0..config.elite_count.max(1));
            let p1 = &population[p1_idx];
            let p2 = &population[p2_idx];

            let alpha = rng.gen_range(0.2f32..0.8f32);

            // Crossover
            let mut child_dt = alpha * p1.dt + (1.0 - alpha) * p2.dt;
            let mut child_a0 = alpha * p1.a0 + (1.0 - alpha) * p2.a0;
            let mut child_c0 = alpha * p1.c0_scale + (1.0 - alpha) * p2.c0_scale;
            let mut child_pen = (alpha as f64) * p1.penalty_scale + (1.0 - alpha as f64) * p2.penalty_scale;

            // Mutation
            if rng.gen_range(0.0f32..1.0f32) < config.mutation_rate {
                child_dt += rng.gen_range(-0.1f32..0.1f32);
            }
            if rng.gen_range(0.0f32..1.0f32) < config.mutation_rate {
                child_a0 += rng.gen_range(-0.15f32..0.15f32);
            }
            if rng.gen_range(0.0f32..1.0f32) < config.mutation_rate {
                child_c0 += rng.gen_range(-0.2f32..0.2f32);
            }
            if rng.gen_range(0.0f32..1.0f32) < config.mutation_rate {
                child_pen += rng.gen_range(-0.5f64..0.5f64);
            }

            // Clamping to valid physical boundaries
            child_dt = child_dt.clamp(0.1, 1.2);
            child_a0 = child_a0.clamp(0.5, 2.0);
            child_c0 = child_c0.clamp(0.2, 5.0);
            child_pen = child_pen.clamp(1.0, 10.0);

            new_population.push(SpecializedSolverProfile {
                dt: child_dt,
                a0: child_a0,
                c0_scale: child_c0,
                penalty_scale: child_pen,
                recommended_steps: p1.recommended_steps,
                recommended_replicas: p1.recommended_replicas,
                fitness: -1e9,
                feasible_rate: 0.0,
                generation: gen,
            });
        }

        // Evaluate generation offspring
        for ind in &mut new_population {
            if ind.fitness <= -1e8 {
                let metrics = evaluate_profile(ind, benchmarks, ctx, config.seed.wrapping_add(gen as u64 * 777));
                ind.fitness = metrics.fitness;
                ind.feasible_rate = metrics.feasible_rate;
            }
        }

        new_population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        if new_population[0].fitness > best_profile.fitness {
            best_profile = new_population[0].clone();
        }

        population = new_population;
    }

    best_profile
}
