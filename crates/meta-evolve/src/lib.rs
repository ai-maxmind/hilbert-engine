//! `meta-evolve` crate:
//!
//! Provides the Meta-Evolutionary Synthesis Engine:
//! Automatically evolves hyperparameter profiles (`dt`, `a0`, `c0_scale`, penalty multiplier $M$)
//! using Evolutionary Strategy, evaluating candidates in parallel on GPU, and filtering
//! through the `verify-bridge`.

pub mod evaluator;
pub mod profile;
pub mod strategy;

pub use evaluator::{evaluate_profile, EvalBenchmark, EvaluationMetrics};
pub use profile::SpecializedSolverProfile;
pub use strategy::{evolve_solver_profile, EvolutionConfig};
