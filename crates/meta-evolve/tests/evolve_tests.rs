use backend_physics::GpuContext;
use frontend_dsl::constraints::Constraint;
use meta_evolve::evaluator::{evaluate_profile, EvalBenchmark};
use meta_evolve::profile::SpecializedSolverProfile;
use meta_evolve::strategy::{evolve_solver_profile, EvolutionConfig};

#[test]
fn test_default_profile_evaluation() {
    let ctx = GpuContext::new().expect("GPU context needed for test");

    let mut prob = EvalBenchmark::new("small_test", 4);
    prob.add_objective(&[0], -2.0);
    prob.add_objective(&[1], -2.0);
    prob.add_constraint(Constraint::ExactOne {
        vars: vec![0, 1],
        penalty_weight: 4.0,
    });

    let profile = SpecializedSolverProfile {
        dt: 0.5,
        a0: 1.0,
        c0_scale: 1.0,
        penalty_scale: 1.0,
        recommended_steps: 300,
        recommended_replicas: 16,
        fitness: 0.0,
        feasible_rate: 0.0,
        generation: 0,
    };

    let metrics = evaluate_profile(&profile, &[prob], &ctx, 42);
    println!("Evaluation Metrics: {:?}", metrics);
    assert!(metrics.fitness > -1e8);
    assert_eq!(metrics.feasible_rate, 100.0);
}

#[test]
fn test_evolutionary_profile_synthesis_gpu() {
    let ctx = GpuContext::new().expect("GPU context needed for evolution test");

    // Construct benchmark suite: 2 complementary problems
    let mut prob1 = EvalBenchmark::new("exact_one_and", 4);
    prob1.add_objective(&[0], -3.0);
    prob1.add_objective(&[1], -3.0);
    prob1.add_constraint(Constraint::ExactOne {
        vars: vec![0, 1],
        penalty_weight: 2.0,
    });
    prob1.add_constraint(Constraint::And {
        x: 0,
        y: 1,
        out: 2,
        penalty_weight: 2.0,
    });

    let mut prob2 = EvalBenchmark::new("or_xor", 4);
    prob2.add_objective(&[0], 1.0);
    prob2.add_constraint(Constraint::Or {
        vars: vec![0, 1],
        penalty_weight: 2.0,
    });
    prob2.add_constraint(Constraint::Xor {
        x: 0,
        y: 1,
        out: 3,
        penalty_weight: 2.0,
    });

    let config = EvolutionConfig {
        population_size: 4,
        num_generations: 2,
        elite_count: 2,
        mutation_rate: 0.3,
        seed: 1337,
    };

    let best_profile = evolve_solver_profile(&[prob1, prob2], &config, &ctx);

    println!("Synthesized Profile:\n{}", best_profile.display_summary());

    assert!(best_profile.fitness > -1e8);
    assert!(best_profile.dt >= 0.1 && best_profile.dt <= 1.2);
    assert!(best_profile.a0 >= 0.5 && best_profile.a0 <= 2.0);
    assert!(best_profile.c0_scale >= 0.2 && best_profile.c0_scale <= 5.0);
    assert!(best_profile.feasible_rate > 0.0);
}
