use backend_physics::GpuContext;
use frontend_dsl::constraints::Constraint;
use verify_bridge::alm::{run_augmented_lagrangian, AlmConfig, AlmProblem};
use verify_bridge::formal::{FormalChecker, ProblemVerifier};

#[test]
fn test_formal_checker_smtlib2_export() {
    let constraints = vec![
        Constraint::ExactOne {
            vars: vec![0, 1, 2],
            penalty_weight: 1.0,
        },
        Constraint::And {
            x: 0,
            y: 1,
            out: 3,
            penalty_weight: 1.0,
        },
        Constraint::Or {
            vars: vec![1, 2],
            penalty_weight: 1.0,
        },
        Constraint::Xor {
            x: 0,
            y: 2,
            out: 4,
            penalty_weight: 1.0,
        },
    ];

    let verifier = ProblemVerifier::new(5, constraints);
    let candidate = vec![1u8, 0u8, 0u8, 0u8, 1u8];

    let smt_script = verifier.export_smtlib2(Some(&candidate));

    // Verify key SMT-LIB2 syntactic components
    assert!(smt_script.contains("(set-logic QF_NIA)"));
    assert!(smt_script.contains("(declare-const x_0 Int)"));
    assert!(smt_script.contains("(declare-const x_4 Int)"));
    assert!(smt_script.contains("(assert (= (+ x_0 x_1 x_2) 1))"));
    assert!(smt_script.contains("(assert (= x_3 (* x_0 x_1)))"));
    assert!(smt_script.contains("(assert (>= (+ x_1 x_2) 1))"));
    assert!(smt_script.contains("(assert (= x_0 1))"));
    assert!(smt_script.contains("(check-sat)"));
    assert!(smt_script.contains("(get-model)"));

    // Verify solution report
    let report = verifier.verify_solution(&candidate);
    // Candidate satisfies ExactOne(1,0,0) and And(1,0 -> 0) and Xor(1^0 -> 1),
    // but violates Or(x_1, x_2) because both are 0!
    assert!(!report.is_feasible);
    assert_eq!(report.violation_count, 1);
    assert_eq!(report.constraint_status, vec![true, true, false, true]);
}

#[test]
fn test_alm_closed_loop_violation_elimination_gpu() {
    // Objective: Wants x_0 = 1, x_1 = 1 (each gives negative cost -5.0)
    // Hard constraint: ExactOne(x_0, x_1) (strictly forbids both being 1!)
    // If penalty is too weak initially (e.g. 0.5), solver picks x_0=1, x_1=1.
    // Closed-loop ALM detects violation, grows penalty (mu -> 1 -> 2 -> 4 -> 8...),
    // until the constraint dominates and exact feasibility is reached!

    let mut problem = AlmProblem::new(3);
    // Strong incentive to violate:
    problem.add_objective(&[0], -5.0);
    problem.add_objective(&[1], -5.0);

    // Hard constraint with intentionally weak initial penalty
    problem.add_constraint(Constraint::ExactOne {
        vars: vec![0, 1],
        penalty_weight: 0.5,
    });
    // Another constraint: x_2 = x_0 AND x_1
    problem.add_constraint(Constraint::And {
        x: 0,
        y: 1,
        out: 2,
        penalty_weight: 0.5,
    });

    let config = AlmConfig {
        max_outer_iterations: 6,
        initial_penalty: 0.5,
        penalty_growth_factor: 2.5,
        gpu_replicas: 32,
        gpu_steps: 400,
        seed: 42,
    };

    let ctx = GpuContext::new().expect("GPU context needed for ALM test");
    let result = run_augmented_lagrangian(&problem, &config, &ctx).expect("ALM loop failed");

    println!("ALM Result:");
    println!("  - Feasible:            {}", result.is_feasible);
    println!("  - Best Violations:     {}", result.best_violations);
    println!("  - Iterations:          {}", result.iterations_executed);
    println!("  - Final Penalties:     {:?}", result.final_penalties);
    println!("  - Best Solution:       {:?}", result.best_solution);
    println!("  - Violation History:   {:?}", result.violation_history);

    assert!(
        result.is_feasible,
        "ALM must drive constraint violations to 0!"
    );
    assert_eq!(result.best_violations, 0);

    // Verify ExactOne: exactly one of x_0, x_1 is 1
    let sum_01 = result.best_solution[0] + result.best_solution[1];
    assert_eq!(sum_01, 1, "ExactOne must be satisfied");

    // Verify And: x_2 must be x_0 & x_1 = 0
    assert_eq!(result.best_solution[2], 0);
}
