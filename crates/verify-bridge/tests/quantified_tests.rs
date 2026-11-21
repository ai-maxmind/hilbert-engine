//! Integration tests for Sprint 9: Adversarial Minimax Bifurcation & Saddle Point Verification.

use backend_physics::{solve_dsb_cpu, solve_dsb_gpu, DsbConfig, GpuContext};
use frontend_dsl::QuantifiedProblemBuilder;
use verify_bridge::{export_quantified_smtlib2, verify_saddle_point};

#[test]
fn test_matching_pennies_exact_nash_equilibrium() {
    // 2-variable game: 1 Minimizer (x), 1 Maximizer (y)
    // Hamiltonian: H(x, y) = - K x y - h_x x - h_y y
    // With h_x = 2.0, h_y = -2.0, K = 0.5:
    // Minimizer force: dH/dx = -(K y + h_x) = -(0.5 y + 2.0) < 0 for all y => x* = +1
    // Maximizer force: dH/dy = -(K x + h_y) = -(0.5(1) - 2.0) = +1.5 > 0 => y* = +1
    // Saddle Point (Nash Equilibrium): x* = +1, y* = +1.

    let mut builder = QuantifiedProblemBuilder::new(1, 1);
    builder.add_adversarial_interaction(0, 0, 0.5);
    builder.add_x_bias(0, 2.0);
    builder.add_y_bias(0, -2.0);

    let (ising, num_x) = builder.build_unified_ising().expect("CSR build failed");
    assert_eq!(ising.num_spins, 2);
    assert_eq!(num_x, 1);

    // 1. Test verification on optimal saddle state (+1, +1)
    let saddle_state = vec![1i8, 1i8];
    let rep_opt = verify_saddle_point(&ising, num_x, &saddle_state);
    assert!(rep_opt.is_local_nash_equilibrium, "Optimal state (+1, +1) must be Nash equilibrium");
    assert_eq!(rep_opt.minimizer_profitable_deviations, 0);
    assert_eq!(rep_opt.maximizer_profitable_deviations, 0);

    // 2. Test verification on non-equilibrium state (-1, +1)
    // Minimizer has profitable deviation to flip to +1
    let state_x_bad = vec![-1i8, 1i8];
    let rep_x_bad = verify_saddle_point(&ising, num_x, &state_x_bad);
    assert!(!rep_x_bad.is_local_nash_equilibrium);
    assert_eq!(rep_x_bad.minimizer_profitable_deviations, 1);

    // 3. Test verification on non-equilibrium state (+1, -1)
    // Maximizer has profitable deviation to flip to +1
    let state_y_bad = vec![1i8, -1i8];
    let rep_y_bad = verify_saddle_point(&ising, num_x, &state_y_bad);
    assert!(!rep_y_bad.is_local_nash_equilibrium);
    assert_eq!(rep_y_bad.maximizer_profitable_deviations, 1);

    // 4. Test SMT-LIB2 export
    let smt = export_quantified_smtlib2(&builder, "MatchingPenniesProof");
    assert!(smt.contains("(set-logic LIA)"));
    assert!(smt.contains("(declare-const x_0 Int)"));
    assert!(smt.contains("(declare-const y_0 Int)"));
    assert!(smt.contains("(check-sat)"));
}

#[test]
fn test_adversarial_minimax_dsb_solver_cpu() {
    // 8-variable Zero-Sum Game: 4 Minimizers (x0..x3), 4 Maximizers (y0..y3)
    // Structured with well-defined pure Nash Equilibrium:
    // Pair 0 & 2: biased toward (+1, +1)
    // Pair 1 & 3: biased toward (-1, -1)
    let num_x = 4;
    let num_y = 4;
    let mut builder = QuantifiedProblemBuilder::new(num_x, num_y);

    for i in 0..num_x {
        builder.add_adversarial_interaction(i, i, 1.0);
    }
    // Internal cooperative couplings
    builder.add_x_coupling(0, 2, 0.5);
    builder.add_x_coupling(1, 3, 0.5);
    builder.add_y_coupling(0, 2, 0.5);
    builder.add_y_coupling(1, 3, 0.5);

    // Well-posed saddle biases
    builder.add_x_bias(0, 1.5);
    builder.add_y_bias(0, -1.5);

    builder.add_x_bias(1, -1.5);
    builder.add_y_bias(1, 1.5);

    builder.add_x_bias(2, 2.0);
    builder.add_y_bias(2, -1.5);

    builder.add_x_bias(3, -2.0);
    builder.add_y_bias(3, 1.5);

    let (ising, split) = builder.build_unified_ising().expect("Unified Ising build failed");
    assert_eq!(split, num_x);
    assert_eq!(ising.num_spins, 8);

    let config = DsbConfig {
        num_steps: 600,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 32,
        seed: 42,
        mode: backend_physics::SolverPhysicsMode::Discrete,
        num_x: Some(num_x),
    };

    let result = solve_dsb_cpu(&ising, &config).expect("Adversarial CPU solver failed");
    assert_eq!(result.best_spins.len(), 8);

    // Verify saddle point properties
    let report = verify_saddle_point(&ising, num_x, &result.best_spins);
    println!("CPU Minimax Solution Report: {:?}", report);
    assert!(
        report.is_local_nash_equilibrium,
        "CPU Adversarial DSB solver must converge exactly to the Nash equilibrium"
    );
    assert_eq!(report.minimizer_profitable_deviations, 0);
    assert_eq!(report.maximizer_profitable_deviations, 0);
}

#[test]
fn test_adversarial_minimax_dsb_solver_gpu() {
    // Test GPU solver if Vulkan / GPU hardware context is available
    if GpuContext::new().is_err() {
        eprintln!("Skipping GPU test: no compatible GPU found");
        return;
    }

    // 16-variable Zero-Sum Game: 8 Minimizers, 8 Maximizers
    let num_x = 8;
    let num_y = 8;
    let mut builder = QuantifiedProblemBuilder::new(num_x, num_y);

    for i in 0..num_x {
        builder.add_adversarial_interaction(i, i, 1.2);
        if i % 2 == 0 {
            builder.add_x_bias(i, 2.0);
            builder.add_y_bias(i, -2.0);
        } else {
            builder.add_x_bias(i, -2.0);
            builder.add_y_bias(i, 2.0);
        }
    }
    builder.add_x_coupling(0, 2, 0.6);
    builder.add_y_coupling(1, 3, 0.6);

    let (ising, split) = builder.build_unified_ising().expect("Unified Ising failed");

    let config = DsbConfig {
        num_steps: 600,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 64,
        seed: 1337,
        mode: backend_physics::SolverPhysicsMode::Discrete,
        num_x: Some(split),
    };

    let result = solve_dsb_gpu(&ising, &config).expect("GPU Minimax solver failed");
    assert_eq!(result.best_spins.len(), 16);

    let report = verify_saddle_point(&ising, split, &result.best_spins);
    println!("GPU Minimax Solution Report: {:?}", report);
    assert!(
        report.is_local_nash_equilibrium,
        "GPU Adversarial DSB solver must converge exactly to the Nash equilibrium"
    );
    assert_eq!(report.minimizer_profitable_deviations, 0);
    assert_eq!(report.maximizer_profitable_deviations, 0);
}
