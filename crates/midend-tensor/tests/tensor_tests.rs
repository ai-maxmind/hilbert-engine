//! Integration and validation tests for `midend-tensor` (Sprint 10 - Tensor Networks).

use backend_physics::{solve_dsb_cpu, DsbConfig};
use midend_tensor::{MpoPolynomialBuilder, TensorTrainIsingProjector};

#[test]
fn test_exact_monomial_mpo_evaluation_degree_8() {
    let n = 8;
    let vars = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let coeff = 4.2;

    let tt = MpoPolynomialBuilder::build_single_monomial_tt(n, &vars, coeff).expect("TT build failed");
    assert_eq!(tt.num_sites(), 8);
    assert_eq!(tt.max_bond_dimension(), 1, "Single monomial TT must have rank 1");

    // All ones bitstring: value must be 4.2
    let all_ones = vec![1u8; 8];
    let val_ones = tt.eval(&all_ones).expect("Eval failed");
    assert!((val_ones - 4.2).abs() < 1e-12);

    // Any bitstring with at least one 0: value must be 0.0
    for i in 0..8 {
        let mut bits = vec![1u8; 8];
        bits[i] = 0;
        let val = tt.eval(&bits).expect("Eval failed");
        assert_eq!(val, 0.0, "Monomial with 0 bit must evaluate to 0");
    }

    // Partition sum: only 1 state has value 4.2, so sum over all 256 states must be 4.2!
    let sum = tt.sum_all();
    assert!((sum - 4.2).abs() < 1e-12);
}

#[test]
fn test_pubo_mpo_evaluation_multiple_terms() {
    let n = 10;
    let mut builder = MpoPolynomialBuilder::new(n);

    // Add high-degree hyper-edges: degree 3, degree 4, degree 6
    builder.add_monomial(&[0, 1, 2], 2.0);
    builder.add_monomial(&[2, 3, 4, 5], -3.5);
    builder.add_monomial(&[1, 3, 5, 7, 8, 9], 5.0);

    let mpo = builder.build_polynomial_mpo().expect("MPO build failed");
    assert_eq!(mpo.num_sites(), 10);
    assert!(mpo.max_bond_dimension() <= 4);

    // Test a bitstring where all 3 terms are active: x = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    let bits_all_active = vec![1, 1, 1, 1, 1, 1, 0, 1, 1, 1];
    let val = mpo.eval(&bits_all_active).expect("Eval failed");
    let expected = 2.0 * (1*1*1) as f64 - 3.5 * (1*1*1*1) as f64 + 5.0 * (1*1*1*1*1*1) as f64;
    assert!((val - expected).abs() < 1e-12);

    // Test zero bitstring
    let bits_zero = vec![0u8; 10];
    let val_zero = mpo.eval(&bits_zero).expect("Eval failed");
    assert_eq!(val_zero, 0.0);
}

#[test]
fn test_tensor_ising_projection_zero_ancilla_degree_10() {
    let n = 10;
    let mut builder = MpoPolynomialBuilder::new(n);

    // Monomial of degree 10: - 10.0 * x0 * x1 * ... * x9
    // Minimizing this PUBO prefers x = [1, 1, ..., 1] with energy -10.0
    let all_vars: Vec<usize> = (0..n).collect();
    builder.add_monomial(&all_vars, -10.0);

    // Project using Tensor-Train Low-Rank Projector
    let ising = TensorTrainIsingProjector::project_pubo_to_ising(&builder)
        .expect("Projection to Ising failed");

    // Critical assertion: 0 ancilla variables added!
    assert_eq!(
        ising.num_spins, 10,
        "Tensor-Train projection must require exactly N spins with 0 ancillae (Rosenberg would require N + 8 = 18)"
    );

    // Solve with dSB
    let config = DsbConfig {
        num_steps: 500,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 32,
        seed: 42,
        ..Default::default()
    };

    let result = solve_dsb_cpu(&ising, &config).expect("Solver failed");
    assert_eq!(result.best_spins.len(), 10);

    // In spin space, x_i = 1 corresponds to s_i = -1
    // The ground state must have all spins aligned!
    let first_spin = result.best_spins[0];
    let all_aligned = result.best_spins.iter().all(|&s| s == first_spin);
    assert!(all_aligned, "Ground state of ferromagnetic degree-10 projector must be fully aligned");
}
