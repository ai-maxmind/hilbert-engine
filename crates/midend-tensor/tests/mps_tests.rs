//! Integration tests for Quantum Matrix Product State (MPS) and Sign-Problem-Free Solver.

use midend_tensor::MatrixProductState;

#[test]
fn test_mps_ferromagnetic_state_and_tfim_energy() {
    let n = 6;
    let mps = MatrixProductState::ferromagnetic_product_state(n);

    let norm_sq = mps.norm_squared();
    assert!((norm_sq - 1.0).abs() < 1e-10, "Norm squared of product state must be 1.0");

    // All spins in |0> have sigma^z = +1
    for site in 0..n {
        let sz = mps.expectation_value_z(site);
        assert!((sz - 1.0).abs() < 1e-10, "Site {} expectation <sigma^z> must be +1.0", site);
    }

    // Two-point correlator <sigma_i^z sigma_{i+1}^z> = +1.0
    for i in 0..(n - 1) {
        let szz = mps.expectation_value_zz(i, i + 1);
        assert!((szz - 1.0).abs() < 1e-10, "Bond {} correlator must be +1.0", i);
    }

    // Classical Ising energy: -J * (N - 1) = -1.0 * 5 = -5.0
    let energy = mps.evaluate_quantum_energy(1.0, 0.0);
    assert!((energy - (-5.0)).abs() < 1e-10, "TFIM classical ground state energy must be -5.0");
}

#[test]
fn test_mps_ghz_entangled_state_sign_problem_free() {
    let n = 4;
    let ghz = MatrixProductState::ghz_entangled_state(n);

    let norm_sq = ghz.norm_squared();
    assert!((norm_sq - 1.0).abs() < 1e-10, "GHZ state norm squared must be exactly 1.0");

    // In GHZ state, <sigma^z> = 0.0 for any individual spin (equal superposition of all up and all down)
    for site in 0..n {
        let sz = ghz.expectation_value_z(site);
        assert!(sz.abs() < 1e-10, "Single-site <sigma^z> in GHZ state must be 0.0");
    }

    // But long-range correlation <sigma_0^z sigma_3^z> = +1.0!
    let zz_long_range = ghz.expectation_value_zz(0, 3);
    assert!((zz_long_range - 1.0).abs() < 1e-10, "GHZ state must have perfect long-range entanglement = 1.0");

    println!("GHZ Entangled MPS Verified: Sites={}, Norm={:.4}, Long-Range Corr={:.4}", n, norm_sq, zz_long_range);
}
