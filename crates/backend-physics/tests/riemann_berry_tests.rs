//! Integration tests for Berry-Keating Hilbert-Pólya Quantum Chaos Operator (Riemann Hypothesis).

use backend_physics::{BerryKeatingConfig, BerryKeatingOperator};

#[test]
fn test_berry_keating_hermiticity_and_real_spectrum() {
    let config = BerryKeatingConfig {
        basis_dim: 16,
        spatial_cutoff: 4.0,
        hbar: 1.0,
    };

    let op = BerryKeatingOperator::new(config);
    let (is_hermitian, asymmetry) = op.verify_hermiticity();

    assert!(is_hermitian, "Berry-Keating operator must be strictly Hermitian (self-adjoint)");
    assert!(asymmetry < 1e-12, "Departure from Hermiticity must be negligible");

    let report = op.compute_spectrum(6);
    println!(
        "Riemann-Berry Spectral Report: Dim={}, Ground={:.4}, Mean Spacing={:.4}, Levels={:?}",
        report.basis_dimension, report.ground_state_energy, report.mean_level_spacing, report.lowest_eigenvalues
    );

    assert!(report.all_eigenvalues_real, "All eigenvalues must be 100% real (no imaginary component)");
    assert_eq!(report.lowest_eigenvalues.len(), 6);
    assert!(report.mean_level_spacing > 0.0, "Must exhibit level repulsion (positive spacing)");

    // Eigenvalues should be strictly non-decreasing
    for i in 0..5 {
        assert!(report.lowest_eigenvalues[i] <= report.lowest_eigenvalues[i + 1] + 1e-6);
    }
}
