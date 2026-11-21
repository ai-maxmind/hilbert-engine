use midend_tensor::{PolynomialSosGramSolver, SosDecompositionReport};

#[test]
fn test_artin_sos_positive_semidefinite_polynomial() {
    // P(x) = (x^2 + 1)^2 = x^4 + 2*x^2 + 1
    // Coeffs: [1.0, 0.0, 2.0, 0.0, 1.0]
    let coeffs = vec![1.0, 0.0, 2.0, 0.0, 1.0];
    let solver = PolynomialSosGramSolver::new(coeffs);
    let report: SosDecompositionReport = solver.decompose_sos();

    println!(
        "SOS Report: Degree={}, Dim={}, PSD={}, MinEig={:.4}, Squares={}",
        report.polynomial_degree, report.gram_dimension, report.is_positive_semidefinite,
        report.min_eigenvalue_estimate, report.num_square_components
    );
    println!("Certificate: {}", report.mathematical_certificate);

    assert!(report.is_positive_semidefinite, "P(x) = (x^2 + 1)^2 must be a Sum of Squares");
    assert!(report.global_non_negativity_proven);
    assert!(report.num_square_components >= 1);
    assert_eq!(solver.eval(0.0), 1.0);
    assert_eq!(solver.eval(1.0), 4.0);
}

#[test]
fn test_artin_sos_negative_polynomial_fails_psd() {
    // P(x) = x^2 - 4*x + 3 = (x - 1)(x - 3)
    // Takes negative values between 1 and 3 (e.g. at x=2, P(2) = -1 < 0)
    let coeffs = vec![3.0, -4.0, 1.0];
    let solver = PolynomialSosGramSolver::new(coeffs);
    let report = solver.decompose_sos();

    println!("Non-SOS Polynomial Report: PSD={}", report.is_positive_semidefinite);
    assert!(!report.is_positive_semidefinite, "Polynomial taking negative values cannot be SOS");
    assert!(!report.global_non_negativity_proven);
}
