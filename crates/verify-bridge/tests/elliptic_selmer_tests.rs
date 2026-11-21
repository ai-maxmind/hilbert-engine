use verify_bridge::{EllipticCurve, SelmerRankReport};

#[test]
fn test_elliptic_selmer_rank_zero_curve() {
    // E: y^2 = x^3 - x (a = -1, b = 0, discriminant = 64)
    // Famous congruent number curve for n=1 (1 is not a congruent number -> rank is 0)
    let curve = EllipticCurve::new(-1, 0);
    assert_eq!(curve.discriminant(), 64);

    let roots = curve.find_integer_2_torsion_roots();
    println!("Rational 2-torsion roots of y^2 = x^3 - x: {:?}", roots);
    assert_eq!(roots, vec![-1, 0, 1]); // Full 2-torsion

    let report: SelmerRankReport = curve.compute_2_selmer_rank_bound();
    println!(
        "Selmer 2-Descent Report: a={}, b={}, Disc={}, DimS2={}, RankBound={}, BSD={}",
        report.a, report.b, report.discriminant, report.selmer_2_dimension, report.mordell_weil_rank_bound, report.bsd_implication
    );

    assert!(report.mordell_weil_rank_bound <= 1, "Rank bound must be small for rank 0 curve");
}

#[test]
fn test_elliptic_selmer_congruent_number_curve() {
    // E: y^2 = x^3 - 25*x (a = -25, b = 0)
    // 5 is a congruent number (3^2 + 4^2 = 5^2, area = 6; for 5: (9/4)^2 + ... area = 5)
    // Curve has rank 1!
    let curve = EllipticCurve::new(-25, 0);
    let report = curve.compute_2_selmer_rank_bound();

    println!(
        "Curve y^2 = x^3 - 25x: Disc={}, DimS2={}, RankBound={}",
        report.discriminant, report.selmer_2_dimension, report.mordell_weil_rank_bound
    );
    assert!(report.discriminant != 0, "Curve must be non-singular");
    assert_eq!(report.num_rational_2_torsion, 3);
}
