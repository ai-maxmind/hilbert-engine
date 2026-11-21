use verify_bridge::{PolyhedronKind, ScissorsCongruenceReport};

#[test]
fn test_hilbert_3_dehn_obstruction_cube_vs_tetrahedron() {
    // Construct a cube of volume 1.0
    let cube = PolyhedronKind::Cube { side: 1.0 };
    assert!((cube.volume() - 1.0).abs() < 1e-9);

    // Construct a regular tetrahedron of the SAME volume 1.0:
    // edge = (Vol * 6 * sqrt(2))^(1/3)
    let tetra_edge = (1.0 * 6.0 * std::f64::consts::SQRT_2).cbrt();
    let tetra = PolyhedronKind::RegularTetrahedron { edge: tetra_edge };
    assert!((tetra.volume() - 1.0).abs() < 1e-6, "Volumes must be equal");

    let report: ScissorsCongruenceReport = PolyhedronKind::verify_congruence(&cube, &tetra);

    println!(
        "Hilbert 3 Test: Vol1={:.4}, Vol2={:.4}, VolumesMatch={}, DehnMatch={}, ScissorsCongruent={}",
        report.volume_1, report.volume_2, report.volumes_match, report.dehn_invariants_match, report.are_scissors_congruent
    );
    println!("Verdict: {}", report.mathematical_verdict);

    assert!(report.volumes_match, "Volumes must be equal");
    assert!(!report.dehn_invariants_match, "Dehn invariants must NOT match between cube and tetrahedron");
    assert!(!report.are_scissors_congruent, "Hilbert 3: Cube and regular tetrahedron cannot be dissected into each other");
    assert!(report.dehn_1_rational, "Cube has rational Dehn invariant D = 0");
    assert!(!report.dehn_2_rational, "Regular tetrahedron has irrational Dehn invariant D != 0");
}

#[test]
fn test_dehn_scissors_congruence_cube_vs_prism() {
    // A cube of side 2 (Vol = 8) and a rectangular prism 1 x 2 x 4 (Vol = 8)
    let cube = PolyhedronKind::Cube { side: 2.0 };
    let prism = PolyhedronKind::RectangularPrism { a: 1.0, b: 2.0, c: 4.0 };

    let report = PolyhedronKind::verify_congruence(&cube, &prism);

    assert!(report.volumes_match);
    assert!(report.dehn_invariants_match);
    assert!(report.are_scissors_congruent, "Two rectangular prisms of equal volume are scissors-congruent");
}
