use verify_bridge::{LatticeBasis, LllReport};

#[test]
fn test_lll_lattice_reduction_svp() {
    // Basis of a 3D lattice
    let basis = vec![
        vec![1.0, 1.0, 1.0],
        vec![-1.0, 0.0, 2.0],
        vec![3.0, 5.0, 6.0],
    ];
    let mut lattice = LatticeBasis::new(basis);
    let report: LllReport = lattice.lll_reduce(0.75);

    println!(
        "LLL Reduction: dim={}, iters={}, shortest_norm={:.4}, shortest_vec={:?}",
        report.dimension, report.iterations, report.shortest_vector_norm, report.shortest_vector
    );

    assert!(report.is_reduced);
    assert!(report.shortest_vector_norm > 0.0);
    assert!(report.shortest_vector_norm <= 3.0, "Shortest vector must be small in LLL basis");
}

#[test]
fn test_lll_bounded_diophantine_solver() {
    // Equation: 2*x_0 + 3*x_1 + 5*x_2 = 19, with |x_i| <= 10
    let coeffs = vec![2, 3, 5];
    let target = 19;
    let bound = 10;

    let solution = LatticeBasis::solve_bounded_diophantine(&coeffs, target, bound);
    println!("Diophantine Solution for 2*x + 3*y + 5*z = 19: {:?}", solution);

    assert!(solution.is_some(), "LLL must find a bounded integer solution");
    let sol = solution.unwrap();
    assert_eq!(sol.len(), 3);
    assert!(sol.iter().all(|&x| x.abs() <= bound));
    let sum: i64 = sol.iter().zip(coeffs.iter()).map(|(&x, &a)| x * a).sum();
    assert_eq!(sum, target, "Solution must satisfy Diophantine equation exactly");
}
