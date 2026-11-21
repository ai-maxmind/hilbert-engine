use egg::*;
use midend_egraph::lang::PhysicsLang;
use midend_egraph::pubo::PuboPolynomial;
use midend_egraph::rosenberg::quadratize_rosenberg;
use midend_egraph::rules::simplify_expr;

#[test]
fn test_egg_binary_idempotence() {
    // Expression: x * x
    let expr: RecExpr<PhysicsLang> = "(* x x)".parse().unwrap();
    let simplified = simplify_expr(&expr);
    let expected: RecExpr<PhysicsLang> = "x".parse().unwrap();
    assert_eq!(simplified, expected);
}

#[test]
fn test_egg_algebraic_simplifications() {
    // Expression: (+ (* x 1) (* y 0)) => x
    let expr: RecExpr<PhysicsLang> = "(+ (* x 1) (* y 0))".parse().unwrap();
    let simplified = simplify_expr(&expr);
    let expected: RecExpr<PhysicsLang> = "x".parse().unwrap();
    assert_eq!(simplified, expected);

    // Expression: (- (- z)) => z
    let expr2: RecExpr<PhysicsLang> = "(- (- z))".parse().unwrap();
    let simplified2 = simplify_expr(&expr2);
    let expected2: RecExpr<PhysicsLang> = "z".parse().unwrap();
    assert_eq!(simplified2, expected2);
}

#[test]
fn test_ancilla_reuse_multiple_terms() {
    // Terms:
    // T1: 3.0 * x_0 * x_1 * x_2
    // T2: -2.0 * x_0 * x_1 * x_3
    // The pair (0, 1) is common to both cubic terms and MUST reuse the same ancilla w!
    let mut pubo = PuboPolynomial::new(4);
    pubo.add_term(&[0, 1, 2], 3.0).unwrap();
    pubo.add_term(&[0, 1, 3], -2.0).unwrap();

    let res = quadratize_rosenberg(&pubo).unwrap();

    // Verify exactly 1 ancilla was introduced, NOT 2!
    assert_eq!(res.num_ancillas, 1, "Ancilla variable must be reused!");
    assert_eq!(res.pair_to_ancilla.len(), 1);
    assert!(res.pair_to_ancilla.contains_key(&(0, 1)));
    assert!(res.qubo.is_quadratic());

    // Verify ground state preservation
    let (pubo_min, _) = pubo.brute_force_min();
    let (qubo_min, qubo_sol) = res.qubo.brute_force_min();
    assert!((pubo_min - qubo_min).abs() < 1e-6);

    // Check optimal original variable values match
    let pubo_eval = pubo.evaluate(&qubo_sol[0..4]).unwrap();
    assert!((pubo_eval - pubo_min).abs() < 1e-6);
}

#[test]
fn test_degree_4_quadratization() {
    // Quartic term: -5.0 * x_0 * x_1 * x_2 * x_3
    let mut pubo = PuboPolynomial::new(4);
    pubo.add_term(&[0, 1, 2, 3], -5.0).unwrap();

    let res = quadratize_rosenberg(&pubo).unwrap();
    assert!(res.qubo.is_quadratic());
    assert!(res.qubo.degree() <= 2);

    let (pubo_min, _) = pubo.brute_force_min();
    let (qubo_min, qubo_sol) = res.qubo.brute_force_min();

    // Min of -5 * x0*x1*x2*x3 is -5 (when all are 1)
    assert_eq!(pubo_min, -5.0);
    assert_eq!(qubo_min, -5.0);
    assert_eq!(&qubo_sol[0..4], &[1, 1, 1, 1]);
}
