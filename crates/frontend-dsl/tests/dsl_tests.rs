use frontend_dsl::ProblemBuilder;

#[test]
fn test_exact_one_constraint() {
    let mut builder = ProblemBuilder::new();
    let vars = builder.add_variables(3);
    builder.add_exact_one(&vars, 10.0);
    let pubo = builder.build().unwrap();

    // Valid assignments (sum = 1) -> Penalty 0
    assert_eq!(pubo.evaluate(&[1, 0, 0]).unwrap(), 0.0);
    assert_eq!(pubo.evaluate(&[0, 1, 0]).unwrap(), 0.0);
    assert_eq!(pubo.evaluate(&[0, 0, 1]).unwrap(), 0.0);

    // Invalid assignments -> Penalty > 0
    assert_eq!(pubo.evaluate(&[0, 0, 0]).unwrap(), 10.0); // sum = 0
    assert_eq!(pubo.evaluate(&[1, 1, 0]).unwrap(), 10.0); // sum = 2
    assert_eq!(pubo.evaluate(&[1, 1, 1]).unwrap(), 40.0); // (3 - 1)^2 * 10 = 40
}

#[test]
fn test_and_constraint() {
    let mut builder = ProblemBuilder::new();
    let x = builder.add_variable(Some("x"));
    let y = builder.add_variable(Some("y"));
    let out = builder.add_variable(Some("out"));

    builder.add_and(x, y, out, 5.0);
    let pubo = builder.build().unwrap();

    // Truth table: out == x & y
    // (x, y, out)
    assert_eq!(pubo.evaluate(&[0, 0, 0]).unwrap(), 0.0); // valid
    assert_eq!(pubo.evaluate(&[0, 1, 0]).unwrap(), 0.0); // valid
    assert_eq!(pubo.evaluate(&[1, 0, 0]).unwrap(), 0.0); // valid
    assert_eq!(pubo.evaluate(&[1, 1, 1]).unwrap(), 0.0); // valid

    // Violations: out != x & y
    assert!(pubo.evaluate(&[0, 0, 1]).unwrap() >= 5.0);
    assert!(pubo.evaluate(&[1, 0, 1]).unwrap() >= 5.0);
    assert!(pubo.evaluate(&[1, 1, 0]).unwrap() >= 5.0);
}

#[test]
fn test_or_constraint() {
    let mut builder = ProblemBuilder::new();
    let vars = builder.add_variables(2);
    builder.add_or(&vars, 8.0);
    let pubo = builder.build().unwrap();

    // Satisfied if at least one is 1
    assert_eq!(pubo.evaluate(&[1, 0]).unwrap(), 0.0);
    assert_eq!(pubo.evaluate(&[0, 1]).unwrap(), 0.0);
    assert_eq!(pubo.evaluate(&[1, 1]).unwrap(), 0.0);

    // Violated if both are 0
    assert_eq!(pubo.evaluate(&[0, 0]).unwrap(), 8.0);
}

#[test]
fn test_xor_constraint() {
    let mut builder = ProblemBuilder::new();
    let x = builder.add_variable(Some("x"));
    let y = builder.add_variable(Some("y"));
    let out = builder.add_variable(Some("out"));
    builder.add_xor(x, y, out, 12.0);
    let pubo = builder.build().unwrap();

    // Valid XOR truth table
    assert_eq!(pubo.evaluate(&[0, 0, 0]).unwrap(), 0.0);
    assert_eq!(pubo.evaluate(&[1, 0, 1]).unwrap(), 0.0);
    assert_eq!(pubo.evaluate(&[0, 1, 1]).unwrap(), 0.0);
    assert_eq!(pubo.evaluate(&[1, 1, 0]).unwrap(), 0.0);

    // Violations
    assert!(pubo.evaluate(&[0, 0, 1]).unwrap() >= 12.0);
    assert!(pubo.evaluate(&[1, 1, 1]).unwrap() >= 12.0);
    assert!(pubo.evaluate(&[1, 0, 0]).unwrap() >= 12.0);
}

#[test]
fn test_clause_3sat() {
    let mut builder = ProblemBuilder::new();
    let vars = builder.add_variables(3);
    // Clause: (x_0 OR NOT x_1 OR x_2)
    builder.add_clause_3sat(
        (vars[0], true),
        (vars[1], false),
        (vars[2], true),
        15.0,
    );
    let pubo = builder.build().unwrap();

    // Only violated when x0=0, x1=1, x2=0
    assert_eq!(pubo.evaluate(&[0, 1, 0]).unwrap(), 15.0);

    // Satisfied in all other 7 cases
    assert_eq!(pubo.evaluate(&[0, 0, 0]).unwrap(), 0.0); // NOT x1 is true
    assert_eq!(pubo.evaluate(&[1, 0, 0]).unwrap(), 0.0); // x0 is true
    assert_eq!(pubo.evaluate(&[0, 0, 1]).unwrap(), 0.0); // x2 is true
    assert_eq!(pubo.evaluate(&[1, 1, 1]).unwrap(), 0.0);
}
