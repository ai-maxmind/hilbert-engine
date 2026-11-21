use midend_egraph::pubo::PuboPolynomial;
use midend_egraph::rosenberg::quadratize_rosenberg;
use proptest::prelude::*;

// Strategy to generate random degree-3 PUBO polynomials
fn pubo_strategy() -> impl Strategy<Value = PuboPolynomial> {
    // Number of variables N between 3 and 5 for fast brute-force evaluation in tests
    (3usize..=5).prop_flat_map(|num_vars| {
        let max_vars = num_vars;
        // Generate a random number of cubic terms
        let cubic_terms = prop::collection::vec(
            (
                0..max_vars,
                0..max_vars,
                0..max_vars,
                -10i32..=10i32,
            ),
            1..=4,
        );
        // Generate quadratic terms
        let quad_terms = prop::collection::vec(
            (
                0..max_vars,
                0..max_vars,
                -10i32..=10i32,
            ),
            0..=3,
        );
        // Generate linear terms
        let linear_terms = prop::collection::vec(
            (
                0..max_vars,
                -10i32..=10i32,
            ),
            0..=2,
        );
        // Constant offset
        let constant = -5i32..=5i32;

        (Just(num_vars), cubic_terms, quad_terms, linear_terms, constant)
    }).prop_map(|(num_vars, cubics, quads, linears, const_val)| {
        let mut pubo = PuboPolynomial::new(num_vars);
        let _ = pubo.add_term(&[], const_val as f64);

        for (u, coeff) in linears {
            let _ = pubo.add_term(&[u], coeff as f64);
        }

        for (u, v, coeff) in quads {
            if u != v {
                let _ = pubo.add_term(&[u, v], coeff as f64);
            }
        }

        for (u, v, w, coeff) in cubics {
            // Require 3 distinct variables for an actual degree-3 term
            if u != v && v != w && u != w {
                let _ = pubo.add_term(&[u, v, w], coeff as f64);
            }
        }

        pubo
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn test_proptest_rosenberg_ground_state_500_pubo(pubo in pubo_strategy()) {
        // 1. Compute ground state of original PUBO
        let (pubo_min, _pubo_opt_assign) = pubo.brute_force_min();

        // 2. Perform Rosenberg Quadratization with Ancilla Reuse
        let quad_res = quadratize_rosenberg(&pubo).expect("Quadratization must succeed");

        // 3. Verify the resulting polynomial is strictly quadratic
        prop_assert!(
            quad_res.qubo.is_quadratic(),
            "Resulting polynomial must be quadratic (degree <= 2), got degree {}",
            quad_res.qubo.degree()
        );

        // 4. Compute ground state of quadratized QUBO
        let (qubo_min, qubo_opt_assign) = quad_res.qubo.brute_force_min();

        // 5. Rigorously verify that global minimum energy is PRESERVED EXACTLY:
        // min(P(x)) == min(Q(x, w))
        let diff = (pubo_min - qubo_min).abs();
        prop_assert!(
            diff < 1e-4,
            "Ground state mismatch: PUBO min = {}, QUBO min = {}, diff = {}",
            pubo_min, qubo_min, diff
        );

        // 6. Verify that the optimal x values from QUBO evaluate to the exact PUBO minimum
        let x_opt = &qubo_opt_assign[0..pubo.num_variables];
        let pubo_eval = pubo.evaluate(x_opt).unwrap();
        prop_assert!(
            (pubo_eval - pubo_min).abs() < 1e-4,
            "Optimal assignment does not achieve PUBO ground state: eval={}, min={}",
            pubo_eval, pubo_min
        );

        // 7. Verify ancilla constraint satisfaction: w* == x_u* * x_v*
        for (&(u, v), &w) in &quad_res.pair_to_ancilla {
            let expected_w = qubo_opt_assign[u] * qubo_opt_assign[v];
            let actual_w = qubo_opt_assign[w];
            prop_assert_eq!(
                actual_w,
                expected_w,
                "Ancilla variable w={} was not equal to x_{} * x_{} at the global minimum!",
                w,
                u,
                v
            );
        }
    }
}
