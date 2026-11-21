//! Integration tests for Bounded Model Checking (BMC) and Inductive Invariant Prover.

use verify_bridge::bmc::{
    InductiveProver, SafetyInvariant, TransitionClause, TransitionLiteral, TransitionSystem,
};

#[test]
fn test_bmc_transition_system_unrolling() {
    // 2-bit counter system: bit 0 and bit 1
    let mut system = TransitionSystem::new(2);
    // Initial: s_0 = 0, s_1 = 0
    system.set_initial(0, 0);
    system.set_initial(1, 0);

    // Transition: s0' = NOT s0 (or encoded as 2-literal clause)
    system.add_transition_clause(TransitionClause::new(vec![
        TransitionLiteral::Curr(0, false),
        TransitionLiteral::Next(0, false),
    ]));

    // Bad state: reaching s_0 = 1 and s_1 = 1
    system.add_bad_condition(0, 1);
    system.add_bad_condition(1, 1);

    // Unroll 3 steps
    let builder = system.unroll_bmc(3);
    let pubo = builder.build().expect("Build unrolled PUBO");
    // (3 + 1) * 2 = 8 variables
    assert_eq!(pubo.num_variables, 8);
}

#[test]
fn test_inductive_invariant_verification_and_smtlib2_export() {
    let mut system = TransitionSystem::new(2);
    // Mutex system: Process 0 and Process 1
    // Initial: both idle (0, 0)
    system.set_initial(0, 0);
    system.set_initial(1, 0);

    // Bad state: both in critical section simultaneously (1, 1)
    system.add_bad_condition(0, 1);
    system.add_bad_condition(1, 1);

    // Candidate Invariant: Process 0 is 0
    let inv = SafetyInvariant {
        safe_conditions: vec![(0, 0)],
    };

    let prover = InductiveProver::new(&system);
    let report = prover.verify_inductive(&inv);

    assert!(report.is_inductive_invariant, "Must prove inductive invariant");
    assert!(report.base_holds);
    assert!(report.step_holds);
    assert!(report.safety_holds);

    // Check SMT-LIB2 export
    let smt = prover.export_inductive_smtlib2(&inv, "MutexTwoProcess");
    assert!(smt.contains("(set-logic QF_BV)"));
    assert!(smt.contains("(define-fun is_init () Bool"));
    assert!(smt.contains("(define-fun is_inv_s () Bool"));
    assert!(smt.contains("(check-sat)"));
}
