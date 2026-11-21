//! Verified rewrite rules for binary algebra and polynomial simplification.
//!
//! # Core Algebraic Axioms:
//!
//! 1. **Binary Idempotence:**
//!    For $x \in \{0, 1\}$, $x^2 = x \cdot x = x$.
//!
//! 2. **Spin Idempotence:**
//!    For $s \in \{-1, +1\}$, $s^2 = s \cdot s = 1$.
//!
//! 3. **Monoid & Ring Properties:**
//!    Commutativity, associativity, additive zero, multiplicative identity and annihilator.

use egg::*;
use crate::lang::PhysicsLang;

/// Construct standard verified rewrite rules for `PhysicsLang`.
pub fn default_rules() -> Vec<Rewrite<PhysicsLang, ()>> {
    let mut rules: Vec<Rewrite<PhysicsLang, ()>> = vec![
        // Binary idempotence: x * x => x
        rewrite!("binary-idempotence"; "(* ?x ?x)" => "?x"),

        // Commutativity
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),

        // Associativity
        rewrite!("assoc-add"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rewrite!("assoc-mul"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),

        // Identity and Annihilation
        rewrite!("add-zero"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-one"; "(* ?a 1)" => "?a"),
        rewrite!("mul-zero"; "(* ?a 0)" => "0"),

        // Negation and Inversion
        rewrite!("double-neg"; "(- (- ?a))" => "?a"),
        rewrite!("neg-zero"; "(- 0)" => "0"),
        rewrite!("neg-mul-left"; "(* (- ?a) ?b)" => "(- (* ?a ?b))"),
        rewrite!("neg-mul-right"; "(* ?a (- ?b))" => "(- (* ?a ?b))"),
        rewrite!("distribute-neg"; "(- (+ ?a ?b))" => "(+ (- ?a) (- ?b))"),
        rewrite!("add-inverse"; "(+ ?a (- ?a))" => "0"),
    ];

    // Constant folding rewrites for small integer constants
    rules.push(rewrite!("fold-add-1"; "(+ 1 1)" => "2"));
    rules.push(rewrite!("fold-mul-neg"; "(* -1 -1)" => "1"));

    rules
}

/// Simplify an expression using equality saturation on the E-Graph.
pub fn simplify_expr(expr: &RecExpr<PhysicsLang>) -> RecExpr<PhysicsLang> {
    let runner = Runner::default()
        .with_expr(expr)
        .with_iter_limit(30)
        .with_node_limit(10_000)
        .run(&default_rules());

    let root = runner.roots[0];
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_best_cost, best_expr) = extractor.find_best(root);
    best_expr
}
