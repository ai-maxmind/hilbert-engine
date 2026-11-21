//! Language definition for E-Graph optimization of polynomial objectives.
//!
//! # Grammar:
//!
//! ```text
//! Expr := Num(i64)
//!       | Var(Symbol)
//!       | Add(Expr, Expr)   // "+"
//!       | Mul(Expr, Expr)   // "*"
//!       | Neg(Expr)         // "-"
//! ```

use egg::*;

define_language! {
    /// Intermediate representation language for combinatorial polynomial expressions.
    pub enum PhysicsLang {
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Neg(Id),
        Num(i64),
        Var(Symbol),
    }
}

/// Helper builders for constructing `RecExpr<PhysicsLang>` ASTs programmatically.
pub struct ExprBuilder;

impl ExprBuilder {
    pub fn var(expr: &mut RecExpr<PhysicsLang>, name: &str) -> Id {
        expr.add(PhysicsLang::Var(Symbol::from(name)))
    }

    pub fn num(expr: &mut RecExpr<PhysicsLang>, val: i64) -> Id {
        expr.add(PhysicsLang::Num(val))
    }

    pub fn add(expr: &mut RecExpr<PhysicsLang>, a: Id, b: Id) -> Id {
        expr.add(PhysicsLang::Add([a, b]))
    }

    pub fn mul(expr: &mut RecExpr<PhysicsLang>, a: Id, b: Id) -> Id {
        expr.add(PhysicsLang::Mul([a, b]))
    }

    pub fn neg(expr: &mut RecExpr<PhysicsLang>, a: Id) -> Id {
        expr.add(PhysicsLang::Neg(a))
    }
}
