//! `midend-egraph` crate:
//!
//! Provides the intermediate representation (IR) based on `egg` e-graphs,
//! verified rewrite rules for binary/spin idempotence ($x^2 = x, s^2 = 1$),
//! and Rosenberg polynomial quadratization with ancilla variable reuse.

pub mod lang;
pub mod pubo;
pub mod rosenberg;
pub mod rules;

pub use lang::PhysicsLang;
pub use pubo::PuboPolynomial;
pub use rosenberg::{quadratize_rosenberg, QuadratizationResult};
pub use rules::default_rules;
