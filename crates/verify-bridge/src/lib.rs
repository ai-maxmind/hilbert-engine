//! `verify-bridge` crate:
//!
//! Provides the verification and penalty refinement bridge:
//! 1. Solution decoding and constraint residual checks (`decoder`)
//! 2. Closed-loop Augmented Lagrangian Method (`alm`)
//! 3. Level-2 formal verification and SMT-LIB2 / Z3 export (`formal`)

pub mod alm;
pub mod bmc;
pub mod decoder;
pub mod dehn;
pub mod elliptic_selmer;
pub mod formal;
pub mod lll;
pub mod quantified;

pub use alm::{run_augmented_lagrangian, AlmConfig, AlmError, AlmProblem, AlmResult};
pub use bmc::{
    InductiveProofResult, InductiveProver, SafetyInvariant, TransitionClause, TransitionLiteral,
    TransitionSystem,
};
pub use decoder::{evaluate_constraint, verify_constraints, VerificationReport};
pub use dehn::{PolyhedronKind, ScissorsCongruenceReport};
pub use elliptic_selmer::{EllipticCurve, SelmerRankReport};
pub use formal::{FormalChecker, FormalError, ProblemVerifier, Z3Result};
pub use lll::{LatticeBasis, LllReport};
pub use quantified::{export_quantified_smtlib2, verify_saddle_point, SaddlePointReport};
