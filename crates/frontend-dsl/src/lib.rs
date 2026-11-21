//! `frontend-dsl` crate:
//!
//! Provides the High-Level Problem Builder API for declaring binary variables,
//! polynomial objective functions (PUBO), and hard combinatorial constraints
//! (`ExactOne`, `And`, `Or`, `Xor`, `3-SAT Clauses`).

pub mod builder;
pub mod constraints;
pub mod protein;
pub mod quantified;

pub use builder::{DslError, ProblemBuilder};
pub use constraints::{Clause3Sat, Constraint};
pub use protein::{AminoAcid, FoldedProtein, HpProteinModel, HpSequence, MonomerCoord};
pub use quantified::QuantifiedProblemBuilder;
