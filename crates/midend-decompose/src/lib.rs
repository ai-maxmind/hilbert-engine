//! `midend-decompose` crate:
//!
//! Provides interaction graph topology analysis with `petgraph`,
//! dynamic range scaling, and compilation of quadratic polynomials into `IsingCsr`.

pub mod decomposer;
pub mod multilevel;

pub use decomposer::{decompose_to_ising, DecomposedIsing, DecomposeError};
pub use multilevel::{
    coarsen_ising_one_level, prolongate, refine_local_search,
    HierarchyLevel, LevelMapping, MultilevelConfig, MultilevelHierarchy,
    MultilevelResult, MultilevelSolver,
};
