//! `backend-physics` crate:
//!
//! Provides the physics-inspired solver engine for Ising spin-glass models,
//! featuring Compressed Sparse Row (CSR) storage and Discrete Simulated
//! Bifurcation (dSB) in CPU Deterministic Simulation Testing (DST) mode.

pub mod dsb;
pub mod entropic_game;
pub mod gpu;
pub mod halo;
pub mod ising;
pub mod riemann_berry;
pub mod vortex;
pub mod wilson_gauge;

pub use dsb::{solve_dsb_cpu, DsbConfig, DsbError, DsbResult, SolverPhysicsMode};
pub use entropic_game::{
    solve_entropic_mixed_nash, MixedNashResult, ZeroSumMatrixGame,
};
pub use gpu::{solve_dsb_gpu, solve_dsb_gpu_with_context, GpuContext, GpuError, SimParams};
pub use halo::{solve_distributed_dsb_cpu, DomainDecomposer, DomainInfo};
pub use ising::{IsingCsr, IsingError};
pub use riemann_berry::{BerryKeatingConfig, BerryKeatingOperator, RiemannSpectralReport};
pub use vortex::{VortexParticle, VortexRegularityReport, VortexSimConfig, VortexSystem};
pub use wilson_gauge::{LatticeGaugeSystem, WilsonConfinementReport};

