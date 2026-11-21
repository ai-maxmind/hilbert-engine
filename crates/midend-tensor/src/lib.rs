//! Crate `midend-tensor` provides Tensor-Train (TT) / Matrix Product Operator (MPO)
//! decompositions and Low-Rank Projections for ultra-high-degree ($k \ge 10$)
//! Polynomial Unconstrained Binary Optimization (PUBO) problems without ancilla explosion.

pub mod mpo_hyperedge;
pub mod mps;
pub mod projection;
pub mod sos;
pub mod tensor_train;

pub use mpo_hyperedge::MpoPolynomialBuilder;
pub use mps::{MatrixProductState, MpsSiteTensor};
pub use projection::TensorTrainIsingProjector;
pub use sos::{PolynomialSosGramSolver, SosDecompositionReport};
pub use tensor_train::{TensorTrain, TtCore, TtError};
