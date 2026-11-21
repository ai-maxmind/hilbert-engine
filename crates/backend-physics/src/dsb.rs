//! Module `dsb` implements the Discrete Simulated Bifurcation (dSB) algorithm
//! in single-threaded Deterministic Simulation Testing (DST) mode.
//!
//! # Mathematical Foundations
//!
//! Simulated Bifurcation solves NP-hard Ising optimization problems by simulating
//! a network of classical nonlinear coupled parametric oscillators undergoing
//! pitchfork bifurcations.
//!
//! ## Continuous Hamiltonian Dynamics:
//!
//! $$\mathcal{H}(\mathbf{x}, \mathbf{y}, t) = \frac{1}{2} \sum_{i=1}^N y_i^2 + \frac{1}{2}(1 - a_t) \sum_{i=1}^N x_i^2 + c_0 H_{\text{Ising}}(\operatorname{sgn}(\mathbf{x}))$$
//!
//! Applying Hamilton's equations of motion:
//!
//! $$\frac{dx_i}{dt} = \frac{\partial \mathcal{H}}{\partial y_i} = y_i$$
//! $$\frac{dy_i}{dt} = -\frac{\partial \mathcal{H}}{\partial x_i} = -(1 - a_t) x_i + c_0 \left( \sum_{j=1}^N J_{ij} \operatorname{sgn}(x_j) + h_i \right)$$
//!
//! ## Symplectic Euler Integrator (Step $t \to t+1$):
//!
//! 1. Pumping schedule parameter:
//!    $$a_t = a_0 \cdot \frac{t}{M}$$
//! 2. Binarized spin proxy:
//!    $$s_j^{(t)} = \operatorname{sgn}(x_j^{(t)}) = \begin{cases} +1.0 & \text{if } x_j^{(t)} \ge 0 \\ -1.0 & \text{if } x_j^{(t)} < 0 \end{cases}$$
//! 3. Interaction field:
//!    $$f_i^{(t)} = \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{values}[k] \cdot s_{\text{col\_indices}[k]}^{(t)} + h_i$$
//! 4. Momentum update:
//!    $$y_i^{(t+1)} = y_i^{(t)} + \Delta t \left[ -(1 - a_t) x_i^{(t)} + c_0 f_i^{(t)} \right]$$
//! 5. Position update:
//!    $$x_i^{(t+1)} = x_i^{(t)} + \Delta t \cdot y_i^{(t+1)}$$
//! 6. Inelastic wall boundary condition (Kerr nonlinearity approximation):
//!    $$\text{if } |x_i^{(t+1)}| > 1.0 \implies x_i^{(t+1)} = \operatorname{sgn}(x_i^{(t+1)}), \quad y_i^{(t+1)} = 0.0$$

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use thiserror::Error;

use crate::ising::{IsingCsr, IsingError};

/// Errors encountered during dSB solver execution.
#[derive(Debug, Error)]
pub enum DsbError {
    #[error("Ising model error: {0}")]
    Ising(#[from] IsingError),

    #[error("Invalid configuration: num_steps must be > 0")]
    ZeroSteps,

    #[error("Invalid configuration: num_replicas must be > 0")]
    ZeroReplicas,

    #[error("Invalid configuration: dt must be positive, got {0}")]
    InvalidDt(f32),
}

/// Simulated Bifurcation physics simulation mode.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverPhysicsMode {
    /// Discrete Simulated Bifurcation (dSB) with inelastic wall boundary condition.
    #[default]
    Discrete = 0,
    /// Ballistic Simulated Bifurcation (bSB) with elastic reflection to overcome flat traps.
    Ballistic = 1,
    /// Chaotic Simulated Bifurcation (cSB) with ergodic chaotic perturbation to escape Spin Glass minima.
    Chaotic = 2,
}

/// Hyperparameter configuration for the Discrete Simulated Bifurcation algorithm.
#[derive(Debug, Clone)]
pub struct DsbConfig {
    /// Number of discrete time steps $M$ in the annealing schedule (default: 1000).
    pub num_steps: usize,
    /// Symplectic Euler integration time step $\Delta t$ (default: 0.5).
    pub dt: f32,
    /// Detuning scale amplitude $a_0$ (default: 1.0).
    pub a0: f32,
    /// Interaction coupling scale $c_0$.
    /// If `None`, automatically computed via Goto's formula: $c_0 = \frac{0.5 \sqrt{N-1}}{\|J\|_F}$.
    pub c0: Option<f32>,
    /// Number of independent simulation replicas (default: 1).
    pub num_replicas: usize,
    /// Seed for `rand_chacha::ChaCha20Rng` ensuring 100% bit-for-bit deterministic simulation testing (DST).
    pub seed: u64,
    /// Simulation physics mode (dSB, bSB, or cSB).
    pub mode: SolverPhysicsMode,
    /// Number of Minimizer variables $N_x$ for Adversarial Minimax dynamics ($\min_x \max_y H(x, y)$).
    /// If `None`, all variables are Minimizers (standard optimization).
    pub num_x: Option<usize>,
}

impl Default for DsbConfig {
    fn default() -> Self {
        Self {
            num_steps: 1000,
            dt: 0.5,
            a0: 1.0,
            c0: None,
            num_replicas: 1,
            seed: 42,
            mode: SolverPhysicsMode::Discrete,
            num_x: None,
        }
    }
}

/// Result of running the dSB solver on an Ising model instance.
#[derive(Debug, Clone)]
pub struct DsbResult {
    /// Optimal spin configuration found $\mathbf{s}^* \in \{-1, +1\}^N$.
    pub best_spins: Vec<i8>,
    /// Minimum Ising Hamiltonian energy achieved $E(\mathbf{s}^*)$.
    pub best_energy: f64,
    /// Number of time integration steps executed.
    pub steps_executed: usize,
    /// Effective coupling parameter $c_0$ utilized.
    pub c0: f32,
    /// Final spin configuration per replica.
    pub replica_spins: Vec<Vec<i8>>,
    /// Final Ising Hamiltonian energy per replica.
    pub replica_energies: Vec<f64>,
}

/// Execute Discrete Simulated Bifurcation (dSB) in CPU DST mode.
///
/// Guarantee: Given identical `ising` and `config` (including `seed`), this function
/// guarantees bit-for-bit identical trajectory and results across consecutive runs.
pub fn solve_dsb_cpu(ising: &IsingCsr, config: &DsbConfig) -> Result<DsbResult, DsbError> {
    if config.num_steps == 0 {
        return Err(DsbError::ZeroSteps);
    }
    if config.num_replicas == 0 {
        return Err(DsbError::ZeroReplicas);
    }
    if config.dt <= 0.0 || !config.dt.is_finite() {
        return Err(DsbError::InvalidDt(config.dt));
    }

    let n = ising.num_spins;
    if n == 0 {
        return Ok(DsbResult {
            best_spins: Vec::new(),
            best_energy: 0.0,
            steps_executed: config.num_steps,
            c0: 0.0,
            replica_spins: vec![Vec::new(); config.num_replicas],
            replica_energies: vec![0.0; config.num_replicas],
        });
    }

    // 1. Determine coupling scale c0
    let c0 = config.c0.unwrap_or_else(|| ising.compute_c0());

    // 2. Initialize deterministic PRNG
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    let mut replica_spins: Vec<Vec<i8>> = Vec::with_capacity(config.num_replicas);
    let mut replica_energies: Vec<f64> = Vec::with_capacity(config.num_replicas);

    let mut best_spins: Vec<i8> = Vec::new();
    let mut best_energy = f64::INFINITY;
    let mut best_deviations = usize::MAX;
    let mut best_violation = f64::INFINITY;

    // Working buffers pre-allocated to avoid allocations in simulation loop
    let mut x: Vec<f32> = vec![0.0f32; n];
    let mut y: Vec<f32> = vec![0.0f32; n];
    let mut s: Vec<f32> = vec![0.0f32; n];

    let dt = config.dt;
    let a0 = config.a0;
    let m_f32 = config.num_steps as f32;

    for replica in 0..config.num_replicas {
        // Initial perturbation: x_i(0), y_i(0) ~ U(-0.1, 0.1) sampled in deterministic sequence
        for i in 0..n {
            x[i] = rng.gen_range(-0.1f32..0.1f32);
            y[i] = rng.gen_range(-0.1f32..0.1f32);
        }

        // Discrete time evolution
        for step in 0..config.num_steps {
            // Pumping schedule a_t = a0 * (step / M)
            let a_t = a0 * (step as f32 / m_f32);
            let detuning = -(1.0f32 - a_t);

            // Discretize spin proxy s_j = sgn(x_j)
            for i in 0..n {
                s[i] = if x[i] >= 0.0f32 { 1.0f32 } else { -1.0f32 };
            }

            // Symplectic Euler update:
            // 1. Compute force and update y
            // 2. Update x
            // 3. Apply inelastic wall condition
            for i in 0..n {
                let start = ising.row_ptr[i] as usize;
                let end = ising.row_ptr[i + 1] as usize;

                let mut field: f32 = ising.h[i];
                for k in start..end {
                    let col = ising.col_indices[k] as usize;
                    field += ising.values[k] * s[col];
                }

                // Chaotic perturbation for cSB
                if config.mode == SolverPhysicsMode::Chaotic {
                    let phi = (((i as f32 * 12.9898 + replica as f32 * 78.233).sin() * 43758.5453).fract()) * 6.2831853;
                    let decay = (1.0f32 - (step as f32 / m_f32)).max(0.0f32);
                    let chaos_force = 0.25f32 * decay * (25.0f32 * (step as f32 / m_f32) + phi).sin();
                    field += chaos_force;
                }

                let num_x = config.num_x.unwrap_or(n);
                let force_sign = if i < num_x { 1.0f32 } else { -1.0f32 };

                // dy/dt = detuning * x_i + c0 * field * force_sign
                let dy = (detuning * x[i] + c0 * field * force_sign) * dt;
                y[i] += dy;

                // dx/dt = y_i
                let new_x = x[i] + dt * y[i];

                // Boundary condition based on mode
                if new_x > 1.0f32 {
                    x[i] = 1.0f32;
                    if config.mode == SolverPhysicsMode::Discrete {
                        y[i] = 0.0f32;
                    } else {
                        y[i] = -0.5f32 * y[i];
                    }
                } else if new_x < -1.0f32 {
                    x[i] = -1.0f32;
                    if config.mode == SolverPhysicsMode::Discrete {
                        y[i] = 0.0f32;
                    } else {
                        y[i] = -0.5f32 * y[i];
                    }
                } else {
                    x[i] = new_x;
                }
            }
        }

        // Read out final binarized spin state s_i* in {-1, +1}
        let final_spins: Vec<i8> = x
            .iter()
            .map(|&val| if val >= 0.0f32 { 1i8 } else { -1i8 })
            .collect();

        let energy = ising.compute_energy(&final_spins)?;

        if let Some(num_x) = config.num_x {
            let (violation, deviations) = ising.compute_saddle_violation(num_x, &final_spins);
            if deviations < best_deviations
                || (deviations == best_deviations && (violation < best_violation || (violation == best_violation && energy < best_energy)))
            {
                best_deviations = deviations;
                best_violation = violation;
                best_energy = energy;
                best_spins = final_spins.clone();
            }
        } else {
            if energy < best_energy {
                best_energy = energy;
                best_spins = final_spins.clone();
            }
        }

        replica_spins.push(final_spins);
        replica_energies.push(energy);
    }

    Ok(DsbResult {
        best_spins,
        best_energy,
        steps_executed: config.num_steps,
        c0,
        replica_spins,
        replica_energies,
    })
}
