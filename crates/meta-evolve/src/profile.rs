//! Specialized Solver Profile holding synthesized, domain-optimized hyperparameters.

/// Specialized hyperparameter profile synthesized by the meta-evolution engine.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecializedSolverProfile {
    /// Symplectic Euler time step $\Delta t \in [0.1, 1.2]$.
    pub dt: f32,
    /// Adiabatic detuning amplitude scale $a_0 \in [0.5, 2.0]$.
    pub a0: f32,
    /// Scaling multiplier applied to Goto's coupling factor $c_0 \in [0.2, 5.0]$.
    pub c0_scale: f32,
    /// Multiplier scaling hard constraint penalties $M \in [1.0, 10.0]$.
    pub penalty_scale: f64,
    /// Recommended number of time-integration steps $M$.
    pub recommended_steps: usize,
    /// Recommended number of parallel replicas on GPU.
    pub recommended_replicas: usize,
    /// Fitness score achieved during evolutionary synthesis.
    pub fitness: f64,
    /// Feasibility percentage across benchmark instances (0.0 to 100.0%).
    pub feasible_rate: f64,
    /// Generation number where this profile was synthesized.
    pub generation: usize,
}

impl Default for SpecializedSolverProfile {
    fn default() -> Self {
        Self {
            dt: 0.5,
            a0: 1.0,
            c0_scale: 1.0,
            penalty_scale: 2.0,
            recommended_steps: 600,
            recommended_replicas: 64,
            fitness: 0.0,
            feasible_rate: 100.0,
            generation: 0,
        }
    }
}

impl SpecializedSolverProfile {
    /// Display a structured summary of the synthesized profile.
    pub fn display_summary(&self) -> String {
        format!(
            "SpecializedSolverProfile {{\n  \
               dt: {:.4},\n  \
               a0: {:.4},\n  \
               c0_scale: {:.4},\n  \
               penalty_scale: {:.2},\n  \
               recommended_steps: {},\n  \
               recommended_replicas: {},\n  \
               fitness: {:.4},\n  \
               feasible_rate: {:.2}%,\n  \
               generation: {}\n\
             }}",
            self.dt,
            self.a0,
            self.c0_scale,
            self.penalty_scale,
            self.recommended_steps,
            self.recommended_replicas,
            self.fitness,
            self.feasible_rate,
            self.generation
        )
    }
}
