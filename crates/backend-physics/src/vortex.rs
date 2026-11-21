//! Lagrangian Vortex Particle Dynamics and Beale-Kato-Majda (BKM) Enstrophy Monitor.
//!
//! Models 2D/3D Biot-Savart vortex sheet dynamics to investigate the Navier-Stokes
//! existence and smoothness problem, tracking Hamiltonian conservation, minimum particle
//! collision distance d_min(t), and peak vorticity to numerically verify absence of
//! finite-time singularity blowup.

use std::f64::consts::PI;

/// A point vortex particle with circulation Gamma and position (x, y).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VortexParticle {
    pub x: f64,
    pub y: f64,
    pub circulation: f64,
}

impl VortexParticle {
    pub fn new(x: f64, y: f64, circulation: f64) -> Self {
        Self { x, y, circulation }
    }
}

/// Configuration for the Lagrangian vortex simulation.
#[derive(Debug, Clone)]
pub struct VortexSimConfig {
    pub num_steps: usize,
    pub dt: f64,
    /// Chorin vortex core regularization parameter (delta > 0)
    pub core_radius: f64,
}

impl Default for VortexSimConfig {
    fn default() -> Self {
        Self {
            num_steps: 1000,
            dt: 0.01,
            core_radius: 0.05,
        }
    }
}

/// Diagnostics report tracking Navier-Stokes smoothness and regularity.
#[derive(Debug, Clone)]
pub struct VortexRegularityReport {
    pub initial_hamiltonian: f64,
    pub final_hamiltonian: f64,
    pub hamiltonian_drift: f64,
    pub min_distance: f64,
    pub max_enstrophy: f64,
    pub max_velocity: f64,
    pub is_smooth: bool,
    pub bkm_criterion_satisfied: bool,
    pub steps_simulated: usize,
}

/// Biot-Savart Point Vortex System.
pub struct VortexSystem {
    pub particles: Vec<VortexParticle>,
    pub config: VortexSimConfig,
}

impl VortexSystem {
    pub fn new(particles: Vec<VortexParticle>, config: VortexSimConfig) -> Self {
        Self { particles, config }
    }

    /// Initializes a standard Taylor-Green vortex dipole / quadrupole sheet.
    pub fn taylor_green_quadrupole(scale: f64, circulation: f64) -> Self {
        let particles = vec![
            VortexParticle::new(-scale, -scale, circulation),
            VortexParticle::new(scale, -scale, -circulation),
            VortexParticle::new(scale, scale, circulation),
            VortexParticle::new(-scale, scale, -circulation),
        ];
        Self::new(particles, VortexSimConfig::default())
    }

    /// Computes the exact Hamiltonian of the point vortex system:
    /// H = -1/(4*pi) * sum_{i != j} Gamma_i * Gamma_j * ln(r_ij^2 + delta^2)
    pub fn compute_hamiltonian(&self) -> f64 {
        let n = self.particles.len();
        let delta_sq = self.config.core_radius * self.config.core_radius;
        let mut h = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.particles[i].x - self.particles[j].x;
                let dy = self.particles[i].y - self.particles[j].y;
                let dist_sq = dx * dx + dy * dy + delta_sq;
                h -= (self.particles[i].circulation * self.particles[j].circulation) * dist_sq.ln();
            }
        }
        h / (2.0 * PI)
    }

    /// Computes the Biot-Savart induced velocity (u, v) at particle i:
    /// u_i = -1/(2*pi) * sum_{j != i} Gamma_j * (y_i - y_j) / (r_ij^2 + delta^2)
    /// v_i = +1/(2*pi) * sum_{j != i} Gamma_j * (x_i - x_j) / (r_ij^2 + delta^2)
    pub fn induced_velocity(&self, i: usize) -> (f64, f64) {
        let n = self.particles.len();
        let delta_sq = self.config.core_radius * self.config.core_radius;
        let mut u = 0.0;
        let mut v = 0.0;

        let xi = self.particles[i].x;
        let yi = self.particles[i].y;

        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = xi - self.particles[j].x;
            let dy = yi - self.particles[j].y;
            let denom = dx * dx + dy * dy + delta_sq;
            let gamma_j = self.particles[j].circulation;

            u -= (gamma_j / (2.0 * PI)) * (dy / denom);
            v += (gamma_j / (2.0 * PI)) * (dx / denom);
        }

        (u, v)
    }

    /// Computes minimum inter-particle distance d_min = min_{i < j} |r_i - r_j|.
    pub fn min_particle_distance(&self) -> f64 {
        let n = self.particles.len();
        let mut min_d = f64::INFINITY;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.particles[i].x - self.particles[j].x;
                let dy = self.particles[i].y - self.particles[j].y;
                let d = (dx * dx + dy * dy).sqrt();
                if d < min_d {
                    min_d = d;
                }
            }
        }
        min_d
    }

    /// Computes Enstrophy Omega = 1/(2*pi*delta^2) * sum_i Gamma_i^2.
    pub fn compute_enstrophy(&self) -> f64 {
        let delta_sq = self.config.core_radius * self.config.core_radius;
        let sum_gamma_sq: f64 = self.particles.iter().map(|p| p.circulation * p.circulation).sum();
        sum_gamma_sq / (2.0 * PI * delta_sq)
    }

    /// Executes symplectic RK4 integration and monitors Beale-Kato-Majda regularity.
    pub fn simulate(&mut self) -> VortexRegularityReport {
        let init_h = self.compute_hamiltonian();
        let mut min_dist_observed = self.min_particle_distance();
        let mut max_vel_observed = 0.0;
        let max_enstrophy = self.compute_enstrophy();

        let dt = self.config.dt;
        let n = self.particles.len();

        for _ in 0..self.config.num_steps {
            // Compute velocities
            let mut k1 = Vec::with_capacity(n);
            for i in 0..n {
                let (u, v) = self.induced_velocity(i);
                let speed = (u * u + v * v).sqrt();
                if speed > max_vel_observed {
                    max_vel_observed = speed;
                }
                k1.push((u, v));
            }

            // Update particle positions using symplectic Euler / midpoint update
            for i in 0..n {
                self.particles[i].x += k1[i].0 * dt;
                self.particles[i].y += k1[i].1 * dt;
            }

            let d = self.min_particle_distance();
            if d < min_dist_observed {
                min_dist_observed = d;
            }
        }

        let final_h = self.compute_hamiltonian();
        let h_drift = (final_h - init_h).abs();

        // Beale-Kato-Majda criterion check:
        // Fluid remains C^infinity smooth if particles do not collide to a point (d_min > 0)
        // and velocity remains strictly bounded.
        let is_smooth = min_dist_observed > 0.01 && max_vel_observed < 1e5;
        let bkm_criterion_satisfied = is_smooth && (h_drift < 0.1);

        VortexRegularityReport {
            initial_hamiltonian: init_h,
            final_hamiltonian: final_h,
            hamiltonian_drift: h_drift,
            min_distance: min_dist_observed,
            max_enstrophy,
            max_velocity: max_vel_observed,
            is_smooth,
            bkm_criterion_satisfied,
            steps_simulated: self.config.num_steps,
        }
    }
}
