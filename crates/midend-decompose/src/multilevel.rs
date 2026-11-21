//! Module `multilevel` implements Hierarchical Multilevel Graph Coarsening
//! and the V-Cycle Multigrid Solver for Ultra-Scale Ising Optimization ($N \ge 10^4 - 10^6$).
//!
//! # Mathematical Foundations of Multilevel Coarsening
//!
//! Given a fine Ising Hamiltonian at level $l$:
//!
//! $$H_f(\mathbf{s}) = - \frac{1}{2} \sum_{u < v} J_{uv} s_u s_v - \sum_u h_u s_u$$
//!
//! ## Heavy-Edge Matching (HEM)
//!
//! Strongly coupled spins have high correlation in the ground state:
//! - If $J_{uv} > 0$ (ferromagnetic), spins tend to align ($s_u = s_v$).
//! - If $J_{uv} < 0$ (antiferromagnetic / Max-Cut), spins tend to anti-align ($s_u = -s_v$).
//!
//! For each matched pair $(u, v)$ contracted into super-vertex $W \in [0, N_c)$:
//!
//! $$s_u = S_W, \quad s_v = \sigma_{uv} S_W, \quad \text{where } \sigma_{uv} = \operatorname{sgn}(J_{uv})$$
//!
//! ## Exact Coarse Hamiltonian Projection:
//!
//! For any coarse spin assignment $\mathbf{S} \in \{-1, +1\}^{N_c}$, under prolongation $s_u = \sigma_u S_{C(u)}$:
//!
//! $$H_f(\mathbf{s}) \equiv H_c(\mathbf{S}) + \Delta E_{\text{internal}}$$
//!
//! where:
//! 1. External field projection:
//!    $$h^c_W = \sum_{u \in C^{-1}(W)} h_u \sigma_u$$
//! 2. Inter-cluster coupling:
//!    $$J^c_{W_1, W_2} = \sum_{u \in C^{-1}(W_1), v \in C^{-1}(W_2)} J_{uv} \sigma_u \sigma_v$$
//! 3. Internal energy offset (exact ground-state shift):
//!    $$\Delta E_{\text{internal}} = - \frac{1}{2} \sum_{u < v, C(u)=C(v)} J_{uv} \sigma_u \sigma_v = - \frac{1}{2} \sum_{(u,v) \in M} |J_{uv}|$$

use std::collections::BTreeMap;
use backend_physics::dsb::{DsbConfig, solve_dsb_cpu};
use backend_physics::gpu::solve_dsb_gpu;
use backend_physics::ising::IsingCsr;
use crate::decomposer::DecomposeError;

/// Configuration for the Multilevel V-Cycle Decomposer & Solver.
#[derive(Debug, Clone)]
pub struct MultilevelConfig {
    /// Maximum size of the coarsest graph to solve directly via dSB (default: 128).
    pub coarse_threshold: usize,
    /// Maximum number of coarsening levels allowed (default: 12).
    pub max_levels: usize,
    /// Number of local search relaxation sweeps per uncoarsening level (default: 3).
    pub refinement_passes: usize,
    /// Number of dSB steps to run at the coarsest level (default: 800).
    pub coarsest_steps: usize,
    /// Number of parallel replicas at the coarsest level (default: 64).
    pub coarsest_replicas: usize,
    /// Random seed for deterministic simulation.
    pub seed: u64,
}

impl Default for MultilevelConfig {
    fn default() -> Self {
        Self {
            coarse_threshold: 128,
            max_levels: 12,
            refinement_passes: 3,
            coarsest_steps: 800,
            coarsest_replicas: 64,
            seed: 42,
        }
    }
}

/// Mapping from fine spins to coarse super-vertices.
#[derive(Debug, Clone)]
pub struct LevelMapping {
    /// Fine spin index $u \mapsto (W, \sigma_u)$ where $s_u = \sigma_u S_W$.
    pub fine_to_coarse: Vec<(usize, i8)>,
    /// Number of coarse super-vertices $N_c$.
    pub num_coarse: usize,
    /// Energy offset induced by internal contracted edges $\Delta E_{\text{internal}}$.
    pub internal_energy_offset: f64,
}

/// A level in the multilevel hierarchy.
#[derive(Debug, Clone)]
pub struct HierarchyLevel {
    pub ising: IsingCsr,
    pub level_index: usize,
}

/// Multilevel hierarchy storing coarse representations and prolongation operators.
#[derive(Debug, Clone)]
pub struct MultilevelHierarchy {
    pub levels: Vec<HierarchyLevel>,
    pub mappings: Vec<LevelMapping>,
    pub total_internal_energy: f64,
}

/// Result of the Multilevel V-Cycle optimization.
#[derive(Debug, Clone)]
pub struct MultilevelResult {
    /// Optimal fine-level spin configuration $\mathbf{s}^* \in \{-1, +1\}^N$.
    pub best_spins: Vec<i8>,
    /// Best fine-level Ising Hamiltonian energy $H_f(\mathbf{s}^*)$.
    pub best_energy: f64,
    /// Total number of hierarchy levels traversed in the V-Cycle.
    pub num_levels: usize,
    /// Number of variables at the coarsest level.
    pub coarsest_spins: usize,
    /// Energy achieved at the coarsest level.
    pub coarsest_energy: f64,
}

/// Coarsen an `IsingCsr` graph by one level using Heavy-Edge Matching (HEM).
pub fn coarsen_ising_one_level(fine: &IsingCsr) -> Result<(IsingCsr, LevelMapping), DecomposeError> {
    let n = fine.num_spins;
    let mut visited = vec![false; n];
    let mut fine_to_coarse = vec![(0usize, 1i8); n];
    let mut next_coarse = 0usize;

    // 1. Heavy-Edge Matching
    for u in 0..n {
        if visited[u] {
            continue;
        }

        let start = fine.row_ptr[u] as usize;
        let end = fine.row_ptr[u + 1] as usize;

        let mut best_v: Option<usize> = None;
        let mut best_weight = 0.0f32;
        let mut best_sign = 1i8;

        for k in start..end {
            let v = fine.col_indices[k] as usize;
            if v != u && !visited[v] {
                let weight = fine.values[k];
                let abs_w = weight.abs();
                if abs_w > best_weight {
                    best_weight = abs_w;
                    best_v = Some(v);
                    best_sign = if weight >= 0.0 { 1 } else { -1 };
                }
            }
        }

        let c_id = next_coarse;
        next_coarse += 1;
        visited[u] = true;
        fine_to_coarse[u] = (c_id, 1i8);

        if let Some(v) = best_v {
            visited[v] = true;
            fine_to_coarse[v] = (c_id, best_sign);
        }
    }

    let num_coarse = next_coarse;

    // 2. Project External Fields and Compute Internal Energy Shift
    let mut coarse_h = vec![0.0f32; num_coarse];
    let mut internal_energy = 0.0f64;

    for u in 0..n {
        let (c_u, s_u) = fine_to_coarse[u];
        coarse_h[c_u] += (fine.h[u] as f64 * s_u as f64) as f32;
    }

    // 3. Project Inter-Cluster Couplings
    let mut coarse_j_map: BTreeMap<(usize, usize), f32> = BTreeMap::new();

    for u in 0..n {
        let (c_u, s_u) = fine_to_coarse[u];
        let start = fine.row_ptr[u] as usize;
        let end = fine.row_ptr[u + 1] as usize;

        for k in start..end {
            let v = fine.col_indices[k] as usize;
            if u < v {
                let (c_v, s_v) = fine_to_coarse[v];
                let val = fine.values[k];
                let eff_val = (val as f64 * s_u as f64 * s_v as f64) as f32;

                if c_u == c_v {
                    // Internal edge in symmetric Hamiltonian contributes -1.0 * J_uv * s_u * s_v
                    internal_energy += -1.0 * (val as f64) * (s_u as f64) * (s_v as f64);
                } else {
                    let key = if c_u < c_v { (c_u, c_v) } else { (c_v, c_u) };
                    *coarse_j_map.entry(key).or_insert(0.0) += eff_val;
                }
            }
        }
    }

    // 4. Assemble Coarse CSR Model
    let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_coarse];
    for ((c1, c2), val) in coarse_j_map {
        if val.abs() > 1e-8 {
            adj[c1].push((c2 as u32, val));
            adj[c2].push((c1 as u32, val));
        }
    }

    let mut row_ptr = Vec::with_capacity(num_coarse + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    let mut current_offset = 0u32;
    row_ptr.push(current_offset);

    for mut neighbors in adj {
        neighbors.sort_by_key(|&(col, _)| col);
        for (col, val) in neighbors {
            col_indices.push(col);
            values.push(val);
            current_offset += 1;
        }
        row_ptr.push(current_offset);
    }

    let coarse_ising = IsingCsr::new(num_coarse, row_ptr, col_indices, values, coarse_h)?;
    let mapping = LevelMapping {
        fine_to_coarse,
        num_coarse,
        internal_energy_offset: internal_energy,
    };

    Ok((coarse_ising, mapping))
}

/// Prolongate a coarse spin configuration $\mathbf{S}$ to fine spins $\mathbf{s}$.
#[inline]
pub fn prolongate(coarse_spins: &[i8], mapping: &LevelMapping) -> Vec<i8> {
    let fine_n = mapping.fine_to_coarse.len();
    let mut fine_spins = vec![1i8; fine_n];
    for u in 0..fine_n {
        let (c_u, s_u) = mapping.fine_to_coarse[u];
        fine_spins[u] = s_u * coarse_spins[c_u];
    }
    fine_spins
}

/// Post-smoothing 1-flip Greedy Relaxation / Local Search.
///
/// Monotonically decreases or preserves the Ising Hamiltonian energy:
/// $$\Delta E_i = 2 s_i \left( \sum_{j} J_{ij} s_j + h_i \right)$$
/// If $\Delta E_i < 0$, flipping $s_i \to -s_i$ strictly decreases energy.
pub fn refine_local_search(ising: &IsingCsr, spins: &mut [i8], max_sweeps: usize) -> f64 {
    let n = ising.num_spins;
    if n == 0 {
        return 0.0;
    }

    for _ in 0..max_sweeps {
        let mut flipped_any = false;
        for i in 0..n {
            let start = ising.row_ptr[i] as usize;
            let end = ising.row_ptr[i + 1] as usize;
            let mut field = ising.h[i] as f64;
            for k in start..end {
                let j = ising.col_indices[k] as usize;
                field += (ising.values[k] as f64) * (spins[j] as f64);
            }

            let delta_e = 2.0 * (spins[i] as f64) * field;
            if delta_e < -1e-6 {
                spins[i] = -spins[i];
                flipped_any = true;
            }
        }
        if !flipped_any {
            break;
        }
    }

    ising.compute_energy(spins).unwrap_or(0.0)
}

/// Hierarchical Multilevel V-Cycle Decomposer and Solver.
pub struct MultilevelSolver {
    pub config: MultilevelConfig,
}

impl MultilevelSolver {
    pub fn new(config: MultilevelConfig) -> Self {
        Self { config }
    }

    /// Build the full coarsening hierarchy from $L_0$ down to $L_k$.
    pub fn build_hierarchy(&self, fine_ising: &IsingCsr) -> Result<MultilevelHierarchy, DecomposeError> {
        let mut levels = Vec::new();
        let mut mappings = Vec::new();
        let mut total_internal = 0.0f64;

        levels.push(HierarchyLevel {
            ising: fine_ising.clone(),
            level_index: 0,
        });

        let mut current_ising = fine_ising.clone();
        for level in 0..self.config.max_levels {
            if current_ising.num_spins <= self.config.coarse_threshold {
                break;
            }

            let (coarse_ising, mapping) = coarsen_ising_one_level(&current_ising)?;
            let reduction = current_ising.num_spins - coarse_ising.num_spins;
            if reduction == 0 || (reduction as f64 / current_ising.num_spins as f64) < 0.02 {
                // Diminishing returns in coarsening
                break;
            }

            total_internal += mapping.internal_energy_offset;
            mappings.push(mapping);
            levels.push(HierarchyLevel {
                ising: coarse_ising.clone(),
                level_index: level + 1,
            });
            current_ising = coarse_ising;
        }

        Ok(MultilevelHierarchy {
            levels,
            mappings,
            total_internal_energy: total_internal,
        })
    }

    /// Execute the complete V-Cycle on GPU or CPU.
    pub fn v_cycle(&self, ising: &IsingCsr, use_gpu: bool) -> Result<MultilevelResult, DecomposeError> {
        // 1. Downward Phase: Build Coarsening Hierarchy
        let hierarchy = self.build_hierarchy(ising)?;
        let num_levels = hierarchy.levels.len();
        let coarsest_level = &hierarchy.levels[num_levels - 1];

        // 2. Coarsest Level: Solve directly via dSB
        let dsb_config = DsbConfig {
            num_steps: self.config.coarsest_steps,
            dt: 0.7,
            a0: 1.0,
            c0: None,
            num_replicas: self.config.coarsest_replicas,
            seed: self.config.seed,
            ..Default::default()
        };

        let coarsest_result = if use_gpu {
            match solve_dsb_gpu(&coarsest_level.ising, &dsb_config) {
                Ok(res) => res,
                Err(_) => {
                    // Transparent fallback to CPU DST mode if GPU device is busy or unavailable
                    solve_dsb_cpu(&coarsest_level.ising, &dsb_config)?
                }
            }
        } else {
            solve_dsb_cpu(&coarsest_level.ising, &dsb_config)?
        };

        let mut current_spins = coarsest_result.best_spins;
        let coarsest_energy = coarsest_result.best_energy;

        // 3. Upward Phase: Prolongation and Post-smoothing Refinement
        for l in (0..num_levels - 1).rev() {
            let mapping = &hierarchy.mappings[l];
            let fine_level = &hierarchy.levels[l];

            // 3.1 Prolongate to finer level
            let mut prolonged = prolongate(&current_spins, mapping);

            // 3.2 Post-smoothing local search refinement
            refine_local_search(&fine_level.ising, &mut prolonged, self.config.refinement_passes);
            current_spins = prolonged;
        }

        let best_energy = ising.compute_energy(&current_spins)?;

        Ok(MultilevelResult {
            best_spins: current_spins,
            best_energy,
            num_levels,
            coarsest_spins: coarsest_level.ising.num_spins,
            coarsest_energy,
        })
    }

    /// Solve with GPU acceleration for the coarsest level.
    pub fn solve_gpu(&self, ising: &IsingCsr) -> Result<MultilevelResult, DecomposeError> {
        self.v_cycle(ising, true)
    }

    /// Solve in CPU Deterministic Simulation Testing (DST) mode.
    pub fn solve_cpu(&self, ising: &IsingCsr) -> Result<MultilevelResult, DecomposeError> {
        self.v_cycle(ising, false)
    }
}
