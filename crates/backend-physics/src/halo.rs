//! Module `halo` provides Domain Decomposition and Multi-Device / Multi-GPU Halo Exchange
//! for ultra-scale Ising systems ($N \gg 10^5$).
//!
//! # Mathematical Principle of Domain Decomposition:
//! Let the Ising spin set $V = \{0, \dots, N-1\}$ be partitioned into $P$ disjoint subdomains:
//!
//! $$V = \Omega_0 \cup \Omega_1 \cup \dots \cup \Omega_{P-1}, \quad \Omega_p \cap \Omega_q = \emptyset \quad (\forall p \ne q)$$
//!
//! For each domain $p$:
//! - **Local Spins $V_p^{\text{local}}$:** Indices belonging to $\Omega_p$.
//! - **Boundary Spins $V_p^{\text{bound}}$:** Spins $i \in \Omega_p$ having at least one neighbor outside $\Omega_p$.
//! - **Halo (Ghost) Spins $\mathcal{H}_p$:** External spins $j \notin \Omega_p$ that are neighbors of spins in $V_p^{\text{bound}}$.
//!
//! The local effective state vector is $\tilde{\mathbf{s}}_p = (\mathbf{s}_{\Omega_p}, \mathbf{s}_{\mathcal{H}_p})$.
//! At each time integration step, local boundary spins are exchanged across halo buffers,
//! guaranteeing bit-for-bit mathematical identity with the monolithic solver.

use std::collections::{HashMap, HashSet};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::dsb::{DsbConfig, DsbError, DsbResult};
use crate::ising::IsingCsr;

/// Information for a single partitioned domain $\Omega_p$.
#[derive(Debug, Clone)]
pub struct DomainInfo {
    /// Domain identifier $p \in [0, P)$.
    pub domain_id: usize,
    /// Global spin indices belonging to this domain $\Omega_p$.
    pub local_spins: Vec<usize>,
    /// Global spin indices in the halo $\mathcal{H}_p$ (needed from other domains).
    pub halo_spins: Vec<usize>,
    /// Local indices within this domain that are boundary spins (must be sent to other domains).
    pub boundary_local_indices: Vec<usize>,
    /// Local CSR interaction matrix of size $N_p^{\text{local}} \times (N_p^{\text{local}} + N_p^{\text{halo}})$.
    pub local_csr: IsingCsr,
    /// Map from neighboring domain ID to list of `(local_spin_idx, remote_halo_idx)` to exchange.
    pub send_map: HashMap<usize, Vec<(usize, usize)>>,
}

/// Domain partitioner for sparse Ising interaction graphs.
pub struct DomainDecomposer;

impl DomainDecomposer {
    /// Partition an `IsingCsr` graph into $P$ balanced domains using contiguous block partitioning
    /// and compute exact halo boundary communication patterns.
    pub fn partition(ising: &IsingCsr, num_domains: usize) -> Vec<DomainInfo> {
        let n = ising.num_spins;
        assert!(num_domains > 0, "Number of domains must be >= 1");
        let p = num_domains.min(n);

        // 1. Assign each spin to a domain
        let chunk_size = (n + p - 1) / p;
        let mut spin_domain = vec![0usize; n];
        for i in 0..n {
            spin_domain[i] = (i / chunk_size).min(p - 1);
        }

        let mut domain_local_spins = vec![Vec::new(); p];
        for i in 0..n {
            domain_local_spins[spin_domain[i]].push(i);
        }

        // 2. Identify halo spins and boundary spins for each domain
        let mut domain_halo_sets = vec![HashSet::new(); p];
        let mut domain_boundary_sets = vec![HashSet::new(); p];

        for u in 0..n {
            let my_domain = spin_domain[u];
            let start = ising.row_ptr[u] as usize;
            let end = ising.row_ptr[u + 1] as usize;

            for k in start..end {
                let v = ising.col_indices[k] as usize;
                let other_domain = spin_domain[v];
                if other_domain != my_domain {
                    domain_boundary_sets[my_domain].insert(u);
                    domain_halo_sets[my_domain].insert(v);
                }
            }
        }

        // 3. Build DomainInfo for each domain
        let mut result = Vec::with_capacity(p);

        for d in 0..p {
            let local_spins = domain_local_spins[d].clone();
            let mut halo_spins: Vec<usize> = domain_halo_sets[d].iter().copied().collect();
            halo_spins.sort_unstable();

            // Map global index -> domain internal index [0..local_len + halo_len)
            let mut global_to_extended = HashMap::new();
            for (idx, &g) in local_spins.iter().enumerate() {
                global_to_extended.insert(g, idx);
            }
            let local_len = local_spins.len();
            for (h_idx, &g) in halo_spins.iter().enumerate() {
                global_to_extended.insert(g, local_len + h_idx);
            }

            let boundary_local: Vec<usize> = domain_boundary_sets[d]
                .iter()
                .map(|&g| *global_to_extended.get(&g).unwrap())
                .collect();

            // Build local CSR
            let extended_n = local_len + halo_spins.len();
            let mut local_row_ptr = vec![0u32; local_len + 1];
            let mut local_col_indices = Vec::new();
            let mut local_values = Vec::new();
            let mut local_h = vec![0.0f32; extended_n];

            for (local_idx, &g) in local_spins.iter().enumerate() {
                local_h[local_idx] = ising.h[g];
                let start = ising.row_ptr[g] as usize;
                let end = ising.row_ptr[g + 1] as usize;

                for k in start..end {
                    let neighbor_g = ising.col_indices[k] as usize;
                    let ext_col = *global_to_extended.get(&neighbor_g).expect("Neighbor must be in local or halo");
                    local_col_indices.push(ext_col as u32);
                    local_values.push(ising.values[k]);
                }
                local_row_ptr[local_idx + 1] = local_col_indices.len() as u32;
            }

            let local_csr = IsingCsr {
                num_spins: extended_n,
                row_ptr: local_row_ptr,
                col_indices: local_col_indices,
                values: local_values,
                h: local_h,
            };

            result.push(DomainInfo {
                domain_id: d,
                local_spins,
                halo_spins,
                boundary_local_indices: boundary_local,
                local_csr,
                send_map: HashMap::new(),
            });
        }

        // 4. Construct reciprocal communication send_map between domains
        let mut transfers = Vec::new();
        for d in 0..p {
            for (h_slot, &halo_global) in result[d].halo_spins.iter().enumerate() {
                let owner_domain = spin_domain[halo_global];
                let owner_local_idx = result[owner_domain]
                    .local_spins
                    .iter()
                    .position(|&g| g == halo_global)
                    .unwrap();
                transfers.push((owner_domain, d, owner_local_idx, h_slot));
            }
        }

        for (owner_domain, dst_domain, owner_local_idx, h_slot) in transfers {
            result[owner_domain]
                .send_map
                .entry(dst_domain)
                .or_default()
                .push((owner_local_idx, h_slot));
        }

        result
    }
}

/// Execute Multi-Device / Multi-Domain Distributed Simulated Bifurcation with synchronous Halo Exchange.
///
/// Guaranteed to produce bit-for-bit identical spin dynamics as monolithic `solve_dsb_cpu`.
pub fn solve_distributed_dsb_cpu(
    ising: &IsingCsr,
    num_domains: usize,
    config: &DsbConfig,
) -> Result<DsbResult, DsbError> {
    if num_domains <= 1 || ising.num_spins <= 1 {
        return crate::dsb::solve_dsb_cpu(ising, config);
    }

    let n = ising.num_spins;
    let domains = DomainDecomposer::partition(ising, num_domains);
    let p = domains.len();

    let c0 = config.c0.unwrap_or_else(|| ising.compute_c0());
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    let mut replica_spins: Vec<Vec<i8>> = Vec::with_capacity(config.num_replicas);
    let mut replica_energies: Vec<f64> = Vec::with_capacity(config.num_replicas);
    let mut best_spins: Vec<i8> = Vec::new();
    let mut best_energy = f64::INFINITY;

    let dt = config.dt;
    let a0 = config.a0;
    let m_f32 = config.num_steps as f32;

    for _replica in 0..config.num_replicas {
        // Initial perturbation sampled globally in identical sequence for exact DST equivalence
        let mut global_x = vec![0.0f32; n];
        let mut global_y = vec![0.0f32; n];
        for i in 0..n {
            global_x[i] = rng.gen_range(-0.1f32..0.1f32);
            global_y[i] = rng.gen_range(-0.1f32..0.1f32);
        }

        // Domain-local continuous states
        let mut domain_x: Vec<Vec<f32>> = Vec::with_capacity(p);
        let mut domain_y: Vec<Vec<f32>> = Vec::with_capacity(p);
        let mut domain_s: Vec<Vec<f32>> = Vec::with_capacity(p);

        for d in 0..p {
            let total_len = domains[d].local_csr.num_spins;
            let mut dx = vec![0.0f32; total_len];
            let mut dy = vec![0.0f32; total_len];

            for (local_idx, &g) in domains[d].local_spins.iter().enumerate() {
                dx[local_idx] = global_x[g];
                dy[local_idx] = global_y[g];
            }
            for (h_idx, &g) in domains[d].halo_spins.iter().enumerate() {
                let offset = domains[d].local_spins.len();
                dx[offset + h_idx] = global_x[g];
                dy[offset + h_idx] = global_y[g];
            }

            domain_x.push(dx);
            domain_y.push(dy);
            domain_s.push(vec![0.0f32; total_len]);
        }

        // Time evolution loop with Halo Exchange at each step
        for step in 0..config.num_steps {
            let a_t = a0 * (step as f32 / m_f32);
            let detuning = -(1.0f32 - a_t);

            // 1. Halo Exchange: Exchange boundary spin proxies s_j = sgn(x_j) between domains
            for d in 0..p {
                let local_len = domains[d].local_spins.len();
                for i in 0..local_len {
                    domain_s[d][i] = if domain_x[d][i] >= 0.0f32 { 1.0f32 } else { -1.0f32 };
                }
            }

            // Communicate halo values
            for src_domain in 0..p {
                for (&dst_domain, exchanges) in &domains[src_domain].send_map {
                    let dst_offset = domains[dst_domain].local_spins.len();
                    for &(src_local_idx, dst_halo_slot) in exchanges {
                        domain_s[dst_domain][dst_offset + dst_halo_slot] = domain_s[src_domain][src_local_idx];
                    }
                }
            }

            // 2. Local Symplectic Euler update within each domain
            for d in 0..p {
                let local_len = domains[d].local_spins.len();
                let csr = &domains[d].local_csr;

                for i in 0..local_len {
                    let start = csr.row_ptr[i] as usize;
                    let end = csr.row_ptr[i + 1] as usize;

                    let mut field = csr.h[i];
                    for k in start..end {
                        let col = csr.col_indices[k] as usize;
                        field += csr.values[k] * domain_s[d][col];
                    }

                    // dy/dt = detuning * x + c0 * field
                    let dy = (detuning * domain_x[d][i] + c0 * field) * dt;
                    domain_y[d][i] += dy;

                    // dx/dt = y
                    let new_x = domain_x[d][i] + dt * domain_y[d][i];

                    if new_x > 1.0f32 {
                        domain_x[d][i] = 1.0f32;
                        domain_y[d][i] = 0.0f32;
                    } else if new_x < -1.0f32 {
                        domain_x[d][i] = -1.0f32;
                        domain_y[d][i] = 0.0f32;
                    } else {
                        domain_x[d][i] = new_x;
                    }
                }
            }
        }

        // Gather final binarized spins into unified global configuration
        let mut final_spins = vec![0i8; n];
        for d in 0..p {
            for (local_idx, &g) in domains[d].local_spins.iter().enumerate() {
                final_spins[g] = if domain_x[d][local_idx] >= 0.0f32 { 1i8 } else { -1i8 };
            }
        }

        let energy = ising.compute_energy(&final_spins)?;
        if energy < best_energy {
            best_energy = energy;
            best_spins = final_spins.clone();
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
