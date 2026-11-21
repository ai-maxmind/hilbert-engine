//! Unit and mathematical equivalence tests for Multi-Device Domain Decomposition
//! and Halo Exchange (Sprint 10).

use backend_physics::{
    solve_distributed_dsb_cpu, solve_dsb_cpu, DomainDecomposer, DsbConfig, IsingCsr,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn test_domain_decomposition_partition_correctness() {
    // Construct 8-spin 1D ring Ising model: edge (0,1), (1,2), ..., (7,0)
    let n = 8;
    let mut edges = Vec::new();
    for i in 0..n {
        edges.push((i, (i + 1) % n, -1.0f32));
    }
    let h = vec![0.0f32; n];
    let ising = IsingCsr::from_undirected_edges(n, &edges, Some(&h)).expect("CSR build failed");

    // Partition into 3 domains:
    // Domain 0: spins [0, 1, 2]
    // Domain 1: spins [3, 4, 5]
    // Domain 2: spins [6, 7]
    let domains = DomainDecomposer::partition(&ising, 3);
    assert_eq!(domains.len(), 3);

    let mut total_local = 0;
    for (d_idx, d) in domains.iter().enumerate() {
        total_local += d.local_spins.len();
        println!(
            "Domain {}: local {:?}, halo {:?}, boundary {:?}",
            d_idx, d.local_spins, d.halo_spins, d.boundary_local_indices
        );
        // Domain boundary spins must not be empty since it's a connected ring
        assert!(!d.boundary_local_indices.is_empty());
        assert!(!d.halo_spins.is_empty());
    }
    assert_eq!(total_local, n, "All spins must be partitioned across domains");
}

#[test]
fn test_distributed_dsb_bit_for_bit_vs_monolithic() {
    // Generate 60-spin random graph with 250 edges
    let n = 60;
    let mut rng = ChaCha20Rng::seed_from_u64(0xDECA_FBAD_1010);
    let mut edges = Vec::new();
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.gen_bool(0.15) {
                let w = rng.gen_range(-2.0f32..2.0f32);
                edges.push((u, v, w));
            }
        }
    }
    let mut h = vec![0.0f32; n];
    for val in h.iter_mut() {
        *val = rng.gen_range(-1.0f32..1.0f32);
    }

    let ising = IsingCsr::from_undirected_edges(n, &edges, Some(&h)).expect("CSR failed");

    let config = DsbConfig {
        num_steps: 600,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 16,
        seed: 42,
        ..Default::default()
    };

    // 1. Monolithic solve
    let res_mono = solve_dsb_cpu(&ising, &config).expect("Monolithic solve failed");

    // 2. Distributed solve across 4 domains
    let res_dist = solve_distributed_dsb_cpu(&ising, 4, &config).expect("Distributed solve failed");

    // 3. Mathematical proof of equivalence:
    // Both solvers MUST achieve identical energy and bit-for-bit identical spin configuration!
    println!("Monolithic Energy:  {:.6}", res_mono.best_energy);
    println!("Distributed Energy: {:.6}", res_dist.best_energy);

    assert_eq!(
        res_mono.best_spins, res_dist.best_spins,
        "Distributed 4-domain solver with Halo Exchange must match monolithic solver 100% bit-for-bit!"
    );
    assert!(
        (res_mono.best_energy - res_dist.best_energy).abs() < 1e-10,
        "Energy must be bit-for-bit identical"
    );
}
