use backend_physics::{
    solve_dsb_cpu, solve_dsb_gpu, DsbConfig, GpuContext, IsingCsr, SolverPhysicsMode,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Create a fully-connected Sherrington-Kirkpatrick (SK) Spin Glass graph.
/// SK models have severe frustration and rugged energy landscapes with numerous local minima.
fn generate_sk_spin_glass(n: usize, seed: u64) -> IsingCsr {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            // Random Gaussian or bimodal coupling ±1 / sqrt(n)
            let val = if rng.gen_bool(0.5) { 1.0f32 } else { -1.0f32 };
            let normalized = val / (n as f32).sqrt();
            adj[i].push((j as u32, normalized));
            adj[j].push((i as u32, normalized));
        }
    }

    let mut row_ptr = Vec::with_capacity(n + 1);
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

    let h: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.2..0.2)).collect();

    IsingCsr::new(n, row_ptr, col_indices, values, h).unwrap()
}

#[test]
fn test_cpu_physics_modes_dsb_bsb_csb() {
    let ising = generate_sk_spin_glass(32, 101);

    for mode in [
        SolverPhysicsMode::Discrete,
        SolverPhysicsMode::Ballistic,
        SolverPhysicsMode::Chaotic,
    ] {
        let config = DsbConfig {
            num_steps: 400,
            dt: 0.6,
            a0: 1.0,
            c0: None,
            num_replicas: 4,
            seed: 42,
            mode,
            ..Default::default()
        };

        let result = solve_dsb_cpu(&ising, &config).expect("CPU solver failed");

        assert_eq!(result.best_spins.len(), 32);
        for &s in &result.best_spins {
            assert!(s == 1 || s == -1, "Spins must be ±1, got {s}");
        }

        let energy = ising.compute_energy(&result.best_spins).unwrap();
        assert!((energy - result.best_energy).abs() < 1e-4);
        println!("Mode {:?}: Best Energy = {:.4}", mode, result.best_energy);
    }
}

#[test]
fn test_gpu_physics_modes_dsb_bsb_csb() {
    let ising = generate_sk_spin_glass(64, 202);
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(_) => return, // Gracefully skip if no Vulkan GPU adapter
    };

    println!("Running GPU Physics Modes on {}", ctx.adapter_info.name);

    for mode in [
        SolverPhysicsMode::Discrete,
        SolverPhysicsMode::Ballistic,
        SolverPhysicsMode::Chaotic,
    ] {
        let config = DsbConfig {
            num_steps: 500,
            dt: 0.6,
            a0: 1.0,
            c0: None,
            num_replicas: 16,
            seed: 42,
            mode,
            ..Default::default()
        };

        let result = solve_dsb_gpu(&ising, &config).expect("GPU solver failed");

        assert_eq!(result.best_spins.len(), 64);
        for &s in &result.best_spins {
            assert!(s == 1 || s == -1);
        }

        let energy = ising.compute_energy(&result.best_spins).unwrap();
        assert!((energy - result.best_energy).abs() < 1e-4);
        println!("GPU Mode {:?}: Best Energy = {:.4}", mode, result.best_energy);
    }
}
