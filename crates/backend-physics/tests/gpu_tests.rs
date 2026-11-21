use backend_physics::{solve_dsb_cpu, solve_dsb_gpu, DsbConfig, GpuContext, IsingCsr};

#[test]
fn test_gpu_context_initialization() {
    let ctx = GpuContext::new();
    match ctx {
        Ok(c) => {
            println!(
                "Successfully initialized GPU: {} ({:?})",
                c.adapter_info.name, c.adapter_info.backend
            );
        }
        Err(e) => {
            panic!("Failed to initialize GPU context: {e}");
        }
    }
}

#[test]
fn test_triangle_graph_gpu_vs_cpu() {
    // Triangle graph K_3 with edges (0,1), (1,2), (2,0) with weight -1.0
    let edges = vec![(0, 1, -1.0f32), (1, 2, -1.0f32), (2, 0, -1.0f32)];
    let ising = IsingCsr::from_undirected_edges(3, &edges, None).expect("Valid CSR");

    let config = DsbConfig {
        num_steps: 400,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 4,
        seed: 42,
        ..Default::default()
    };

    let cpu_res = solve_dsb_cpu(&ising, &config).expect("CPU solver failed");
    let gpu_res = solve_dsb_gpu(&ising, &config).expect("GPU solver failed");

    println!("Triangle Graph - CPU Energy: {}", cpu_res.best_energy);
    println!("Triangle Graph - GPU Energy: {}", gpu_res.best_energy);

    // Max-Cut on K_3 has ground state energy = -1.0 (cutting 2 edges)
    assert_eq!(cpu_res.best_energy, -1.0);
    assert_eq!(gpu_res.best_energy, -1.0);
    assert_eq!(cpu_res.best_energy, gpu_res.best_energy);
}

#[test]
fn test_cross_comparison_cpu_gpu_random_ising() {
    // 30-node random Ising model
    let n = 30;
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if (i * 11 + j * 17) % 4 == 0 {
                edges.push((i, j, -1.0f32));
            }
        }
    }
    let ising = IsingCsr::from_undirected_edges(n, &edges, None).unwrap();

    let config = DsbConfig {
        num_steps: 600,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 16,
        seed: 1337,
        ..Default::default()
    };

    let cpu_res = solve_dsb_cpu(&ising, &config).expect("CPU solver failed");
    let gpu_res = solve_dsb_gpu(&ising, &config).expect("GPU solver failed");

    println!("Random Ising (N={n}) - CPU Best Energy: {}", cpu_res.best_energy);
    println!("Random Ising (N={n}) - GPU Best Energy: {}", gpu_res.best_energy);

    // Verify that GPU reaches an optimal or near-optimal state consistent with CPU
    let diff = (cpu_res.best_energy - gpu_res.best_energy).abs();
    println!("Energy difference |E_cpu - E_gpu|: {diff}");
    assert!(
        diff <= 4.0,
        "Energy disparity too large: CPU={}, GPU={}",
        cpu_res.best_energy,
        gpu_res.best_energy
    );
}

#[test]
fn test_gpu_1024_replicas_scaling() {
    // Test that the GPU can successfully execute 1,024 replicas in a single CommandEncoder
    let n = 50;
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if (i + j) % 5 == 0 {
                edges.push((i, j, -1.0f32));
            }
        }
    }
    let ising = IsingCsr::from_undirected_edges(n, &edges, None).unwrap();

    let config = DsbConfig {
        num_steps: 200,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 1024,
        seed: 777,
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let gpu_res = solve_dsb_gpu(&ising, &config).expect("GPU 1024-replica solver failed");
    let elapsed = start.elapsed();

    println!(
        "GPU 1024 Replicas completed in {:.2} ms, Best Energy: {}",
        elapsed.as_secs_f64() * 1000.0,
        gpu_res.best_energy
    );

    assert_eq!(gpu_res.replica_energies.len(), 1024);
    assert_eq!(gpu_res.replica_spins.len(), 1024);
    for r in 0..1024 {
        assert_eq!(gpu_res.replica_spins[r].len(), n);
    }
}
