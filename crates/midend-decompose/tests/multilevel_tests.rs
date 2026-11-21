use backend_physics::ising::IsingCsr;
use midend_decompose::multilevel::{
    coarsen_ising_one_level, prolongate, refine_local_search,
    MultilevelConfig, MultilevelSolver,
};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Generate an Erdős–Rényi random Ising model for stress testing.
fn generate_random_ising(num_spins: usize, edge_prob: f64, seed: u64) -> IsingCsr {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut adj: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_spins];

    for i in 0..num_spins {
        for j in (i + 1)..num_spins {
            if rng.gen_bool(edge_prob) {
                let weight: f32 = rng.gen_range(-2.0..2.0);
                if weight.abs() > 0.05 {
                    adj[i].push((j as u32, weight));
                    adj[j].push((i as u32, weight));
                }
            }
        }
    }

    let mut row_ptr = Vec::with_capacity(num_spins + 1);
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

    let h: Vec<f32> = (0..num_spins).map(|_| rng.gen_range(-1.0..1.0)).collect();

    IsingCsr::new(num_spins, row_ptr, col_indices, values, h).unwrap()
}

#[test]
fn test_exact_energy_identity_after_coarsening() {
    let fine = generate_random_ising(30, 0.4, 12345);
    let (coarse, mapping) = coarsen_ising_one_level(&fine).expect("Coarsening failed");

    assert!(coarse.num_spins < fine.num_spins, "Coarsening must reduce variables");
    assert!(coarse.num_spins >= 15, "Coarsening should reduce by at most ~50%");

    let mut rng = ChaCha20Rng::seed_from_u64(999);

    // Test across 10 random coarse spin assignments
    for _ in 0..10 {
        let coarse_spins: Vec<i8> = (0..coarse.num_spins)
            .map(|_| if rng.gen_bool(0.5) { 1 } else { -1 })
            .collect();

        let fine_spins = prolongate(&coarse_spins, &mapping);

        let e_fine = fine.compute_energy(&fine_spins).unwrap();
        let e_coarse = coarse.compute_energy(&coarse_spins).unwrap();
        let e_projected = e_coarse + mapping.internal_energy_offset;

        let diff = (e_fine - e_projected).abs();
        assert!(
            diff < 1e-4,
            "Exact energy identity violated! E_fine={e_fine}, E_projected={e_projected}, diff={diff}"
        );
    }
}

#[test]
fn test_refine_local_search_monotonicity() {
    let ising = generate_random_ising(100, 0.3, 42);
    let mut spins = vec![1i8; 100];
    let initial_energy = ising.compute_energy(&spins).unwrap();

    let refined_energy = refine_local_search(&ising, &mut spins, 5);

    assert!(
        refined_energy <= initial_energy + 1e-6,
        "Local search must monotonically decrease or preserve energy! initial={initial_energy}, refined={refined_energy}"
    );
}

#[test]
fn test_multilevel_solver_v_cycle_1000_variables() {
    let ising = generate_random_ising(1000, 0.05, 777);
    let config = MultilevelConfig {
        coarse_threshold: 64,
        max_levels: 8,
        refinement_passes: 3,
        coarsest_steps: 400,
        coarsest_replicas: 16,
        seed: 42,
    };

    let solver = MultilevelSolver::new(config);
    let result = solver.solve_cpu(&ising).expect("V-Cycle solve failed");

    assert_eq!(result.best_spins.len(), 1000);
    assert!(result.num_levels >= 3, "Expected at least 3 hierarchy levels, got {}", result.num_levels);
    assert!(result.coarsest_spins <= 128, "Coarsest spins should be reduced, got {}", result.coarsest_spins);

    // Initial naive all-ones energy
    let all_ones = vec![1i8; 1000];
    let naive_energy = ising.compute_energy(&all_ones).unwrap();

    assert!(
        result.best_energy < naive_energy,
        "Multilevel V-Cycle energy ({}) should beat naive all-ones ({})",
        result.best_energy,
        naive_energy
    );
}
