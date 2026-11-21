use backend_physics::{solve_dsb_cpu, DsbConfig, IsingCsr, IsingError};

#[test]
fn test_csr_validation_errors() {
    // 1. Invalid row_ptr length
    let res = IsingCsr::new(3, vec![0, 1, 2], vec![1, 2], vec![1.0, 1.0], vec![0.0; 3]);
    assert!(matches!(res, Err(IsingError::InvalidRowPtrLen { .. })));

    // 2. Row ptr not starting with 0
    let res = IsingCsr::new(
        2,
        vec![1, 2, 2],
        vec![1, 0],
        vec![1.0, 1.0],
        vec![0.0; 2],
    );
    assert!(matches!(res, Err(IsingError::InvalidRowPtrStart(1))));

    // 3. Col index out of bounds
    let res = IsingCsr::new(
        2,
        vec![0, 1, 1],
        vec![5],
        vec![1.0],
        vec![0.0; 2],
    );
    assert!(matches!(res, Err(IsingError::ColIndexOutOfBounds { col: 5, .. })));

    // 4. Non-zero diagonal
    let res = IsingCsr::new(
        2,
        vec![0, 1, 1],
        vec![0],
        vec![1.0],
        vec![0.0; 2],
    );
    assert!(matches!(res, Err(IsingError::NonZeroDiagonal { spin: 0, .. })));
}

#[test]
fn test_triangle_graph_maxcut_energy() {
    // Triangle graph K_3 with edges (0,1), (1,2), (2,0)
    // Max-Cut minimizes H(s) = - 0.5 * s^T J s with J_ij = -1
    let edges = vec![(0, 1, -1.0f32), (1, 2, -1.0f32), (2, 0, -1.0f32)];
    let ising = IsingCsr::from_undirected_edges(3, &edges, None).expect("Valid CSR");

    // All same spin: H = -(-1)*3 = +3
    let all_up = [1i8, 1i8, 1i8];
    assert_eq!(ising.compute_energy(&all_up).unwrap(), 3.0);

    // Two up, one down: H = -1
    // Cut = (|E| - H) / 2 = (3 - (-1)) / 2 = 2 (Max-Cut)
    let cut_state = [1i8, 1i8, -1i8];
    assert_eq!(ising.compute_energy(&cut_state).unwrap(), -1.0);
}

#[test]
fn test_dst_bit_for_bit_reproducibility() {
    // Construct a 20-node random graph
    let mut edges = Vec::new();
    for i in 0..20 {
        for j in (i + 1)..20 {
            if (i * 7 + j * 13) % 3 == 0 {
                edges.push((i, j, -1.0f32));
            }
        }
    }
    let ising = IsingCsr::from_undirected_edges(20, &edges, None).unwrap();

    let config = DsbConfig {
        num_steps: 500,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 4,
        seed: 0xDEADBEEF_CAFE1337,
        ..Default::default()
    };

    // Run 1
    let run1 = solve_dsb_cpu(&ising, &config).expect("Run 1 failed");

    // Run 2
    let run2 = solve_dsb_cpu(&ising, &config).expect("Run 2 failed");

    // Strict 100% bit-for-bit assertion
    assert_eq!(
        run1.best_spins, run2.best_spins,
        "Best spins must be identical bit-for-bit"
    );
    assert_eq!(
        run1.best_energy.to_bits(),
        run2.best_energy.to_bits(),
        "Best energy must be identical bit-for-bit (IEEE 754 bitwise match)"
    );
    assert_eq!(
        run1.c0.to_bits(),
        run2.c0.to_bits(),
        "c0 must be identical bit-for-bit"
    );
    assert_eq!(
        run1.replica_energies.len(),
        run2.replica_energies.len()
    );
    for r in 0..config.num_replicas {
        assert_eq!(
            run1.replica_energies[r].to_bits(),
            run2.replica_energies[r].to_bits(),
            "Replica {r} energy bitwise mismatch"
        );
        assert_eq!(
            run1.replica_spins[r], run2.replica_spins[r],
            "Replica {r} spins mismatch"
        );
    }
}

#[test]
fn test_spmv_vs_dense() {
    let dense = vec![
        vec![0.0, -1.0, 2.5],
        vec![-1.0, 0.0, -0.5],
        vec![2.5, -0.5, 0.0],
    ];
    let ising = IsingCsr::from_dense(&dense, None).unwrap();

    let x = [0.8f32, -0.4f32, 1.2f32];
    let mut out = [0.0f32; 3];
    ising.spmv(&x, &mut out);

    // Expected:
    // row 0: 0*0.8 + (-1)*(-0.4) + 2.5*1.2 = 0.4 + 3.0 = 3.4
    // row 1: (-1)*0.8 + 0*(-0.4) + (-0.5)*1.2 = -0.8 - 0.6 = -1.4
    // row 2: 2.5*0.8 + (-0.5)*(-0.4) + 0*1.2 = 2.0 + 0.2 = 2.2
    assert!((out[0] - 3.4).abs() < 1e-6);
    assert!((out[1] - (-1.4)).abs() < 1e-6);
    assert!((out[2] - 2.2).abs() < 1e-6);
}
