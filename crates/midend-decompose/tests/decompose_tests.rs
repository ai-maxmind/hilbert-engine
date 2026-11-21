use midend_decompose::decompose_to_ising;
use midend_egraph::pubo::PuboPolynomial;

#[test]
fn test_qubo_to_ising_energy_exact_identity() {
    // Construct a random 4-variable QUBO
    let mut qubo = PuboPolynomial::new(4);
    qubo.add_term(&[], 10.0).unwrap();
    qubo.add_term(&[0], -3.0).unwrap();
    qubo.add_term(&[1], 4.5).unwrap();
    qubo.add_term(&[0, 1], 2.0).unwrap();
    qubo.add_term(&[1, 2], -6.0).unwrap();
    qubo.add_term(&[2, 3], 5.0).unwrap();
    qubo.add_term(&[0, 3], -1.5).unwrap();

    let decomposed = decompose_to_ising(&qubo, 4).expect("Decomposition must succeed");

    // Dynamic range scaling must be positive
    assert!(decomposed.scale_factor > 0.0);

    // Verify all 2^4 = 16 states have exact energy equivalence
    for state in 0..16 {
        let mut binary = vec![0u8; 4];
        let mut spins = vec![1i8; 4];
        for i in 0..4 {
            let bit = ((state >> i) & 1) as u8;
            binary[i] = bit;
            spins[i] = if bit == 1 { -1i8 } else { 1i8 }; // s_i = 1 - 2*x_i
        }

        let qubo_energy = qubo.evaluate(&binary).unwrap();
        let (decoded_binary, decoded_energy) = decomposed.decode_spins(&spins).unwrap();

        assert_eq!(decoded_binary, binary);
        let diff = (qubo_energy - decoded_energy).abs();
        assert!(
            diff < 1e-5,
            "Energy mismatch at state {state:04b}: QUBO={}, Ising Decoded={}, diff={diff}",
            qubo_energy,
            decoded_energy
        );
    }
}

#[test]
fn test_petgraph_topology_metrics() {
    let mut qubo = PuboPolynomial::new(5);
    // Component 1: Triangle (0, 1, 2)
    qubo.add_term(&[0, 1], 1.0).unwrap();
    qubo.add_term(&[1, 2], 1.0).unwrap();
    qubo.add_term(&[0, 2], 1.0).unwrap();

    // Component 2: Edge (3, 4)
    qubo.add_term(&[3, 4], 2.0).unwrap();

    let decomposed = decompose_to_ising(&qubo, 5).unwrap();

    // Interaction graph should have exactly 2 connected components
    assert_eq!(decomposed.connected_components, 2);
    assert_eq!(decomposed.num_edges, 4);
    assert_eq!(decomposed.num_variables, 5);
    assert_eq!(decomposed.max_degree, 2); // Node 0,1,2 have degree 2
}
