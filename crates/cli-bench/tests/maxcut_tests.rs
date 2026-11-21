use backend_physics::{solve_dsb_cpu, DsbConfig};
use cli_bench::maxcut::MaxCutInstance;

#[test]
fn test_maxcut_100_dst_bit_for_bit() {
    let seed = 0x1234_5678_9ABC;
    let instance = MaxCutInstance::generate_erdos_renyi(100, 0.5, seed).unwrap();

    let config = DsbConfig {
        num_steps: 600,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 2,
        seed: 42,
        ..Default::default()
    };

    let run1 = solve_dsb_cpu(&instance.ising, &config).unwrap();
    let run2 = solve_dsb_cpu(&instance.ising, &config).unwrap();

    // 100% bit-for-bit assertion
    assert_eq!(run1.best_energy.to_bits(), run2.best_energy.to_bits());
    assert_eq!(run1.c0.to_bits(), run2.c0.to_bits());
    assert_eq!(run1.best_spins, run2.best_spins);
    assert_eq!(run1.replica_energies, run2.replica_energies);
    assert_eq!(run1.replica_spins, run2.replica_spins);

    // Verify mathematical relation between Cut and Ising Energy
    let cut_direct = instance.calculate_cut_direct(&run1.best_spins);
    let cut_energy = instance.calculate_cut_from_energy(run1.best_energy);
    assert_eq!(cut_direct as f64, cut_energy);
}

#[test]
fn test_maxcut_500_dst_bit_for_bit() {
    let seed = 0xFEED_FACE_CAFE;
    let instance = MaxCutInstance::generate_erdos_renyi(500, 0.1, seed).unwrap();

    let config = DsbConfig {
        num_steps: 400,
        dt: 0.5,
        a0: 1.0,
        c0: None,
        num_replicas: 1,
        seed: 9999,
        ..Default::default()
    };

    let run1 = solve_dsb_cpu(&instance.ising, &config).unwrap();
    let run2 = solve_dsb_cpu(&instance.ising, &config).unwrap();

    // 100% bit-for-bit assertion
    assert_eq!(run1.best_energy.to_bits(), run2.best_energy.to_bits());
    assert_eq!(run1.c0.to_bits(), run2.c0.to_bits());
    assert_eq!(run1.best_spins, run2.best_spins);

    let cut_direct = instance.calculate_cut_direct(&run1.best_spins);
    let cut_energy = instance.calculate_cut_from_energy(run1.best_energy);
    assert_eq!(cut_direct as f64, cut_energy);
}
