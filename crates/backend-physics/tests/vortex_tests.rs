//! Integration tests for Navier-Stokes Lagrangian Vortex Dynamics and BKM Regularity.

use backend_physics::VortexSystem;

#[test]
fn test_taylor_green_vortex_hamiltonian_conservation() {
    let mut system = VortexSystem::taylor_green_quadrupole(1.0, 2.0);
    let report = system.simulate();

    println!(
        "Vortex Simulation: Init H={:.4}, Final H={:.4}, Drift={:.6}, Min Dist={:.4}, Smooth={}",
        report.initial_hamiltonian, report.final_hamiltonian, report.hamiltonian_drift,
        report.min_distance, report.is_smooth
    );

    // Hamiltonian should be conserved within numerical tolerance
    assert!(report.hamiltonian_drift < 0.1, "Hamiltonian drift must be small");
    assert!(report.min_distance > 0.05, "Vortices must not collide into a singularity");
    assert!(report.is_smooth, "Fluid flow must satisfy Beale-Kato-Majda smoothness criterion");
    assert!(report.bkm_criterion_satisfied, "BKM criterion must be satisfied");
}
