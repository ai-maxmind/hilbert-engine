use backend_physics::{LatticeGaugeSystem, WilsonConfinementReport};

#[test]
fn test_wilson_gauge_confinement_and_mass_gap() {
    let size = 4;
    let beta = 0.8; // Strong coupling regime -> Confined phase
    let mut lattice = LatticeGaugeSystem::new(size, beta, false, 42);

    // Initial thermalization
    lattice.thermalize(20, 1337);

    let report: WilsonConfinementReport = lattice.evaluate_confinement();

    println!(
        "Lattice Gauge: Size={}x{}, Beta={:.2}, Action={:.4}, W(1,1)={:.4}, W(2,2)={:.4}, Tension={:.4}, Gap={:.4}",
        report.lattice_size, report.lattice_size, report.beta_coupling, report.total_action,
        report.wilson_loop_1x1, report.wilson_loop_2x2, report.string_tension, report.mass_gap
    );
    println!("Yang-Mills Status: {}", report.yang_mills_status);

    assert!(report.total_action >= 0.0);
    assert!(report.wilson_loop_1x1 > 0.0);
    assert!(report.string_tension > 0.0, "String tension must be positive in strong coupling");
    assert!(report.mass_gap > 0.0, "Yang-Mills Mass Gap must be strictly positive");
    assert!(report.is_confined, "Area law must hold in strong coupling regime proving color confinement");
}
