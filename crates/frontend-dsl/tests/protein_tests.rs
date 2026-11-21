//! Integration tests for 2D HP Protein Folding Lattice Model.

use frontend_dsl::{HpProteinModel, HpSequence};
use midend_decompose::decompose_to_ising;
use backend_physics::{solve_dsb_cpu, DsbConfig, SolverPhysicsMode};

#[test]
fn test_hp_sequence_parsing_and_model_build() {
    let seq = HpSequence::parse("HPPHPH").expect("Valid HP sequence");
    assert_eq!(seq.len(), 6);

    let model = HpProteinModel::new(seq, 4, 4);
    assert_eq!(model.total_variables(), 6 * 16); // 96 variables

    let builder = model.build_problem();
    let pubo = builder.build().expect("PUBO build success");
    assert_eq!(pubo.num_variables, 96);
}

#[test]
fn test_solve_hp_protein_folding_cpu_dsb() {
    // Sequence with potential H-H contact: H P P H
    // In 2D, monomer 0 (H) at (0,0), 1 (P) at (0,1), 2 (P) at (1,1), 3 (H) at (1,0)
    // Distance between 0 and 3 is Manhattan 1 -> valid H-H non-covalent contact!
    let seq = HpSequence::parse("HPPH").expect("Valid sequence");
    let model = HpProteinModel::new(seq, 3, 3);
    let builder = model.build_problem();
    let pubo = builder.build().expect("PUBO build");

    let n_vars = pubo.num_variables;
    let ising = decompose_to_ising(&pubo, n_vars).expect("decompose").ising_csr;
    let config = DsbConfig {
        num_steps: 600,
        dt: 0.4,
        a0: 1.0,
        c0: None,
        num_replicas: 64,
        seed: 42,
        mode: SolverPhysicsMode::Ballistic,
        ..Default::default()
    };

    let result = solve_dsb_cpu(&ising, &config).expect("Solve dSB");
    let binary_sol = result.best_spins.iter().map(|&s| if s > 0 { 1u8 } else { 0u8 }).collect::<Vec<_>>();
    let folded = model.decode_solution(&binary_sol);

    println!("\nFolded Protein Result:\n{}", folded);
    assert_eq!(folded.coords.len(), 4);
}
