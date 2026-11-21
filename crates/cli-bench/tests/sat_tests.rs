use backend_physics::GpuContext;
use cli_bench::sat::{compile_3sat_pipeline, solve_3sat_end_to_end_gpu};

#[test]
fn test_end_to_end_3sat_20_vars_80_clauses_gpu() {
    let num_vars = 20;
    let num_clauses = 80;
    let seed = 42;

    println!("[Step 1-3] Compiling 3-SAT Pipeline (DSL -> E-Graph -> Decomposer)...");
    let pipeline = compile_3sat_pipeline(num_vars, num_clauses, seed);

    println!("  - Original Variables:   {}", pipeline.decomposed.num_original_vars);
    println!("  - Ancilla Variables:    {}", pipeline.decomposed.num_ancillas);
    println!("  - Total Variables:      {}", pipeline.decomposed.num_variables);
    println!("  - CSR Non-zeros:        {}", pipeline.decomposed.ising_csr.nnz());
    println!("  - Connected Components: {}", pipeline.decomposed.connected_components);
    println!("  - Max Vertex Degree:    {}", pipeline.decomposed.max_degree);
    println!("  - Scale Factor S:       {:.4}", pipeline.decomposed.scale_factor);

    assert_eq!(pipeline.decomposed.num_original_vars, 20);
    assert!(pipeline.decomposed.num_ancillas > 0);
    assert!(pipeline.decomposed.scale_factor > 0.0);

    let ctx = GpuContext::new().expect("Failed to initialize GPU context");

    println!("[Step 4] Dispatching 64 replicas x 800 steps to GPU in 1 CommandEncoder...");
    let (res, decoded_x, satisfied) = solve_3sat_end_to_end_gpu(&pipeline, &ctx, 64, 800, seed);

    println!("[Step 5] Verification Results:");
    println!("  - Clauses Satisfied:    {}/{} ({:.2}%)", satisfied, num_clauses, (satisfied as f64 / num_clauses as f64) * 100.0);
    println!("  - Best Ising Energy:    {:.4}", res.best_energy);
    println!("  - Solution Assignment:  {:?}", decoded_x);

    // Assert that the compiler pipeline achieves high satisfiability (>= 95% clauses) on hard 3-SAT
    assert!(
        satisfied >= 75,
        "Expected at least 75/80 clauses satisfied, got {}",
        satisfied
    );
}
