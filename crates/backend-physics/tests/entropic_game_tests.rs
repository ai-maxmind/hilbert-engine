//! Integration tests for Entropic Mixed-Strategy Nash Solver.

use backend_physics::entropic_game::{solve_entropic_mixed_nash, ZeroSumMatrixGame};

#[test]
fn test_matching_pennies_mixed_nash_convergence() {
    let game = ZeroSumMatrixGame::matching_pennies();
    let result = solve_entropic_mixed_nash(&game, 1500, 0.05, 1.0);

    println!(
        "Matching Pennies Solved: p={:?}, q={:?}, value={:.4}, exploitability={:.4}",
        result.p_strategy, result.q_strategy, result.game_value, result.exploitability
    );

    // Theory: Mixed Nash for Matching Pennies is p = [0.5, 0.5], q = [0.5, 0.5], value = 0.0
    assert!((result.p_strategy[0] - 0.5).abs() < 0.1, "p0 ~ 0.5");
    assert!((result.p_strategy[1] - 0.5).abs() < 0.1, "p1 ~ 0.5");
    assert!((result.q_strategy[0] - 0.5).abs() < 0.1, "q0 ~ 0.5");
    assert!((result.q_strategy[1] - 0.5).abs() < 0.1, "q1 ~ 0.5");
    assert!(result.exploitability < 0.1, "Exploitability must be near zero");
}

#[test]
fn test_rock_paper_scissors_mixed_nash_convergence() {
    let game = ZeroSumMatrixGame::rock_paper_scissors();
    let result = solve_entropic_mixed_nash(&game, 2000, 0.05, 1.0);

    println!(
        "Rock-Paper-Scissors Solved: p={:?}, q={:?}, value={:.4}, exploitability={:.4}",
        result.p_strategy, result.q_strategy, result.game_value, result.exploitability
    );

    // Theory: Mixed Nash is uniform (1/3, 1/3, 1/3) ~ 0.3333
    for &pi in &result.p_strategy {
        assert!((pi - 0.3333).abs() < 0.1, "Player 1 should play ~1/3 for each action");
    }
    for &qj in &result.q_strategy {
        assert!((qj - 0.3333).abs() < 0.1, "Player 2 should play ~1/3 for each action");
    }
    assert!(result.game_value.abs() < 0.1, "Fair zero-sum game value ~ 0");
    assert!(result.exploitability < 0.1, "Nash exploitability must be small");
}
