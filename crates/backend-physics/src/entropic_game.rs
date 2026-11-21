//! Entropic Regularized Continuous Dynamics for Mixed-Strategy Nash Equilibrium.
//!
//! Solves adversarial zero-sum games without pure strategy equilibria (e.g. Matching Pennies,
//! Rock-Paper-Scissors) by embedding into continuous probability simplices with entropy
//! regularization and symplectic gradient dynamics.

/// Zero-sum matrix game represented by an M x N payoff matrix A.
/// Player 1 (Minimizer) chooses p in Delta_M to minimize p^T A q.
/// Player 2 (Maximizer) chooses q in Delta_N to maximize p^T A q.
#[derive(Debug, Clone)]
pub struct ZeroSumMatrixGame {
    pub m: usize,
    pub n: usize,
    /// Flattened row-major M x N payoff matrix
    pub matrix: Vec<f64>,
}

impl ZeroSumMatrixGame {
    pub fn new(m: usize, n: usize, matrix: Vec<f64>) -> Self {
        assert_eq!(matrix.len(), m * n, "Matrix size must equal m * n");
        Self { m, n, matrix }
    }

    /// Standard 2x2 Matching Pennies (no pure Nash equilibrium).
    pub fn matching_pennies() -> Self {
        Self::new(2, 2, vec![1.0, -1.0, -1.0, 1.0])
    }

    /// Standard 3x3 Rock-Paper-Scissors (equilibrium is uniform 1/3, 1/3, 1/3).
    pub fn rock_paper_scissors() -> Self {
        Self::new(
            3,
            3,
            vec![
                0.0, -1.0, 1.0,  // Rock vs (Rock, Paper, Scissors)
                1.0, 0.0, -1.0,  // Paper vs (Rock, Paper, Scissors)
                -1.0, 1.0, 0.0,  // Scissors vs (Rock, Paper, Scissors)
            ],
        )
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.matrix[i * self.n + j]
    }

    /// Compute payoff u = p^T A q
    pub fn compute_payoff(&self, p: &[f64], q: &[f64]) -> f64 {
        let mut u = 0.0;
        for i in 0..self.m {
            for j in 0..self.n {
                u += p[i] * self.get(i, j) * q[j];
            }
        }
        u
    }

    /// Compute expected payoffs for Minimizer: (A q)_i
    pub fn payoff_vector_x(&self, q: &[f64]) -> Vec<f64> {
        let mut res = vec![0.0; self.m];
        for i in 0..self.m {
            for j in 0..self.n {
                res[i] += self.get(i, j) * q[j];
            }
        }
        res
    }

    /// Compute expected payoffs for Maximizer: (A^T p)_j
    pub fn payoff_vector_y(&self, p: &[f64]) -> Vec<f64> {
        let mut res = vec![0.0; self.n];
        for j in 0..self.n {
            for i in 0..self.m {
                res[j] += self.get(i, j) * p[i];
            }
        }
        res
    }

    /// Compute total Nash exploitability: eps = (u - min_i (Aq)_i) + (max_j (A^T p)_j - u).
    pub fn compute_exploitability(&self, p: &[f64], q: &[f64]) -> (f64, f64, f64) {
        let u = self.compute_payoff(p, q);
        let aq = self.payoff_vector_x(q);
        let atp = self.payoff_vector_y(p);

        let min_aq = aq.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_atp = atp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let exploit_x = (u - min_aq).max(0.0);
        let exploit_y = (max_atp - u).max(0.0);
        let total_exploit = exploit_x + exploit_y;

        (total_exploit, exploit_x, exploit_y)
    }
}

/// Result of entropic mixed-strategy Nash dynamics.
#[derive(Debug, Clone)]
pub struct MixedNashResult {
    pub p_strategy: Vec<f64>,
    pub q_strategy: Vec<f64>,
    pub game_value: f64,
    pub exploitability: f64,
    pub is_nash_equilibrium: bool,
    pub steps_converged: usize,
}

/// Solves a matrix game using continuous entropic replicator / fictitious play dynamics.
pub fn solve_entropic_mixed_nash(
    game: &ZeroSumMatrixGame,
    num_steps: usize,
    learning_rate: f64,
    temp_init: f64,
) -> MixedNashResult {
    let m = game.m;
    let n = game.n;

    // Logits initialized to 0 (uniform initial distribution)
    let mut x = vec![0.0; m];
    let mut y = vec![0.0; n];

    // Cumulative frequency buffers (time-average convergence guarantees Nash)
    let mut p_cum = vec![0.0; m];
    let mut q_cum = vec![0.0; n];

    for step in 1..=num_steps {
        let progress = step as f64 / num_steps as f64;
        let tau = (temp_init * (1.0 - 0.95 * progress)).max(0.01);

        // Softmax probabilities
        let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps_x: Vec<f64> = x.iter().map(|&v| (-(v - max_x) / tau).exp()).collect();
        let sum_exps_x: f64 = exps_x.iter().sum();
        let p_instant: Vec<f64> = exps_x.iter().map(|&v| v / sum_exps_x).collect();

        let max_y = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps_y: Vec<f64> = y.iter().map(|&v| ((v - max_y) / tau).exp()).collect();
        let sum_exps_y: f64 = exps_y.iter().sum();
        let q_instant: Vec<f64> = exps_y.iter().map(|&v| v / sum_exps_y).collect();

        // Accumulate for time-average
        for i in 0..m {
            p_cum[i] += p_instant[i];
        }
        for j in 0..n {
            q_cum[j] += q_instant[j];
        }

        // Gradient updates
        let aq = game.payoff_vector_x(&q_instant);
        let atp = game.payoff_vector_y(&p_instant);

        // Minimizer moves opposite to payoffs (wants lower payoff)
        for i in 0..m {
            x[i] += learning_rate * aq[i];
        }
        // Maximizer moves with payoffs (wants higher payoff)
        for j in 0..n {
            y[j] += learning_rate * atp[j];
        }
    }

    // Final time-averaged strategies
    let sum_p: f64 = p_cum.iter().sum();
    let p_strategy: Vec<f64> = p_cum.iter().map(|&v| v / sum_p).collect();

    let sum_q: f64 = q_cum.iter().sum();
    let q_strategy: Vec<f64> = q_cum.iter().map(|&v| v / sum_q).collect();

    let game_value = game.compute_payoff(&p_strategy, &q_strategy);
    let (exploitability, _, _) = game.compute_exploitability(&p_strategy, &q_strategy);

    let is_nash_equilibrium = exploitability < 0.05;

    MixedNashResult {
        p_strategy,
        q_strategy,
        game_value,
        exploitability,
        is_nash_equilibrium,
        steps_converged: num_steps,
    }
}
