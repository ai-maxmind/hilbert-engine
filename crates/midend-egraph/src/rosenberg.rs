//! Module `rosenberg` implements polynomial quadratization (PUBO -> QUBO)
//! using Rosenberg's reduction with global ancilla variable reuse.
//!
//! # Mathematical Foundations
//!
//! Given a higher-order binary polynomial $P(\mathbf{x})$ with terms of degree $\ge 3$:
//! Rosenberg (1975) showed that any product $x_i x_j$ can be replaced by an auxiliary
//! binary variable $w \in \{0, 1\}$ subject to the quadratic penalty function:
//!
//! $$P(x_i, x_j, w) = x_i x_j - 2 x_i w - 2 x_j w + 3 w$$
//!
//! ## Truth Table of $P(x_i, x_j, w)$:
//!
//! | $x_i$ | $x_j$ | $w$ | $P(x_i, x_j, w)$ | Status |
//! | :---: | :---: | :-: | :--------------: | :----- |
//! | 0 | 0 | 0 | **0** | $w = x_i x_j$ (Valid) |
//! | 0 | 0 | 1 | 3 | $w \ne x_i x_j$ (Penalized +3) |
//! | 1 | 0 | 0 | **0** | $w = x_i x_j$ (Valid) |
//! | 1 | 0 | 1 | 1 | $w \ne x_i x_j$ (Penalized +1) |
//! | 0 | 1 | 0 | **0** | $w = x_i x_j$ (Valid) |
//! | 0 | 1 | 1 | 1 | $w \ne x_i x_j$ (Penalized +1) |
//! | 1 | 1 | 0 | 1 | $w \ne x_i x_j$ (Penalized +1) |
//! | 1 | 1 | 1 | **0** | $w = x_i x_j$ (Valid) |
//!
//! For any configuration, $\min_{w \in \{0, 1\}} P(x_i, x_j, w) = 0$, achieved
//! if and only if $w = x_i x_j$.
//!
//! ## Penalty Weight $M$:
//!
//! Let $\mathcal{T}$ be the set of monomials where $x_i x_j$ is substituted by $w$.
//! Choosing:
//!
//! $$M \ge \sum_{t \in \mathcal{T}} |c_t| + 1.0$$
//!
//! guarantees that any violation $w \ne x_i x_j$ strictly raises the objective
//! value above the true minimum, rigorously preserving the global ground state:
//!
//! $$\min_{\mathbf{x}, \mathbf{w}} Q(\mathbf{x}, \mathbf{w}) = \min_{\mathbf{x}} P(\mathbf{x})$$

use std::collections::{BTreeMap, HashMap};
use crate::pubo::{PuboError, PuboPolynomial};

/// Outcome of Rosenberg polynomial quadratization.
#[derive(Debug, Clone)]
pub struct QuadratizationResult {
    /// The resulting quadratic polynomial with degree <= 2.
    pub qubo: PuboPolynomial,
    /// Number of original problem variables.
    pub original_variables: usize,
    /// Number of auxiliary ancilla variables introduced.
    pub num_ancillas: usize,
    /// Mapping of (variable_u, variable_v) -> assigned ancilla variable index.
    pub pair_to_ancilla: HashMap<(usize, usize), usize>,
    /// Penalty multipliers $M$ applied per ancilla variable.
    pub penalty_multipliers: Vec<f64>,
}

/// Perform Rosenberg reduction with greedy maximum ancilla reuse.
///
/// Converts any PUBO polynomial with terms up to degree 4 down to a strictly
/// quadratic polynomial (QUBO) of degree $\le 2$.
pub fn quadratize_rosenberg(pubo: &PuboPolynomial) -> Result<QuadratizationResult, PuboError> {
    let original_n = pubo.num_variables;
    let mut current_n = original_n;

    // Working copy of polynomial terms
    let mut current_terms: BTreeMap<Vec<usize>, f64> = pubo.terms.clone();

    let mut pair_to_ancilla: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ancilla_to_pair: Vec<(usize, usize)> = Vec::new();
    let mut ancilla_coefficients: Vec<f64> = Vec::new();

    // Loop until all terms have degree <= 2
    loop {
        // 1. Identify all terms of degree >= 3
        let higher_terms: Vec<(Vec<usize>, f64)> = current_terms
            .iter()
            .filter(|(vars, &coeff)| vars.len() >= 3 && coeff.abs() > 1e-9)
            .map(|(vars, &coeff)| (vars.clone(), coeff))
            .collect();

        if higher_terms.is_empty() {
            break;
        }

        // 2. Count pair frequencies across all higher-order terms
        let mut pair_freq: HashMap<(usize, usize), (usize, f64)> = HashMap::new();
        for (vars, coeff) in &higher_terms {
            let len = vars.len();
            for i in 0..len {
                for j in (i + 1)..len {
                    let u = vars[i];
                    let v = vars[j];
                    let pair = if u < v { (u, v) } else { (v, u) };
                    let entry = pair_freq.entry(pair).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += coeff.abs();
                }
            }
        }

        // 3. Greedy selection: Pick the pair with the highest occurrence frequency
        // (to maximize ancilla reuse across multiple higher-order monomials)
        let (best_pair, (_, coeff_sum)) = pair_freq
            .into_iter()
            .max_by(|a, b| {
                // Prioritize frequency of reuse, then total coefficient magnitude
                a.1 .0.cmp(&b.1 .0).then_with(|| {
                    a.1 .1
                        .partial_cmp(&b.1 .1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            })
            .expect("Higher order terms exist");

        // 4. Allocate or reuse ancilla variable w
        let (ancilla_idx, _is_new) = if let Some(&existing_idx) = pair_to_ancilla.get(&best_pair) {
            (existing_idx, false)
        } else {
            let new_idx = current_n;
            current_n += 1;
            pair_to_ancilla.insert(best_pair, new_idx);
            ancilla_to_pair.push(best_pair);
            ancilla_coefficients.push(0.0);
            (new_idx, true)
        };

        let ancilla_sub_idx = ancilla_idx - original_n;
        ancilla_coefficients[ancilla_sub_idx] += coeff_sum;

        // 5. Replace pair in all higher-order terms containing best_pair
        let mut new_terms = BTreeMap::new();
        for (vars, coeff) in current_terms {
            if coeff.abs() <= 1e-12 {
                continue;
            }
            if vars.len() >= 3 && vars.contains(&best_pair.0) && vars.contains(&best_pair.1) {
                // Remove u and v, insert w
                let mut substituted_vars: Vec<usize> = vars
                    .into_iter()
                    .filter(|&x| x != best_pair.0 && x != best_pair.1)
                    .collect();
                substituted_vars.push(ancilla_idx);
                substituted_vars.sort_unstable();
                substituted_vars.dedup(); // x_i^2 = x_i

                let entry = new_terms.entry(substituted_vars).or_insert(0.0);
                *entry += coeff;
            } else {
                let entry = new_terms.entry(vars).or_insert(0.0);
                *entry += coeff;
            }
        }
        current_terms = new_terms;
    }

    // 6. Add Rosenberg penalties for all ancilla variables:
    // P(u, v, w) = M * (u*v - 2*u*w - 2*v*w + 3*w)
    let num_ancillas = current_n - original_n;
    let mut penalty_multipliers = Vec::with_capacity(num_ancillas);

    for (ancilla_sub, &pair) in ancilla_to_pair.iter().enumerate() {
        let w = original_n + ancilla_sub;
        let u = pair.0;
        let v = pair.1;

        // M >= sum(|coeffs|) + 1.0
        let m = ancilla_coefficients[ancilla_sub] + 2.0;
        penalty_multipliers.push(m);

        // Term 1: +M * u * v
        let uv_key = if u < v { vec![u, v] } else { vec![v, u] };
        *current_terms.entry(uv_key).or_insert(0.0) += m;

        // Term 2: -2M * u * w
        let uw_key = if u < w { vec![u, w] } else { vec![w, u] };
        *current_terms.entry(uw_key).or_insert(0.0) -= 2.0 * m;

        // Term 3: -2M * v * w
        let vw_key = if v < w { vec![v, w] } else { vec![w, v] };
        *current_terms.entry(vw_key).or_insert(0.0) -= 2.0 * m;

        // Term 4: +3M * w
        *current_terms.entry(vec![w]).or_insert(0.0) += 3.0 * m;
    }

    // 7. Assemble final verified QUBO polynomial
    let mut qubo = PuboPolynomial::new(current_n);
    for (vars, coeff) in current_terms {
        if coeff.abs() > 1e-12 {
            qubo.add_term(&vars, coeff)?;
        }
    }

    debug_assert!(
        qubo.is_quadratic(),
        "Internal Compiler Error: Polynomial not quadratic after Rosenberg reduction!"
    );

    Ok(QuadratizationResult {
        qubo,
        original_variables: original_n,
        num_ancillas,
        pair_to_ancilla,
        penalty_multipliers,
    })
}
