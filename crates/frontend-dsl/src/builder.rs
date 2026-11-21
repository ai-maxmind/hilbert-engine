//! Fluent API Builder for specifying combinatorial optimization problems.

use midend_egraph::pubo::{PuboError, PuboPolynomial};
use thiserror::Error;
use crate::constraints::{Clause3Sat, Constraint, Literal};

#[derive(Debug, Error)]
pub enum DslError {
    #[error("Pubo construction error: {0}")]
    Pubo(#[from] PuboError),

    #[error("Variable index {0} out of bounds (registered count: {1})")]
    InvalidVariable(usize, usize),

    #[error("Empty variable set provided for constraint")]
    EmptyConstraintVars,
}

/// Problem Builder for declaring variables, soft objectives, and hard logical constraints.
#[derive(Debug, Clone)]
pub struct ProblemBuilder {
    num_variables: usize,
    var_names: Vec<String>,
    objectives: Vec<(Vec<usize>, f64)>,
    constraints: Vec<Constraint>,
}

impl Default for ProblemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProblemBuilder {
    /// Initialize an empty problem builder.
    pub fn new() -> Self {
        Self {
            num_variables: 0,
            var_names: Vec::new(),
            objectives: Vec::new(),
            constraints: Vec::new(),
        }
    }

    /// Register a new binary variable $x_i \in \{0, 1\}$ with an optional identifier name.
    pub fn add_variable(&mut self, name: Option<&str>) -> usize {
        let idx = self.num_variables;
        let vname = name.map(|s| s.to_string()).unwrap_or_else(|| format!("x_{idx}"));
        self.var_names.push(vname);
        self.num_variables += 1;
        idx
    }

    /// Register $N$ binary variables simultaneously.
    pub fn add_variables(&mut self, count: usize) -> Vec<usize> {
        (0..count).map(|_| self.add_variable(None)).collect()
    }

    /// Add a linear cost term $c \cdot x_i$ to the objective.
    pub fn add_cost_linear(&mut self, var: usize, weight: f64) -> &mut Self {
        self.objectives.push((vec![var], weight));
        self
    }

    /// Add a quadratic interaction cost $c \cdot x_u x_v$ to the objective.
    pub fn add_cost_quadratic(&mut self, u: usize, v: usize, weight: f64) -> &mut Self {
        self.objectives.push((vec![u, v], weight));
        self
    }

    /// Add a higher-order polynomial monomial $c \cdot \prod_{i \in \text{vars}} x_i$.
    pub fn add_cost_polynomial(&mut self, vars: &[usize], weight: f64) -> &mut Self {
        self.objectives.push((vars.to_vec(), weight));
        self
    }

    /// Add an `ExactOne` hard constraint: $\sum_{i \in \text{vars}} x_i = 1$.
    pub fn add_exact_one(&mut self, vars: &[usize], penalty_weight: f64) -> &mut Self {
        self.constraints.push(Constraint::ExactOne {
            vars: vars.to_vec(),
            penalty_weight,
        });
        self
    }

    /// Add an `And` gate constraint: $\text{out} = x \land y$.
    pub fn add_and(&mut self, x: usize, y: usize, out: usize, penalty_weight: f64) -> &mut Self {
        self.constraints.push(Constraint::And {
            x,
            y,
            out,
            penalty_weight,
        });
        self
    }

    /// Add an `Or` clause constraint: $\bigvee_{i} x_i = 1$.
    pub fn add_or(&mut self, vars: &[usize], penalty_weight: f64) -> &mut Self {
        self.constraints.push(Constraint::Or {
            vars: vars.to_vec(),
            penalty_weight,
        });
        self
    }

    /// Add an `Xor` gate constraint: $\text{out} = x \oplus y$.
    pub fn add_xor(&mut self, x: usize, y: usize, out: usize, penalty_weight: f64) -> &mut Self {
        self.constraints.push(Constraint::Xor {
            x,
            y,
            out,
            penalty_weight,
        });
        self
    }

    /// Add a 3-SAT clause: $(\ell_1 \lor \ell_2 \lor \ell_3) = 1$.
    pub fn add_clause_3sat(
        &mut self,
        l1: (usize, bool),
        l2: (usize, bool),
        l3: (usize, bool),
        penalty_weight: f64,
    ) -> &mut Self {
        self.constraints.push(Constraint::Clause3Sat {
            clause: Clause3Sat {
                l1: Literal(l1.0, l1.1),
                l2: Literal(l2.0, l2.1),
                l3: Literal(l3.0, l3.1),
            },
            penalty_weight,
        });
        self
    }

    /// Compile the declared variables, objectives, and constraints into a canonical `PuboPolynomial`.
    pub fn build(self) -> Result<PuboPolynomial, DslError> {
        let mut pubo = PuboPolynomial::new(self.num_variables);

        // 1. Add objective terms
        for (vars, weight) in self.objectives {
            pubo.add_term(&vars, weight)?;
        }

        // 2. Add constraint penalty expansions
        for constraint in self.constraints {
            constraint.apply_to_pubo(&mut pubo)?;
        }

        Ok(pubo)
    }

    /// Current number of registered variables.
    pub fn num_variables(&self) -> usize {
        self.num_variables
    }
}
