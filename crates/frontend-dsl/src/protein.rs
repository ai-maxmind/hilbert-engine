//! 2D HP Protein Folding Lattice Model.
//!
//! Models the hydrophobic-polar (HP) protein folding problem on a 2D square lattice,
//! mapping primary amino acid sequences into binary quadratic optimization (QUBO / Ising)
//! solvable on physics-inspired bifurcation backends.

use crate::builder::ProblemBuilder;
use std::fmt;

/// Amino acid type in Dill's HP model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AminoAcid {
    /// Hydrophobic (attracted to other H monomers, energy bonus for contact)
    H,
    /// Polar (hydrophilic, neutral)
    P,
}

impl AminoAcid {
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'H' => Some(AminoAcid::H),
            'P' => Some(AminoAcid::P),
            _ => None,
        }
    }
}

/// A primary amino acid sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HpSequence {
    pub monomers: Vec<AminoAcid>,
}

impl HpSequence {
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut monomers = Vec::new();
        for (idx, c) in s.chars().enumerate() {
            if c.is_whitespace() {
                continue;
            }
            let aa = AminoAcid::from_char(c)
                .ok_or_else(|| format!("Invalid amino acid '{}' at index {}", c, idx))?;
            monomers.push(aa);
        }
        if monomers.is_empty() {
            return Err("Sequence cannot be empty".to_string());
        }
        Ok(Self { monomers })
    }

    pub fn len(&self) -> usize {
        self.monomers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.monomers.is_empty()
    }
}

/// 2D folded monomer position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonomerCoord {
    pub index: usize,
    pub aa: AminoAcid,
    pub x: usize,
    pub y: usize,
}

/// Solved protein folding result.
#[derive(Debug, Clone)]
pub struct FoldedProtein {
    pub sequence: HpSequence,
    pub grid_w: usize,
    pub grid_h: usize,
    pub coords: Vec<MonomerCoord>,
    pub hh_contacts: usize,
    pub is_valid_walk: bool,
    pub is_self_avoiding: bool,
}

impl fmt::Display for FoldedProtein {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "FoldedProtein [L={}, H-H contacts={}, ValidWalk={}, SelfAvoiding={}]",
            self.sequence.len(),
            self.hh_contacts,
            self.is_valid_walk,
            self.is_self_avoiding
        )?;

        let min_x = self.coords.iter().map(|c| c.x).min().unwrap_or(0);
        let max_x = self.coords.iter().map(|c| c.x).max().unwrap_or(0);
        let min_y = self.coords.iter().map(|c| c.y).min().unwrap_or(0);
        let max_y = self.coords.iter().map(|c| c.y).max().unwrap_or(0);

        let w = max_x - min_x + 1;
        let h = max_y - min_y + 1;
        let mut grid = vec![vec!['.'; w]; h];

        for c in &self.coords {
            let gx = c.x - min_x;
            let gy = c.y - min_y;
            grid[gy][gx] = match c.aa {
                AminoAcid::H => 'H',
                AminoAcid::P => 'P',
            };
        }

        for row in grid.iter().rev() {
            let row_str: String = row.iter().collect();
            writeln!(f, "  {}", row_str)?;
        }
        Ok(())
    }
}

/// 2D Square Lattice HP Protein Model Synthesizer.
pub struct HpProteinModel {
    pub sequence: HpSequence,
    pub grid_w: usize,
    pub grid_h: usize,
    pub penalty_overlap: f64,
    pub penalty_connectivity: f64,
    pub contact_energy: f64,
}

impl HpProteinModel {
    pub fn new(sequence: HpSequence, grid_w: usize, grid_h: usize) -> Self {
        Self {
            sequence,
            grid_w,
            grid_h,
            penalty_overlap: 5.0,
            penalty_connectivity: 4.0,
            contact_energy: 1.0,
        }
    }

    /// Auto-size the grid based on sequence length.
    pub fn with_auto_grid(sequence: HpSequence) -> Self {
        let l = sequence.len();
        let side = (l as f64).sqrt().ceil() as usize + 2;
        let side = side.max(4);
        Self::new(sequence, side, side)
    }

    #[inline]
    pub fn var_index(&self, monomer: usize, gx: usize, gy: usize) -> usize {
        let grid_cells = self.grid_w * self.grid_h;
        monomer * grid_cells + gy * self.grid_w + gx
    }

    pub fn total_variables(&self) -> usize {
        self.sequence.len() * self.grid_w * self.grid_h
    }

    /// Build the QUBO/PUBO optimization problem.
    pub fn build_problem(&self) -> ProblemBuilder {
        let num_vars = self.total_variables();
        let mut builder = ProblemBuilder::new();
        builder.add_variables(num_vars);

        let l = self.sequence.len();
        let gw = self.grid_w;
        let gh = self.grid_h;
        let g_total = gw * gh;

        // 1. ExactOne constraint: Each monomer i must be placed at exactly one grid cell
        for i in 0..l {
            let mut monomer_vars = Vec::with_capacity(g_total);
            for gy in 0..gh {
                for gx in 0..gw {
                    monomer_vars.push(self.var_index(i, gx, gy));
                }
            }
            builder.add_exact_one(&monomer_vars, 6.0);
        }

        // 2. Overlap penalty: No two monomers i < j can occupy the same cell g
        for gy in 0..gh {
            for gx in 0..gw {
                for i in 0..l {
                    let vi = self.var_index(i, gx, gy);
                    for j in (i + 1)..l {
                        let vj = self.var_index(j, gx, gy);
                        builder.add_cost_quadratic(vi, vj, self.penalty_overlap);
                    }
                }
            }
        }

        // Precompute lattice neighbors for each cell (gx, gy)
        let mut neighbors = vec![Vec::new(); g_total];
        for gy in 0..gh {
            for gx in 0..gw {
                let g = gy * gw + gx;
                if gx + 1 < gw {
                    neighbors[g].push((gx + 1, gy));
                }
                if gx > 0 {
                    neighbors[g].push((gx - 1, gy));
                }
                if gy + 1 < gh {
                    neighbors[g].push((gx, gy + 1));
                }
                if gy > 0 {
                    neighbors[g].push((gx, gy - 1));
                }
            }
        }

        // 3. Chain Connectivity: Monomer i and i + 1 must be adjacent
        for i in 0..(l - 1) {
            builder.add_cost_linear(self.var_index(i, 0, 0), 0.0); // touch
            for gy in 0..gh {
                for gx in 0..gw {
                    let g = gy * gw + gx;
                    let vi = self.var_index(i, gx, gy);
                    for &(nx, ny) in &neighbors[g] {
                        let vi_next = self.var_index(i + 1, nx, ny);
                        builder.add_cost_quadratic(vi, vi_next, -self.penalty_connectivity);
                    }
                }
            }
        }

        // 4. Hydrophobic Contact Energy:
        for i in 0..l {
            if self.sequence.monomers[i] != AminoAcid::H {
                continue;
            }
            for j in (i + 2)..l {
                if self.sequence.monomers[j] != AminoAcid::H {
                    continue;
                }
                for gy in 0..gh {
                    for gx in 0..gw {
                        let g = gy * gw + gx;
                        let vi = self.var_index(i, gx, gy);
                        for &(nx, ny) in &neighbors[g] {
                            let vj = self.var_index(j, nx, ny);
                            builder.add_cost_quadratic(vi, vj, -self.contact_energy);
                        }
                    }
                }
            }
        }

        builder
    }

    /// Decode binary solution vector into folded 2D coordinates.
    pub fn decode_solution(&self, binary_solution: &[u8]) -> FoldedProtein {
        let l = self.sequence.len();
        let gw = self.grid_w;
        let gh = self.grid_h;
        let mut coords = Vec::with_capacity(l);

        for i in 0..l {
            let mut best_cell = (0, 0);
            let mut found = false;
            for gy in 0..gh {
                for gx in 0..gw {
                    let idx = self.var_index(i, gx, gy);
                    if idx < binary_solution.len() && binary_solution[idx] == 1 {
                        best_cell = (gx, gy);
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            coords.push(MonomerCoord {
                index: i,
                aa: self.sequence.monomers[i],
                x: best_cell.0,
                y: best_cell.1,
            });
        }

        let mut is_valid_walk = true;
        for i in 0..(l - 1) {
            let dx = (coords[i].x as isize - coords[i + 1].x as isize).abs();
            let dy = (coords[i].y as isize - coords[i + 1].y as isize).abs();
            if dx + dy != 1 {
                is_valid_walk = false;
                break;
            }
        }

        let mut is_self_avoiding = true;
        for i in 0..l {
            for j in (i + 1)..l {
                if coords[i].x == coords[j].x && coords[i].y == coords[j].y {
                    is_self_avoiding = false;
                    break;
                }
            }
            if !is_self_avoiding {
                break;
            }
        }

        let mut hh_contacts = 0;
        for i in 0..l {
            if coords[i].aa != AminoAcid::H {
                continue;
            }
            for j in (i + 2)..l {
                if coords[j].aa != AminoAcid::H {
                    continue;
                }
                let dx = (coords[i].x as isize - coords[j].x as isize).abs();
                let dy = (coords[i].y as isize - coords[j].y as isize).abs();
                if dx + dy == 1 {
                    hh_contacts += 1;
                }
            }
        }

        FoldedProtein {
            sequence: self.sequence.clone(),
            grid_w: gw,
            grid_h: gh,
            coords,
            hh_contacts,
            is_valid_walk,
            is_self_avoiding,
        }
    }
}
