//! Dehn Invariant & Scissors Congruence Verifier for Hilbert's 3rd Problem
//!
//! Evaluates volume and Dehn invariant D(P) = sum l_i \otimes (\theta_i / \pi)
//! to mathematically prove whether two 3D polyhedra can be dissected into each other.

use std::f64::consts::PI;

/// Type of 3D polyhedra supported.
#[derive(Debug, Clone, PartialEq)]
pub enum PolyhedronKind {
    /// Cube of side length a.
    Cube { side: f64 },
    /// Regular tetrahedron of edge length a.
    RegularTetrahedron { edge: f64 },
    /// Rectangular prism of dimensions a x b x c.
    RectangularPrism { a: f64, b: f64, c: f64 },
}

/// Verification report for Scissors Congruence (Hilbert 3).
#[derive(Debug, Clone)]
pub struct ScissorsCongruenceReport {
    pub poly1_name: String,
    pub poly2_name: String,
    pub volume_1: f64,
    pub volume_2: f64,
    pub volumes_match: bool,
    pub dehn_1_rational: bool,
    pub dehn_2_rational: bool,
    pub dehn_invariants_match: bool,
    pub are_scissors_congruent: bool,
    pub mathematical_verdict: String,
}

impl PolyhedronKind {
    /// Computes the exact Euclidean volume of the polyhedron.
    pub fn volume(&self) -> f64 {
        match self {
            PolyhedronKind::Cube { side } => side.powi(3),
            PolyhedronKind::RegularTetrahedron { edge } => {
                // Vol = a^3 / (6 * sqrt(2))
                edge.powi(3) / (6.0 * std::f64::consts::SQRT_2)
            }
            PolyhedronKind::RectangularPrism { a, b, c } => a * b * c,
        }
    }

    /// Computes total edge length and dihedral angle in units of pi: \theta / pi.
    /// Also returns whether \theta / pi is a rational number.
    ///
    /// By Dehn's theorem:
    /// - For a cube / prism: all dihedral angles are pi/2, so theta/pi = 1/2 in Q.
    ///   Hence the Dehn invariant D(Cube) = 0 in R \otimes (R / pi Q).
    /// - For a regular tetrahedron: dihedral angle is arccos(1/3).
    ///   Dehn proved arccos(1/3) / pi is IRRATIONAL.
    ///   Hence D(Tetrahedron) != 0.
    pub fn dehn_components(&self) -> (f64, f64, bool) {
        match self {
            PolyhedronKind::Cube { side } => {
                let total_edge = 12.0 * side;
                let angle_over_pi = 0.5; // pi / 2
                (total_edge, angle_over_pi, true)
            }
            PolyhedronKind::RegularTetrahedron { edge } => {
                let total_edge = 6.0 * edge;
                let dihedral = (1.0f64 / 3.0f64).acos();
                let angle_over_pi = dihedral / PI; // ~0.391826... (irrational)
                (total_edge, angle_over_pi, false)
            }
            PolyhedronKind::RectangularPrism { a, b, c } => {
                let total_edge = 4.0 * (a + b + c);
                let angle_over_pi = 0.5;
                (total_edge, angle_over_pi, true)
            }
        }
    }

    /// Evaluates scissors congruence between two polyhedra.
    pub fn verify_congruence(p1: &PolyhedronKind, p2: &PolyhedronKind) -> ScissorsCongruenceReport {
        let vol1 = p1.volume();
        let vol2 = p2.volume();
        let volumes_match = (vol1 - vol2).abs() < 1e-6;

        let (_l1, _a1, rat1) = p1.dehn_components();
        let (_l2, _a2, rat2) = p2.dehn_components();

        // Dehn invariants match if both are rational (both D = 0 in R \otimes (R / pi Q))
        // or if both have identical irrational components and edge lengths.
        let dehn_match = if rat1 && rat2 {
            true
        } else if !rat1 && !rat2 {
            // Both irrational: compare edge length and dihedral
            (_l1 - _l2).abs() < 1e-6 && (_a1 - _a2).abs() < 1e-6
        } else {
            false
        };

        let are_scissors_congruent = volumes_match && dehn_match;

        let verdict = if are_scissors_congruent {
            "SCISSORS-CONGRUENT: Volumes match and Dehn invariants match (Dissection possible)".to_string()
        } else if volumes_match && !dehn_match {
            "NOT SCISSORS-CONGRUENT (Hilbert 3 Obstruction): Volumes are EQUAL, but Dehn invariants DIFFER (Dissection mathematically impossible)".to_string()
        } else {
            "NOT CONGRUENT: Volumes do not match".to_string()
        };

        ScissorsCongruenceReport {
            poly1_name: format!("{:?}", p1),
            poly2_name: format!("{:?}", p2),
            volume_1: vol1,
            volume_2: vol2,
            volumes_match,
            dehn_1_rational: rat1,
            dehn_2_rational: rat2,
            dehn_invariants_match: dehn_match,
            are_scissors_congruent,
            mathematical_verdict: verdict,
        }
    }
}
