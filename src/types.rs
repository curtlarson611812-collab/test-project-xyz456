//! Shared types and structures for SpeedBitCrack V3
//!
//! Contains structs/enums: DpEntry, KangarooState, AlphaBeta, Point coords, SearchMode enum

use serde::{Deserialize, Serialize};
use std::fmt;
use crate::math::bigint::BigInt256;

/// Rho algorithm state for GPU kernel execution (common definition)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct RhoState {
    pub current: Point,
    pub steps: BigInt256,
    pub is_dp: bool,
}

impl RhoState {
    /// Create random state within range
    pub fn random_in_range(_range: &(BigInt256, BigInt256)) -> Self {
        // TODO: Implement proper random generation
        Self {
            current: Point::infinity(),
            steps: BigInt256::zero(),
            is_dp: false,
        }
    }
}

impl Default for RhoState {
    fn default() -> Self {
        Self {
            current: Point::infinity(),
            steps: BigInt256::zero(),
            is_dp: false,
        }
    }
}

/// secp256k1 point representation (Jacobian coordinates)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Point {
    /// X coordinate (256-bit)
    pub x: [u64; 4],
    /// Y coordinate (256-bit)
    pub y: [u64; 4],
    /// Z coordinate (256-bit, for Jacobian coordinates)
    pub z: [u64; 4],
}

impl Point {
    /// Create point at infinity (neutral element)
    pub fn infinity() -> Self {
        Point {
            x: [0; 4],
            y: [0; 4],
            z: [0; 4],
        }
    }

    /// Create point from affine coordinates (x, y, z=1)
    pub fn from_affine(x: [u64; 4], y: [u64; 4]) -> Self {
        Point {
            x,
            y,
            z: [1, 0, 0, 0],
        }
    }

    /// Check if point is at infinity
    pub fn is_infinity(&self) -> bool {
        self.z == [0; 4]
    }

    /// Get x coordinate as BigInt256
    pub fn x_bigint(&self) -> BigInt256 {
        BigInt256 { limbs: self.x }
    }

    /// Get y coordinate as BigInt256
    pub fn y_bigint(&self) -> BigInt256 {
        BigInt256 { limbs: self.y }
    }

    /// Get low 32 bits of x coordinate
    pub fn x_low_u32(&self) -> u32 {
        self.x[0] as u32
    }

    /// Validate that point lies on the secp256k1 curve
    /// Checks if y² = x³ + ax + b mod p
    pub fn validate_curve(&self, curve: &super::math::secp::Secp256k1) -> bool {
        if self.is_infinity() {
            return true; // Point at infinity is valid
        }

        let x = super::math::bigint::BigInt256::from_u64_array(self.x);
        let y = super::math::bigint::BigInt256::from_u64_array(self.y);

        // Compute y²
        let y_squared = curve.montgomery_p.mul(&y, &y);

        // Compute x³ + ax + b
        let x_squared = curve.montgomery_p.mul(&x, &x);
        let x_cubed = curve.montgomery_p.mul(&x_squared, &x);
        let ax = curve.montgomery_p.mul(&curve.a, &x);
        let rhs = curve.barrett_p.add(&x_cubed, &ax);
        let rhs = curve.barrett_p.add(&rhs, &curve.b);

        // Check if y² ≡ x³ + ax + b mod p
        y_squared == rhs
    }

    /// Check if point is in the prime order subgroup
    /// Validates that n*P = infinity where n is the curve order
    pub fn validate_subgroup(&self, curve: &super::math::secp::Secp256k1) -> bool {
        if self.is_infinity() {
            return true;
        }

        // Compute n*P where n is the curve order
        let result = curve.mul(&curve.n, self);

        // Should be point at infinity
        result.is_infinity()
    }

    /// Comprehensive point validation
    /// Checks curve membership and subgroup membership
    pub fn validate(&self, curve: &super::math::secp::Secp256k1) -> Result<(), &'static str> {
        if !self.validate_curve(curve) {
            return Err("Point does not lie on the secp256k1 curve");
        }

        if !self.validate_subgroup(curve) {
            return Err("Point is not in the prime order subgroup (potential small subgroup attack)");
        }

        Ok(())
    }

    /// Convert Jacobian point to affine coordinates
    pub fn to_affine(&self, curve: &super::math::secp::Secp256k1) -> Point {
        if self.is_infinity() {
            return *self;
        }

        let z_inv = super::math::secp::Secp256k1::mod_inverse(&super::math::bigint::BigInt256::from_u64_array(self.z), &curve.p).unwrap();
        let z2 = curve.montgomery_p.mul(&z_inv, &z_inv);
        let z3 = curve.montgomery_p.mul(&z2, &z_inv);
        let x_aff = curve.barrett_p.mul(&super::math::bigint::BigInt256::from_u64_array(self.x), &z2);
        let y_aff = curve.barrett_p.mul(&super::math::bigint::BigInt256::from_u64_array(self.y), &z3);

        Point {
            x: x_aff.to_u64_array(),
            y: y_aff.to_u64_array(),
            z: [1, 0, 0, 0], // Z=1 for affine
        }
    }

    /// Scalar multiplication with k256::Scalar
    /// Converts scalar to our BigInt256 format and uses curve multiplication
    pub fn mul_scalar(&self, scalar: &k256::Scalar, curve: &super::math::secp::Secp256k1) -> Point {
        let scalar_bytes = scalar.to_bytes();
        let scalar_array: [u8; 32] = scalar_bytes.into();
        let scalar_bigint = super::math::bigint::BigInt256::from_bytes_be(&scalar_array);
        curve.mul(&scalar_bigint, self)
    }
}

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinity() {
            write!(f, "Infinity")
        } else {
            write!(f, "({:032x}, {:032x}, {:032x})",
                   u128::from(self.x[3]) << 96 | u128::from(self.x[2]) << 64 |
                   u128::from(self.x[1]) << 32 | u128::from(self.x[0]),
                   u128::from(self.y[3]) << 96 | u128::from(self.y[2]) << 64 |
                   u128::from(self.y[1]) << 32 | u128::from(self.y[0]),
                   u128::from(self.z[3]) << 96 | u128::from(self.z[2]) << 64 |
                   u128::from(self.z[1]) << 32 | u128::from(self.z[0]))
        }
    }
}

/// Kangaroo state for Pollard's rho method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KangarooState {
    /// Current position (point on curve)
    pub position: Point,
    /// Distance traveled (steps)
    pub distance: u64,
    /// Alpha coefficient for collision solving
    pub alpha: [u64; 4],
    /// Beta coefficient for collision solving
    pub beta: [u64; 4],
    /// Whether this is a tame kangaroo (starts from G) or wild (starts from target)
    pub is_tame: bool,
    /// Whether this kangaroo has reached a distinguished point
    pub is_dp: bool,
    /// Kangaroo ID for tracking
    pub id: u64,
}

/// Tagged kangaroo state for multi-target solving
#[derive(Debug, Clone)]
pub struct TaggedKangarooState {
    /// Current position (point on curve)
    pub point: Point,
    /// Distance traveled (BigInt256 for large ranges)
    pub distance: BigInt256,
    /// Target index this kangaroo is solving for
    pub target_idx: u32,
    /// Initial offset used for wild kangaroo generation (d in P - d*G)
    pub initial_offset: BigInt256,
}

impl KangarooState {
    /// Create new kangaroo state
    pub fn new(position: Point, distance: u64, alpha: [u64; 4], beta: [u64; 4], is_tame: bool, is_dp: bool, id: u64) -> Self {
        KangarooState {
            position,
            distance,
            alpha,
            beta,
            is_tame,
            is_dp,
            id,
        }
    }
}

/// Alpha and Beta coefficients for collision solving
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AlphaBeta {
    /// Alpha coefficient (256-bit)
    pub alpha: [u64; 4],
    /// Beta coefficient (256-bit)
    pub beta: [u64; 4],
}

impl AlphaBeta {
    /// Create zero coefficients
    pub fn zero() -> Self {
        AlphaBeta {
            alpha: [0; 4],
            beta: [0; 4],
        }
    }

    /// Check if coefficients are zero
    pub fn is_zero(&self) -> bool {
        self.alpha == [0; 4] && self.beta == [0; 4]
    }
}

/// Distinguished Point entry in DP table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpEntry {
    /// Point coordinates (x,y in affine)
    pub point: Point,
    /// Kangaroo state when DP was found
    pub state: KangarooState,
    /// Hash of x-coordinate for DP checking
    pub x_hash: u64,
    /// Timestamp when DP was added
    pub timestamp: u64,
    /// Cluster ID for grouping related points
    pub cluster_id: u32,
    /// Value score for pruning decisions
    pub value_score: f64,
}

impl DpEntry {
    /// Create new DP entry
    pub fn new(point: Point, state: KangarooState, x_hash: u64, cluster_id: u32) -> Self {
        DpEntry {
            point,
            state,
            x_hash,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cluster_id,
            value_score: 0.0,
        }
    }
}

/// Solution found by collision detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Recovered private key (256-bit)
    pub private_key: [u64; 4],
    /// Target point that was solved
    pub target_point: Point,
    /// Total operations performed
    pub total_ops: u64,
    /// Time taken (seconds)
    pub time_seconds: f64,
    /// Verification status
    pub verified: bool,
}

impl Solution {
    /// Create new solution
    pub fn new(private_key: [u64; 4], target_point: Point, total_ops: u64, time_seconds: f64) -> Self {
        Solution {
            private_key,
            target_point,
            total_ops,
            time_seconds,
            verified: false,
        }
    }

    /// Format private key as hex string
    pub fn private_key_hex(&self) -> String {
        format!("{:032x}{:032x}{:032x}{:032x}",
                self.private_key[3], self.private_key[2],
                self.private_key[1], self.private_key[0])
    }
}

/// Search mode enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    /// Full range search for P2PK/Magic 9 clusters
    FullRange,
    /// Interval search for specific puzzle ranges
    Interval { low: u64, high: u64 },
}

impl Default for SearchMode {
    fn default() -> Self {
        SearchMode::FullRange
    }
}

/// Jump operation for kangaroo stepping
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum JumpOp {
    /// Add G (generator point)
    AddG,
    /// Subtract G
    SubG,
    /// Add kG where k is target key
    AddKG,
    /// Subtract kG
    SubKG,
}

impl JumpOp {
    /// All 8 basic jump operations
    pub const ALL: [JumpOp; 8] = [
        JumpOp::AddG, JumpOp::SubG,
        JumpOp::AddKG, JumpOp::SubKG,
        // Negation variants for symmetry
        JumpOp::AddG, JumpOp::SubG,
        JumpOp::AddKG, JumpOp::SubKG,
    ];
}

/// Target information for solving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    /// Target point to solve for
    pub point: Point,
    /// Expected key range (for puzzles)
    pub key_range: Option<(u64, u64)>,
    /// Target ID for tracking
    pub id: u64,
    /// Priority score (higher = more valuable)
    pub priority: f64,
    /// Bitcoin address (if known)
    pub address: Option<String>,
    /// Value in BTC (if known)
    pub value_btc: Option<f64>,
}

impl Target {
    /// Create new target
    pub fn new(point: Point, id: u64) -> Self {
        Target {
            point,
            key_range: None,
            id,
            priority: 0.0,
            address: None,
            value_btc: None,
        }
    }
}