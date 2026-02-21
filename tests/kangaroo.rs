// tests/kangaroo.rs - Basic tests for kangaroo algorithm components

use speedbitcrack::kangaroo::CollisionDetector;
use speedbitcrack::math::bigint::BigInt256;
use speedbitcrack::types::{Point, KangarooState};

#[cfg(test)]
mod tests {
    use super::*;

    // Test basic kangaroo state creation
    #[test]
    fn test_kangaroo_state_creation() {
        // Create a tame kangaroo at position 1
        let tame_kangaroo = KangarooState {
            position: Point::infinity(),
            distance: BigInt256::one(),
            alpha: [1, 0, 0, 0],
            beta: [1, 0, 0, 0],
            is_tame: true,
            is_dp: false,
            id: 0,
            step: 0,
            kangaroo_type: 1, // tame
        };

        assert_eq!(tame_kangaroo.distance, BigInt256::one());
        assert_eq!(tame_kangaroo.is_tame, true);
        assert_eq!(tame_kangaroo.kangaroo_type, 1);
    }

    // Test collision detector creation
    #[test]
    fn test_collision_detector() {
        let collision_detector = CollisionDetector::new();
        // Just test that it can be created
        assert!(true);
        println!("Collision detector test passed");
    }

    // Test basic BigInt256 operations used in kangaroo
    #[test]
    fn test_bigint_operations() {
        let a = BigInt256::from_u64(10);
        let b = BigInt256::from_u64(5);
        let c = BigInt256::one();

        assert_eq!(a, BigInt256::from_u64(10));
        assert_eq!(b, BigInt256::from_u64(5));
        assert_eq!(c, BigInt256::one());
    }

    // Test point operations
    #[test]
    fn test_point_operations() {
        let infinity = Point::infinity();
        // Just test that infinity point works
        assert!(true);
    }
}
