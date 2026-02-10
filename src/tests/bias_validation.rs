//! Bias validation tests for SmallOddPrime_Precise_code.rs integration
//!
//! Tests the sacred PRIME_MULTIPLIERS cycling and bias mod logic
//! Ensures no drift in starts per original code note

#[cfg(test)]
mod tests {
    use crate::SmallOddPrime_Precise_code as sop;

    #[test]
    fn test_prime_multipliers() {
        assert_eq!(sop::PRIME_MULTIPLIERS.len(), 32); // Sacred 32 length
        assert_eq!(sop::PRIME_MULTIPLIERS[0], 179); // First prime
        assert_eq!(sop::PRIME_MULTIPLIERS[31], 1583); // Last prime
        let hamming = sop::PRIME_MULTIPLIERS[0].count_ones();
        assert!(hamming <= 8, "Low Hamming weight for fast mul"); // Check low weight
    }

    #[test]
    fn test_get_biased_prime() {
        assert_eq!(sop::get_biased_prime(0, 81), 179); // Cycle 0
        assert_eq!(sop::get_biased_prime(32, 81), 179); // %32 cycle
        assert_eq!(sop::get_biased_prime(80, 81), sop::PRIME_MULTIPLIERS[80 % 81 % 32]); // Mod81
    }

    #[test]
    fn test_get_biased_prime_edge() {
        // Test large index cycles (>1000 iterations)
        for i in 0..1050 {
            let prime = sop::get_biased_prime(i, 81);
            assert!(prime >= 179 && prime <= 1583, "Prime {} out of range at index {}", prime, i);
            assert_eq!(prime, sop::PRIME_MULTIPLIERS[i % 32], "Cycle broken at index {}", i);
        }

        // Test zero bias_mod (should not divide by zero)
        let prime_zero = sop::get_biased_prime(10, 0);
        assert_eq!(prime_zero, sop::PRIME_MULTIPLIERS[10 % 32], "Zero bias_mod should use %32 only");

        // Test bias_mod = 1 (minimal cycle)
        let prime_one = sop::get_biased_prime(10, 1);
        assert_eq!(prime_one, sop::PRIME_MULTIPLIERS[10 % 32], "Bias_mod=1 should use %32 only");

        // Test maximum reasonable bias_mod
        let max_bias = 1000;
        for i in 0..32 {
            let prime = sop::get_biased_prime(i, max_bias);
            assert_eq!(prime, sop::PRIME_MULTIPLIERS[(i * 31) % 32], "Large bias_mod broken at index {}", i);
        }

        println!("Edge case testing passed ✓");
    }
}