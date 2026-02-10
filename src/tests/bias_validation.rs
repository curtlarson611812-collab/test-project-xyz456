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
        // Test basic cycling
        assert_eq!(sop::get_biased_prime(0, 81), 179);  // First prime
        assert_eq!(sop::get_biased_prime(1, 81), 257);  // Second prime
        assert_eq!(sop::get_biased_prime(31, 81), 1583);  // Last prime

        // Test cycling (32 % 81 % 32 = 0)
        assert_eq!(sop::get_biased_prime(32, 81), 179);

        // Test different bias mods
        assert_eq!(sop::get_biased_prime(0, 9), 179);   // mod9
        assert_eq!(sop::get_biased_prime(0, 27), 179);  // mod27
        assert_eq!(sop::get_biased_prime(0, 81), 179);  // mod81 (gold)

        // Test edge case: bias_mod = 1 (should cycle through all)
        assert_eq!(sop::get_biased_prime(0, 1), 179);
        assert_eq!(sop::get_biased_prime(1, 1), 257);
    }

    #[test]
    fn test_prime_properties() {
        // Verify low Hamming weight (fast GPU multiplication)
        for &prime in &sop::PRIME_MULTIPLIERS {
            let hamming = (prime as u64).count_ones();
            assert!(hamming <= 8, "Prime {} has high Hamming weight: {}", prime, hamming);
        }

        // Verify primes are distinct
        let mut seen = std::collections::HashSet::new();
        for &prime in &sop::PRIME_MULTIPLIERS {
            assert!(!seen.contains(&prime), "Duplicate prime: {}", prime);
            seen.insert(prime);
        }
    }
}