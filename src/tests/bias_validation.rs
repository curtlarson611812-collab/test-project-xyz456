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
}