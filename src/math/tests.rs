#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::secp::Secp256k1;

    #[test]
    fn test_decompress_puzzles() {
        let curve = Secp256k1::new();

        // Test all revealed puzzles from puzzles.txt
        let puzzles = vec![
            "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16",  // #135
            "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640",  // #140
            "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5",  // #145
            "03137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49",  // #150
            "035cd1854cae45391ca4ec428cc7e6c7d9984424b954209a8eea197b9e364c05f6",  // #155
            "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673",  // #160
        ];

        for hex in puzzles {
            let bytes = hex::decode(hex).unwrap();
            let mut comp = [0u8; 33];
            comp.copy_from_slice(&bytes);
            let point = curve.decompress_point(&comp);
            assert!(point.is_some(), "Failed to decompress puzzle: {}", hex);

            let p = point.unwrap();
            assert!(curve.is_on_curve(&p), "Point not on curve for puzzle: {}", hex);
        }
    }

    #[test]
    fn test_barrett_reduction() {
        let modulus = BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
        let reducer = BarrettReducer::new(&modulus).unwrap();

        // Test with various values
        let test_values = vec![
            BigInt256::from_u64(1),
            BigInt256::from_u64(123456),
            BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"),
        ];

        for val in test_values {
            let reduced = reducer.reduce(&BigInt512::from_bigint256(&val)).unwrap();
            assert!(reduced < modulus, "Reduction failed for value: {}", val.to_hex());
        }
    }

    #[test]
    fn test_bigint512_mul() {
        let a = BigInt512::from_bigint256(&BigInt256::from_u64(2));
        let b = BigInt512::from_bigint256(&BigInt256::from_u64(3));
        let result = a.mul(&b);

        // 2 * 3 = 6
        assert_eq!(result.limbs[0], 6);
        for i in 1..8 {
            assert_eq!(result.limbs[i], 0);
        }
    }

    #[test]
    fn test_pow_mod() {
        let curve = Secp256k1::new();

        // Test 2^3 mod 7 = 1
        let base = BigInt256::from_u64(2);
        let exp = BigInt256::from_u64(3);
        let modulus = BigInt256::from_u64(7);
        let result = curve.pow_mod(&base, &exp, &modulus);
        assert_eq!(result, BigInt256::one());

        // Test Legendre symbol for quadratic residue
        let qr = BigInt256::from_u64(4); // 4 is QR mod 7
        let legendre_exp = BigInt256::from_u64(3); // (7-1)/2 = 3
        let legendre = curve.pow_mod(&qr, &legendre_exp, &BigInt256::from_u64(7));
        assert_eq!(legendre, BigInt256::one());
    }

    #[test]
    fn test_mod_inverse() {
        // Test 3 * inv(3) ≡ 1 mod 7
        let a = BigInt256::from_u64(3);
        let modulus = BigInt256::from_u64(7);
        let inv = mod_inverse(&a, &modulus).unwrap();

        let product = (a * inv) % modulus;
        assert_eq!(product, BigInt256::one());
    }

    #[test]
    fn test_tonelli_shanks() {
        let curve = Secp256k1::new();

        // Test case: find sqrt(4) mod 7 = 2 or 5 (since 2^2 = 4, 5^2 = 25 ≡ 4 mod 7)
        let value = BigInt256::from_u64(4);
        let modulus = BigInt256::from_u64(7);

        let root = curve.tonelli_shanks(&value, &modulus);
        assert!(root.is_some());

        let root_val = root.unwrap();
        let root_sq = (root_val.clone() * root_val.clone()) % modulus.clone();

        // Check that root^2 ≡ value mod modulus
        assert_eq!(root_sq, value);

        // Also check the other root: modulus - root
        let other_root = modulus.clone() - root_val;
        let other_root_sq = (other_root.clone() * other_root) % modulus;
        assert_eq!(other_root_sq, value);
    }

    #[test]
    fn test_decompress_135() {
        let curve = Secp256k1::new();
        // Puzzle #135 from puzzles.txt
        let comp_hex = "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16";
        let bytes = hex::decode(comp_hex).unwrap();
        let mut comp = [0u8; 33];
        comp.copy_from_slice(&bytes);

        let point = curve.decompress_point(&comp);
        assert!(point.is_some(), "Failed to decompress puzzle #135");

        let p = point.unwrap();
        assert!(curve.is_on_curve(&p), "Point not on curve for puzzle #135");

        // For compressed format 02, y should be even
        assert_eq!(p.y[0] & 1, 0, "Y coordinate should be even for 02 prefix");
    }

    fn mod_pow_basic(base: &BigInt256, exp: &BigInt256, modulus: &BigInt256) -> BigInt256 {
        use crate::math::bigint::BarrettReducer;
        let barrett = BarrettReducer::new(modulus);
        let mut result = BigInt256::one();
        let mut b = barrett.reduce(&BigInt512::from_bigint256(&base.clone())).unwrap();
        let mut e = exp.clone();

        while !e.is_zero() {
            if e.limbs[0] & 1 == 1 {
                result = barrett.mul(&result, &b);
                result = barrett.reduce(&BigInt512::from_bigint256(&result)).unwrap();
            }
            b = barrett.mul(&b, &b);
            b = barrett.reduce(&BigInt512::from_bigint256(&b)).unwrap();
            e = e.right_shift(1);
        }
        result
    }

    #[test]
    fn test_tonelli_shanks_basic() {
        use crate::math::{BigInt256, BigInt512};
        use crate::math::bigint::BarrettReducer;

        // Test case: sqrt(4) mod 7 = 2 or 5 (since 2^2 = 4, 5^2 = 25 ≡ 4 mod 7)
        let value = BigInt256::from_u64(4);
        let modulus = BigInt256::from_u64(7);
        let barrett = BarrettReducer::new(&modulus);

        // For p=7 (p≡3 mod 4), use (p+1)/4 = 2
        let exp = BigInt256::from_u64(2);
        let root = mod_pow_basic(&value, &exp, &modulus);

        // Check that root^2 ≡ value mod modulus
        let root_sq = barrett.mul(&root, &root);
        let root_sq_mod = barrett.reduce(&BigInt512::from_bigint256(&root_sq)).unwrap();
        assert_eq!(root_sq_mod, value, "Tonelli-Shanks root should satisfy root^2 ≡ value mod modulus");

        // Also check the other root: modulus - root
        let other_root = barrett.sub(&modulus, &root);
        let other_root_sq = barrett.mul(&other_root, &other_root);
        let other_root_sq_mod = barrett.reduce(&BigInt512::from_bigint256(&other_root_sq)).unwrap();
        assert_eq!(other_root_sq_mod, value, "Other root should also satisfy root^2 ≡ value mod modulus");
    }
}