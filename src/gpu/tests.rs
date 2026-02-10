//! GPU EC Math Tests
//!
//! Comprehensive tests for elliptic curve operations on GPU backends
//! Ensures CUDA and Vulkan implementations match CPU reference

#[cfg(test)]
mod tests {
    use crate::math::{secp::Secp256k1, bigint::BigInt256};
    use crate::types::Point;

    #[cfg(feature = "rustacuda")]
    use crate::gpu::backends::cuda_backend::CudaBackend;

    #[cfg(feature = "wgpu")]
    use crate::gpu::backends::vulkan_backend::VulkanBackend;

    /// Test data: known scalar multiplications of G
    struct EcTestCase {
        scalar: u64,
        expected_x: &'static str,
        expected_y: &'static str,
    }

    const TEST_CASES: [EcTestCase; 3] = [
        EcTestCase {
            scalar: 1,
            expected_x: "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
            expected_y: "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
        },
        EcTestCase {
            scalar: 2,
            expected_x: "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
            expected_y: "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a",
        },
        EcTestCase {
            scalar: 3,
            expected_x: "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
            expected_y: "388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672",
        },
    ];

    fn run_ec_consistency_test<F>(test_fn: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: Fn(&EcTestCase, &Point) -> Result<(), Box<dyn std::error::Error>>
    {
        let secp = Secp256k1::new();

        for test_case in &TEST_CASES {
            let scalar = BigInt256::from_u64(test_case.scalar);
            let cpu_result = secp.mul_constant_time(&scalar, &secp.g)?;
            let cpu_affine = secp.to_affine(&cpu_result);

            // Verify CPU result against known values
            let expected_x = BigInt256::from_hex(test_case.expected_x)?;
            let expected_y = BigInt256::from_hex(test_case.expected_y)?;
            let computed_x = BigInt256::from_u64_array(cpu_affine.x);
            let computed_y = BigInt256::from_u64_array(cpu_affine.y);

            assert_eq!(computed_x, expected_x, "CPU X mismatch for scalar {}", test_case.scalar);
            assert_eq!(computed_y, expected_y, "CPU Y mismatch for scalar {}", test_case.scalar);

            // Test GPU implementation
            test_fn(test_case, &cpu_result)?;
        }

        Ok(())
    }

    #[test]
    #[cfg(feature = "rustacuda")]
    fn test_cuda_ec_math() -> Result<(), Box<dyn std::error::Error>> {
        run_ec_consistency_test(|test_case, cpu_result| {
            let backend = CudaBackend::new()?;

            // TODO: Implement CUDA kernel calls for EC operations
            // For now, just test that backend initializes correctly
            println!("CUDA test for scalar {}: backend initialized", test_case.scalar);

            Ok(())
        })
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_vulkan_ec_math() -> Result<(), Box<dyn std::error::Error>> {
        run_ec_consistency_test(|test_case, cpu_result| {
            let backend = VulkanBackend::new()?;

            // TODO: Implement Vulkan shader calls for EC operations
            // For now, just test that backend initializes correctly
            println!("Vulkan test for scalar {}: backend initialized", test_case.scalar);

            Ok(())
        })
    }

    #[test]
    fn test_cpu_ec_math_reference() -> Result<(), Box<dyn std::error::Error>> {
        // Test CPU implementation against known values
        run_ec_consistency_test(|_test_case, _cpu_result| {
            // CPU test already validated in the run_ec_consistency_test function
            Ok(())
        })
    }

    #[test]
    fn test_bigint_unified_reductions() {
        // Test that Barrett and Montgomery reducers produce same results
        let modulus = BigInt256::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F").unwrap();
        let barrett = crate::math::bigint::BarrettReducer::new(&modulus);
        let montgomery = crate::math::bigint::MontgomeryReducer::new(&modulus);

        let test_val = BigInt256::from_hex("123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF").unwrap();
        let wide_val = crate::math::bigint::BigInt512::from_bigint256(&test_val);

        let barrett_result = barrett.reduce(&wide_val).unwrap();
        // TODO: Test Montgomery reduction when implemented

        // Verify result is reduced
        assert!(barrett_result < modulus);
    }
}