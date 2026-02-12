// Concise Block: CUDA Prime Mul Kernel for Test
// In src/gpu/cuda/kernels.cu — batch mul primes * target, check on-curve.
// Use our montgomery_point_mul (ladder for const-time).
#include <stdint.h>
__global__ void test_prime_mul(Point* outputs, const Point target, const uint64_t primes[32], uint64_t modulus[4]) {
    int idx = threadIdx.x;
    if (idx >= 32) return;

    uint64_t prime_limbs[4] = {primes[idx], 0, 0, 0}; // u64 to BigInt256
    Point result = montgomery_point_mul(&target, prime_limbs, modulus); // Our Phase 3 impl

    // On-curve check (y^2 == x^3 +7 mod p)
    uint64_t y2[4], x3[4], temp[4];
    montgomery_mul_opt(result.y, result.y, modulus, n_prime, y2);
    montgomery_mul_opt(result.x, result.x, modulus, n_prime, temp);
    montgomery_mul_opt(temp, result.x, modulus, n_prime, x3);
    add_limbs(x3, BigInt256_from_u64(7), x3); // +7
    bool on_curve = compare_limbs(y2, x3) == 0;

    // Store result if on_curve (atomic or shared flag)
    outputs[idx] = on_curve ? result : Point_infinity();
}

// Deep note: Launch <<<1,32>>> for test, sync, check all on-curve == true.
// Hamming low primes: Fewer 1s mean faster ladder (add/dbl ratios).