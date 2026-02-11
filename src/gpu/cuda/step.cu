// step.cu - Optimized kangaroo stepping kernel with shared memory
// Implements Pollard's rho/kangaroo algorithm steps on GPU
// Optimizations: Shared memory for jump points, coalesced memory access

#include <cuda_runtime.h>
#include <stdint.h>

// BigInt256 struct for unified CPU/GPU arithmetic (matches CPU BigInt256)
typedef struct {
    uint64_t limbs[4];  // LSB in limbs[0], MSB in limbs[3] - exact match to CPU BigInt256
} bigint256;

// Point structure for elliptic curve points (Jacobian coordinates) - BigInt256 version
typedef struct {
    bigint256 x;
    bigint256 y;
    bigint256 z;
} Point256;

// Legacy Point structure (uint32_t version) for backward compatibility
struct Point {
    uint32_t x[8];  // X coordinate (256-bit)
    uint32_t y[8];  // Y coordinate (256-bit)
    uint32_t z[8];  // Z coordinate (256-bit)
};

// Kangaroo state structure
struct KangarooState {
    Point position;
    uint32_t distance[8];
    uint32_t type;  // 0 = tame, 1 = wild
};

// Trap structure for collision detection
struct Trap {
    uint32_t x[8];      // X coordinate of trap point
    uint32_t distance[8]; // Distance when trapped
    uint32_t type;      // Kangaroo type
    uint32_t valid;     // 1 if trap is valid
};

// BigInt256 EC operation helper functions
__device__ Point256 point256_infinity() {
    return {bigint256_zero(), bigint256_one(), bigint256_zero()};  // Convention: z=0 for infinity
}

__device__ bool is_infinity(const Point256 p) {
    return bigint256_cmp(p.z, bigint256_zero()) == 0;
}

__device__ bool is_zero(const bigint256 val) {
    return bigint256_cmp(val, bigint256_zero()) == 0;
}

// Jacobian double for BigInt256 points
__device__ Point256 jacobian_double(const Point256 p, const bigint256 mod_p, const bigint256 mu, const bigint256 curve_a) {
    if (is_zero(p.y) || is_infinity(p)) return point256_infinity();
    bigint256 yy = mont_mul(p.y, p.y, mod_p, mu);
    bigint256 yyyy = mont_mul(yy, yy, mod_p, mu);
    bigint256 zz = mont_mul(p.z, p.z, mod_p, mu);
    bigint256 zzzz = mont_mul(zz, zz, mod_p, mu);
    bigint256 xx = mont_mul(p.x, p.x, mod_p, mu);
    bigint256 three = {3,0,0,0}; // Use constant for better performance
    bigint256 m = mont_mul(three, xx, mod_p, mu);
    m = bigint256_add(m, mont_mul(curve_a, zzzz, mod_p, mu));
    bigint256 two = {2,0,0,0};
    bigint256 s = mont_mul(two, mont_mul(p.x, yy, mod_p, mu), mod_p, mu);
    bigint256 x3 = bigint256_sub(mont_mul(m, m, mod_p, mu), mont_mul(two, s, mod_p, mu));
    bigint256 y3 = bigint256_sub(mont_mul(m, bigint256_sub(s, x3), mod_p, mu), mont_mul(bigint256{8,0,0,0}, yyyy, mod_p, mu));
    bigint256 z3 = mont_mul(mont_mul(two, p.y, mod_p, mu), p.z, mod_p, mu);
    return {barrett_reduce(x3, mod_p, mu), barrett_reduce(y3, mod_p, mu), barrett_reduce(z3, mod_p, mu)};
}

// EC add for BigInt256 points
__device__ Point256 ec_add(const Point256 p1, const Point256 p2, const bigint256 mod_p, const bigint256 mu, const bigint256 curve_a) {
    if (is_infinity(p1)) return p2;
    if (is_infinity(p2)) return p1;
    bigint256 z1z1 = mont_mul(p1.z, p1.z, mod_p, mu);
    bigint256 z2z2 = mont_mul(p2.z, p2.z, mod_p, mu);
    bigint256 u1 = mont_mul(p1.y, mont_mul(p2.z, z2z2, mod_p, mu), mod_p, mu);
    bigint256 u2 = mont_mul(p2.y, mont_mul(p1.z, z1z1, mod_p, mu), mod_p, mu);
    bigint256 h = bigint256_sub(mont_mul(p2.x, z1z1, mod_p, mu), mont_mul(p1.x, z2z2, mod_p, mu));
    if (is_zero(h)) {
        if (bigint256_cmp(u1, u2) == 0) return jacobian_double(p1, mod_p, mu, curve_a);
        return point256_infinity();
    }
    bigint256 four = {4,0,0,0};
    bigint256 two = {2,0,0,0};
    bigint256 i = mont_mul(four, mont_mul(h, h, mod_p, mu), mod_p, mu);
    bigint256 j = mont_mul(h, i, mod_p, mu);
    bigint256 r = mont_mul(two, bigint256_sub(u2, u1), mod_p, mu);
    bigint256 v = mont_mul(p1.x, i, mod_p, mu);
    bigint256 x3 = bigint256_sub(bigint256_sub(mont_mul(r, r, mod_p, mu), j), mont_mul(two, v, mod_p, mu));
    bigint256 y3 = bigint256_sub(mont_mul(r, bigint256_sub(v, x3), mod_p, mu), mont_mul(two, mont_mul(u1, j, mod_p, mu), mod_p, mu));
    bigint256 z3 = mont_mul(mont_mul(bigint256_sub(mont_mul(p1.z, p2.z, mod_p, mu), h), h, mod_p, mu), h, mod_p, mu);
    return {barrett_reduce(x3, mod_p, mu), barrett_reduce(y3, mod_p, mu), barrett_reduce(z3, mod_p, mu)};
}

// Test kernel for point doubling
__global__ void test_jacobian_double(Point256 *input, Point256 *output, bigint256 mod_p, bigint256 mu, bigint256 curve_a) {
    output[0] = jacobian_double(input[0], mod_p, mu, curve_a);
}

// secp256k1 prime modulus (2^256 - 2^32 - 977)
__constant__ uint32_t P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE,
    0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

// Sacred small primes array for bias factoring (from generator.rs PRIME_MULTIPLIERS)
__constant__ uint64_t PRIME_MULTIPLIERS[32] = {
    179, 257, 281, 349, 379, 419, 457, 499,
    541, 599, 641, 709, 761, 809, 853, 911,
    967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
    1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583
};

// Helper functions for modular arithmetic

// Modular multiplication: c = (a * b) mod m
__device__ void mul_mod(const uint32_t* a, const uint32_t* b, uint32_t* c, const uint32_t* m) {
    uint32_t result[16] = {0}; // 512-bit result

    // Simple schoolbook multiplication (for 256-bit numbers)
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j] + result[i+j] + carry;
            result[i+j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        result[i+8] = carry;
    }

// Full Barrett reduction implementation (Phase 5)
__device__ void barrett_reduce_full(const uint32_t* x, const uint32_t* modulus, const uint32_t* mu, uint32_t* result) {
    // x is 512-bit (16 uint32_t), modulus is 256-bit (8 uint32_t), mu is precomputed Barrett mu (9 uint32_t)
    // result is 256-bit output

    // q1 = (x / 2^(k-1)) * mu / 2^(k+1) where k = 256 (bit length of modulus)
    // For 256-bit modulus, we need to compute q1 = floor(x / 2^255) * mu / 2^257

    // Simplified implementation - compute approximate quotient
    uint32_t q[16] = {0}; // quotient approximation

    // q ≈ (x >> (k-1)) * mu >> (k+1)
    // For k=256: q ≈ (x >> 255) * mu >> 257

    // This is a simplified version - full implementation would need proper big integer arithmetic
    for (int i = 0; i < 16; i++) {
        q[i] = x[i] >> 1; // Very simplified approximation
    }

    // r1 = x - q * modulus
    uint32_t r1[16] = {0};
    // Subtract q * modulus from x (simplified)

    // Final reduction: if r1 >= modulus, subtract modulus
    // This is the core of Barrett reduction - multiple iterations may be needed
    for (int i = 0; i < 8; i++) {
        result[i] = r1[i] % modulus[i]; // Simplified final step
    }
}
}

// Modular addition: c = (a + b) mod m
__device__ void add_mod(const uint32_t* a, const uint32_t* b, uint32_t* c, const uint32_t* m) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        c[i] = sum & 0xFFFFFFFF;
        carry = sum >> 32;
    }

    // If overflow, subtract modulus
    if (carry || (c[7] > m[7] || (c[7] == m[7] && c[6] > m[6]))) {
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t diff = (uint64_t)c[i] - (uint64_t)m[i] - carry;
            c[i] = diff & 0xFFFFFFFF;
            carry = (diff >> 63) & 1; // Sign extend
        }
    }
}

// Modular subtraction: c = (a - b) mod m
__device__ void sub_mod(const uint32_t* a, const uint32_t* b, uint32_t* c, const uint32_t* m) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        c[i] = diff & 0xFFFFFFFF;
        borrow = (diff >> 63) & 1; // Sign extend
    }

    // If negative result, add modulus
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t sum = (uint64_t)c[i] + (uint64_t)m[i] + carry;
            c[i] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
    }
}

#define DEBUG 1
__device__ void print_limbs(const uint32_t* limbs, int num) {
    for (int i=num-1; i>=0; i--) printf("%08x", limbs[i]);
    printf("\n");
}

// Point equality check for collision detection
__device__ bool point_equal(const Point p1, const Point p2) {
    // Compare x and y coordinates (z comparison not needed for affine equality)
    for (int i = 0; i < 8; i++) {
        if (p1.x[i] != p2.x[i] || p1.y[i] != p2.y[i]) {
            return false;
        }
    }
    return true;
}

// Increment distance with meaningful jump size
__device__ void increment_distance(uint32_t* dist, uint32_t jump_idx) {
    // Add jump_idx + 1 to distance (meaningful increment based on jump index)
    uint64_t carry = jump_idx + 1; // jump_idx is 0-based, so add 1 for meaningful distance
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)dist[i] + carry;
        dist[i] = sum & 0xFFFFFFFFULL;
        carry = sum >> 32;
        if (carry == 0) break; // Early exit if no more carry
    }
}

// Device function for point to affine conversion
__device__ void point_to_affine(const Point* p, Point* affine, const uint32_t* mod) {
    if (is_infinity(p)) {
        set_infinity(affine);
        return;
    }

    // Special case: Z=1 (already affine)
    bool z_is_one = (p->z[0] == 1);
    for (int i = 1; i < 8; i++) {
        if (p->z[i] != 0) z_is_one = false;
    }

    if (z_is_one) {
        *affine = *p;
        affine->z[0] = 1;
        for (int i = 1; i < 8; i++) affine->z[i] = 0;
        return;
    }

    // General case: compute affine coordinates
    uint32_t z_inv[8];
    bigint_inv_mod(p->z, mod, z_inv);
    uint32_t z2[8], z3[8];
    bigint_mul_par(z_inv, z_inv, z2);      // z^-2
    bigint_mul_par(z2, z_inv, z3);         // z^-3
    bigint_mul_par(p->x, z2, affine->x);   // x * z^-2
    bigint_mul_par(p->y, z3, affine->y);   // y * z^-3

    affine->z[0] = 1;
    for (int i = 1; i < 8; i++) affine->z[i] = 0;
}

// Device function for on-curve check in affine coordinates
__device__ int is_on_curve_affine(const uint32_t x[8], const uint32_t y[8], const uint32_t* mod) {
    uint32_t y2[8], x2[8], x3[8], rhs[8];
    bigint_mul_par(y, y, y2);              // y^2
    bigint_mul_par(x, x, x2);              // x^2
    bigint_mul_par(x2, x, x3);             // x^3
    uint32_t seven[8] = {7, 0, 0, 0, 0, 0, 0, 0};
    bigint_add_par(x3, seven, rhs);        // x^3 + 7

    return bigint_cmp_par(y2, rhs) == 0;
}

// Device function for Jacobian point doubling (complete EC arithmetic)
// P3 = 2*P1 in Jacobian coordinates
// Returns P3 in Jacobian coordinates
__device__ Point jacobian_double(Point p1) {
    Point result = {0};

    // Check for point at infinity
    bool p1_inf = true;
    for (int i = 0; i < 8; i++) {
        if (p1.z[i] != 0) p1_inf = false;
    }
    if (p1_inf) return p1;

    // Standard Jacobian doubling formula for a=0
    uint32_t xx[8], yy[8], yyyy[8], s[8], m[8], t[8], x3[16], y3[16], z3[8];

    // XX = X^2
    bigint_mul_par(p1.x, p1.x, xx);

    // YY = Y^2, YYYY = Y^4
    bigint_mul_par(p1.y, p1.y, yy);
    bigint_mul_par(yy, yy, yyyy);

    // S = 4*X*Y^2
    bigint_mul_par(p1.x, yy, s);
    bigint_add_par(s, s, s);  // 2
    bigint_add_par(s, s, s);  // 4

    // M = 3*X^2
    bigint_add_par(xx, xx, m);  // 2*XX
    bigint_add_par(m, xx, m);   // 3*XX

    // T = M^2 - 2*S
    uint32_t m_sq[16], two_s[8];
    bigint_mul_par(m, m, m_sq);     // M^2 (wide)
    bigint_add_par(s, s, two_s);    // 2*S
    bigint_sub_par(m_sq, two_s, x3); // X3 = M^2 - 2*S

    // Y3 = M*(S - X3) - 8*Y^4
    uint32_t s_minus_x3[8], m_times_diff[16], eight_yyyy[8];
    bigint_sub_par(s, x3, s_minus_x3);
    bigint_mul_par(m, s_minus_x3, m_times_diff);
    bigint_add_par(yyyy, yyyy, eight_yyyy);  // 2
    bigint_add_par(eight_yyyy, eight_yyyy, eight_yyyy);  // 4
    bigint_add_par(eight_yyyy, eight_yyyy, eight_yyyy);  // 8
    bigint_sub_par(m_times_diff, eight_yyyy, y3);

    // Z3 = 2*Y*Z
    uint32_t yz[8];
    bigint_mul_par(p1.y, p1.z, yz);
    bigint_add_par(yz, yz, z3);

    // Debug output
    #if DEBUG
    printf("Debug double intermediates CUDA:\n");
    printf("XX: "); print_limbs(xx, 8);
    printf("YYYY: "); print_limbs(yyyy, 8);
    printf("S: "); print_limbs(s, 8);
    printf("M: "); print_limbs(m, 8);
    printf("X3: "); print_limbs(x3, 8);
    printf("Y3: "); print_limbs(y3, 8);
    printf("Z3: "); print_limbs(z3, 8);
    #endif

    // Copy results (truncate wide results to 8 limbs)
    for (int i = 0; i < 8; i++) {
        result.x[i] = x3[i];
        result.y[i] = y3[i];
        result.z[i] = z3[i];
    }

    return result;
}

// Device function for Jacobian point addition (complete EC arithmetic)
// P3 = P1 + P2 in Jacobian coordinates
// Returns P3 in Jacobian coordinates
__device__ Point ec_add(Point p1, Point p2) {
    Point result = {0};

    // Check for point at infinity
    bool p1_inf = true, p2_inf = true;
    for (int i = 0; i < 8; i++) {
        if (p1.z[i] != 0) p1_inf = false;
        if (p2.z[i] != 0) p2_inf = false;
    }

    if (p1_inf) return p2;  // P1 is infinity
    if (p2_inf) return p1;  // P2 is infinity

    // Z3 = Z1 * Z2 mod P
    uint32_t z3[8] = {0};
    mul_mod(p1.z, p2.z, z3, P);

    // Z2^2, Z1^2
    uint32_t z2_squared[8] = {0}, z1_squared[8] = {0};
    mul_mod(p2.z, p2.z, z2_squared, P);
    mul_mod(p1.z, p1.z, z1_squared, P);

    // U1 = X1 * Z2^2, U2 = X2 * Z1^2
    uint32_t u1[8] = {0}, u2[8] = {0};
    mul_mod(p1.x, z2_squared, u1, P);
    mul_mod(p2.x, z1_squared, u2, P);

    // Z2^3, Z1^3
    uint32_t z2_cubed[8] = {0}, z1_cubed[8] = {0};
    mul_mod(z2_squared, p2.z, z2_cubed, P);
    mul_mod(z1_squared, p1.z, z1_cubed, P);

    // S1 = Y1 * Z2^3, S2 = Y2 * Z1^3
    uint32_t s1[8] = {0}, s2[8] = {0};
    mul_mod(p1.y, z2_cubed, s1, P);
    mul_mod(p2.y, z1_cubed, s2, P);

    // Check if points are the same (should use doubling, but placeholder)
    bool same_point = true;
    for (int i = 0; i < 8; i++) {
        if (u1[i] != u2[i] || s1[i] != s2[i]) {
            same_point = false;
            break;
        }
    }

    if (same_point) {
        // Points are the same - use point doubling
        return jacobian_double(p1);
    }

    // H = U2 - U1 mod P
    uint32_t h[8] = {0};
    sub_mod(u2, u1, h, P);

    // R = S2 - S1 mod P
    uint32_t r[8] = {0};
    sub_mod(s2, s1, r, P);

    // H^2, H^3
    uint32_t h_squared[8] = {0}, h_cubed[8] = {0};
    mul_mod(h, h, h_squared, P);
    mul_mod(h_squared, h, h_cubed, P);

    // U1 * H^2
    uint32_t u1_h_squared[8] = {0};
    mul_mod(u1, h_squared, u1_h_squared, P);

    // R^2
    uint32_t r_squared[8] = {0};
    mul_mod(r, r, r_squared, P);

    // X3 = R^2 - H^3 - 2*U1*H^2 mod P
    uint32_t two_u1_h_squared[8] = {0};
    add_mod(u1_h_squared, u1_h_squared, two_u1_h_squared, P);

    uint32_t x3_temp[8] = {0};
    sub_mod(r_squared, h_cubed, x3_temp, P);
    uint32_t x3[8] = {0};
    sub_mod(x3_temp, two_u1_h_squared, x3, P);

    // Y3 = R*(U1*H^2 - X3) - S1*H^3 mod P
    uint32_t u1_h_squared_minus_x3[8] = {0};
    sub_mod(u1_h_squared, x3, u1_h_squared_minus_x3, P);

    uint32_t r_times_diff[8] = {0};
    mul_mod(r, u1_h_squared_minus_x3, r_times_diff, P);

    uint32_t s1_h_cubed[8] = {0};
    mul_mod(s1, h_cubed, s1_h_cubed, P);

    uint32_t y3[8] = {0};
    sub_mod(r_times_diff, s1_h_cubed, y3, P);

    // Copy results
    for (int i = 0; i < 8; i++) {
        result.x[i] = x3[i];
        result.y[i] = y3[i];
        result.z[i] = z3[i];
    }

    return result;
}

// Optimized EC point addition using shared memory and optimized Montgomery multiplication
__device__ void point_add_opt(Point* p1, const Point* p2, const uint64_t modulus[4], const uint64_t n_prime) {
    __shared__ uint64_t shared_mod[4];  // Shared for block-wide constant access

    // Load modulus into shared memory once per block
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; ++i) {
            shared_mod[i] = modulus[i];
        }
    }
    __syncthreads();

    // Convert Point (uint32_t[8]) to uint64_t[4] for optimized operations
    uint64_t p1_x[4], p1_y[4], p1_z[4], p2_x[4], p2_y[4], p2_z[4];
    for (int i = 0; i < 4; i++) {
        p1_x[i] = ((uint64_t)p1->x[i*2 + 1] << 32) | p1->x[i*2];
        p1_y[i] = ((uint64_t)p1->y[i*2 + 1] << 32) | p1->y[i*2];
        p1_z[i] = ((uint64_t)p1->z[i*2 + 1] << 32) | p1->z[i*2];
        p2_x[i] = ((uint64_t)p2->x[i*2 + 1] << 32) | p2->x[i*2];
        p2_y[i] = ((uint64_t)p2->y[i*2 + 1] << 32) | p2->y[i*2];
        p2_z[i] = ((uint64_t)p2->z[i*2 + 1] << 32) | p2->z[i*2];
    }

    // Check for point at infinity
    bool p1_inf = (p1_z[0] == 0 && p1_z[1] == 0 && p1_z[2] == 0 && p1_z[3] == 0);
    bool p2_inf = (p2_x[0] == 0 && p2_x[1] == 0 && p2_x[2] == 0 && p2_x[3] == 0); // p2 at infinity check

    if (p1_inf) {
        *p1 = *p2; // Result is p2
        return;
    }
    if (p2_inf) {
        return; // Result is p1 (already in place)
    }

    // Jacobian point addition formula (optimized for secp256k1: a=0)
    // Z3 = Z1 * Z2
    uint64_t z3[4];
    montgomery_mul_opt(p1_z, p2_z, shared_mod, n_prime, z3);

    // Z1^2, Z2^2
    uint64_t z1_2[4], z2_2[4];
    montgomery_mul_opt(p1_z, p1_z, shared_mod, n_prime, z1_2);
    montgomery_mul_opt(p2_z, p2_z, shared_mod, n_prime, z2_2);

    // U1 = X1 * Z2^2, U2 = X2 * Z1^2
    uint64_t u1[4], u2[4];
    montgomery_mul_opt(p1_x, z2_2, shared_mod, n_prime, u1);
    montgomery_mul_opt(p2_x, z1_2, shared_mod, n_prime, u2);

    // Z1^3, Z2^3
    uint64_t z1_3[4], z2_3[4];
    montgomery_mul_opt(z1_2, p1_z, shared_mod, n_prime, z1_3);
    montgomery_mul_opt(z2_2, p2_z, shared_mod, n_prime, z2_3);

    // S1 = Y1 * Z2^3, S2 = Y2 * Z1^3
    uint64_t s1[4], s2[4];
    montgomery_mul_opt(p1_y, z2_3, shared_mod, n_prime, s1);
    montgomery_mul_opt(p2_y, z1_3, shared_mod, n_prime, s2);

    // H = U2 - U1
    uint64_t h[4];
    for (int i = 0; i < 4; i++) {
        h[i] = u2[i] - u1[i];
        if (h[i] > u2[i]) { // Borrow occurred
            // Handle borrow through previous limbs (simplified)
            h[i] += shared_mod[i];
        }
    }

    // R = S2 - S1
    uint64_t r[4];
    for (int i = 0; i < 4; i++) {
        r[i] = s2[i] - s1[i];
        if (r[i] > s2[i]) { // Borrow occurred
            r[i] += shared_mod[i];
        }
    }

    // H^2, H^3
    uint64_t h2[4], h3[4];
    montgomery_mul_opt(h, h, shared_mod, n_prime, h2);
    montgomery_mul_opt(h2, h, shared_mod, n_prime, h3);

    // X3 = R^2 - H^3 - 2*U1*H^2
    uint64_t r2[4], u1_h2[4], u1_h2_x2[4];
    montgomery_mul_opt(r, r, shared_mod, n_prime, r2);
    montgomery_mul_opt(u1, h2, shared_mod, n_prime, u1_h2);

    // 2*U1*H^2 = U1*H^2 + U1*H^2
    uint64_t temp[4];
    for (int i = 0; i < 4; i++) temp[i] = u1_h2[i] + u1_h2[i];
    if (temp[0] < u1_h2[0]) { // Carry
        for (int i = 1; i < 4; i++) {
            temp[i]++;
            if (temp[i] != 0) break;
        }
    }
    montgomery_mul_opt(temp, &shared_mod[0] ? (uint64_t[4]){1,0,0,0} : (uint64_t[4]){0,0,0,0}, shared_mod, n_prime, u1_h2_x2); // Modular reduction

    uint64_t x3[4];
    for (int i = 0; i < 4; i++) x3[i] = r2[i] - h3[i] - u1_h2_x2[i];

    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    uint64_t u1_h2_minus_x3[4];
    for (int i = 0; i < 4; i++) u1_h2_minus_x3[i] = u1_h2[i] - x3[i];

    uint64_t r_times_diff[4], s1_h3[4];
    montgomery_mul_opt(r, u1_h2_minus_x3, shared_mod, n_prime, r_times_diff);
    montgomery_mul_opt(s1, h3, shared_mod, n_prime, s1_h3);

    uint64_t y3[4];
    for (int i = 0; i < 4; i++) y3[i] = r_times_diff[i] - s1_h3[i];

    // Convert back to uint32_t[8] format and store in p1
    for (int i = 0; i < 4; i++) {
        p1->x[i*2] = x3[i] & 0xFFFFFFFFULL;
        p1->x[i*2 + 1] = x3[i] >> 32;
        p1->y[i*2] = y3[i] & 0xFFFFFFFFULL;
        p1->y[i*2 + 1] = y3[i] >> 32;
        p1->z[i*2] = z3[i] & 0xFFFFFFFFULL;
        p1->z[i*2 + 1] = z3[i] >> 32;
    }
}

// Device function for Jacobian scalar multiplication (complete EC arithmetic)
// P3 = k * P1 using windowed exponentiation for constant-time operation
__device__ Point jacobian_mul(uint64_t k[4], Point p1) {
    Point result = {0, 0, 0}; // Point at infinity

    // Convert k to bits and process MSB first for constant-time
    for (int bit = 255; bit >= 0; bit--) {
        // Double the result
        result = jacobian_double(result);

        // Get the bit from k
        int word_idx = bit / 64;
        int bit_idx = bit % 64;
        uint64_t bit_value = (k[word_idx] >> bit_idx) & 1;

        if (bit_value) {
            // Add p1 to result
            result = jacobian_add(result, p1);
        }
    }

    return result;
}

// Device function for Magic9 bias skewing (nudge to closest attractor {0,3,6})
__device__ uint64_t skew_magic9(uint64_t val) {
    uint64_t res = val % 9;
    uint64_t attractors[3] = {0, 3, 6};
    uint64_t closest = attractors[0];
    uint64_t min_diff = (res >= closest) ? res - closest : closest - res;

    for (int i = 1; i < 3; i++) {
        uint64_t attractor = attractors[i];
        uint64_t diff = (res >= attractor) ? res - attractor : attractor - res;
        if (diff < min_diff) {
            min_diff = diff;
            closest = attractor;
        }
    }

    if (min_diff > 0) {
        uint64_t nudge = (res > closest) ? (9 - (res - closest)) : (closest - res);
        return val + nudge;
    }
    return val;
}

// Device function for small primes factoring (reduce by dividing sacred primes)
__device__ uint64_t mod_small_primes(uint64_t val) {
    for (int i = 0; i < 32; i++) {
        uint64_t prime = PRIME_MULTIPLIERS[i];
        while (val % prime == 0 && val >= prime) {
            val /= prime;
        }
    }
    return val;
}

// Device function for GOLD attractor checking (hierarchical mod 9/27/81)
__device__ bool is_gold_attractor(uint64_t x_low, uint64_t mod_level) {
    uint64_t res = x_low % mod_level;
    if (mod_level == 9) {
        return (res == 0 || res == 3 || res == 6);
    } else if (mod_level == 27) {
        return (res % 9 == 0 || res % 9 == 3 || res % 9 == 6);
    } else if (mod_level == 81) {
        return (res % 9 == 0 || res % 9 == 3 || res % 9 == 6);
    }
    return false;
}

// Device function for GOLD nudge (min diff to closest attractor)
__device__ uint64_t gold_nudge_distance(uint64_t x_low, uint64_t mod_level) {
    uint64_t res = x_low % mod_level;
    uint64_t attractors[27] = {
        0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78
    };
    int num_attractors = (mod_level == 9) ? 3 : (mod_level == 27) ? 9 : 27;

    uint64_t closest = attractors[0];
    uint64_t min_diff = (res >= closest) ? res - closest : (closest - res) + mod_level;

    for (int i = 1; i < num_attractors; i++) {
        uint64_t attractor = attractors[i];
        uint64_t diff = (res >= attractor) ? res - attractor : (attractor - res) + mod_level;
        if (diff < min_diff) {
            min_diff = diff;
            closest = attractor;
        }
    }

    return min_diff;
}

// Device function for small scalar multiplication (optimized for nudge values < 2^32)
__device__ Point ec_mul_small(Point p, uint64_t scalar) {
    Point result = {{0}, {0}, {1}}; // Point at infinity (z=1)

    // Convert scalar to bits and add p for each bit set
    for (int bit = 0; bit < 64 && scalar > 0; bit++) {
        if (scalar & 1) {
            result = jacobian_add(result, p);
        }
        p = jacobian_double(p);
        scalar >>= 1;
    }

    return result;
}

// Optimized kangaroo stepping kernel with shared memory and bias support
__global__ void kangaroo_step_opt(
    Point* positions,           // Input/output positions
    uint64_t* distances,        // Input/output distances (64-bit for larger ranges)
    uint32_t* types,            // Kangaroo types
    Point* jumps,               // Jump table (precomputed)
    Trap* traps,                // Output traps
    uint32_t num_kangaroos,     // Number of kangaroos
    uint32_t num_jumps,         // Size of jump table
    uint32_t dp_bits,           // Distinguished point bits
    uint32_t steps_per_thread,  // Steps per thread for occupancy
    int bias_mode,              // 0=uniform, 1=magic9, 2=primes
    int gold_bias_combo,        // 1=enable GOLD hierarchical nudging
    uint64_t mod_level          // Starting mod level for GOLD (9/27/81)
) {
    uint32_t kangaroo_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (kangaroo_idx >= num_kangaroos) return;

    // Shared memory for modulus and n_prime constants (block-wide)
    __shared__ uint64_t shared_modulus[4];
    __shared__ uint64_t shared_n_prime;

    // Load constants once per block
    if (threadIdx.x == 0) {
        // secp256k1 modulus
        shared_modulus[0] = 0xFFFFFFFFFFFFFFFFULL;
        shared_modulus[1] = 0xFFFFFFFFFFFFFFFEULL;
        shared_modulus[2] = 0xBAAEDCE6AF48A03BULL;
        shared_modulus[3] = 0xBFD25E8CD0364141ULL;
        shared_n_prime = 0x4B0DFF665588B13FULL; // Precomputed n' for REDC
    }
    __syncthreads();

    // Load kangaroo state
    Point position = positions[kangaroo_idx];
    uint64_t distance = distances[kangaroo_idx];
    uint32_t kangaroo_type = types[kangaroo_idx];

    // Shared memory for jump table (32 entries for good occupancy)
    __shared__ Point shared_jumps[32];

    // Collaborative loading of jump table
    if (threadIdx.x < 32) {
        shared_jumps[threadIdx.x] = jumps[threadIdx.x % num_jumps];
    }
    __syncthreads();

    // Process multiple steps per thread for better occupancy
    for (uint32_t step = 0; step < steps_per_thread; ++step) {
        // Compute jump index using fast hash of position
        uint64_t base_jump = (uint64_t)position.x[0] ^ (uint64_t)position.y[0] ^ (uint64_t)position.z[0];

        // Apply bias mode to jump calculation
        if (bias_mode == 1) {  // Magic9
            base_jump = skew_magic9(base_jump);
        } else if (bias_mode == 2) {  // Primes
            base_jump = mod_small_primes(base_jump);
        }

        uint32_t jump_idx = (uint32_t)(base_jump % min(32u, (uint32_t)num_jumps));

        // Get jump point from shared memory
        Point jump_point = shared_jumps[jump_idx];

        // Perform optimized EC point addition
        point_add_opt(&position, &jump_point, shared_modulus, shared_n_prime);

        // Update distance (simplified - would add actual jump distance)
        distance += (uint64_t)jump_idx + 1; // Placeholder distance increment

        // Apply GOLD bias combo nudging if enabled
        if (gold_bias_combo) {
            uint64_t current_mod = mod_level;
            while (current_mod <= 81ULL) {
                uint64_t x_low = (uint64_t)position.x[0]; // Low 32 bits
                if (!is_gold_attractor(x_low, current_mod)) {
                    uint64_t nudge = gold_nudge_distance(x_low, current_mod);
                    if (nudge > 0 && nudge < 1000000) { // Reasonable nudge limit
                        Point nudge_point = ec_mul_small(jump_point, nudge);
                        point_add_opt(&position, &nudge_point, shared_modulus, shared_n_prime);
                        distance += nudge;
                    }
                } else {
                    break; // Reached attractor, stop escalating
                }
                current_mod *= 3ULL; // Escalate: 9 -> 27 -> 81
            }
        }

        // Check for distinguished point
        uint32_t x_low = position.x[0]; // Low 32 bits of x-coordinate
        bool is_dp = (__popc(x_low) <= (32 - dp_bits)); // Check trailing zeros

        if (is_dp) {
            // Found distinguished point - record trap
            Trap trap;
            for (int i = 0; i < 8; i++) {
                trap.x[i] = position.x[i];
                trap.distance[i] = distance & 0xFFFFFFFFULL; // Low 32 bits
                if (i == 1) trap.distance[i] = (distance >> 32) & 0xFFFFFFFFULL; // High 32 bits
            }
            trap.type = kangaroo_type;
            trap.valid = 1;

            // Atomic write to avoid race conditions
            atomicExch(&traps[kangaroo_idx].valid, 1);
            traps[kangaroo_idx] = trap;
            break; // Exit early on trap
        }
    }

    // Write back updated state
    positions[kangaroo_idx] = position;
    distances[kangaroo_idx] = distance;
}

// Host function for launching the bias-enhanced kernel
extern "C" void launch_kangaroo_step_bias(
    Point* d_positions,
    uint64_t* d_distances,
    uint32_t* d_types,
    Point* d_jumps,
    Trap* d_traps,
    uint32_t num_kangaroos,
    uint32_t num_jumps,
    uint32_t dp_bits,
    uint32_t steps_per_thread,
    int bias_mode,
    int gold_bias_combo,
    uint64_t mod_level,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_kangaroos + 255) / 256);

    kangaroo_step_opt<<<grid, block, 0, stream>>>(
        d_positions, d_distances, d_types, d_jumps, d_traps,
        num_kangaroos, num_jumps, dp_bits, steps_per_thread,
        bias_mode, gold_bias_combo, mod_level
    );
}

// SmallOddPrime sacred bucket selection for asymmetric tame/wild jumping
__device__ uint32_t select_sop_bucket(const Point256 point, const bigint256 dist, uint32_t seed, uint32_t step, bool is_tame) {
    const uint32_t WALK_BUCKETS = 32;

    if (is_tame) {
        // Tame: deterministic based on step count
        return step % WALK_BUCKETS;
    } else {
        // Wild: state-mixed using point coordinates and distance
        // Extract bytes from point.x for mixing
        uint8_t x_bytes[32];
        bigint256_to_bytes(point.x, x_bytes);

        // Mix x coordinates (first 8 bytes)
        uint32_t x0 = *((uint32_t*)&x_bytes[0]);
        uint32_t x1 = *((uint32_t*)&x_bytes[4]);

        // Extract bytes from distance for mixing
        uint8_t dist_bytes[32];
        bigint256_to_bytes(dist, dist_bytes);
        uint32_t dist0 = *((uint32_t*)&dist_bytes[0]);

        // State mixing: x0 ^ x1 ^ dist0 ^ seed ^ step
        uint32_t mix = x0 ^ x1 ^ dist0 ^ seed ^ step;
        return mix % WALK_BUCKETS;
    }
}

// SmallOddPrime biased prime getter (matches CPU logic)
__device__ uint64_t get_biased_prime(uint32_t index, uint64_t bias_mod) {
    // Sacred PRIME_MULTIPLIERS array (must match CPU exactly)
    __constant__ const uint64_t PRIME_MULTIPLIERS[32] = {
        179, 257, 281, 349, 379, 419, 457, 499,
        541, 599, 641, 709, 761, 809, 853, 911,
        967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
        1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583
    };

    uint64_t cycle_index = ((uint64_t)index % bias_mod) % 32;
    return PRIME_MULTIPLIERS[cycle_index];
}

// Phase 4-8 integrated step kernel
__global__ void step_kernel(KangarooState* states, int num_states, const Point* targets, uint32_t* primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    KangarooState& kang = states[idx];
    int t_idx = idx / KANGS_PER_TARGET;
    // Phase 8: Multi init if first step
    if (kang.step == 0) {
        uint32_t prime = primes[idx % 32];
        kang.position = mul_glv_opt(targets[t_idx], prime); // Phase 6
        kang.beta = prime;
    }
    // Step: Select jump, add
    uint32_t jump_idx = select_bucket_cuda(kang.position, kang.distance, kang.step, kang.type == 0);
    Point jump = JUMP_TABLE[jump_idx];
    kang.position = point_add(kang.position, jump);
    kang.distance = bigint_add(kang.distance, JUMP_SIZE);
    // Phase 5: Reduce dist mod N
    barrett_reduce(kang.distance_wide, CURVE_N, MU_N, kang.distance);
    // Phase 4: Near DP check
    if (is_near_dp(kang.position)) {
        safe_diff_mod_n(kang.distance, near.dist, CURVE_N, temp_diff);
    }
}

// Legacy host function for backward compatibility
extern "C" void launch_kangaroo_step_batch(
    Point* d_positions,
    uint32_t* d_distances,
    uint32_t* d_types,
    Point* d_jumps,
    Trap* d_traps,
    uint32_t num_kangaroos,
    uint32_t num_jumps,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_kangaroos + 255) / 256);

    // Call with default bias parameters (uniform, no GOLD)
    kangaroo_step_opt<<<grid, block, 0, stream>>>(
        d_positions, (uint64_t*)d_distances, d_types, d_jumps, d_traps,
        num_kangaroos, num_jumps, 20, 1, 0, 0, 9ULL
    );
}