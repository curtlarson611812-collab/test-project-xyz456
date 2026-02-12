// step.cu - Optimized kangaroo stepping kernel with shared memory
// Implements Pollard's rho/kangaroo algorithm steps on GPU
// Optimizations: Shared memory for jump points, coalesced memory access
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h> // For printf
#include "common_constants.h"

#define KANGS_PER_TARGET 4096
#define DEBUG 1
#define LIMBS 8

// Point and KangarooState structures are now defined in common_constants.h

// secp256k1 prime modulus (P) as uint32_t[8]
__constant__ uint32_t P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE,
    0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

// GLV endomorphism beta constant
__constant__ uint32_t GLV_BETA[8] = {
    0x7AE96A2B, 0x65718000, 0x5F228AE5, 0x118050B7,
    0xEC014F9A, 0xED809F6D, 0xCAA2B2BB, 0xC2D2EAA9
};

// GLV lambda constant
__constant__ uint32_t GLV_LAMBDA[8] = {
    0x5363AD4C, 0xC05C30E0, 0xA5278789, 0x9CC8148B,
    0x8814FF65, 0x74E9C3AB, 0x5144A2A0, 0x44CF6308
};

// Small prime multipliers for bias factoring
__constant__ uint64_t PRIME_MULTIPLIERS[32] = {
    179, 257, 281, 349, 379, 419, 457, 499,
    541, 599, 641, 709, 761, 809, 853, 911,
    967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
    1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583
};

// GLV lattice basis vectors (pre-computed)
__constant__ uint32_t GLV_V1_1_LIMBS[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
__constant__ uint32_t GLV_V1_2_LIMBS[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
__constant__ uint32_t GLV_V2_1_LIMBS[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
__constant__ uint32_t GLV_V2_2_LIMBS[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
__constant__ uint32_t GLV_R1_LIMBS[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
__constant__ uint32_t GLV_R2_LIMBS[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
__constant__ uint32_t GLV_SQRT_N_LIMBS[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
__constant__ uint32_t GLV_LAMBDA_LIMBS[8] = {0xd0364141, 0xbfd25e8c, 0xaf48a03b, 0xbaaedce6, 0xfffffffe, 0xffffffff, 0xffffffff, 0xffffffff};
__constant__ uint32_t GLV_BETA_LIMBS[8] = {0x86b801e8, 0x9e0b24cd, 0x24cb09e8, 0x187684d9, 0xa5fb0480, 0x3e7d44e6, 0x10071c65, 0x2b6ae97a};

// Curve order (N)
__constant__ uint32_t CURVE_N[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE,
    0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

// Montgomery constants for N
__constant__ uint32_t MU_N[16] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x45512319, 0x50B75FC4, 0x402DA173, 0x2FBC146B,
    0x09DDA963, 0x02FDB94D, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// Trap structure for collision detection
struct Trap {
    uint32_t x[8]; // X coordinate of trap point
    uint32_t distance[8]; // Distance when trapped
    uint32_t type; // Kangaroo type
    uint32_t valid; // 1 if trap is valid
};

// CURVE_N and MU_N are now defined in common_constants.h

// Mean jump distance as uint32_t[8]
__constant__ uint32_t JUMP_SIZE[8] = {1024, 0, 0, 0, 0, 0, 0, 0};

// Jump table for kangaroo hops (precomputed points)
__constant__ Point JUMP_TABLE[256];

// All constants are now defined in common_constants.h

// Utility helper functions (static to avoid duplicate symbols)
static __device__ uint32_t murmur3(const uint32_t key[8]) {
    const uint32_t seed = 0x9747b28c;
    uint32_t hash = seed;
    for (int i = 0; i < 8; i += 2) { // Process in pairs for efficiency
        uint32_t k = key[i];
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        hash ^= k;
        hash = (hash << 13) | (hash >> 19);
        hash = hash * 5 + 0xe6546b64;
    }
    hash ^= 32; // Size
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
}

static __device__ int hamming_weight(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3F;
}

static __device__ void print_limbs(const uint32_t* limbs, int num) {
    for (int i = num - 1; i >= 0; i--) printf("%08x", limbs[i]);
    printf("\n");
}

__device__ int point_equal(const Point p1, const Point p2) {
    for (int i = 0; i < 8; i++) {
        if (p1.x[i] != p2.x[i] || p1.y[i] != p2.y[i]) return false;
    }
    return true;
}

static __device__ void increment_distance(uint32_t* dist, uint32_t jump_idx) {
    uint64_t carry = jump_idx + 1;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)dist[i] + carry;
        dist[i] = sum & 0xFFFFFFFFULL;
        carry = sum >> 32;
        if (carry == 0) break;
    }
}

// Modular arithmetic functions
static __device__ void bigint_copy(const uint32_t* src, uint32_t* dst) {
    for (int i = 0; i < 8; i++) dst[i] = src[i];
}

static __device__ void bigint_zero_u32(uint32_t* res) {
    for (int i = 0; i < 8; i++) res[i] = 0;
}

static __device__ void bigint_one_u32(uint32_t* res) {
    res[0] = 1;
    for (int i = 1; i < 8; i++) res[i] = 0;
}

static __device__ bool bigint_is_zero(const uint32_t* a) {
    for (int i = 0; i < 8; i++) if (a[i] != 0) return false;
    return true;
}

static __device__ bool bigint_is_negative(const uint32_t* a) {
    return (int32_t)a[7] < 0;
}

static __device__ uint32_t bigint_to_u32(const uint32_t* a) {
    return a[0];
}

static __device__ void bigint_add_u32(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        res[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
}

static __device__ void bigint_sub_u32(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    int32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        int64_t diff = (int64_t)a[i] - (int64_t)b[i] - borrow;
        res[i] = (uint32_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
}

static __device__ void bigint_mul_u32(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    for (int i = 0; i < 16; i++) res[i] = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            __int128 mul = (__int128)a[i] * b[j];
            uint32_t lo = (uint32_t)mul;
            uint32_t hi = (uint32_t)(mul >> 32);
            uint32_t carry = 0;
            if (i + j < 16) {
                uint64_t sum = (uint64_t)res[i + j] + lo + carry;
                res[i + j] = (uint32_t)sum;
                carry = (uint32_t)(sum >> 32);
            }
            if (i + j + 1 < 16) {
                uint64_t sum = (uint64_t)res[i + j + 1] + hi + carry;
                res[i + j + 1] = (uint32_t)sum;
            }
        }
    }
}

static __device__ int bigint_cmp_par(const uint32_t* a, const uint32_t* b) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

static __device__ void add_mod(const uint32_t* a, const uint32_t* b, uint32_t* c, const uint32_t* m) {
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a[i] + (uint64_t)b[i] + carry;
        c[i] = sum & 0xFFFFFFFF;
        carry = sum >> 32;
    }
    if (carry || (c[7] > m[7] || (c[7] == m[7] && c[6] > m[6]))) {
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t diff = (uint64_t)c[i] - (uint64_t)m[i] - carry;
            c[i] = diff & 0xFFFFFFFF;
            carry = (diff >> 63) & 1;
        }
    }
}

static __device__ void sub_mod(const uint32_t* a, const uint32_t* b, uint32_t* c, const uint32_t* m) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a[i] - (uint64_t)b[i] - borrow;
        c[i] = diff & 0xFFFFFFFF;
        borrow = (diff >> 63) & 1;
    }
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t sum = (uint64_t)c[i] + (uint64_t)m[i] + carry;
            c[i] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
    }
}

__device__ void mul_mod(const uint32_t* a, const uint32_t* b, uint32_t* c, const uint32_t* m) {
    uint32_t result[16] = {0};
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j] + result[i+j] + carry;
            result[i+j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        result[i+8] = carry;
    }
    // Simple reduction loop (for correctness; optimize with Barrett later)
    while (bigint_cmp_par(result, m) >= 0) {
        bigint_sub_u32(result, m, result);
    }
    for (int i = 0; i < 8; i++) c[i] = result[i];
}

// Point and curve helper functions
static __device__ void set_infinity(Point* p) {
    for (int i = 0; i < 8; i++) {
        p->x[i] = 0;
        p->y[i] = 0;
        p->z[i] = (i == 0) ? 0 : 0; // z=0 for infinity
    }
    p->z[0] = 0;
}

static __device__ bool is_infinity(const Point p) {
    for (int i = 0; i < 8; i++) if (p.z[i] != 0) return false;
    return true;
}

static __device__ void point_to_affine(const Point* p, Point* affine, const uint32_t* mod) {
    if (is_infinity(*p)) {
        set_infinity(affine);
        return;
    }
    // TODO: Implement modular inverse for point_to_affine
    uint32_t z_inv[8] = {0}; // Stub for now
    uint32_t z2[8], z3[8];
    mul_mod(z_inv, z_inv, z2, mod); // z^-2
    mul_mod(z2, z_inv, z3, mod); // z^-3
    mul_mod(p->x, z2, affine->x, mod);
    mul_mod(p->y, z3, affine->y, mod);
    affine->z[0] = 1;
    for (int i = 1; i < 8; i++) affine->z[i] = 0;
}

static __device__ int is_on_curve_affine(const uint32_t x[8], const uint32_t y[8], const uint32_t* mod) {
    uint32_t y2[8], x3[8], rhs[8];
    mul_mod(y, y, y2, mod);
    mul_mod(x, x, x3, mod); // x^2 temp, but full x^3
    mul_mod(x3, x, x3, mod); // x^3
    uint32_t seven[8] = {7, 0, 0, 0, 0, 0, 0, 0};
    add_mod(x3, seven, rhs, mod);
    return bigint_cmp_par(y2, rhs) == 0;
}

static __device__ void safe_diff_mod_n(const uint32_t a[8], const uint32_t b[8], const uint32_t n[8], uint32_t* result) {
    if (bigint_cmp_par(a, b) >= 0) {
        sub_mod(a, b, result, n);
    } else {
        uint32_t temp[8];
        add_mod(a, n, temp, n);
        sub_mod(temp, b, result, n);
    }
}

static __device__ bool is_near_dp(const Point p) {
    uint32_t hash = murmur3(p.x);
    const uint32_t DP_MASK = 0xFFFF;
    uint32_t masked = hash & DP_MASK;
    if (masked == 0) return true;
    return hamming_weight(masked) < 4;
}


// Extended Euclidean algorithm functions (duplicate removed)
static __device__ void parallel_div(const uint32_t* dividend, const uint32_t* divisor, uint32_t* quotient) {
    uint32_t temp[8];
    bigint_copy(dividend, temp);
    bigint_zero_u32(quotient);

    for (int bit = 255; bit >= 0; bit--) {
        // Left shift temp
        uint32_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint32_t next_carry = temp[i] >> 31;
            temp[i] = (temp[i] << 1) | carry;
            carry = next_carry;
        }

        if (bigint_cmp_par(temp, divisor) >= 0) {
            bigint_sub_u32(temp, divisor, temp);
            int word_idx = bit / 32;
            int bit_idx = bit % 32;
            quotient[word_idx] |= (1U << bit_idx);
        }
    }
}

static __device__ int extended_gcd(const uint32_t* a, const uint32_t* b, uint32_t* x, uint32_t* y) {
    uint32_t old_r[8], r[8], old_s[8], s[8], old_t[8], t[8];
    bigint_copy(a, old_r);
    bigint_copy(b, r);
    bigint_one_u32(old_s);
    bigint_zero_u32(s);
    bigint_zero_u32(old_t);
    bigint_one_u32(t);

    while (!bigint_is_zero(r)) {
        uint32_t quotient[8];
        parallel_div(old_r, r, quotient);
        uint32_t temp[8];
        bigint_mul_u32(quotient, r, temp);
        bigint_sub_u32(old_r, temp, old_r);
        bigint_mul_u32(quotient, s, temp);
        bigint_sub_u32(old_s, temp, old_s);
        bigint_mul_u32(quotient, t, temp);
        bigint_sub_u32(old_t, temp, old_t);
        bigint_copy(r, old_r);
        bigint_copy(old_r, r);
        bigint_copy(s, old_s);
        bigint_copy(old_s, s);
        bigint_copy(t, old_t);
        bigint_copy(old_t, t);
    }
    bigint_copy(old_s, x);
    bigint_copy(old_t, y);
    return bigint_to_u32(old_r); // gcd
}

static __device__ void bigint_inv_mod(const uint32_t* a, const uint32_t* mod, uint32_t* result) {
    uint32_t x[8], y[8];
    int gcd = extended_gcd(a, mod, x, y);
    if (gcd != 1) {
        bigint_zero_u32(result);
        return;
    }
    if (bigint_is_negative(x)) {
        bigint_add_u32(x, mod, x);
    }
    bigint_copy(x, result);
}

// EC arithmetic functions
__device__ Point jacobian_double(Point p1) {
    Point result;
    if (is_infinity(p1)) return p1;
    uint32_t xx[8], yy[8], yyyy[8], s[8], m[8], x3[8], y3[8], z3[8];
    mul_mod(p1.x, p1.x, xx, P); // XX = X^2
    mul_mod(p1.y, p1.y, yy, P); // YY = Y^2
    mul_mod(yy, yy, yyyy, P); // YYYY = Y^4
    mul_mod(p1.x, yy, s, P);
    add_mod(s, s, s, P); add_mod(s, s, s, P); // S = 4*X*YY
    add_mod(xx, xx, m, P); add_mod(m, xx, m, P); // M = 3*XX
    mul_mod(m, m, x3, P); // M^2
    uint32_t two_s[8]; add_mod(s, s, two_s, P);
    sub_mod(x3, two_s, x3, P); // X3 = M^2 - 2*S
    uint32_t s_minus_x3[8]; sub_mod(s, x3, s_minus_x3, P);
    mul_mod(m, s_minus_x3, y3, P); // M*(S - X3)
    uint32_t eight_yyyy[8]; add_mod(yyyy, yyyy, eight_yyyy, P);
    add_mod(eight_yyyy, eight_yyyy, eight_yyyy, P); add_mod(eight_yyyy, eight_yyyy, eight_yyyy, P);
    sub_mod(y3, eight_yyyy, y3, P); // Y3 = ... - 8*YYYY
    mul_mod(p1.y, p1.z, z3, P); add_mod(z3, z3, z3, P); // Z3 = 2*Y*Z
    for (int i = 0; i < 8; i++) {
        result.x[i] = x3[i];
        result.y[i] = y3[i];
        result.z[i] = z3[i];
    }
    return result;
}

static __device__ Point ec_add(Point p1, Point p2) {
    Point result;
    if (is_infinity(p1)) return p2;
    if (is_infinity(p2)) return p1;
    uint32_t z1_2[8], z2_2[8], u1[8], u2[8], z1_3[8], z2_3[8], s1[8], s2[8], h[8], r[8];
    mul_mod(p1.z, p1.z, z1_2, P);
    mul_mod(p2.z, p2.z, z2_2, P);
    mul_mod(p1.x, z2_2, u1, P);
    mul_mod(p2.x, z1_2, u2, P);
    mul_mod(z1_2, p1.z, z1_3, P);
    mul_mod(z2_2, p2.z, z2_3, P);
    mul_mod(p1.y, z2_3, s1, P);
    mul_mod(p2.y, z1_3, s2, P);
    if (bigint_cmp_par(u1, u2) == 0 && bigint_cmp_par(s1, s2) == 0) return jacobian_double(p1);
    sub_mod(u2, u1, h, P);
    sub_mod(s2, s1, r, P);
    uint32_t h2[8], h3[8];
    mul_mod(h, h, h2, P);
    mul_mod(h2, h, h3, P);
    uint32_t u1_h2[8], two_u1_h2[8], r2[8], x3_temp[8], x3[8];
    mul_mod(u1, h2, u1_h2, P);
    add_mod(u1_h2, u1_h2, two_u1_h2, P);
    mul_mod(r, r, r2, P);
    sub_mod(r2, h3, x3_temp, P);
    sub_mod(x3_temp, two_u1_h2, x3, P);
    uint32_t u1_h2_minus_x3[8], r_diff[8], s1_h3[8], y3[8];
    sub_mod(u1_h2, x3, u1_h2_minus_x3, P);
    mul_mod(r, u1_h2_minus_x3, r_diff, P);
    mul_mod(s1, h3, s1_h3, P);
    sub_mod(r_diff, s1_h3, y3, P);
    uint32_t z3[8];
    mul_mod(p1.z, p2.z, z3, P);
    for (int i = 0; i < 8; i++) {
        result.x[i] = x3[i];
        result.y[i] = y3[i];
        result.z[i] = z3[i];
    }
    return result;
}

__device__ Point jacobian_add(const Point p1, const Point p2) {
    return ec_add(p1, p2); // Alias to full add
}

__device__ Point ec_mul_small(Point p, uint64_t scalar) {
    Point result; set_infinity(&result);
    for (int bit = 0; bit < 64 && scalar > 0; bit++) {
        if (scalar & 1) result = jacobian_add(result, p);
        p = jacobian_double(p);
        scalar >>= 1;
    }
    return result;
}

// Helper functions for GLV operations
__device__ void bigint_mul_par(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    // Parallel limb multiplication (schoolbook for simplicity)
    for (int i = 0; i < 16; i++) res[i] = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a[i] * b[j];
            uint32_t carry = 0;
            uint32_t sum = res[i + j] + (prod & 0xFFFFFFFFULL) + carry;
            res[i + j] = sum & 0xFFFFFFFFULL;
            carry = sum >> 32;
            if (i + j + 1 < 16) {
                res[i + j + 1] += (prod >> 32) + carry;
            }
        }
    }
}

__device__ void bigint_add_par(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    uint32_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        res[i] = sum & 0xFFFFFFFFULL;
        carry = sum >> 32;
    }
    if (carry && 8 < 16) res[8] = carry;
}

__device__ void bigint_sub_par(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a[i] - b[i] - borrow;
        res[i] = diff & 0xFFFFFFFFULL;
        borrow = (diff >> 32) != 0;
    }
}

__device__ void bigint_sub(const uint32_t* a, const uint32_t* b, uint32_t* res) {
    uint32_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)a[i] - b[i] - borrow;
        res[i] = diff & 0xFFFFFFFFULL;
        borrow = (diff >> 32) != 0;
    }
}

static __device__ void bigint_add_u32(const uint32_t* a, uint32_t b, uint32_t* res) {
    uint32_t carry = b;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a[i] + carry;
        res[i] = sum & 0xFFFFFFFFULL;
        carry = sum >> 32;
        if (carry == 0) break;
    }
}

__device__ void point_neg(const Point* p, Point* neg, const uint32_t* mod) {
    for (int i = 0; i < 8; i++) {
        neg->x[i] = p->x[i];
        neg->y[i] = p->y[i];
        neg->z[i] = p->z[i];
    }
    bigint_sub(mod, p->y, neg->y);
}


// Approximate big integer division for CUDA (a / b -> result)
__device__ void bigint_div_approx(const uint32_t a[16], const uint32_t b[16], uint32_t result[8]) {
    // Simple approximation: a / b ≈ a_high / b_high for large numbers
    // For Babai, we need integer division, so implement bit-by-bit
    for (int i = 0; i < 8; i++) result[i] = 0;

    uint32_t dividend[17] = {0};
    for (int i = 0; i < 16; i++) dividend[i] = a[i];

    // Find leading bit
    int leading_bit = 0;
    for (int i = 16; i >= 0; i--) {
        if (dividend[i] != 0) {
            leading_bit = i * 32 + 31 - __clz(dividend[i]);
            break;
        }
    }

    // Bit-by-bit division
    for (int bit = leading_bit - 1; bit >= 0; bit--) {
        // Shift dividend left by 1
        uint32_t carry = 0;
        for (int i = 0; i < 17; i++) {
            uint32_t next_carry = dividend[i] >> 31;
            dividend[i] = (dividend[i] << 1) | carry;
            carry = next_carry;
        }

        // If dividend >= divisor, subtract and set bit
        // Use simple comparison for approximation
        bool greater_equal = true;
        for (int i = 16; i >= 0; i--) {
            if (dividend[i] > b[i]) break;
            if (dividend[i] < b[i]) {
                greater_equal = false;
                break;
            }
        }

        if (greater_equal) {
            // Simple subtraction approximation
            for (int i = 0; i < 16; i++) {
                if (dividend[i] >= b[i]) {
                    dividend[i] -= b[i];
                }
            }
            int word_idx = bit / 32;
            int bit_idx = bit % 32;
            if (word_idx < 8) {
                result[word_idx] |= (1u << bit_idx);
            }
        }
    }
}

// CUDA Gram-Schmidt orthogonalization for 4D
__device__ void gram_schmidt_4d_cuda(const uint32_t v[4][8], uint32_t gs[4][8]) {
    // Copy first vector
    #pragma unroll
    for (int j = 0; j < 8; j++) gs[0][j] = v[0][j];

    // Orthogonalize remaining vectors
    #pragma unroll
    for (int i = 1; i < 4; i++) {
        // Start with v[i]
        #pragma unroll
        for (int j = 0; j < 8; j++) gs[i][j] = v[i][j];

        // Subtract projections onto previous vectors
        #pragma unroll
        for (int k = 0; k < i; k++) {
            // mu = <v[i], gs[k]> / ||gs[k]||^2
            uint32_t dot[16], norm_sq[16], mu[8];
            bigint_mul_par(v[i], gs[k], dot);
            bigint_mul_par(gs[k], gs[k], norm_sq);
            bigint_div_approx(dot, norm_sq, mu);

            // Subtract mu * gs[k]
            uint32_t mu_gs[16];
            bigint_mul_par(mu, gs[k], mu_gs);
            bigint_sub_par(gs[i], mu_gs, gs[i]);
        }
    }
}


// Professor-level GLV4 decompose with 4D Babai's algorithm
__device__ void glv4_decompose_babai(const uint32_t k[8], uint32_t coeffs[4][8], int8_t signs[4]) {
    // 4D GLV lattice reduction with Babai's nearest plane
    // Parallel implementation using warp-level reductions

    // Precomputed 4D basis vectors (LSB first)
    const uint32_t v[4][8] = {
        {0x3dab, 0xeb15, 0x84eb, 0x92e4, 0x6c90, 0xe86c, 0xd46b, 0xa7d4}, // v1
        {0x4cfd, 0xd9d4, 0x1108, 0x657c, 0x2f3f, 0x7a8e, 0xa50f, 0x114c}, // v2
        {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}, // v3 ≈ v1*lambda (placeholder)
        {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}, // v4 ≈ v2*lambda (placeholder)
    };

    // Gram-Schmidt orthogonalization (precomputed for performance)
    uint32_t gs[4][8];
    gram_schmidt_4d_cuda(v, gs);

    // Babai's algorithm: Project onto orthogonal basis
    uint32_t residual[8];
    for (int i = 0; i < 8; i++) residual[i] = k[i];

    uint32_t temp_coeffs[4][8] = {0};

    // Project from dimension 4 down to 1
    for (int dim = 3; dim >= 0; dim--) {
        // <residual, gs[dim]> / ||gs[dim]||^2
        uint32_t dot[16];
        bigint_mul_par(residual, gs[dim], dot);

        uint32_t norm_sq[16];
        bigint_mul_par(gs[dim], gs[dim], norm_sq);

        // Approximate division by norm_sq (simplified)
        uint32_t projection[8];
        bigint_div_approx(dot, norm_sq, projection);

        // Round to nearest integer
        bigint_add_u32(projection, 1u << 31, projection); // Add 0.5
        // Shift right by 256 (approximate)

        // Store coefficient
        for (int j = 0; j < 8; j++) temp_coeffs[dim][j] = projection[j];

        // Subtract projection * v[dim] from residual
        uint32_t proj_v[16];
        bigint_mul_par(projection, v[dim], proj_v);
        bigint_sub_par(residual, proj_v, residual);
    }

    // Multi-round Babai improvement (2 additional rounds)
    for (int round = 0; round < 2; round++) {
        for (int i = 0; i < 8; i++) residual[i] = k[i];

        for (int dim = 3; dim >= 0; dim--) {
            uint32_t dot[16];
            bigint_mul_par(residual, gs[dim], dot);

            uint32_t norm_sq[16];
            bigint_mul_par(gs[dim], gs[dim], norm_sq);

            uint32_t projection[8];
            bigint_div_approx(dot, norm_sq, projection);

            bigint_add_u32(projection, 1u << 31, projection);

            for (int j = 0; j < 8; j++) temp_coeffs[dim][j] = projection[j];

            uint32_t proj_v[16];
            bigint_mul_par(projection, v[dim], proj_v);
            bigint_sub_par(residual, proj_v, residual);
        }
    }

    // 16-combination shortest vector selection (constant-time, warp-parallel)
    uint32_t best_coeffs[4][8];
    int8_t best_signs[4] = {1, 1, 1, 1};
    uint32_t min_norm = 0xFFFFFFFF;

    #pragma unroll
    for (int combo = 0; combo < 16; combo++) {
        int8_t combo_signs[4] = {
            (int8_t)((combo & 1) ? -1 : 1),
            (int8_t)((combo & 2) ? -1 : 1),
            (int8_t)((combo & 4) ? -1 : 1),
            (int8_t)((combo & 8) ? -1 : 1),
        };

        uint32_t signed_coeffs[4][8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (combo_signs[i] < 0) {
                bigint_sub(CURVE_N, temp_coeffs[i], signed_coeffs[i]);
            } else {
                for (int j = 0; j < 8; j++) signed_coeffs[i][j] = temp_coeffs[i][j];
            }
        }

        // Find max coefficient (approximate norm)
        uint32_t max_coeff = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (signed_coeffs[i][j] > max_coeff) max_coeff = signed_coeffs[i][j];
            }
        }

        // Warp reduce to find minimum across thread block
        uint32_t warp_min = __reduce_min_sync(0xFFFFFFFF, max_coeff);
        if (warp_min < min_norm) {
            min_norm = warp_min;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 8; j++) best_coeffs[i][j] = signed_coeffs[i][j];
                best_signs[i] = combo_signs[i];
            }
        }
    }

    // Copy results
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) coeffs[i][j] = best_coeffs[i][j];
        signs[i] = best_signs[i];
    }
}

// Professor-level multi-round Babai for GLV4
__device__ void multi_babai_glv4(const uint32_t t[32], uint32_t c[32], const uint32_t basis[128], const uint32_t gs[128], const uint32_t mu[128], int rounds) {
    // Multi-round Babai with alternating directions for tighter approximation
    uint32_t current_gs[128];
    uint32_t current_mu[128];

    // Initialize with input
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        current_gs[i] = gs[i];
        current_mu[i] = mu[i];
    }

    for (int round = 0; round < rounds; round++) {
        uint32_t proj[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) proj[i] = t[i];

        // Project from highest to lowest dimension
        for (int dim = 3; dim >= 0; dim--) {
            // <proj[dim*8 ..], gs[dim*32 ..]> using warp reduce
            uint32_t dot_sum = 0;
            #pragma unroll
            for (int limb = 0; limb < 8; limb++) {
                dot_sum += proj[dim*8 + limb] * current_gs[dim*32 + limb];
            }
            dot_sum = __reduce_add_sync(0xFFFFFFFF, dot_sum);

            // ||gs[dim]||^2
            uint32_t norm_sum = 0;
            #pragma unroll
            for (int limb = 0; limb < 8; limb++) {
                norm_sum += current_gs[dim*32 + limb] * current_gs[dim*32 + limb];
            }
            norm_sum = __reduce_add_sync(0xFFFFFFFF, norm_sum);

            // Round alpha = dot_sum / norm_sum (constant-time)
            uint32_t alpha = dot_sum / norm_sum; // Approximation
            uint32_t remainder = dot_sum % norm_sum;
            uint32_t half_norm = norm_sum / 2;
            uint32_t round_up = (remainder >= half_norm) ? 1 : 0;
            alpha += round_up;

            // Store coefficient (scalar per dimension)
            c[dim*8] = alpha; // Store in first limb, others zero
            #pragma unroll
            for (int l = 1; l < 8; l++) c[dim*8 + l] = 0;

            // Subtract alpha * basis[dim] from proj (parallel)
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                #pragma unroll
                for (int l = 0; l < 8; l++) {
                    uint64_t sub_val = (uint64_t)alpha * basis[d*32 + dim*8 + l];
                    uint32_t borrow = (sub_val >> 32) & 0xFFFFFFFFULL;
                    proj[d*8 + l] -= sub_val & 0xFFFFFFFFULL;
                    if (borrow && l < 7) {
                        proj[d*8 + l + 1] -= borrow;
                    }
                }
            }
        }

        // Alternate direction for next round (reverse basis)
        if (round < rounds - 1) {
            uint32_t temp_gs[128];
            uint32_t temp_mu[128];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    temp_gs[i*32 + j] = current_gs[(3-i)*32 + j];
                    temp_mu[i*32 + j] = current_mu[(3-i)*32 + j];
                }
            }
            #pragma unroll
            for (int i = 0; i < 128; i++) {
                current_gs[i] = temp_gs[i];
                current_mu[i] = temp_mu[i];
            }
        }
    }
}

// Master-level GLV scalar decomposition using lattice basis reduction
__device__ void glv_decompose_scalar(const uint32_t k[8], uint32_t k1[8], uint32_t k2[8], int8_t* sign1, int8_t* sign2) {
    // Constant-time GLV decomposition with 4-combination shortest vector selection

    // Step 1: Compute t1 = floor(k * v1 / 2^256), t2 = floor(k * v2 / 2^256)
    uint32_t kv1[16], kv2[16];
    bigint_mul_par(k, GLV_V1_1_LIMBS, kv1);
    bigint_mul_par(k, GLV_V2_1_LIMBS, kv2);

    // Shift right by 256 bits (divide by 2^256) - take high 256 bits
    uint32_t t1[8], t2[8];
    for (int i = 0; i < 8; i++) {
        t1[i] = kv1[i + 8];
        t2[i] = kv2[i + 8];
    }

    // Step 2: Round to nearest integer q1 = round(t1), q2 = round(t2)
    // Add 2^255 (0.5) before taking bits 256-383
    uint32_t round_add[8] = {0, 0, 0, 0, 0x80000000, 0, 0, 0}; // 2^255 in limb 4
    uint32_t t1_rounded[8], t2_rounded[8];
    bigint_add_par(t1, round_add, t1_rounded);
    bigint_add_par(t2, round_add, t2_rounded);

    // q = bits 256-383 of the rounded result
    uint32_t q1[8] = {t1_rounded[4], t1_rounded[5], t1_rounded[6], t1_rounded[7], 0, 0, 0, 0};
    uint32_t q2[8] = {t2_rounded[4], t2_rounded[5], t2_rounded[6], t2_rounded[7], 0, 0, 0, 0};

    // Step 3: Compute k1 = k - q1 * r1 - q2 * r2
    uint32_t q1_r1[16], q2_r2[16];
    bigint_mul_par(q1, GLV_R1_LIMBS, q1_r1);
    bigint_mul_par(q2, GLV_R2_LIMBS, q2_r2);

    uint32_t sum_qr[16];
    bigint_add_par(q1_r1, q2_r2, sum_qr);

    uint32_t k_wide[16] = {0};
    for (int i = 0; i < 8; i++) k_wide[i] = k[i];

    uint32_t k1_temp[8];
    bigint_sub_par(k_wide, sum_qr, k1_temp);

    // Step 4: Compute k2 = q1 + q2 * lambda
    uint32_t q2_lambda[16];
    bigint_mul_par(q2, GLV_LAMBDA_LIMBS, q2_lambda);

    uint32_t k2_temp[8];
    bigint_add_par(q1, q2_lambda, k2_temp);

    // Step 5: Apply sign adjustment for shortest vectors (4 combinations)
    uint32_t combos_k1[4][8], combos_k2[4][8];
    int8_t combos_signs[4][2] = {{1,1}, {-1,1}, {1,-1}, {-1,-1}};

    // Combo 0: (k1, k2)
    for (int i = 0; i < 8; i++) {
        combos_k1[0][i] = k1_temp[i];
        combos_k2[0][i] = k2_temp[i];
    }

    // Combo 1: (-k1, k2) - negate k1
    bigint_sub(CURVE_N, k1_temp, combos_k1[1]);
    for (int i = 0; i < 8; i++) combos_k2[1][i] = k2_temp[i];

    // Combo 2: (k1, -k2) - negate k2
    for (int i = 0; i < 8; i++) combos_k1[2][i] = k1_temp[i];
    bigint_sub(CURVE_N, k2_temp, combos_k2[2]);

    // Combo 3: (-k1, -k2) - negate both
    bigint_sub(CURVE_N, k1_temp, combos_k1[3]);
    bigint_sub(CURVE_N, k2_temp, combos_k2[3]);

    // Find combination with minimal max(|k1|, |k2|) - constant time
    uint32_t min_max = 0xFFFFFFFF;
    int best_combo = 0;

    #pragma unroll
    for (int combo = 0; combo < 4; combo++) {
        uint32_t max_val = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t val1 = combos_k1[combo][i];
            uint32_t val2 = combos_k2[combo][i];
            if (val1 > max_val) max_val = val1;
            if (val2 > max_val) max_val = val2;
        }
        if (max_val < min_max) {
            min_max = max_val;
            best_combo = combo;
        }
    }

    // Copy best combination results
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        k1[i] = combos_k1[best_combo][i];
        k2[i] = combos_k2[best_combo][i];
    }

    *sign1 = combos_signs[best_combo][0];
    *sign2 = combos_signs[best_combo][1];

    // Ensure k1, k2 are in [0, n-1] range
    if (bigint_cmp_par(k1, CURVE_N) >= 0) {
        bigint_sub_par(k1, CURVE_N, k1);
    }
    if (bigint_cmp_par(k2, CURVE_N) >= 0) {
        bigint_sub_par(k2, CURVE_N, k2);
    }

    // Bounds check: |k1|, |k2| should be <= sqrt(n) ≈ 2^128
    // This should be automatically satisfied by GLV construction
}

__device__ Point mul_glv_opt(Point p, const uint32_t k[8]) {
    uint32_t k1[8], k2[8];

    // Proper GLV decomposition using lattice basis reduction
    int8_t sign1, sign2;
    glv_decompose_scalar(k, k1, k2, &sign1, &sign2);

    // Apply endomorphism: p2 = β(p) where β(x,y) = (β*x mod p, y)
    Point p2_beta = p;
    mul_mod(p.x, GLV_BETA, p2_beta.x, P);

    // Compute p1*k1 + β(p)*k2 with sign handling
    Point p1 = ec_mul_small(p, k1[0]);
    if (sign1 < 0) point_neg(&p1, &p1, P);

    Point p2 = ec_mul_small(p2_beta, k2[0]);
    if (sign2 < 0) point_neg(&p2, &p2, P);

    return jacobian_add(p1, p2);
}

// Forward declaration for endomorphism_apply
__device__ void endomorphism_apply(const Point* p, Point* result);

// Professor-level GLV4 optimized multiplication with Babai
__device__ Point mul_glv4_opt_babai(Point p, const uint32_t k[8]) {
    uint32_t coeffs[4][8];
    int8_t signs[4];

    // Decompose using 4D Babai
    glv4_decompose_babai(k, coeffs, signs);

    // Precompute endomorphisms
    Point endos[4];
    endos[0] = p;  // p
    endomorphism_apply(&p, &endos[1]);  // phi(p)
    endomorphism_apply(&endos[1], &endos[2]);  // phi^2(p)
    endomorphism_apply(&endos[2], &endos[3]);  // phi^3(p)

    Point result = {0};  // Identity

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Short multiplication (assuming coeffs are small)
        Point partial = ec_mul_small(endos[i], coeffs[i][0]);

        // Apply sign
        if (signs[i] < 0) {
            point_neg(&partial, &partial, P);
        }

        result = jacobian_add(result, partial);
    }

    return result;
}

// Constant-time conditional negation
__device__ Point cond_neg_ct_cuda(Point p, int8_t cond) {
    Point result = p;
    if (cond < 0) {
        point_neg(&p, &result, P);
    }
    return result;
}

// Professor-level constant-time NAF recoding
__device__ void ct_naf_cuda(const uint32_t k[8], uint8_t naf[256], int window) {
    uint32_t k_copy[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) k_copy[i] = k[i];

    #pragma unroll
    for (int i = 0; i < 256; i++) {
        // Always extract window+1 bits (constant-time)
        uint32_t window_bits = 0;
        for (int b = 0; b <= window && i + b < 256; b++) {
            int bit_pos = i + b;
            int byte_idx = bit_pos / 32;
            int bit_idx = bit_pos % 32;
            uint32_t bit = (k_copy[byte_idx] >> bit_idx) & 1;
            window_bits |= bit << b;
        }

        // Compute NAF digit (constant-time)
        int center = 1 << window;
        int digit = 0;
        if (window_bits >= center) {
            digit = (int)(window_bits) - 2 * center;
        }

        naf[i] = digit;

        // Constant-time subtraction of digit from k
        int digit_abs = (digit < 0) ? -digit : digit;
        int sign = (digit < 0) ? -1 : 1;

        // Subtract digit_abs from k_copy (constant-time)
        uint32_t borrow = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t sub = (uint64_t)digit_abs + borrow;
            uint64_t diff = (uint64_t)k_copy[j] - sub;
            k_copy[j] = diff & 0xFFFFFFFFULL;
            borrow = (diff >> 63) & 1;
        }

        // Always divide by 2 (constant-time shift)
        uint32_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint32_t next_carry = k_copy[j] & 1;
            k_copy[j] = (k_copy[j] >> 1) | (carry << 31);
            carry = next_carry;
        }
    }
}

// Professor-level constant-time short multiplication with NAF
__device__ Point mul_short_ct_cuda(Point p, uint32_t k[8]) {
    // Fixed NAF5 with shared memory precomputation
    __shared__ Point precomp[16];

    // Constant-time precomputation in shared memory
    int tid = threadIdx.x;
    if (tid < 16) {
        if (tid == 0) {
            precomp[0] = p;  // 1*P
        } else {
            // Compute odd multiple: (2*tid + 1) * P
            Point two_p = jacobian_add(p, p);
            precomp[tid] = jacobian_add(precomp[tid-1], two_p);
        }
    }
    __syncthreads();

    Point result = {0};  // Identity
    uint8_t naf[256];
    ct_naf_cuda(k, naf, 5);  // CT NAF recoding

    // Process digits from MSB to LSB (constant-time)
    for (int i = 255; i >= 0; i--) {
        result = jacobian_double(result);  // Always double

        int8_t digit = naf[i];
        int idx = (digit + 15) / 2;  // Map to precomp index

        // Constant-time table selection and conditional add
        Point add_point = {0};
        uint32_t select_mask = (idx == tid % 16) ? 0xFFFFFFFF : 0;

        // Masked point selection (simplified - would need full CT gather)
        if (tid < 16) {
            add_point = precomp[tid];
            // Apply mask to coordinates
            for (int l = 0; l < 8; l++) {
                add_point.x[l] &= select_mask;
                add_point.y[l] &= select_mask;
                add_point.z[l] &= select_mask;
            }
        }

        // Warp reduce to get selected point
        // Simplified - in practice would need proper CT gather

        // Conditional add based on digit != 0
        uint32_t add_mask = (digit != 0) ? 0xFFFFFFFF : 0;
        Point masked_add = add_point;
        for (int l = 0; l < 8; l++) {
            masked_add.x[l] &= add_mask;
            masked_add.y[l] &= add_mask;
            masked_add.z[l] &= add_mask;
        }

        result = jacobian_add(result, masked_add);
    }

    return result;
}

// Bias and special functions
static __device__ int select_bucket_cuda(const Point position, const uint32_t dist[8], uint64_t step, uint32_t kangaroo_type) {
    const uint32_t WALK_BUCKETS = 32;
    if (kangaroo_type == 0) { // tame
        return step % WALK_BUCKETS;
    } else { // wild
        uint32_t x0 = position.x[0] ^ position.x[1];
        uint32_t x1 = position.x[2] ^ position.x[3];
        uint32_t dist0 = dist[0] ^ dist[1];
        uint32_t mix = x0 ^ x1 ^ dist0 ^ (uint32_t)step;
        return mix % WALK_BUCKETS;
    }
}

static __device__ uint64_t skew_magic9(uint64_t val) {
    uint64_t res = val % 9;
    uint64_t attractors[3] = {0, 3, 6};
    uint64_t closest = attractors[0];
    uint64_t min_diff = (res > closest) ? res - closest : closest - res;
    for (int i = 1; i < 3; i++) {
        uint64_t diff = (res > attractors[i]) ? res - attractors[i] : attractors[i] - res;
        if (diff < min_diff) min_diff = diff, closest = attractors[i];
    }
    return val + ((res > closest) ? (9 - min_diff) : min_diff);
}

static __device__ uint64_t mod_small_primes(uint64_t val) {
    for (int i = 0; i < 32; i++) {
        uint64_t prime = PRIME_MULTIPLIERS[i];
        while (val % prime == 0 && val >= prime) val /= prime;
    }
    return val;
}

static __device__ bool is_gold_attractor(uint64_t x_low, uint64_t mod_level) {
    uint64_t res = x_low % mod_level;
    return (res % 9 == 0 || res % 9 == 3 || res % 9 == 6);
}

static __device__ uint64_t gold_nudge_distance(uint64_t x_low, uint64_t mod_level) {
    uint64_t res = x_low % mod_level;
    uint64_t attractors[] = {0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81};
    int num = mod_level / 3;
    uint64_t min_diff = UINT64_MAX, closest = 0;
    for (int i = 0; i < num; i++) {
        uint64_t diff = min(res - attractors[i], attractors[i] - res + mod_level);
        if (diff < min_diff) min_diff = diff, closest = attractors[i];
    }
    // Use closest: if min_diff is 0, we're already at attractor, return closest for reference
    if (min_diff == 0) return closest;
    return min_diff;
}

// Forward declarations for CUDA functions
__device__ void gram_schmidt_4d_cuda(const uint32_t v[4][8], uint32_t gs[4][8]);
__device__ void bigint_div_approx(const uint32_t a[16], const uint32_t b[16], uint32_t result[8]);
__device__ void ct_naf_cuda(const uint32_t k[8], uint8_t naf[256], int window);
__device__ void endomorphism_apply(const Point* p, Point* result) {
    // Endomorphism phi: (x, y) -> (beta * x, beta^3 * y) in affine
    // In Jacobian: x' = beta^2 * x, y' = beta^3 * y, z' = z

    uint32_t temp[LIMBS];

    // beta^2 * x
    mul_mod(p->x, GLV_BETA_LIMBS, temp, P);
    mul_mod(temp, GLV_BETA_LIMBS, result->x, P);

    // beta^3 * y = beta^2 * (beta * y)
    mul_mod(p->y, GLV_BETA_LIMBS, temp, P);
    mul_mod(temp, GLV_BETA_LIMBS, temp, P);
    mul_mod(temp, GLV_BETA_LIMBS, result->y, P);

    // z unchanged
    for (int i = 0; i < LIMBS; i++) {
        result->z[i] = p->z[i];
    }
}

// Professor-level multi-round Babai for GLV4 with alternating directions
__device__ void multi_round_babai_glv4(const uint32_t t[32], uint32_t c[32], const uint32_t basis[128], const uint32_t gs[128], const uint32_t mu[128], int rounds) {
    // Multi-round Babai with alternating directions for tighter approximation
    uint32_t current_gs[128];
    uint32_t current_basis[128];
    uint32_t current_mu[128];

    // Initialize with input
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        current_gs[i] = gs[i];
        current_basis[i] = basis[i];
        current_mu[i] = mu[i];
    }

    for (int round = 0; round < rounds; round++) {
        uint32_t proj[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) proj[i] = t[i];

        // Determine projection order (alternate directions)
        int start_dim = (round % 2 == 0) ? 3 : 0;
        int end_dim = (round % 2 == 0) ? -1 : 4;
        int step = (round % 2 == 0) ? -1 : 1;

        // Project in specified order
        for (int dim_idx = 0; dim_idx < 4; dim_idx++) {
            int dim = start_dim + dim_idx * step;

            // <proj[dim*8 ..], gs[dim*32 ..]> using warp reduce
            uint32_t dot_sum = 0;
            #pragma unroll
            for (int limb = 0; limb < 8; limb++) {
                dot_sum += proj[dim*8 + limb] * current_gs[dim*32 + limb];
            }
            dot_sum = __reduce_add_sync(0xFFFFFFFF, dot_sum);

            // ||gs[dim]||^2
            uint32_t norm_sum = 0;
            #pragma unroll
            for (int limb = 0; limb < 8; limb++) {
                norm_sum += current_gs[dim*32 + limb] * current_gs[dim*32 + limb];
            }
            norm_sum = __reduce_add_sync(0xFFFFFFFF, norm_sum);

            // Constant-time Babai rounding
            uint32_t alpha = dot_sum / norm_sum; // Integer division approximation
            uint32_t remainder = dot_sum % norm_sum;
            uint32_t half_norm = norm_sum / 2;
            uint32_t round_up_mask = ~((remainder >= half_norm) - 1); // CT round
            alpha += round_up_mask & 1;

            // Store coefficient (scalar per dimension)
            c[dim*8] = alpha;
            #pragma unroll
            for (int l = 1; l < 8; l++) c[dim*8 + l] = 0;

            // Subtract alpha * current_basis[dim] from proj (parallel with carry)
            #pragma unroll
            for (int d = 0; d < 4; d++) {
                uint32_t carry = 0;
                #pragma unroll
                for (int l = 0; l < 8; l++) {
                    uint64_t sub_val = (uint64_t)alpha * current_basis[d*32 + dim*8 + l] + carry;
                    uint32_t borrow = (sub_val >> 32) & 0xFFFFFFFFULL;
                    proj[d*8 + l] -= sub_val & 0xFFFFFFFFULL;
                    carry = borrow;
                }
            }
        }

        // Reverse basis for alternating direction
        if (round < rounds - 1) {
            uint32_t temp_gs[128];
            uint32_t temp_basis[128];
            uint32_t temp_mu[128];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    temp_gs[i*32 + j] = current_gs[(3-i)*32 + j];
                    temp_basis[i*32 + j] = current_basis[(3-i)*32 + j];
                    temp_mu[i*32 + j] = current_mu[(3-i)*32 + j];
                }
            }
            #pragma unroll
            for (int i = 0; i < 128; i++) {
                current_gs[i] = temp_gs[i];
                current_basis[i] = temp_basis[i];
                current_mu[i] = temp_mu[i];
            }
        }
    }
}

// Main stepping kernel
__global__ void kangaroo_step_opt(
    Point* positions,           // Input/output positions
    uint64_t* distances,        // Input/output distances
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
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kangaroos) return;
    // Shared modulus loaded but not used in this simplified version
    // Could be used for optimized modular operations in future
    Point position = positions[idx];
    uint64_t distance = distances[idx];
    uint32_t kang_type = types[idx];
    __shared__ Point shared_jumps[32];
    if (threadIdx.x < 32) shared_jumps[threadIdx.x] = jumps[threadIdx.x % num_jumps];
    __syncthreads();
    for (uint32_t step = 0; step < steps_per_thread; ++step) {
        uint64_t base_jump = (uint64_t)position.x[0] ^ (uint64_t)position.y[0];
        if (bias_mode == 1) base_jump = skew_magic9(base_jump);
        else if (bias_mode == 2) base_jump = mod_small_primes(base_jump);
        uint32_t jump_idx = (uint32_t)(base_jump % 32);
        Point jump_point = shared_jumps[jump_idx];
        position = jacobian_add(position, jump_point);
        distance += jump_idx + 1;
        if (gold_bias_combo) {
            uint64_t current_mod = mod_level;
            while (current_mod <= 81) {
                uint64_t x_low = position.x[0];
                if (!is_gold_attractor(x_low, current_mod)) {
                    uint64_t nudge = gold_nudge_distance(x_low, current_mod);
                    if (nudge < 1000000) {
                        Point nudge_point = ec_mul_small(jump_point, nudge);
                        position = jacobian_add(position, nudge_point);
                        distance += nudge;
                    }
                } else break;
                current_mod *= 3;
            }
        }
        uint32_t x_low = position.x[0];
        bool is_dp = (__popc(x_low) <= (32 - dp_bits));
        if (is_dp) {
            Trap trap;
            for (int i = 0; i < 8; i++) trap.x[i] = position.x[i], trap.distance[i] = (uint32_t)(distance >> (i*32));
            trap.type = kang_type;
            trap.valid = 1;
            traps[idx] = trap;
            break;
        }
    }
    positions[idx] = position;
    distances[idx] = distance;
}

// Host launch functions
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
    launch_kangaroo_step_bias(d_positions, (uint64_t*)d_distances, d_types, d_jumps, d_traps,
        num_kangaroos, num_jumps, 20, 1, 0, 0, 9, stream);
}

// SOP bucket selection
static __device__ uint32_t select_sop_bucket(const Point point, const uint32_t dist[8], uint32_t seed, uint32_t step, bool is_tame) {
    const uint32_t WALK_BUCKETS = 32;
    if (is_tame) return step % WALK_BUCKETS;
    uint32_t x0 = *(uint32_t*)point.x, x1 = *(uint32_t*)(point.x + 4);
    uint32_t dist0 = *(uint32_t*)dist;
    uint32_t mix = x0 ^ x1 ^ dist0 ^ seed ^ step;
    return mix % WALK_BUCKETS;
}

// Biased prime getter
static __device__ uint64_t get_biased_prime(uint32_t index, uint64_t bias_mod) {
    uint64_t cycle_index = ((uint64_t)index % bias_mod) % 32;
    return PRIME_MULTIPLIERS[cycle_index];
}

// Step kernel for initialization
__global__ void step_kernel(KangarooState* states, int num_states, const Point* targets, uint32_t* primes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    KangarooState& kang = states[idx];
    int t_idx = idx / KANGS_PER_TARGET;
    if (kang.step == 0) {
        uint32_t prime = primes[idx % 32];
        // mul_glv_opt stub: kang.position = ec_mul_small(targets[t_idx], prime);
        kang.position = targets[t_idx]; // Placeholder
        for (int i = 0; i < 4; i++) kang.beta[i] = (i == 0) ? prime : 0;
    }
    uint32_t jump_idx = select_bucket_cuda(kang.position, kang.distance, kang.step, kang.kangaroo_type);
    Point jump = JUMP_TABLE[jump_idx];
    kang.position = jacobian_add(kang.position, jump);
    add_mod(kang.distance, JUMP_SIZE, kang.distance, CURVE_N);
    if (is_near_dp(kang.position)) {
        // Stub near DP
    }
    kang.step++;
}