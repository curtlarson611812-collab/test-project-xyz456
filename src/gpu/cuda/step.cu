// step.cu - Optimized kangaroo stepping kernel with shared memory
// Implements Pollard's rho/kangaroo algorithm steps on GPU
// Optimizations: Shared memory for jump points, coalesced memory access
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h> // For printf

#define KANGS_PER_TARGET 4096
#define DEBUG 1

// Point structure for elliptic curve points (Jacobian coordinates)
struct Point {
    uint32_t x[8]; // X coordinate (256-bit)
    uint32_t y[8]; // Y coordinate (256-bit)
    uint32_t z[8]; // Z coordinate (256-bit)
};

// Kangaroo state structure (exact match to Rust)
struct KangarooState {
    Point position;
    uint32_t distance[8];
    uint32_t alpha[4];
    uint32_t beta[4];
    bool is_tame;
    bool is_dp;
    uint64_t id;
    uint64_t step;
    uint32_t kangaroo_type; // 0 = tame, 1 = wild (renamed from 'type')
};

// Trap structure for collision detection
struct Trap {
    uint32_t x[8]; // X coordinate of trap point
    uint32_t distance[8]; // Distance when trapped
    uint32_t type; // Kangaroo type
    uint32_t valid; // 1 if trap is valid
};

// secp256k1 order (N) as uint32_t[8]
__constant__ uint32_t CURVE_N[8] = {
    0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B,
    0xBFD25E8C, 0xD0364141, 0x00000000, 0x00000000
};

// Barrett mu for secp256k1 order (floor(2^512 / N)) as uint32_t[16]
__constant__ uint32_t MU_N[16] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
    0x45512319, 0x50B75FC4, 0x402DA173, 0x2FBC146B,
    0x09DDA963, 0x02FDB94D, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000
};

// Mean jump distance as uint32_t[8]
__constant__ uint32_t JUMP_SIZE[8] = {1024, 0, 0, 0, 0, 0, 0, 0};

// Jump table for kangaroo hops (precomputed points)
__constant__ Point JUMP_TABLE[256];

// secp256k1 prime modulus (P) as uint32_t[8]
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

static __device__ bool point_equal(const Point p1, const Point p2) {
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

static __device__ void mul_mod(const uint32_t* a, const uint32_t* b, uint32_t* c, const uint32_t* m) {
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
static __device__ Point jacobian_double(Point p1) {
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

static __device__ Point jacobian_add(const Point p1, const Point p2) {
    return ec_add(p1, p2); // Alias to full add
}

static __device__ Point ec_mul_small(Point p, uint64_t scalar) {
    Point result; set_infinity(&result);
    for (int bit = 0; bit < 64 && scalar > 0; bit++) {
        if (scalar & 1) result = jacobian_add(result, p);
        p = jacobian_double(p);
        scalar >>= 1;
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