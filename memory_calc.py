#!/usr/bin/env python3

def calculate_memory(herd_size, dp_bits):
    # KangarooState memory estimate
    point_size = 32 * 3  # x,y,z as [u64;4]
    bigint_size = 32     # BigInt256
    arrays_size = 32 * 2  # alpha, beta as [u64;4]
    other_size = 24       # bools + integers
    kangaroo_size = point_size + bigint_size + arrays_size + other_size
    
    # DP table estimate
    dp_entries = 2 ** dp_bits
    dp_entry_size = 64  # rough estimate per DP entry
    
    kangaroo_memory = herd_size * kangaroo_size
    dp_memory = dp_entries * dp_entry_size
    total_gb = (kangaroo_memory + dp_memory) / (1024**3)
    
    print(f'Herd Size: {herd_size:,} kangaroos')
    print(f'DP Bits: {dp_bits} (2^{dp_bits} = {dp_entries:,} entries)')
    print(f'Kangaroo memory: {kangaroo_memory/(1024**3):.1f} GB')
    print(f'DP table memory: {dp_memory/(1024**3):.1f} GB')
    print(f'Total estimated: {total_gb:.1f} GB')
    
    if total_gb > 64:
        recommended = int(64 * (1024**3) / kangaroo_size)
        print(f'⚠️  TOO BIG! Recommended herd_size: {recommended:,}')
    else:
        print('✅ Memory usage looks reasonable')

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        calculate_memory(int(sys.argv[1]), int(sys.argv[2]))
    else:
        calculate_memory(750000000, 26)  # Your current settings
