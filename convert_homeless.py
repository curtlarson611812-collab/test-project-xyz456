#!/usr/bin/env python3
"""
Convert homeless.txt to SpeedBitCrack format with all required fields
"""

def get_puzzle_data(n):
    """Get the missing fields for each puzzle"""
    # Format: (status, btc_reward, range_min_hex, range_max_hex, search_space_bits)

    if n <= 66:  # Solved puzzles
        status = "SOLVED"
        btc_reward = f"{n/100:.2f}"
    elif n <= 160:  # Revealed but unsolved
        status = "REVEALED"
        btc_reward = f"{n/100:.2f}"
    else:
        status = "UNSOLVED"
        btc_reward = f"{n/100:.2f}"

    # Calculate ranges based on puzzle number
    if n == 1:
        range_min = "1"
        range_max = "1"
        search_bits = "1.0"
    elif n == 2:
        range_min = "2"
        range_max = "3"
        search_bits = "2.0"
    elif n == 3:
        range_min = "4"
        range_max = "7"
        search_bits = "3.0"
    else:
        # For puzzles 4+, calculate based on pattern
        # Each puzzle covers 2^(n-1) to 2^n - 1
        range_min = hex(2**(n-1))[2:]  # Remove 0x prefix
        range_max = hex(2**n - 1)[2:]
        search_bits = f"{n}.0"

    return status, btc_reward, range_min, range_max, search_bits

def main():
    with open('homeless.txt', 'r') as f:
        lines = f.readlines()

    with open('puzzles.txt', 'w') as f:
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                f.write(line + '\n')
                continue

            # Parse the homeless.txt format
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 6:
                f.write(line + '\n')
                continue

            n = int(parts[0])
            priv_key = parts[1]
            address = parts[2]
            upper_range = parts[3]
            pub_key = parts[4]
            date = parts[5] if len(parts) > 5 else ""

            # Get the missing fields
            status, btc_reward, range_min, range_max, search_bits = get_puzzle_data(n)

            # Write in SpeedBitCrack format (9 fields)
            f.write(f"{n}|{status}|{btc_reward}|{pub_key}|{priv_key}|{address}|{range_min}|{range_max}|{search_bits}\n")

if __name__ == "__main__":
    main()