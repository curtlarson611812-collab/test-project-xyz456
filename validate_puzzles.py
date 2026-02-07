#!/usr/bin/env python3
"""
Simple puzzle validation script using only standard library
Validates puzzle data format and basic consistency
"""

import hashlib
import binascii

def validate_hex_string(hex_str, expected_length=None, name="hex string"):
    """Validate hex string format"""
    if not hex_str:
        return False, f"{name} is empty"

    try:
        # Try to decode
        data = binascii.unhexlify(hex_str)
        if expected_length and len(data) != expected_length:
            return False, f"{name} length {len(data)} != expected {expected_length}"
        return True, data
    except Exception as e:
        return False, f"{name} invalid hex: {e}"

def validate_puzzle_line(line, line_num):
    """Validate a single puzzle line"""
    parts = line.split('|')
    if len(parts) != 9:
        return False, f"Expected 9 fields, got {len(parts)} - parts: {[p[:20] + '...' if len(p) > 20 else p for p in parts]}"

    try:
        if line_num == 7:  # Debug line 7
            print(f"DEBUG Line 7: {repr(line)}")
            print(f"DEBUG Parts: {[repr(p) for p in parts]}")

        puzzle_num = int(parts[0])
        status = parts[1]
        btc_reward = float(parts[2])
        pubkey_hex = parts[3]
        privkey_hex = parts[4].strip()
        target_address = parts[5]

        if line_num == 7:  # Debug privkey parsing
            print(f"DEBUG privkey_hex: {repr(privkey_hex)} len={len(privkey_hex) if privkey_hex else 0}")
            print(f"DEBUG privkey_hex count: {[c for c in privkey_hex]} has len {len([c for c in privkey_hex])}")
        range_min_hex = parts[6]
        range_max_hex = parts[7]
        search_space_bits = float(parts[8])

        # Validate status
        if status not in ['SOLVED', 'UNSOLVED', 'REVEALED']:
            return False, f"Invalid status: {status}"

        # For solved puzzles, validate crypto data
        if status == 'SOLVED':
            if not pubkey_hex:
                return False, "Solved puzzle missing pubkey"
            if not privkey_hex:
                return False, "Solved puzzle missing privkey"
            if not target_address:
                return False, "Solved puzzle missing address"

            # Validate hex formats
            valid, msg = validate_hex_string(pubkey_hex, 33, "pubkey")
            if not valid:
                return False, f"Pubkey: {msg}"

            valid, msg = validate_hex_string(privkey_hex, 32, "privkey")
            if not valid:
                print(f"DEBUG: privkey_hex repr: {repr(privkey_hex)}")
                print(f"DEBUG: privkey_hex bytes: {[ord(c) for c in privkey_hex]}")
                return False, f"Privkey '{privkey_hex[:20]}...' ({len(privkey_hex)} chars): {msg}"

        # Validate range hex
        if range_min_hex:
            valid, msg = validate_hex_string(range_min_hex, None, "range_min")
            if not valid:
                return False, f"Range min: {msg}"

        if range_max_hex:
            valid, msg = validate_hex_string(range_max_hex, None, "range_max")
            if not valid:
                return False, f"Range max: {msg}"

        # Validate search space
        if search_space_bits <= 0:
            return False, f"Invalid search space bits: {search_space_bits}"

        return True, f"Valid {status} puzzle #{puzzle_num}"

    except ValueError as e:
        return False, f"Parse error: {e}"

def validate_puzzles_file(filename="puzzles.txt"):
    """Validate entire puzzles file"""
    print(f"Validating puzzles from {filename}...")
    total_lines = 0
    valid_lines = 0
    solved_puzzles = 0
    errors = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                total_lines += 1

                if not line or line.startswith('#'):
                    continue

                valid, message = validate_puzzle_line(line, line_num)
                if valid:
                    valid_lines += 1
                    if "SOLVED" in message:
                        solved_puzzles += 1
                    if len(errors) < 10:  # Show first few successes
                        print(f"✅ Line {line_num}: {message}")
                else:
                    errors.append(f"Line {line_num}: {message}")
                    if len(errors) <= 5:  # Show first few errors
                        print(f"❌ Line {line_num}: {message}")

        print(f"\n📊 Validation Summary:")
        print(f"   Total lines processed: {total_lines}")
        print(f"   Valid puzzle entries: {valid_lines}")
        print(f"   Solved puzzles: {solved_puzzles}")
        print(f"   Errors found: {len(errors)}")

        if errors:
            print(f"\n🚨 First {min(5, len(errors))} errors:")
            for error in errors[:5]:
                print(f"   {error}")

        success_rate = valid_lines / max(1, total_lines) * 100
        print(f"   Success rate: {success_rate:.1f}%")

        return len(errors) == 0

    except FileNotFoundError:
        print(f"❌ Error: {filename} not found")
        return False

if __name__ == "__main__":
    success = validate_puzzles_file()
    exit(0 if success else 1)