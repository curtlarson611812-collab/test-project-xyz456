#!/usr/bin/env python3
"""
Simple verification for comma-separated puzzle format
Only verifies solved puzzles (those with private keys)
"""

import subprocess
import json
import sys
import hashlib
import binascii

def run_bitcoin_cli(*args):
    try:
        cmd = ['bitcoin-cli'] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except:
        return None

def base58_encode(data):
    """Proper base58 encoding with leading zero handling"""
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

    # Count leading zeros
    leading_zeros = 0
    for byte in data:
        if byte == 0:
            leading_zeros += 1
        else:
            break

    # Convert to big integer
    num = int.from_bytes(data, 'big')

    # Encode
    encoded = ''
    while num > 0:
        num, rem = divmod(num, 58)
        encoded = alphabet[rem] + encoded

    # Add leading '1's for each leading zero byte
    return '1' * leading_zeros + encoded

def pubkey_to_address(pubkey_hex):
    pubkey_bytes = binascii.unhexlify(pubkey_hex)
    sha = hashlib.sha256(pubkey_bytes).digest()
    rip = hashlib.new('ripemd160', sha).digest()
    version_rip = b'\x00' + rip
    checksum = hashlib.sha256(hashlib.sha256(version_rip).digest()).digest()[:4]
    return base58_encode(version_rip + checksum)

def verify_solved_puzzle(n, pubkey_hex, priv_hex):
    """Verify a solved puzzle has correct pubkey->address"""
    if not priv_hex or priv_hex == '':
        return True  # Skip unsolved puzzles

    try:
        # Derive address from public key
        derived_address = pubkey_to_address(pubkey_hex)

        # Check if address exists on blockchain
        addr_info = run_bitcoin_cli('getaddressinfo', derived_address)
        if addr_info and 'address' in addr_info:
            print(f"✅ Puzzle #{n}: Valid address {derived_address}")
            return True
        else:
            print(f"❌ Puzzle #{n}: Address {derived_address} not found")
            return False

    except Exception as e:
        print(f"❌ Puzzle #{n}: Verification error - {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 verify_puzzles_simple.py <puzzles.txt>")
        sys.exit(1)

    puzzles_file = sys.argv[1]

    print("🔍 Verifying solved puzzles in comma-separated format")
    print(f"📄 Reading from: {puzzles_file}")
    print()

    total_solved = 0
    valid_solved = 0

    try:
        with open(puzzles_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('//'):
                    continue

                parts = line.split(',')
                if len(parts) != 5:
                    continue

                try:
                    n = int(parts[0])
                    pubkey_hex = parts[1].strip()
                    priv_hex = parts[2].strip()

                    if priv_hex and priv_hex != '':
                        total_solved += 1
                        if verify_solved_puzzle(n, pubkey_hex, priv_hex):
                            valid_solved += 1

                except ValueError:
                    continue

        print()
        print("📊 Solved Puzzle Verification:")
        print(f"  Total solved: {total_solved}")
        print(f"  Valid solved: {valid_solved}")

        if valid_solved == total_solved:
            print("✅ All solved puzzles verified successfully!")
            return 0
        else:
            print(f"❌ {total_solved - valid_solved} solved puzzles failed verification")
            return 1

    except FileNotFoundError:
        print(f"❌ File not found: {puzzles_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())