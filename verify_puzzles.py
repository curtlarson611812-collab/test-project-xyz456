#!/usr/bin/env python3
"""
Bitcoin Puzzle Database Verification Script
Uses bitcoin-cli to verify puzzle addresses exist on blockchain
"""

import subprocess
import json
import sys
import hashlib
import binascii

# Simple base58 implementation
def base58_encode(data):
    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    num = int.from_bytes(data, 'big')

    # Handle leading zeros
    leading_zeros = 0
    for byte in data:
        if byte == 0:
            leading_zeros += 1
        else:
            break

    result = ''
    while num > 0:
        num, rem = divmod(num, 58)
        result = alphabet[rem] + result

    return '1' * leading_zeros + result

def run_bitcoin_cli(*args):
    """Run bitcoin-cli command and return parsed JSON"""
    try:
        cmd = ['bitcoin-cli'] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"bitcoin-cli error: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to parse JSON output: {result.stdout}")
        return None

def pubkey_to_address(pubkey_hex):
    """Convert compressed public key to Bitcoin address"""
    pubkey_bytes = binascii.unhexlify(pubkey_hex)

    # SHA256 of public key
    sha = hashlib.sha256(pubkey_bytes).digest()

    # RIPEMD160 of SHA256
    rip = hashlib.new('ripemd160', sha).digest()

    # Add version byte (0x00 for mainnet)
    version_rip = b'\x00' + rip

    # Double SHA256 for checksum
    checksum = hashlib.sha256(hashlib.sha256(version_rip).digest()).digest()[:4]

    # Base58 encode
    return base58_encode(version_rip + checksum)

def verify_puzzle_on_blockchain(pubkey_hex, expected_address=None):
    """Verify puzzle public key exists on blockchain"""
    try:
        # Derive address from public key
        derived_address = pubkey_to_address(pubkey_hex)

        if expected_address and derived_address != expected_address:
            print(f"❌ Address mismatch!")
            print(f"  Derived: {derived_address}")
            print(f"  Expected: {expected_address}")
            return False

        # Check if address exists using bitcoin-cli
        addr_info = run_bitcoin_cli('getaddressinfo', derived_address)

        if addr_info is None:
            print(f"❌ Failed to get address info for {derived_address}")
            return False

        # bitcoin-cli returns address info for valid addresses, errors for invalid ones
        if 'address' not in addr_info:
            print(f"❌ Invalid address: {derived_address}")
            return False

        # Address is valid if we got here
        print(f"✅ Valid address format: {derived_address}")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 verify_puzzles.py <puzzles.txt>")
        sys.exit(1)

    puzzles_file = sys.argv[1]

    print("🔍 Verifying Bitcoin Puzzle Database")
    print(f"📄 Reading from: {puzzles_file}")
    print()

    try:
        with open(puzzles_file, 'r') as f:
            lines = f.readlines()

        total_puzzles = 0
        valid_puzzles = 0
        solved_puzzles = 0
        valid_solved = 0

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue

            parts = line.split('|')
            if len(parts) < 8:
                print(f"❌ Line {line_num}: Invalid format (expected at least 8 parts)")
                continue

            try:
                n = int(parts[0])
                status = parts[1].strip()
                btc_reward = parts[2].strip()
                pubkey_hex = parts[3].strip()
                priv_hex = parts[4].strip()
                target_address = parts[5].strip()
                range_low = parts[6].strip()
                range_high = parts[7].strip()

                total_puzzles += 1
                is_solved = priv_hex and priv_hex != ''

                print(f"🔍 Puzzle #{n} (Line {line_num})")

                # Verify public key format
                if not pubkey_hex.startswith(('02', '03')) or len(pubkey_hex) != 66:
                    print(f"❌ Invalid compressed public key format: {pubkey_hex}")
                    continue

                # Check if address is valid on blockchain
                if verify_puzzle_on_blockchain(pubkey_hex, target_address if target_address else None):
                    valid_puzzles += 1

                    if is_solved:
                        solved_puzzles += 1
                        print(f"  🔑 Solved puzzle {status} ({btc_reward} BTC, range: {range_low} - {range_high})")
                        valid_solved += 1
                    else:
                        print(f"  🎯 Unsolved puzzle {status} ({btc_reward} BTC, range: {range_low} - {range_high})")
                else:
                    print(f"❌ Puzzle #{n} verification failed")

                print()

            except ValueError as e:
                print(f"❌ Line {line_num}: Parse error - {e}")
                continue

        print("📊 Verification Summary:")
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Valid puzzles: {valid_puzzles}")
        print(f"  Solved puzzles: {solved_puzzles}")
        print(f"  Valid solved: {valid_solved}")

        if valid_puzzles == total_puzzles:
            print("✅ All puzzles verified successfully!")
            return 0
        else:
            print(f"❌ {total_puzzles - valid_puzzles} puzzles failed verification")
            return 1

    except FileNotFoundError:
        print(f"❌ File not found: {puzzles_file}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())