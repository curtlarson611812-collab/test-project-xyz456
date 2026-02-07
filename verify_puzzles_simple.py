#!/usr/bin/env python3
"""
Simple puzzle verification script using only standard library
Verifies that private keys generate correct public keys and addresses
"""

import hashlib
import binascii
import ecdsa
import base58

def pubkey_to_address(pubkey_hex):
    """Convert compressed pubkey to Bitcoin address"""
    # Add version byte (0x00 for mainnet)
    pubkey_bytes = binascii.unhexlify(pubkey_hex)
    sha256_hash = hashlib.sha256(pubkey_bytes).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()

    # Add version byte for mainnet P2PKH
    version_ripemd = b'\x00' + ripemd160_hash

    # Double SHA256 for checksum
    checksum = hashlib.sha256(hashlib.sha256(version_ripemd).digest()).digest()[:4]

    # Base58Check encode
    address_bytes = version_ripemd + checksum
    return base58.b58encode(address_bytes).decode('ascii')

def verify_puzzle_crypto(puzzle_num, pubkey_hex, privkey_hex, target_address):
    """Verify cryptographic consistency of a puzzle"""
    try:
        # Parse private key
        privkey_int = int(privkey_hex, 16)
        privkey_bytes = privkey_int.to_bytes(32, byteorder='big')

        # Generate public key using ecdsa
        sk = ecdsa.SigningKey.from_secret_exponent(privkey_int, curve=ecdsa.SECP256k1)
        vk = sk.verifying_key
        pubkey_compressed = b'\x02' + vk.to_string()[:32] if vk.to_string()[63] % 2 == 0 else b'\x03' + vk.to_string()[:32]
        computed_pubkey_hex = pubkey_compressed.hex()

        # Verify pubkey matches
        if computed_pubkey_hex != pubkey_hex:
            print(f"❌ Puzzle {puzzle_num}: Pubkey mismatch!")
            print(f"   Expected: {pubkey_hex}")
            print(f"   Computed: {computed_pubkey_hex}")
            return False

        # Generate address
        computed_address = pubkey_to_address(pubkey_hex)

        # Verify address matches
        if computed_address != target_address:
            print(f"❌ Puzzle {puzzle_num}: Address mismatch!")
            print(f"   Expected: {target_address}")
            print(f"   Computed: {computed_address}")
            return False

        print(f"✅ Puzzle {puzzle_num}: Cryptography verified")
        return True

    except Exception as e:
        print(f"❌ Puzzle {puzzle_num}: Verification failed - {e}")
        return False

def load_and_verify_puzzles(filename="puzzles.txt"):
    """Load puzzles and verify solved ones"""
    print(f"Loading puzzles from {filename}...")
    solved_count = 0
    verified_count = 0

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('|')
                if len(parts) < 9:
                    continue

                try:
                    puzzle_num = int(parts[0])
                    status = parts[1]
                    btc_reward = float(parts[2]) if parts[2] else 0.0
                    pubkey_hex = parts[3]
                    privkey_hex = parts[4] if len(parts) > 4 and parts[4] != '' else None
                    target_address = parts[5] if len(parts) > 5 else ''

                    if status == 'SOLVED' and pubkey_hex and privkey_hex and target_address:
                        solved_count += 1
                        print(f"\n🔍 Verifying Puzzle #{puzzle_num} (BTC: {btc_reward})")
                        print(f"   Address: {target_address}")

                        if verify_puzzle_crypto(puzzle_num, pubkey_hex, privkey_hex, target_address):
                            verified_count += 1
                        else:
                            print(f"   ❌ Puzzle {puzzle_num} FAILED verification")

                except ValueError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return

    print(f"\n📊 Verification Summary:")
    print(f"   Solved puzzles found: {solved_count}")
    print(f"   Successfully verified: {verified_count}")
    print(f"   Success rate: {verified_count/solved_count*100:.1f}%" if solved_count > 0 else "   No solved puzzles to verify")

if __name__ == "__main__":
    load_and_verify_puzzles()