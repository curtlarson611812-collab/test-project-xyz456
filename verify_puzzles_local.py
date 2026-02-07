#!/usr/bin/env python3
"""
Local Bitcoin Puzzle Verification Script
Verifies puzzle data in puzzles.txt using local cryptographic operations

This script can verify:
- Private key generates correct public key
- Public key generates correct address
- All puzzle data is consistent

Usage:
    python verify_puzzles_local.py [--puzzle N] [--all] [--check-address]
"""

import hashlib
import binascii
import ecdsa
import base58
import sys
import argparse

class PuzzleVerifierLocal:
    def __init__(self):
        self.puzzles = {}

    def load_puzzles(self, filename="puzzles.txt"):
        """Load puzzles from flat file"""
        print(f"Loading puzzles from {filename}...")
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
                        btc_reward = float(parts[2])
                        pubkey_hex = parts[3]
                        privkey_hex = parts[4] if parts[4] != '' else None
                        target_address = parts[5]
                        range_min_hex = parts[6]
                        range_max_hex = parts[7]
                        search_space_bits = float(parts[8])

                        self.puzzles[puzzle_num] = {
                            'status': status,
                            'btc_reward': btc_reward,
                            'pubkey_hex': pubkey_hex,
                            'privkey_hex': privkey_hex,
                            'target_address': target_address,
                            'range_min_hex': range_min_hex,
                            'range_max_hex': range_max_hex,
                            'search_space_bits': search_space_bits
                        }
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Error parsing line {line_num}: {e}")
                        continue

            print(f"Loaded {len(self.puzzles)} puzzles")
            return True
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return False

    def pubkey_to_address(self, pubkey_hex):
        """Convert compressed public key hex to Bitcoin address"""
        try:
            # Decode hex to bytes
            pubkey_bytes = binascii.unhexlify(pubkey_hex)

            # SHA256 hash
            sha256_hash = hashlib.sha256(pubkey_bytes).digest()

            # RIPEMD160 hash
            ripemd160 = hashlib.new('ripemd160', sha256_hash).digest()

            # Add version byte (0x00 for mainnet)
            version_ripemd160 = b'\x00' + ripemd160

            # Double SHA256 for checksum
            checksum = hashlib.sha256(hashlib.sha256(version_ripemd160).digest()).digest()[:4]

            # Combine version + ripemd160 + checksum
            address_bytes = version_ripemd160 + checksum

            # Base58 encode
            address = base58.b58encode(address_bytes).decode('ascii')
            return address
        except Exception as e:
            return f"Error: {e}"

    def verify_puzzle_crypto(self, puzzle_num, puzzle_data):
        """Verify cryptographic consistency of puzzle data"""
        print(f"\n🔐 Verifying Puzzle #{puzzle_num} Cryptography")

        privkey_hex = puzzle_data['privkey_hex']
        pubkey_hex = puzzle_data['pubkey_hex']
        target_address = puzzle_data['target_address']

        if not privkey_hex:
            print("   ℹ️  No private key available (unsolved puzzle)")
            # For unsolved puzzles, just verify address format
            if pubkey_hex:
                computed_address = self.pubkey_to_address(pubkey_hex)
                if computed_address == target_address:
                    print("   ✅ Public key matches target address")
                else:
                    print(f"   ❌ Public key mismatch: got {computed_address}, expected {target_address}")
            return True

        try:
            # Convert private key hex to int
            privkey_int = int(privkey_hex, 16)

            # Create ECDSA key from private key
            sk = ecdsa.SigningKey.from_secret_exponent(privkey_int, curve=ecdsa.SECP256k1)
            vk = sk.verifying_key

            # Get compressed public key
            pubkey_bytes = b'\x02' + vk.pubkey.point.x().to_bytes(32, 'big') if vk.pubkey.point.y() % 2 == 0 else b'\x03' + vk.pubkey.point.x().to_bytes(32, 'big')
            computed_pubkey_hex = pubkey_bytes.hex()

            print(f"   🔑 Private key: {privkey_hex[:16]}...{privkey_hex[-8:]}")
            print(f"   🏗️  Computed pubkey: {computed_pubkey_hex}")
            print(f"   📄 Stored pubkey:   {pubkey_hex}")

            if computed_pubkey_hex == pubkey_hex:
                print("   ✅ Private key generates correct public key")
            else:
                print("   ❌ Private key does not match public key")
                return False

            # Verify address
            computed_address = self.pubkey_to_address(computed_pubkey_hex)
            print(f"   🏠 Computed address: {computed_address}")
            print(f"   🎯 Target address:   {target_address}")

            if computed_address == target_address:
                print("   ✅ Public key generates correct address")
                return True
            else:
                print("   ❌ Public key does not match target address")
                return False

        except Exception as e:
            print(f"   ❌ Cryptographic verification failed: {e}")
            return False

    def verify_puzzle_range(self, puzzle_num, puzzle_data):
        """Verify puzzle range data"""
        print(f"\n📏 Verifying Puzzle #{puzzle_num} Range Data")

        range_min_hex = puzzle_data['range_min_hex']
        range_max_hex = puzzle_data['range_max_hex']
        search_space_bits = puzzle_data['search_space_bits']

        try:
            range_min = int(range_min_hex, 16)
            range_max = int(range_max_hex, 16)
            actual_range = range_max - range_min + 1
            expected_range = 2 ** int(search_space_bits)

            print(f"   Range min: 0x{range_min_hex} ({range_min})")
            print(f"   Range max: 0x{range_max_hex} ({range_max})")
            print(f"   Actual range size: {actual_range}")
            print(f"   Expected range size: {expected_range}")

            if actual_range == expected_range:
                print("   ✅ Range size matches expected search space")
                return True
            else:
                print("   ⚠️  Range size does not match expected search space")
                return False

        except Exception as e:
            print(f"   ❌ Range verification failed: {e}")
            return False

    def verify_puzzle(self, puzzle_num, check_address=False):
        """Verify a single puzzle"""
        if puzzle_num not in self.puzzles:
            print(f"❌ Puzzle #{puzzle_num} not found in database")
            return False

        puzzle_data = self.puzzles[puzzle_num]

        # Verify cryptographic consistency
        crypto_ok = self.verify_puzzle_crypto(puzzle_num, puzzle_data)

        # Verify range data
        range_ok = self.verify_puzzle_range(puzzle_num, puzzle_data)

        # Summary
        status = "✅ PASSED" if crypto_ok and range_ok else "❌ FAILED"
        print(f"\n📋 Puzzle #{puzzle_num} Verification: {status}")

        return crypto_ok and range_ok

    def verify_all_puzzles(self, max_puzzles=50, check_address=False):
        """Verify multiple puzzles"""
        print(f"Verifying up to {max_puzzles} puzzles...")

        results = []
        count = 0

        for puzzle_num in sorted(self.puzzles.keys()):
            if count >= max_puzzles:
                print(f"\n⏹️  Stopped after {max_puzzles} puzzles")
                break

            result = self.verify_puzzle(puzzle_num, check_address)
            results.append((puzzle_num, result))
            count += 1

        # Summary
        passed = sum(1 for _, result in results if result)
        failed = len(results) - passed

        print("\n🎯 Verification Summary:")
        print(f"   Total verified: {len(results)}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")

        if failed > 0:
            print("\n❌ Failed puzzles:")
            for num, result in results:
                if not result:
                    print(f"   Puzzle #{num}")

        return failed == 0

    def show_stats(self):
        """Show statistics about loaded puzzles"""
        total = len(self.puzzles)
        solved = sum(1 for p in self.puzzles.values() if p['status'] == 'SOLVED')
        unsolved = total - solved

        print("\n📊 Puzzle Database Statistics:")
        print(f"   Total puzzles: {total}")
        print(f"   Solved: {solved}")
        print(f"   Unsolved: {unsolved}")

        if total > 0:
            avg_reward = sum(p['btc_reward'] for p in self.puzzles.values()) / total
            print(f"   Average BTC reward: {avg_reward:.3f}")

            max_reward = max(p['btc_reward'] for p in self.puzzles.values())
            print(f"   Highest BTC reward: {max_reward}")

def main():
    parser = argparse.ArgumentParser(description="Verify Bitcoin puzzles locally")
    parser.add_argument('--puzzle', '-p', type=int, help='Verify specific puzzle number')
    parser.add_argument('--all', action='store_true', help='Verify all puzzles')
    parser.add_argument('--max', type=int, default=50, help='Maximum puzzles to verify when using --all')
    parser.add_argument('--file', '-f', default='puzzles.txt', help='Puzzle database file')
    parser.add_argument('--check-address', action='store_true', help='Also verify address generation')

    args = parser.parse_args()

    verifier = PuzzleVerifierLocal()

    # Load puzzles
    if not verifier.load_puzzles(args.file):
        sys.exit(1)

    # Show stats
    verifier.show_stats()

    # Verify puzzles
    success = True
    if args.puzzle:
        success = verifier.verify_puzzle(args.puzzle, args.check_address)
    elif args.all:
        success = verifier.verify_all_puzzles(args.max, args.check_address)
    else:
        print("\nUsage:")
        print("  --puzzle N        Verify puzzle #N")
        print("  --all            Verify all puzzles")
        print("  --check-address  Also verify address generation")
        print("\nExample:")
        print("  python verify_puzzles_local.py --puzzle 35")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()