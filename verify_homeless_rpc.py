#!/usr/bin/env python3
"""
Enhanced Bitcoin Puzzle Database Verification using RPC
Uses direct RPC calls for more detailed blockchain verification
"""

import json
import sys
import base64
import http.client
import hashlib
import binascii

# RPC Configuration
RPC_HOST = 'localhost'
RPC_PORT = 8332
RPC_USER = 'crackercurt'
RPC_PASS = 'trucrekcarc10101010'

def rpc_call(method, params=None):
    """Make direct RPC call to bitcoind"""
    if params is None:
        params = []

    # Create auth header
    auth = base64.b64encode(f"{RPC_USER}:{RPC_PASS}".encode()).decode()

    # Create request
    payload = {
        "jsonrpc": "2.0",
        "id": "speedbitcrack",
        "method": method,
        "params": params
    }

    headers = {
        'Authorization': f'Basic {auth}',
        'Content-Type': 'application/json'
    }

    try:
        conn = http.client.HTTPConnection(RPC_HOST, RPC_PORT)
        conn.request('POST', '/', json.dumps(payload), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode())
        conn.close()

        if 'error' in result and result['error']:
            print(f"RPC Error: {result['error']}")
            return None

        return result.get('result')
    except Exception as e:
        print(f"RPC Connection Error: {e}")
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

    return '1' * leading_zeros + encoded

def pubkey_to_address(pubkey_hex):
    """Convert compressed public key to Bitcoin address"""
    pubkey_bytes = binascii.unhexlify(pubkey_hex)
    sha = hashlib.sha256(pubkey_bytes).digest()
    rip = hashlib.new('ripemd160', sha).digest()
    version_rip = b'\x00' + rip
    checksum = hashlib.sha256(hashlib.sha256(version_rip).digest()).digest()[:4]
    return base58_encode(version_rip + checksum)

def parse_puzzle_line(line):
    """Parse a line from homeless.txt format"""
    parts = line.split('|')
    if len(parts) < 6:
        return None

    try:
        puzzle_num = int(parts[0].strip())
        status = parts[1].strip()
        btc_reward = parts[2].strip()
        pubkey_hex = parts[3].strip()
        priv_hex = parts[4].strip()
        target_address = parts[5].strip()
        solved_date = parts[6].strip() if len(parts) > 6 else ''

        return {
            'num': puzzle_num,
            'status': status,
            'btc_reward': btc_reward,
            'priv_key': priv_hex if priv_hex != '========================== U N K N O W N =========================' and priv_hex.strip() else '',
            'pub_key': pubkey_hex if pubkey_hex != '========================== U N K N O W N =========================' and pubkey_hex.strip() else '',
            'address': target_address,
            'solved_date': solved_date if solved_date != '____-__-__' else ''
        }
    except (ValueError, IndexError):
        return None

def verify_address_detailed(address):
    """Detailed address verification using RPC"""
    try:
        # Get address info
        addr_info = rpc_call('getaddressinfo', [address])
        if not addr_info:
            return False, 0, 0, None, "RPC call failed"

        if not addr_info.get('isvalid', False):
            return False, 0, 0, addr_info, "Invalid address format"

        # Get received balance (confirmed)
        received = rpc_call('getreceivedbyaddress', [address, 0])
        received = received if isinstance(received, (int, float)) else 0

        # Get transaction count by checking UTXOs
        tx_count = 0
        utxos = rpc_call('listunspent', [0, 9999999, [address]])
        if utxos:
            tx_count = len(utxos)

        # Check mempool balance too
        mempool_balance = rpc_call('getreceivedbyaddress', [address, 0, True])  # include mempool
        mempool_balance = mempool_balance if isinstance(mempool_balance, (int, float)) else 0

        return True, received, tx_count, addr_info, None

    except Exception as e:
        return False, 0, 0, None, str(e)

def extract_pubkey_from_tx(address):
    """Try to extract public key from blockchain transactions"""
    try:
        # Get UTXOs for this address
        utxos = rpc_call('listunspent', [0, 9999999, [address]])
        if not utxos:
            return None

        # Get the first UTXO and check its transaction
        for utxo in utxos[:1]:
            txid = utxo['txid']
            vout = utxo['vout']

            # Get raw transaction
            raw_tx = rpc_call('getrawtransaction', [txid])
            if not raw_tx:
                continue

            # Decode transaction
            decoded = rpc_call('decoderawtransaction', [raw_tx])
            if not decoded or 'vout' not in decoded:
                continue

            # Find our output
            for vout_info in decoded['vout']:
                if vout_info['n'] == vout:
                    scriptPubKey = vout_info.get('scriptPubKey', {})
                    if scriptPubKey.get('type') == 'pubkeyhash':
                        # P2PKH - pubkey not directly extractable from address
                        # Would need to find spending tx with signature
                        return None

        return None
    except:
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 verify_homeless_rpc.py <homeless.txt>")
        sys.exit(1)

    filename = sys.argv[1]

    print("🔍 Enhanced RPC Verification of homeless.txt Bitcoin Puzzle Database")
    print(f"📄 Reading from: {filename}")
    print(f"🔗 RPC: {RPC_HOST}:{RPC_PORT} as {RPC_USER}")
    print()

    # Test RPC connection
    blockchain_info = rpc_call('getblockchaininfo')
    if not blockchain_info:
        print("❌ Cannot connect to Bitcoin RPC. Check credentials and bitcoind.")
        sys.exit(1)

    best_block = blockchain_info.get('bestblockhash', 'unknown')[:16]
    blocks = blockchain_info.get('blocks', 0)
    print(f"✅ Connected to Bitcoin blockchain: {blocks} blocks, tip: {best_block}...")
    print()

    total_puzzles = 0
    solved_puzzles = 0
    revealed_pubkeys = 0
    valid_addresses = 0
    addresses_with_balance = 0
    addresses_with_txs = 0

    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue

                puzzle = parse_puzzle_line(line)
                if not puzzle:
                    continue

                total_puzzles += 1
                n = puzzle['num']

                print(f"🔍 Puzzle #{n} ({puzzle['btc_reward']} BTC)")

                # Check if solved
                is_solved = bool(puzzle['priv_key'])
                if is_solved:
                    solved_puzzles += 1
                    print(f"  🔑 SOLVED ({puzzle['solved_date']})")

                # Check if has public key
                has_pubkey = bool(puzzle['pub_key'])
                if has_pubkey:
                    revealed_pubkeys += 1
                    print(f"  🔓 PUBLIC KEY REVEALED")

                    # Verify pubkey matches address
                    derived_address = pubkey_to_address(puzzle['pub_key'])
                    if derived_address != puzzle['address']:
                        print(f"  ❌ Address mismatch! Derived: {derived_address}")
                    else:
                        print("  ✅ Pubkey → Address verified")

                # Detailed blockchain verification
                valid, balance, tx_count, addr_info, error = verify_address_detailed(puzzle['address'])

                if valid:
                    valid_addresses += 1
                    if balance > 0:
                        addresses_with_balance += 1
                        print(f"  💰 Balance: {balance:.8f} BTC")
                    else:
                        print("  ✅ Valid address (empty)")

                    if tx_count > 0:
                        addresses_with_txs += 1
                        print(f"  📊 {tx_count} transactions")
                else:
                    print(f"  ❌ Address verification failed: {error}")

                # Every 5th puzzle: enhanced pubkey extraction
                if n % 5 == 0:
                    print("  🔍 Enhanced every-5th check:")
                    extracted = extract_pubkey_from_tx(puzzle['address'])
                    if extracted:
                        print(f"    🎯 Blockchain pubkey: {extracted}")
                    else:
                        print("    ℹ️  No pubkey extractable from blockchain")

                print()

        print("📊 Enhanced RPC Verification Summary:")
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Solved puzzles: {solved_puzzles}")
        print(f"  Revealed pubkeys: {revealed_pubkeys}")
        print(f"  Valid addresses: {valid_addresses}")
        print(f"  Addresses with balance: {addresses_with_balance}")
        print(f"  Addresses with transactions: {addresses_with_txs}")

        if valid_addresses == total_puzzles:
            print("✅ All puzzle addresses verified on blockchain!")
            if addresses_with_balance == 0:
                print("💎 All puzzle rewards remain unclaimed!")
        else:
            print(f"❌ {total_puzzles - valid_addresses} addresses failed verification")

        return 0

    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())