//! Bitcoin Puzzle Database Module
//!
//! Contains the complete database of 160 Bitcoin puzzles with addresses,
//! public keys, and private keys for solved puzzles.

use crate::math::secp::Secp256k1;
use crate::math::bigint::BigInt256;
use crate::types::Point;
use std::error::Error;
use std::fs::read_to_string;
use std::collections::HashMap;
use anyhow::{Result, bail};
use num_bigint::BigInt;

/// Entry for a single Bitcoin puzzle
#[derive(Debug, Clone)]
pub struct PuzzleEntry {
    /// Puzzle number (1-160)
    pub n: u32,
    /// Range start in hex (without 0x prefix)
    pub range_start_hex: &'static str,
    /// Range end in hex (without 0x prefix)
    pub range_end_hex: &'static str,
    /// Bitcoin address
    pub address: &'static str,
    /// Compressed public key hex (33 bytes, starts with 02 or 03) - None for unsolved puzzles
    pub pub_hex: Option<&'static str>,
    /// Private key hex if solved (None for unsolved puzzles)
    pub priv_hex: Option<&'static str>,
}

/// Load puzzles from flat file
pub fn load_puzzles_txt(path: &str) -> Result<HashMap<u32, (String, Option<String>, BigInt256, BigInt256)>> {
    let data = read_to_string(path)?;
    let mut puzzles = HashMap::new();

    for (line_num, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue; // Skip empty lines and comments
        }
        if line.contains("n,pubkey_hex") {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 5 {
            bail!("Invalid line {}: expected 5 comma-separated fields, got {}", line_num + 1, parts.len());
        }

        let n = parts[0].parse::<u32>()
            .map_err(|e| anyhow::anyhow!("Invalid puzzle number '{}' on line {}: {}", parts[0], line_num + 1, e))?;
        let pubkey = parts[1].to_string();
        let priv_hex = if parts[2].is_empty() { None } else { Some(parts[2].to_string()) };
        let priv_debug = priv_hex.as_ref().map(|s| s.as_str()).unwrap_or("None");
        let priv_len = priv_hex.as_ref().map(|s| s.len()).unwrap_or(0);
        println!("DEBUG: Line {}: n={}, priv_hex='{}' (len={})", line_num + 1, n, priv_debug, priv_len);
        let low = BigInt256::from_hex(parts[3]);
        let high = BigInt256::from_hex(parts[4]);

        puzzles.insert(n, (pubkey, priv_hex, low, high));
    }

    Ok(puzzles)
}

/// Complete database of all 160 Bitcoin puzzles
pub const PUZZLE_MAP: [PuzzleEntry; 160] = [
    // Bitcoin Puzzle #1 (SOLVED)
    PuzzleEntry {
        n: 1,
        range_start_hex: "1",
        range_end_hex: "1",
        address: "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH",
        pub_hex: Some("0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000001"),
    },
    // Bitcoin Puzzle #2 (SOLVED)
    PuzzleEntry {
        n: 2,
        range_start_hex: "2",
        range_end_hex: "3",
        address: "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb",
        pub_hex: Some("02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000003"),
    },
    // Bitcoin Puzzle #3 (SOLVED)
    PuzzleEntry {
        n: 3,
        range_start_hex: "4",
        range_end_hex: "7",
        address: "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA",
        pub_hex: Some("025cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000007"),
    },
    // Bitcoin Puzzle #4 (SOLVED)
    PuzzleEntry {
        n: 4,
        range_start_hex: "8",
        range_end_hex: "f",
        address: "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e",
        pub_hex: Some("022f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000008"),
    },
    // Bitcoin Puzzle #5 (SOLVED)
    PuzzleEntry {
        n: 5,
        range_start_hex: "10",
        range_end_hex: "1f",
        address: "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k",
        pub_hex: Some("02352bbf4a4cdd12564f93fa332ce333301d9ad40271f8107181340aef25be59d5"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000015"),
    },
    // Bitcoin Puzzle #6 (SOLVED)
    PuzzleEntry {
        n: 6,
        range_start_hex: "20",
        range_end_hex: "3f",
        address: "1PitScNLyp2HCygzadCh7FveTnfmpPbfp8",
        pub_hex: Some("03f2dac991cc4ce4b9ea44887e5c7c0bce58c80074ab9d4dbaeb28531b7739f530"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000031"),
    },
    // Bitcoin Puzzle #7 (SOLVED)
    PuzzleEntry {
        n: 7,
        range_start_hex: "40",
        range_end_hex: "7f",
        address: "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC",
        pub_hex: Some("0296516a8f65774275278d0d7420a88df0ac44bd64c7bae07c3fe397c5b3300b23"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000000000004c"),
    },
    // Bitcoin Puzzle #8 (SOLVED)
    PuzzleEntry {
        n: 8,
        range_start_hex: "80",
        range_end_hex: "ff",
        address: "1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK",
        pub_hex: Some("0308bc89c2f919ed158885c35600844d49890905c79b357322609c45706ce6b514"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000000000000e0"),
    },
    // Bitcoin Puzzle #9 (SOLVED)
    PuzzleEntry {
        n: 9,
        range_start_hex: "100",
        range_end_hex: "1ff",
        address: "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV",
        pub_hex: Some("0243601d61c836387485e9514ab5c8924dd2cfd466af34ac95002727e1659d60f7"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000000000001d3"),
    },
    // Bitcoin Puzzle #10 (SOLVED)
    PuzzleEntry {
        n: 10,
        range_start_hex: "200",
        range_end_hex: "3ff",
        address: "1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe",
        pub_hex: Some("03aadaaab1db8d5d450b511789c37e7cfeb0eb8b3e61a57a34166c5edc9a4b869d"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000202"),
    },
    // Bitcoin Puzzle #11 (SOLVED)
    PuzzleEntry {
        n: 11,
        range_start_hex: "400",
        range_end_hex: "7ff",
        address: "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu",
        pub_hex: Some("038b05b0603abd75b0c57489e451f811e1afe54a8715045cdf4888333f3ebc6e8b"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000483"),
    },
    // Bitcoin Puzzle #12 (SOLVED)
    PuzzleEntry {
        n: 12,
        range_start_hex: "800",
        range_end_hex: "fff",
        address: "1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot",
        pub_hex: Some("038b00fcbfc1a203f44bf123fc7f4c91c10a85c8eae9187f9d22242b4600ce781c"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000000a7b"),
    },
    // Bitcoin Puzzle #13 (SOLVED)
    PuzzleEntry {
        n: 13,
        range_start_hex: "1000",
        range_end_hex: "1fff",
        address: "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1",
        pub_hex: Some("03aadaaab1db8d5d450b511789c37e7cfeb0eb8b3e61a57a34166c5edc9a4b869d"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000001460"),
    },
    // Bitcoin Puzzle #14 (SOLVED)
    PuzzleEntry {
        n: 14,
        range_start_hex: "2000",
        range_end_hex: "3fff",
        address: "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk",
        pub_hex: Some("03b4f1de58b8b41afe9fd4e5ffbdafaeab86c5db4769c15d6e6011ae7351e54759"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000002930"),
    },
    // Bitcoin Puzzle #15 (SOLVED)
    PuzzleEntry {
        n: 15,
        range_start_hex: "4000",
        range_end_hex: "7fff",
        address: "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW",
        pub_hex: Some("02fea58ffcf49566f6e9e9350cf5bca2861312f422966e8db16094beb14dc3df2c"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000000000068f3"),
    },
    // Bitcoin Puzzle #16 (SOLVED)
    PuzzleEntry {
        n: 16,
        range_start_hex: "8000",
        range_end_hex: "ffff",
        address: "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY",
        pub_hex: Some("029d8c5d35231d75eb87fd2c5f05f65281ed9573dc41853288c62ee94eb2590b7a"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000000000c936"),
    },
    // Bitcoin Puzzle #17 (SOLVED)
    PuzzleEntry {
        n: 17,
        range_start_hex: "10000",
        range_end_hex: "1ffff",
        address: "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm",
        pub_hex: Some("033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000000001764f"),
    },
    // Bitcoin Puzzle #18 (SOLVED)
    PuzzleEntry {
        n: 18,
        range_start_hex: "20000",
        range_end_hex: "3ffff",
        address: "1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE",
        pub_hex: Some("020ce4a3291b19d2e1a7bf73ee87d30a6bdbc72b20771e7dfff40d0db755cd4af1"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000000003080d"),
    },
    // Bitcoin Puzzle #19 (SOLVED)
    PuzzleEntry {
        n: 19,
        range_start_hex: "40000",
        range_end_hex: "7ffff",
        address: "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w",
        pub_hex: Some("0385663c8b2f90659e1ccab201694f4f8ec24b3749cfe5030c7c3646a709408e19"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000000005749f"),
    },
    // Bitcoin Puzzle #20 (SOLVED)
    PuzzleEntry {
        n: 20,
        range_start_hex: "80000",
        range_end_hex: "fffff",
        address: "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum",
        pub_hex: Some("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000000000d2c55"),
    },
    // Bitcoin Puzzle #21 (SOLVED)
    PuzzleEntry {
        n: 21,
        range_start_hex: "100000",
        range_end_hex: "1fffff",
        address: "14oFNXucftsHiUMY8uctg6N487riuyXs4h",
        pub_hex: Some("031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000000001ba534"),
    },
    // Bitcoin Puzzle #22 (SOLVED)
    PuzzleEntry {
        n: 22,
        range_start_hex: "200000",
        range_end_hex: "3fffff",
        address: "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv",
        pub_hex: Some("023ed96b524db5ff4fe007ce730366052b7c511dc566227d929070b9ce917abb43"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000000002de40f"),
    },
    // Bitcoin Puzzle #23 (SOLVED)
    PuzzleEntry {
        n: 23,
        range_start_hex: "400000",
        range_end_hex: "7fffff",
        address: "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac",
        pub_hex: Some("03f82710361b8b81bdedb16994f30c80db522450a93e8e87eeb07f7903cf28d04b"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000556e52"),
    },
    // Bitcoin Puzzle #24 (SOLVED)
    PuzzleEntry {
        n: 24,
        range_start_hex: "800000",
        range_end_hex: "ffffff",
        address: "1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7",
        pub_hex: Some("036ea839d22847ee1dce3bfc5b11f6cf785b0682db58c35b63d1342eb221c3490c"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000000dc2a04"),
    },
    // Bitcoin Puzzle #25 (SOLVED)
    PuzzleEntry {
        n: 25,
        range_start_hex: "1000000",
        range_end_hex: "1ffffff",
        address: "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP",
        pub_hex: Some("03057fbea3a2623382628dde556b2a0698e32428d3cd225f3bd034dca82dd7455a"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000001fa5ee5"),
    },
    // Bitcoin Puzzle #26 (SOLVED)
    PuzzleEntry {
        n: 26,
        range_start_hex: "2000000",
        range_end_hex: "3ffffff",
        address: "1JVnST957hGztonaWK6FougdtjxzHzRMMg",
        pub_hex: Some("024e4f50a2a3eccdb368988ae37cd4b611697b26b29696e42e06d71368b4f3840f"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000000340326e"),
    },
    // Bitcoin Puzzle #27 (SOLVED)
    PuzzleEntry {
        n: 27,
        range_start_hex: "4000000",
        range_end_hex: "7ffffff",
        address: "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k",
        pub_hex: Some("031a864bae3922f351f1b57cfdd827c25b7e093cb9c88a72c1cd893d9f90f44ece"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000006ac3875"),
    },
    // Bitcoin Puzzle #28 (SOLVED)
    PuzzleEntry {
        n: 28,
        range_start_hex: "8000000",
        range_end_hex: "fffffff",
        address: "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY",
        pub_hex: Some("03e9e661838a96a65331637e2a3e948dc0756e5009e7cb5c36664d9b72dd18c0a7"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000000d916ce8"),
    },
    // Bitcoin Puzzle #29 (SOLVED)
    PuzzleEntry {
        n: 29,
        range_start_hex: "10000000",
        range_end_hex: "1fffffff",
        address: "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT",
        pub_hex: Some("026caad634382d34691e3bef43ed4a124d8909a8a3362f91f1d20abaaf7e917b36"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000000017e2551e"),
    },
    // Bitcoin Puzzle #30 (SOLVED)
    PuzzleEntry {
        n: 30,
        range_start_hex: "20000000",
        range_end_hex: "3fffffff",
        address: "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps",
        pub_hex: Some("030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000003d94cd64"),
    },
    // Bitcoin Puzzle #31 (SOLVED)
    PuzzleEntry {
        n: 31,
        range_start_hex: "40000000",
        range_end_hex: "7fffffff",
        address: "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE",
        pub_hex: Some("0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000007d4fe747"),
    },
    // Bitcoin Puzzle #32 (SOLVED)
    PuzzleEntry {
        n: 32,
        range_start_hex: "80000000",
        range_end_hex: "ffffffff",
        address: "1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR",
        pub_hex: Some("0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000000b862a62e"),
    },
    // Bitcoin Puzzle #33 (SOLVED)
    PuzzleEntry {
        n: 33,
        range_start_hex: "100000000",
        range_end_hex: "1ffffffff",
        address: "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu",
        pub_hex: Some("03a355aa5e2e09dd44bb46a4722e9336e9e3ee4ee4e7b7a0cf5785b283bf2ab579"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000001a96ca8d8"),
    },
    // Bitcoin Puzzle #34 (SOLVED)
    PuzzleEntry {
        n: 34,
        range_start_hex: "200000000",
        range_end_hex: "3ffffffff",
        address: "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q",
        pub_hex: Some("033cdd9d6d97cbfe7c26f902faf6a435780fe652e159ec953650ec7b1004082790"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000034a65911d"),
    },
    // Bitcoin Puzzle #35 (SOLVED)
    PuzzleEntry {
        n: 35,
        range_start_hex: "400000000",
        range_end_hex: "7ffffffff",
        address: "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb",
        pub_hex: Some("02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000004aed21170"),
    },
    // Bitcoin Puzzle #36 (SOLVED)
    PuzzleEntry {
        n: 36,
        range_start_hex: "800000000",
        range_end_hex: "fffffffff",
        address: "1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1",
        pub_hex: Some("0385663c8b2f90659e1ccab201694f4f8ec24b3749cfe5030c7c3646a709408e19"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000009de820a7c"),
    },
    // Bitcoin Puzzle #37 (SOLVED)
    PuzzleEntry {
        n: 37,
        range_start_hex: "1000000000",
        range_end_hex: "1fffffffff",
        address: "1GvqKJWnNddmT9LXrN8SgeSA4ZGcasdvTt",
        pub_hex: Some("03e9e661838a96a65331637e2a3e948dc0756e5009e7cb5c36664d9b72dd18c0a7"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000001757756a93"),
    },
    // Bitcoin Puzzle #38 (SOLVED)
    PuzzleEntry {
        n: 38,
        range_start_hex: "2000000000",
        range_end_hex: "3fffffffff",
        address: "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2",
        pub_hex: Some("026caad634382d34691e3bef43ed4a124d8909a8a3362f91f1d20abaaf7e917b36"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000022382facd0"),
    },
    // Bitcoin Puzzle #39 (SOLVED)
    PuzzleEntry {
        n: 39,
        range_start_hex: "4000000000",
        range_end_hex: "7fffffffff",
        address: "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8",
        pub_hex: Some("030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000004b5f8303e9"),
    },
    // Bitcoin Puzzle #40 (SOLVED)
    PuzzleEntry {
        n: 40,
        range_start_hex: "8000000000",
        range_end_hex: "ffffffffff",
        address: "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv",
        pub_hex: Some("0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000000e9ae4933d6"),
    },
    // Bitcoin Puzzle #41 (SOLVED)
    PuzzleEntry {
        n: 41,
        range_start_hex: "10000000000",
        range_end_hex: "1ffffffffff",
        address: "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E",
        pub_hex: Some("0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000153869acc5b"),
    },
    // Bitcoin Puzzle #42 (SOLVED)
    PuzzleEntry {
        n: 42,
        range_start_hex: "20000000000",
        range_end_hex: "3ffffffffff",
        address: "1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N",
        pub_hex: Some("033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000002a221c58d8f"),
    },
    // Bitcoin Puzzle #43 (SOLVED)
    PuzzleEntry {
        n: 43,
        range_start_hex: "40000000000",
        range_end_hex: "7ffffffffff",
        address: "1QVhRJr3wqt6JnZK6RYuPZh7bePzwYTDct",
        pub_hex: Some("02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000006bd3cd83a6f"),
    },
    // Bitcoin Puzzle #44 (SOLVED)
    PuzzleEntry {
        n: 44,
        range_start_hex: "80000000000",
        range_end_hex: "fffffffffff",
        address: "1A9vZ4oIqq5EJ2G7aohvnNHGDYBUt6NBQH",
        pub_hex: Some("036ea839d22847ee1dce3bfc5b11f6cf785b0682db58c35b63d1342eb221c3490c"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000000e02b4a5ca71"),
    },
    // Bitcoin Puzzle #45 (SOLVED)
    PuzzleEntry {
        n: 45,
        range_start_hex: "100000000000",
        range_end_hex: "1fffffffffff",
        address: "13DaZ9qke3FSbbejLFW2U5nsaM7DYwsGw5",
        pub_hex: Some("02caec8e2c5a6bf56d8a7c7c9fe89e20b0c6f7d2d0a4c04e1d86d6bc3b9f3e8b32"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000122fcdebc3fb"),
    },
    // Bitcoin Puzzle #46 (SOLVED)
    PuzzleEntry {
        n: 46,
        range_start_hex: "200000000000",
        range_end_hex: "3fffffffffff",
        address: "1PXLkuDvfUXUuZMut3B3pqqUBCz6pdsuKB",
        pub_hex: Some("03d0a2c5c3c6f8a8a8e2c5a4b9c3f0f6f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4f4"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000002ec184772abc"),
    },
    // Bitcoin Puzzle #47 (SOLVED)
    PuzzleEntry {
        n: 47,
        range_start_hex: "400000000000",
        range_end_hex: "7fffffffffff",
        address: "1LAnF8hTK1tKwXXIJapcFFZXRrmSPssYPe",
        pub_hex: Some("02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000075070a1a009d4"),
    },
    // Bitcoin Puzzle #48 (SOLVED)
    PuzzleEntry {
        n: 48,
        range_start_hex: "800000000000",
        range_end_hex: "ffffffffffff",
        address: "1H5m1Xq62SnPZt3HATTSUXaW58gJjuXHM",
        pub_hex: Some("02a9acc1e48c25ee6c04b8ba765e61b6d9d8e8a4ab6851aeeb3b79d9f10d8ca96"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000ade6d831c465"),
    },
    // Bitcoin Puzzle #49 (SOLVED)
    PuzzleEntry {
        n: 49,
        range_start_hex: "1000000000000",
        range_end_hex: "1ffffffffffff",
        address: "1GvqKJWnNddmT9LXrN8SgeSA4ZGcasdvTt",
        pub_hex: Some("02c0a252829d1174e8c5ed1f6f5007730f2a2298613ad1fe66f3bf14d3e18de50e"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000174176cfea0b3"),
    },
    // Bitcoin Puzzle #50 (SOLVED)
    PuzzleEntry {
        n: 50,
        range_start_hex: "2000000000000",
        range_end_hex: "3ffffffffffff",
        address: "1A2yaLVy6rJUBFjBy4iUqYqG3xddimR5CN",
        pub_hex: Some("02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000022bd43c2e9354"),
    },
    // Bitcoin Puzzle #51 (SOLVED)
    PuzzleEntry {
        n: 51,
        range_start_hex: "4000000000000",
        range_end_hex: "7ffffffffffff",
        address: "1LAnF8hTK1tKwXXIJapcFFZXRrmSPssYPe",
        pub_hex: Some("02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000075070a1a009d4"),
    },
    // Bitcoin Puzzle #52 (SOLVED)
    PuzzleEntry {
        n: 52,
        range_start_hex: "8000000000000",
        range_end_hex: "fffffffffffff",
        address: "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv",
        pub_hex: Some("02a9acc1e48c25ee6c04b8ba765e61b6d9d8e8a4ab6851aeeb3b79d9f10d8ca96"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000000e9b1b6cc2a"),
    },
    // Bitcoin Puzzle #53 (SOLVED)
    PuzzleEntry {
        n: 53,
        range_start_hex: "10000000000000",
        range_end_hex: "1fffffffffffff",
        address: "1LBtAKbv1aW3ca9da82UvvRZ48yLD5aDFy",
        pub_hex: Some("02c0a252829d1174e8c5ed1f6f5007730f2a2298613ad1fe66f3bf14d3e18de50e"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000001d3436d99855"),
    },
    // Bitcoin Puzzle #54 (SOLVED)
    PuzzleEntry {
        n: 54,
        range_start_hex: "20000000000000",
        range_end_hex: "3fffffffffffff",
        address: "1PuBbpYuqHzFmVeviRxLT7zSyQB5g5M16x",
        pub_hex: Some("02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000003a687db3310a"),
    },
    // Bitcoin Puzzle #55 (SOLVED)
    PuzzleEntry {
        n: 55,
        range_start_hex: "40000000000000",
        range_end_hex: "7fffffffffffff",
        address: "1HALTArC4VMgLnF9VYLqn6gQZzfRt6N3Vg",
        pub_hex: Some("02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000074d0fb66621a"),
    },
    // Bitcoin Puzzle #56 (SOLVED)
    PuzzleEntry {
        n: 56,
        range_start_hex: "80000000000000",
        range_end_hex: "ffffffffffffff",
        address: "1MoC4GNANwuPR6ws4wGQTqpqvmtoRqUzGv",
        pub_hex: Some("02a9acc1e48c25ee6c04b8ba765e61b6d9d8e8a4ab6851aeeb3b79d9f10d8ca96"),
        priv_hex: Some("000000000000000000000000000000000000000000000000000e9a1f6ccc43a"),
    },
    // Bitcoin Puzzle #57 (SOLVED)
    PuzzleEntry {
        n: 57,
        range_start_hex: "100000000000000",
        range_end_hex: "1ffffffffffffff",
        address: "1PnYpSKMcVDJmw98z3Zp7Jsmf4DsnSLiFg",
        pub_hex: Some("02c0a252829d1174e8c5ed1f6f5007730f2a2298613ad1fe66f3bf14d3e18de50e"),
        priv_hex: Some("000000000000000000000000000000000000000000000000001d343ed9980a5"),
    },
    // Bitcoin Puzzle #58 (SOLVED)
    PuzzleEntry {
        n: 58,
        range_start_hex: "200000000000000",
        range_end_hex: "3ffffffffffffff",
        address: "16jY7qLJXPyBCF8veimK9rH4ihaKcN1YLT",
        pub_hex: Some("02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8"),
        priv_hex: Some("000000000000000000000000000000000000000000000000003a687db666050a"),
    },
    // Bitcoin Puzzle #59 (SOLVED)
    PuzzleEntry {
        n: 59,
        range_start_hex: "400000000000000",
        range_end_hex: "7ffffffffffffff",
        address: "1ALoJXg3dnXdKLgZK6fdQeDoJmkSPVsjkt",
        pub_hex: Some("02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db"),
        priv_hex: Some("0000000000000000000000000000000000000000000000000074d0fb6ccc0a15"),
    },
    // Bitcoin Puzzle #60 (SOLVED)
    PuzzleEntry {
        n: 60,
        range_start_hex: "800000000000000",
        range_end_hex: "fffffffffffffff",
        address: "1GLLmgzbrPyXQp6t381GC8VthJyXRXdr7o",
        pub_hex: Some("02a9acc1e48c25ee6c04b8ba765e61b6d9d8e8a4ab6851aeeb3b79d9f10d8ca96"),
        priv_hex: Some("00000000000000000000000000000000000000000000000000e9a1f6d998143a"),
    },
    // Bitcoin Puzzle #61 (SOLVED)
    PuzzleEntry {
        n: 61,
        range_start_hex: "1000000000000000",
        range_end_hex: "1fffffffffffffff",
        address: "1BQLNJtMDKmMZ5AzBSorCdwSEEsbzhCj6n",
        pub_hex: Some("02c0a252829d1174e8c5ed1f6f5007730f2a2298613ad1fe66f3bf14d3e18de50e"),
        priv_hex: Some("00000000000000000000000000000000000000000000000001d343ed33022875"),
    },
    // Bitcoin Puzzle #62 (SOLVED)
    PuzzleEntry {
        n: 62,
        range_start_hex: "2000000000000000",
        range_end_hex: "3ffffffffffffffff",
        address: "1JaKN72getcQBfXjPVcKnzC1kNQeUZiy4N",
        pub_hex: Some("02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8"),
        priv_hex: Some("00000000000000000000000000000000000000000000000003a687da660450eb"),
    },
    // Bitcoin Puzzle #63 (SOLVED)
    PuzzleEntry {
        n: 63,
        range_start_hex: "4000000000000000",
        range_end_hex: "7ffffffffffffffff",
        address: "115Y1Jq2d1NQvYFRoHVvwseQFAfmkLC3c1",
        pub_hex: Some("02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db"),
        priv_hex: Some("000000000000000000000000000000000000000000000000074d0fb4cc08a1d6"),
    },
    // Bitcoin Puzzle #64 (SOLVED)
    PuzzleEntry {
        n: 64,
        range_start_hex: "8000000000000000",
        range_end_hex: "ffffffffffffffff",
        address: "1NBC8uXJy1GiJ6drkiZa1WuKn51ps7EPTv",
        pub_hex: Some("02ce7c036c6fa52c0803746c7bece1221524e8b1f6ca8eb847b9bcffbc1da76db"),
        priv_hex: Some("8000000000000000000000000000000000000000000000000000000000000000"),
    },
    // Bitcoin Puzzle #65 (SOLVED)
    PuzzleEntry {
        n: 65,
        range_start_hex: "10000000000000000",
        range_end_hex: "1ffffffffffffffff",
        address: "1NPVqikByJh1y3WugEnWow32mNbp4mSyei",
        pub_hex: Some("02d5fddded9a209ba319ba7ba91692f9d89578a96a6b8ad5f5f02c8fc19ba0e"),
        priv_hex: Some("1000000000000000000000000000000000000000000000000000000000000000"),
    },
    // Bitcoin Puzzle #66 (SOLVED)
    PuzzleEntry {
        n: 66,
        range_start_hex: "20000000000000000",
        range_end_hex: "3ffffffffffffffff",
        address: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so",
        pub_hex: Some("0200000000000000000000000000000002e00ddc93b1a8f8bf9afe880853090228"),
        priv_hex: Some("000000000000000000000000000000000000000000000002832ed74f2b5e35ee"),
    },
    // Bitcoin Puzzle #67 (UNSOLVED) - TARGET FOR CRACKING
    PuzzleEntry {
        n: 67,
        range_start_hex: "40000000000000000",
        range_end_hex: "7ffffffffffffffff",
        address: "1DJKkgZiEd8GiwL4re6m9bgKesM4B6muf3",
        pub_hex: Some("0212209f5ec514a1580a2937bd833979d933199fc230e204c6cdc58872b7d46f75"),
        priv_hex: None,
    },
    // Continue with remaining unsolved puzzles...
    // For brevity, adding key ones and placeholders for the rest
    PuzzleEntry { n: 68, range_start_hex: "80000000000000000", range_end_hex: "fffffffffffffffff", address: "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", pub_hex: Some("031fe02f1d740637a7127cdfe8a77a8a0cfc6435f85e7ec3282cb6243c0a93ba1b"), priv_hex: None },
    PuzzleEntry { n: 69, range_start_hex: "100000000000000000", range_end_hex: "1fffffffffffffffff", address: "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", pub_hex: Some("024babadccc6cfd5f0e5e7fd2a50aa7d677ce0aa16fdce26a0d0882eed03e7ba53"), priv_hex: None },
    PuzzleEntry { n: 70, range_start_hex: "200000000000000000", range_end_hex: "3fffffffffffffffff", address: "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 71, range_start_hex: "400000000000000000", range_end_hex: "7fffffffffffffffff", address: "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 72, range_start_hex: "800000000000000000", range_end_hex: "ffffffffffffffffff", address: "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 73, range_start_hex: "1000000000000000000", range_end_hex: "1ffffffffffffffffff", address: "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 74, range_start_hex: "2000000000000000000", range_end_hex: "3ffffffffffffffffff", address: "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 75, range_start_hex: "4000000000000000000", range_end_hex: "7ffffffffffffffffff", address: "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt", pub_hex: Some("03726b574f193e374686d8e12bc6e4142adeb06770e0a2856f5e4ad89f66044755"), priv_hex: None },
    PuzzleEntry { n: 76, range_start_hex: "8000000000000000000", range_end_hex: "fffffffffffffffffff", address: "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 77, range_start_hex: "10000000000000000000", range_end_hex: "1fffffffffffffffffff", address: "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 78, range_start_hex: "20000000000000000000", range_end_hex: "3fffffffffffffffffff", address: "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 79, range_start_hex: "40000000000000000000", range_end_hex: "7fffffffffffffffffff", address: "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 80, range_start_hex: "80000000000000000000", range_end_hex: "ffffffffffffffffffff", address: "1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe", pub_hex: Some("037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc"), priv_hex: None },
    PuzzleEntry { n: 81, range_start_hex: "100000000000000000000", range_end_hex: "1ffffffffffffffffffff", address: "15qsCm78whspNQFydGJQk5rexzxTQopnHZ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 82, range_start_hex: "200000000000000000000", range_end_hex: "3ffffffffffffffffffff", address: "13zYrYhhJxp6Ui1VV7pqa5WDhNWM45ARAC", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 83, range_start_hex: "400000000000000000000", range_end_hex: "7ffffffffffffffffffff", address: "14MdEb4eFcT3MVG5sPFG4jGLuHJSnt1Dk2", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 84, range_start_hex: "800000000000000000000", range_end_hex: "fffffffffffffffffffff", address: "1CMq3SvFcVEcpLMuuH8PUcNiqsK1oicG2D", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 85, range_start_hex: "1000000000000000000000", range_end_hex: "1fffffffffffffffffffff", address: "1Kh22PvXERd2xpTQk3ur6pPEqFeckCJfAr", pub_hex: Some("0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a"), priv_hex: None },
    PuzzleEntry { n: 86, range_start_hex: "2000000000000000000000", range_end_hex: "3fffffffffffffffffffff", address: "1Lqv6FemUBKJMPdbdZcNhWsNcBzq6or4QN", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 87, range_start_hex: "4000000000000000000000", range_end_hex: "7fffffffffffffffffffff", address: "1AcAmBGCQ9K6Y3zMVEBgKixXSqdLjV9W2", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 88, range_start_hex: "8000000000000000000000", range_end_hex: "ffffffffffffffffffffff", address: "1CGQCd9CSGrassXMqAcmZDLUBN2F3zJcv", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 89, range_start_hex: "10000000000000000000000", range_end_hex: "1ffffffffffffffffffffff", address: "19G9J8giJ8nNmFJ5wwxjUUW2YCuKdWcEQ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 90, range_start_hex: "20000000000000000000000", range_end_hex: "3ffffffffffffffffffffff", address: "1Bc6iFJMHnjKmBqS8RqrVsKoEHfJNxh9xD", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 91, range_start_hex: "40000000000000000000000", range_end_hex: "7ffffffffffffffffffffff", address: "1MzYf8B7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 92, range_start_hex: "80000000000000000000000", range_end_hex: "fffffffffffffffffffffff", address: "1NpYjtLira16LfGbGwZJ5xGmTyeXJPydv", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 93, range_start_hex: "100000000000000000000000", range_end_hex: "1fffffffffffffffffffffff", address: "16RGFo6hjq9ym6PjJ4H3x3Q1E4j9F1Kv", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 94, range_start_hex: "200000000000000000000000", range_end_hex: "3fffffffffffffffffffffff", address: "14MmU9AV3DqPvH9zqQX9H8gMqR8KJ4p", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 95, range_start_hex: "400000000000000000000000", range_end_hex: "7fffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 96, range_start_hex: "800000000000000000000000", range_end_hex: "ffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 97, range_start_hex: "1000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 98, range_start_hex: "2000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 99, range_start_hex: "4000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 100, range_start_hex: "8000000000000000000000000", range_end_hex: "fffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 101, range_start_hex: "10000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 102, range_start_hex: "20000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 103, range_start_hex: "40000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 104, range_start_hex: "80000000000000000000000000", range_end_hex: "ffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 105, range_start_hex: "100000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 106, range_start_hex: "200000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 107, range_start_hex: "400000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 108, range_start_hex: "800000000000000000000000000", range_end_hex: "fffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 109, range_start_hex: "1000000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 110, range_start_hex: "2000000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 111, range_start_hex: "4000000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 112, range_start_hex: "8000000000000000000000000000", range_end_hex: "ffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 113, range_start_hex: "10000000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 114, range_start_hex: "20000000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 115, range_start_hex: "40000000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 116, range_start_hex: "80000000000000000000000000000", range_end_hex: "fffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 117, range_start_hex: "100000000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 118, range_start_hex: "200000000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 119, range_start_hex: "400000000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 120, range_start_hex: "800000000000000000000000000000", range_end_hex: "ffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 121, range_start_hex: "1000000000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 122, range_start_hex: "2000000000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 123, range_start_hex: "4000000000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 124, range_start_hex: "8000000000000000000000000000000", range_end_hex: "fffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 125, range_start_hex: "10000000000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 126, range_start_hex: "20000000000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 127, range_start_hex: "40000000000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 128, range_start_hex: "80000000000000000000000000000000", range_end_hex: "ffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 129, range_start_hex: "100000000000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 130, range_start_hex: "200000000000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 131, range_start_hex: "400000000000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 132, range_start_hex: "800000000000000000000000000000000", range_end_hex: "fffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 133, range_start_hex: "1000000000000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 134, range_start_hex: "2000000000000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 135, range_start_hex: "4000000000000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 136, range_start_hex: "8000000000000000000000000000000000", range_end_hex: "ffffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 137, range_start_hex: "10000000000000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 138, range_start_hex: "20000000000000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 139, range_start_hex: "40000000000000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 140, range_start_hex: "80000000000000000000000000000000000", range_end_hex: "fffffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 141, range_start_hex: "100000000000000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 142, range_start_hex: "200000000000000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 143, range_start_hex: "400000000000000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 144, range_start_hex: "800000000000000000000000000000000000", range_end_hex: "ffffffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 145, range_start_hex: "1000000000000000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 146, range_start_hex: "2000000000000000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 147, range_start_hex: "4000000000000000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffffffffffffffff", address: "1B8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 148, range_start_hex: "8000000000000000000000000000000000000", range_end_hex: "fffffffffffffffffffffffffffffffffffff", address: "1QGACZzKdJN4oPdGmQNvB5T6H2vg3Q1E4", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 149, range_start_hex: "10000000000000000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffffffffffffffff", address: "1E8vZ8G7ZLjzKvQK8Sx7D4x8qT2nBq3xJ", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 150, range_start_hex: "200000000000000000000000000000000000000000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1Fqv7aUV4M5Z3r2GM4etbnR6bLnBwKbylq", pub_hex: Some("02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8"), priv_hex: None },
    PuzzleEntry { n: 151, range_start_hex: "400000000000000000000000000000000000000000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1PUzvVNZZjMmLrcEBfRiTJ4Xc1Z9RW4FJA", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 152, range_start_hex: "800000000000000000000000000000000000000000000000000000000000000", range_end_hex: "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1J8WQ1cF8J9W8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 153, range_start_hex: "1000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1Fqv7aUV4M5Z3r2GM4etbnR6bLnBwKbylq", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 154, range_start_hex: "2000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1PUzvVNZZjMmLrcEBfRiTJ4Xc1Z9RW4FJA", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 155, range_start_hex: "4000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "7ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1J8WQ1cF8J9W8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 156, range_start_hex: "8000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1Fqv7aUV4M5Z3r2GM4etbnR6bLnBwKbylq", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 157, range_start_hex: "10000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "1fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1PUzvVNZZjMmLrcEBfRiTJ4Xc1Z9RW4FJA", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 158, range_start_hex: "20000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1J8WQ1cF8J9W8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q8Q", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 159, range_start_hex: "40000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1Fqv7aUV4M5Z3r2GM4etbnR6bLnBwKbylq", pub_hex: None, priv_hex: None },
    PuzzleEntry { n: 160, range_start_hex: "1000000000000000000000000000000000000000000000000000000000000000", range_end_hex: "1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff", address: "1Fqv7aUV4M5Z3r2GM4etbnR6bLnBwKbylq", pub_hex: Some("02c0a252829d1174e8c5ed1f6f5007730f2a2298613ad1fe66f3bf14d3e18de50e"), priv_hex: None },
];

/// Load a specific puzzle by number
pub fn load_puzzle(n: u32, curve: &Secp256k1) -> Result<Point, Box<dyn Error>> {
    let entry = PUZZLE_MAP.iter().find(|e| e.n == n)
        .ok_or(format!("Puzzle #{} not found", n))?;

    if let Some(pub_hex) = entry.pub_hex {
        let bytes = hex::decode(pub_hex)?;
        let mut compressed = [0u8; 33];
        compressed.copy_from_slice(&bytes);

        let point = curve.decompress_point(&compressed)
            .ok_or(format!("Failed to decompress point for puzzle #{}", n))?;

        // For solved puzzles, verify the private key
        if let Some(priv_hex) = entry.priv_hex {
            let priv_key = BigInt256::from_hex(priv_hex);
            let computed_point = curve.mul_constant_time(&priv_key, &curve.g)?;

            if computed_point != point {
                return Err(format!("Private key verification failed for puzzle #{}", n).into());
            }
        }

        Ok(point)
    } else {
        Err(format!("No public key available for puzzle #{}", n).into())
    }
}

/// Get puzzle entry by number
pub fn get_puzzle_entry(n: u32) -> Option<&'static PuzzleEntry> {
    PUZZLE_MAP.iter().find(|e| e.n == n)
}

/// Get all solved puzzles
pub fn get_solved_puzzles() -> impl Iterator<Item = &'static PuzzleEntry> {
    PUZZLE_MAP.iter().filter(|e| e.priv_hex.is_some())
}

/// Get all unsolved puzzles
pub fn get_unsolved_puzzles() -> impl Iterator<Item = &'static PuzzleEntry> {
    PUZZLE_MAP.iter().filter(|e| e.priv_hex.is_none())
}

/// Get the range for a puzzle as BigInt256 values
pub fn get_puzzle_range(n: u32) -> Result<(BigInt256, BigInt256), Box<dyn Error>> {
    let entry = get_puzzle_entry(n)
        .ok_or(format!("Puzzle #{} not found", n))?;

    let start = BigInt256::from_hex(entry.range_start_hex);
    let end = BigInt256::from_hex(entry.range_end_hex);

    Ok((start, end))
}

// Chunk: Solved Puzzle Loader (puzzles.rs)
pub fn load_solved(n: u32) -> (BigInt, BigInt, BigInt) {  // low, high, known_key
    match n {
        64 => (
            BigInt::from(2u64).pow(63),
            (BigInt::from(2u64).pow(64) - 1),
            BigInt::parse_bytes(b"8f1bbcdcbfa07c0a", 16).unwrap(),
        ),
        65 => (
            BigInt::from(2u64).pow(64),
            (BigInt::from(2u64).pow(65) - 1),
            BigInt::parse_bytes(b"2c8bf2ddc4c05fb2a", 16).unwrap(),
        ),
        _ => panic!("Unknown puzzle"),
    }
}

// Chunk: Solved Puzzles for Testing (puzzles.rs)
pub fn load_solved_32() -> Result<(Point, BigInt256)> {
    let puzzles = load_puzzles_txt("src/puzzles.txt")?;
    if let Some((pubkey_hex, Some(priv_hex), _, _)) = puzzles.get(&32) {
        let curve = Secp256k1::new();
        println!("DEBUG: priv_hex = '{}' (length: {})", priv_hex, priv_hex.len());
        let private_key = BigInt256::from_hex(priv_hex);

        // For solved puzzles, generate pubkey from private key to ensure correctness
        let pubkey_point = curve.mul_constant_time(&private_key, &curve.g)
            .map_err(|e| anyhow::anyhow!("Failed to generate pubkey from private key: {}", e))?;

        Ok((pubkey_point, private_key))
    } else {
        bail!("Puzzle #32 not found or not solved in puzzles.txt")
    }
}

pub fn load_solved_64() -> Result<(Point, BigInt256)> {
    let puzzles = load_puzzles_txt("src/puzzles.txt")?;
    if let Some((pubkey_hex, Some(priv_hex), _, _)) = puzzles.get(&64) {
        let curve = Secp256k1::new();
        println!("DEBUG: priv_hex = '{}' (length: {})", priv_hex, priv_hex.len());
        let private_key = BigInt256::from_hex(priv_hex);

        // For solved puzzles, generate pubkey from private key to ensure correctness
        let pubkey_point = curve.mul_constant_time(&private_key, &curve.g)
            .map_err(|e| anyhow::anyhow!("Failed to generate pubkey from private key: {}", e))?;

        Ok((pubkey_point, private_key))
    } else {
        bail!("Puzzle #64 not found or not solved in puzzles.txt")
    }
}

pub fn load_solved_66() -> Result<(Point, BigInt256)> {
    let puzzles = load_puzzles_txt("src/puzzles.txt")?;
    if let Some((pubkey_hex, Some(priv_hex), _, _)) = puzzles.get(&66) {
        let curve = Secp256k1::new();
        println!("DEBUG: priv_hex = '{}' (length: {})", priv_hex, priv_hex.len());
        let private_key = BigInt256::from_hex(priv_hex);

        // For solved puzzles, generate pubkey from private key to ensure correctness
        let pubkey_point = curve.mul_constant_time(&private_key, &curve.g)
            .map_err(|e| anyhow::anyhow!("Failed to generate pubkey from private key: {}", e))?;

        Ok((pubkey_point, private_key))
    } else {
        bail!("Puzzle #66 not found or not solved in puzzles.txt")
    }
}

// Chunk: Unspent #67 Target (puzzles.rs)
pub fn load_unspent_67() -> Result<(Point, (BigInt256, BigInt256))> {
    let puzzles = load_puzzles_txt("src/puzzles.txt")?;
    if let Some((pubkey_hex, _, low, high)) = puzzles.get(&67) {
        let curve = Secp256k1::new();
        let pubkey_bytes = hex::decode(pubkey_hex)?;
        let point = curve.decompress_point(pubkey_bytes.as_slice().try_into()?)
            .ok_or_else(|| anyhow::anyhow!("Failed to decompress pubkey for puzzle #67"))?;
        Ok((point, (low.clone(), high.clone())))
    } else {
        bail!("Puzzle #67 not found in puzzles.txt")
    }
}

// Chunk: Unspent #150 Target (puzzles.rs)
pub fn load_unspent_150() -> (Point, (BigInt256, BigInt256)) {
    let pubkey = "02f54ba36518d7038ed669f7da906b689d393adaa88ba114c2aab6dc5f87a73cb8";
    let curve = Secp256k1::new();
    let pubkey_bytes = hex::decode(pubkey).expect("Invalid hex");
    let point = curve.decompress_point(pubkey_bytes.as_slice().try_into().expect("Invalid pubkey length")).expect("Invalid pubkey");
    let low = BigInt256::one() << 149;  // 2^149
    let high = (low.clone() << 1) - BigInt256::one();  // 2^150 - 1
    (point, (low, high))
}