// Build script to auto-populate MAGIC9_BIASES const from valuable_p2pk_pubkeys.txt
// Computes cluster-specific biases (mod3/9/27/81, Hamming weight) at build time

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=valuable_p2pk_pubkeys.txt");
    println!("cargo:rerun-if-changed=src/gpu/cuda");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("magic9_biases.rs");

    // Try to read the pubkey file with enhanced error handling
    let pubkey_file = match File::open("valuable_p2pk_pubkeys.txt") {
        Ok(file) => {
            println!("cargo:warning=Successfully opened valuable_p2pk_pubkeys.txt for bias computation");
            file
        },
        Err(e) => {
            eprintln!("Error: Could not open valuable_p2pk_pubkeys.txt: {}. This file is required for Magic 9 GOLD cluster analysis.", e);
            eprintln!("Please ensure valuable_p2pk_pubkeys.txt exists in the project root.");
            eprintln!("Falling back to placeholder biases - GOLD cluster optimizations will be limited.");
            // Generate placeholder biases based on typical patterns
            generate_placeholder_biases(&dest_path);
            return;
        }
    };

    let lines: Vec<String> = BufReader::new(pubkey_file)
        .lines()
        .filter_map(Result::ok)
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        eprintln!("Warning: valuable_p2pk_pubkeys.txt is empty. Using placeholder biases.");
        generate_placeholder_biases(&dest_path);
        return;
    }

    // Magic 9 indices (0-based)
    let indices = [9379, 28687, 33098, 12457, 18902, 21543, 27891, 31234, 4567];

    let mut biases = Vec::new();

    for &idx in &indices {
        if idx >= lines.len() {
            eprintln!("Warning: Index {} out of bounds (file has {} lines). Using placeholder.", idx, lines.len());
            biases.push((0u8, 0u8, 0u8, 0u8, 128u32)); // Placeholder
            continue;
        }

        let hex_str = &lines[idx];
        match compute_pubkey_biases(hex_str) {
            Ok(bias) => biases.push(bias),
            Err(e) => {
                eprintln!("Warning: Failed to compute bias for index {}: {}. Using placeholder.", idx, e);
                biases.push((0u8, 0u8, 0u8, 0u8, 128u32)); // Placeholder
            }
        }
    }

    // Write the const arrays
    let mut output = String::from("pub const MAGIC9_BIASES: [(u8, u8, u8, u8, u32); 9] = [\n");
    for (i, (mod3, mod9, mod27, mod81, hamming)) in biases.iter().enumerate() {
        output.push_str(&format!("    ({}, {}, {}, {}, {}),\n", mod3, mod9, mod27, mod81, hamming));
        println!("Magic9 key {} (index {}): mod3={}, mod9={}, mod27={}, mod81={}, hamming={}",
                i, indices[i], mod3, mod9, mod27, mod81, hamming);
    }
    output.push_str("];\n\n");

    // Add pre-computed prime sets for GOLD cluster
    let primes: [u64; 32] = [
        179, 257, 281, 349, 379, 419, 457, 499,
        541, 599, 641, 709, 761, 809, 853, 911,
        967, 1013, 1061, 1091, 1151, 1201, 1249, 1297,
        1327, 1381, 1423, 1453, 1483, 1511, 1553, 1583,
    ];

    // Generate GOLD cluster primes (mod81 == 0) - fixed size fallback
    let gold_primes: Vec<u64> = primes.iter().filter(|&&p| p % 81 == 0).cloned().collect();
    let gold_array = if gold_primes.len() >= 4 {
        format!("[{}, {}, {}, {}]", gold_primes[0], gold_primes[1], gold_primes[2], gold_primes[3])
    } else {
        // Use primes that are at least mod27==0 as fallback
        let fallback: Vec<u64> = primes.iter().filter(|&&p| p % 27 == 0).take(4).cloned().collect();
        if fallback.len() >= 4 {
            format!("[{}, {}, {}, {}]", fallback[0], fallback[1], fallback[2], fallback[3])
        } else {
            // Ultimate fallback: first 4 primes
            format!("[{}, {}, {}, {}]", primes[0], primes[1], primes[2], primes[3])
        }
    };
    output.push_str(&format!("pub const GOLD_CLUSTER_PRIMES: [u64; 4] = {};\n\n", gold_array));

    // Generate secondary primes (mod27 == 0) for fallback - fixed 8 elements
    let secondary_primes: Vec<u64> = primes.iter().filter(|&&p| p % 27 == 0).cloned().collect();
    let secondary_array = if secondary_primes.len() >= 8 {
        format!("[{}, {}, {}, {}, {}, {}, {}, {}]",
                secondary_primes[0], secondary_primes[1], secondary_primes[2], secondary_primes[3],
                secondary_primes[4], secondary_primes[5], secondary_primes[6], secondary_primes[7])
    } else {
        // Fallback to first 8 primes
        format!("[{}, {}, {}, {}, {}, {}, {}, {}]",
                primes[0], primes[1], primes[2], primes[3],
                primes[4], primes[5], primes[6], primes[7])
    };
    output.push_str(&format!("pub const SECONDARY_PRIMES: [u64; 8] = {};\n", secondary_array));

    println!("Generated prime sets - GOLD: {} primes, Secondary: {} primes",
             gold_primes.len(), secondary_primes.len());

    // Compile CUDA kernels using cc crate
    compile_cuda_kernels();

    match std::fs::write(&dest_path, &output) {
        Ok(_) => println!("Generated MAGIC9_BIASES and prime sets at {:?}", dest_path),
        Err(e) => eprintln!("Error writing to {:?}: {}", dest_path, e),
    }
}

fn generate_placeholder_biases(dest_path: &Path) {
    // Placeholder biases based on typical secp256k1 patterns
    // These would be replaced by actual computed values
    let placeholder_biases = r#"pub const MAGIC9_BIASES: [(u8, u8, u8, u8, u32); 9] = [
    (0, 0, 0, 0, 128),  // Placeholder for index 9379
    (1, 1, 1, 1, 129),  // Placeholder for index 28687
    (2, 2, 2, 2, 127),  // Placeholder for index 33098
    (0, 3, 3, 3, 130),  // Placeholder for index 12457
    (1, 4, 4, 4, 126),  // Placeholder for index 18902
    (2, 5, 5, 5, 131),  // Placeholder for index 21543
    (0, 6, 6, 6, 125),  // Placeholder for index 27891
    (1, 7, 7, 7, 132),  // Placeholder for index 31234
    (2, 8, 8, 8, 124),  // Placeholder for index 4567
];"#;

    if let Err(e) = std::fs::write(dest_path, placeholder_biases) {
        eprintln!("Error writing placeholder biases: {}", e);
    }
}

fn compute_pubkey_biases(hex_str: &str) -> Result<(u8, u8, u8, u8, u32), Box<dyn std::error::Error>> {
    // Parse compressed pubkey hex
    let hex_clean = hex_str.trim().trim_start_matches("0x");
    let bytes = hex::decode(hex_clean)?;

    if bytes.len() != 33 || (bytes[0] != 0x02 && bytes[0] != 0x03) {
        return Err("Invalid compressed pubkey format".into());
    }

    // Extract x coordinate
    let x_bytes: [u8; 32] = bytes[1..33].try_into()?;
    let x_big = num_bigint::BigUint::from_bytes_be(&x_bytes);

    // Compute modular residues
    let mod3 = (x_big.clone() % 3u32).to_u32_digits()[0] as u8;
    let mod9 = (x_big.clone() % 9u32).to_u32_digits()[0] as u8;
    let mod27 = (x_big.clone() % 27u32).to_u32_digits()[0] as u8;
    let mod81 = (x_big.clone() % 81u32).to_u32_digits()[0] as u8;

    // Compute Hamming weight
    let hamming = x_bytes.iter().map(|b| b.count_ones()).sum::<u32>();

    Ok((mod3, mod9, mod27, mod81, hamming))
}
// Compile CUDA kernels using cc crate
fn compile_cuda_kernels() {
    let cuda_path = std::env::var("CUDA_HOME").unwrap_or("/usr/local/cuda".to_string());
    println!("cargo:rustc-env=CUDA_HOME={}", cuda_path);
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");

    let mut builder = cc::Build::new();
    builder.cuda(true)
        .include("src/gpu/cuda")
        .flag("-O3")
        .flag("-arch=sm_86") // Default RTX 3070; override via env
        .flag("-diag-suppress=63") // Existing shift warnings
        .flag("-diag-suppress=177") // Unused declarations
        .flag("-diag-suppress=550"); // Unused variables/shared mem

    // Allow override via environment variable
    if let Ok(arch) = env::var("GPU_ARCH") {
        builder.flag(&format!("-arch={}", arch));
    } // RTX 5090, suppress shift warnings

    let cu_files = vec![
        "bigint_mul", "step",
        "bias_check_kernel", "gold_cluster", "mod27_kernel", "mod81_kernel"
    ];

    for file in cu_files {
        builder.file(format!("src/gpu/cuda/{}.cu", file));
    }

    builder.compile("gpu_kernels");
}
