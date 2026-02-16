use std::env;
use std::process;

fn main() {
    // Simple test to verify GLV4 basis loads
    println!("Testing GLV4 basis...");

    // Try to access the GLV4 basis
    let _basis = speedbitcrack::math::constants::GLV4_BASIS;
    println!("✅ GLV4 basis accessed successfully!");

    // Try to access the first element
    let first_col = &_basis[0];
    println!("✅ GLV4 basis first column: {:?}", first_col[0]);

    // Exit successfully
    process::exit(0);
}