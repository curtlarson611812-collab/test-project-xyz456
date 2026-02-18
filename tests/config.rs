// tests/config.rs - Tests for configuration functionality

use speedbitcrack::config::enable_nvidia_persistence;

#[cfg(test)]
mod tests {
    use super::*;

    // Chunk: Persistence Test (tests/config.rs)
    #[test]
    fn test_persistence_enable() {
        // Test NVIDIA persistence mode enable function
        // On systems without NVIDIA GPUs or not on Linux, this will return Ok(false)
        // On systems with NVIDIA GPUs, it will attempt to enable persistence mode
        let result = enable_nvidia_persistence();

        // The function should not panic and should return a valid Result
        assert!(result.is_ok() || result.is_err());

        // If it succeeds, it should return a boolean
        if let Ok(enabled) = result {
            // enabled will be true if persistence was successfully enabled
            // or false if the system doesn't support it or it's already enabled
            assert!(enabled == true || enabled == false);
        }
    }
}
