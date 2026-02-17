use anyhow::Result;
use std::sync::{Arc, Mutex};

struct ResourceAllocation {
    cpu_threads: usize,
}

impl ResourceAllocation {
    fn new() -> Self {
        Self {
            cpu_threads: num_cpus::get(),
        }
    }
}
