#!/bin/bash

# Create a minimal working version
cat > test_compile.rs << 'EOL'
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone)]
struct FlowExecutionMode;
#[derive(Clone)]
struct FlowInstance;
#[derive(Clone)]
struct ResourceAllocation;
#[derive(Clone)]
struct PerformanceTargets;
#[derive(Clone)]
struct AdaptationMetrics;

struct FlowControlState {
    current_mode: FlowExecutionMode,
    active_flows: HashMap<String, FlowInstance>,
    flow_priorities: Vec<String>,
    resource_allocation: ResourceAllocation,
    performance_targets: PerformanceTargets,
    adaptation_metrics: AdaptationMetrics,
}

struct HybridGpuManager {
    flow_control: Arc<Mutex<FlowControlState>>,
    scheduler: Arc<Mutex<()>>,
}

impl HybridGpuManager {
    pub fn test(&self) -> Result<()> {
        Ok(())
    }
}
EOL

echo "Testing basic impl..."
rustc --crate-type lib test_compile.rs --extern anyhow=/usr/lib/rustlib/x86_64-unknown-linux-gnu/lib/libanyhow-*.rlib 2>&1
if [ $? -ne 0 ]; then
    echo "Basic impl failed"
    exit 1
fi

echo "Basic impl works. Now testing with async function..."
# Test async function
cat > test_compile2.rs << 'EOL'
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::Instant;

#[derive(Clone)]
struct FlowExecutionMode;
#[derive(Clone)]
struct FlowInstance;
#[derive(Clone)]
struct ResourceAllocation;
#[derive(Clone)]
struct PerformanceTargets;
#[derive(Clone)]
struct AdaptationMetrics;

struct FlowControlState {
    current_mode: FlowExecutionMode,
    active_flows: HashMap<String, FlowInstance>,
    flow_priorities: Vec<String>,
    resource_allocation: ResourceAllocation,
    performance_targets: PerformanceTargets,
    adaptation_metrics: AdaptationMetrics,
}

struct HybridGpuManager {
    flow_control: Arc<Mutex<FlowControlState>>,
    scheduler: Arc<Mutex<()>>,
}

impl HybridGpuManager {
    pub async fn test_async(&self) -> Result<()> {
        Ok(())
    }
}
EOL

echo "Testing async function..."
rustc --crate-type lib test_compile2.rs --extern anyhow=/usr/lib/rustlib/x86_64-unknown-linux-gnu/lib/libanyhow-*.rlib 2>&1
if [ $? -ne 0 ]; then
    echo "Async function failed"
    exit 1
fi

echo "Async function works. Now testing tokio spawn..."
# Test tokio spawn
cat > test_compile3.rs << 'EOL'
use anyhow::Result;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::Instant;
use tokio;

#[derive(Clone)]
struct FlowExecutionMode;
#[derive(Clone)]
struct FlowInstance;
#[derive(Clone)]
struct ResourceAllocation;
#[derive(Clone)]
struct PerformanceTargets;
#[derive(Clone)]
struct AdaptationMetrics;

struct FlowControlState {
    current_mode: FlowExecutionMode,
    active_flows: HashMap<String, FlowInstance>,
    flow_priorities: Vec<String>,
    resource_allocation: ResourceAllocation,
    performance_targets: PerformanceTargets,
    adaptation_metrics: AdaptationMetrics,
}

struct HybridGpuManager {
    flow_control: Arc<Mutex<FlowControlState>>,
    scheduler: Arc<Mutex<()>>,
}

impl HybridGpuManager {
    pub async fn test_tokio(&self) -> Result<()> {
        let handles = vec!["test".to_string()].into_iter().map(|flow_id| {
            tokio::spawn(async move {
                println!("Executing {}", flow_id);
            })
        });
        
        for handle in handles {
            let _ = handle.await;
        }
        
        Ok(())
    }
}
EOL

echo "Testing tokio spawn..."
rustc --crate-type lib test_compile3.rs --extern tokio=/usr/lib/rustlib/x86_64-unknown-linux-gnu/lib/libtokio-*.rlib --extern anyhow=/usr/lib/rustlib/x86_64-unknown-linux-gnu/lib/libanyhow-*.rlib 2>&1
if [ $? -ne 0 ]; then
    echo "Tokio spawn failed"
    exit 1
fi

echo "All basic tests pass. The issue must be in the complex code."
