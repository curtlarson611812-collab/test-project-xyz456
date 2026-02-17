#!/bin/bash

# SpeedBitCrackV3 Quality Assurance Gate
# MANDATORY: Run before EVERY commit
# STRICT: Blocks commits with critical issues

set -e  # Exit on any error

echo "🔍 SpeedBitCrackV3 Quality Assurance Check"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERROR_COUNT=0
WARNING_COUNT=0

# Function to report errors
report_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
    ((ERROR_COUNT++))
}

# Function to report warnings
report_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
    ((WARNING_COUNT++))
}

# Function to report success
report_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo ""
echo "1. 🚨 Critical Blocker Detection"
echo "-------------------------------"

# Check for unimplemented!() calls
if grep -r "unimplemented!()" src/ --include="*.rs" > /dev/null 2>&1; then
    report_error "Found unimplemented!() calls in source code"
    grep -r "unimplemented!()" src/ --include="*.rs"
else
    report_success "No unimplemented!() calls found"
fi

# Check for blocking TODO comments
if grep -r "TODO.*block" src/ --include="*.rs" > /dev/null 2>&1; then
    report_error "Found blocking TODO comments"
    grep -r "TODO.*block" src/ --include="*.rs"
else
    report_success "No blocking TODO comments found"
fi

# Check for placeholder functions
if grep -r "placeholder\|stub\|not implemented" src/ --include="*.rs" > /dev/null 2>&1; then
    report_error "Found placeholder/stub implementations"
    grep -r "placeholder\|stub\|not implemented" src/ --include="*.rs" | head -10
else
    report_success "No placeholder implementations found"
fi

echo ""
echo "2. 🔨 Compilation Verification"
echo "-----------------------------"

# Check basic compilation
echo "Running cargo check..."
if cargo check --quiet 2>&1; then
    report_success "Basic compilation successful"
else
    report_error "Basic compilation failed"
    cargo check 2>&1 | head -20
fi

# Check CUDA compilation if feature enabled
echo "Running cargo check --features cuda..."
if cargo check --features cuda --quiet 2>&1; then
    report_success "CUDA compilation successful"
else
    report_error "CUDA compilation failed"
    cargo check --features cuda 2>&1 | head -20
fi

# Check release build compilation
echo "Running cargo build --release..."
if cargo build --release --quiet 2>&1; then
    report_success "Release build compilation successful"
else
    report_error "Release build compilation failed"
    cargo build --release 2>&1 | head -20
fi

echo ""
echo "3. 🔐 Security & Correctness Review"
echo "----------------------------------"

# Check for unwrap/expect in hot paths
if grep -r "\.unwrap()" src/gpu/ src/kangaroo/ src/math/ --include="*.rs" > /dev/null 2>&1; then
    report_warning "Found unwrap() calls in hot paths (GPU/kangaroo/math)"
    grep -r "\.unwrap()" src/gpu/ src/kangaroo/ src/math/ --include="*.rs" | wc -l
    echo "  unwrap() calls found - review for security"
fi

# Check for expect() in hot paths
if grep -r "\.expect(" src/gpu/ src/kangaroo/ src/math/ --include="*.rs" > /dev/null 2>&1; then
    report_warning "Found expect() calls in hot paths (GPU/kangaroo/math)"
    grep -r "\.expect(" src/gpu/ src/kangaroo/ src/math/ --include="*.rs" | wc -l
    echo "  expect() calls found - review for security"
fi

echo ""
echo "4. 🎨 Code Quality Standards"
echo "---------------------------"

# Check rustfmt
echo "Running cargo fmt --check..."
if cargo fmt --check 2>&1; then
    report_success "Code formatting correct"
else
    report_error "Code formatting issues found"
    echo "Run: cargo fmt"
fi

# Check clippy (allow some warnings for performance)
echo "Running cargo clippy..."
if cargo clippy --quiet -- -W clippy::pedantic -W clippy::nursery 2>&1; then
    report_success "Clippy checks passed"
else
    CLIPPY_OUTPUT=$(cargo clippy -- -W clippy::pedantic -W clippy::nursery 2>&1)
    if echo "$CLIPPY_OUTPUT" | grep -q "error:"; then
        report_error "Clippy found errors"
        echo "$CLIPPY_OUTPUT" | head -10
    else
        report_warning "Clippy found warnings (review but may be acceptable)"
        echo "$CLIPPY_OUTPUT" | grep "warning:" | wc -l
        echo "  warnings found"
    fi
fi

echo ""
echo "5. 🧪 Basic Test Verification"
echo "----------------------------"

# Run basic tests
echo "Running cargo test (basic)..."
if cargo test --quiet 2>&1; then
    report_success "Basic tests passed"
else
    report_error "Basic tests failed"
    cargo test 2>&1 | head -20
fi

# Check if GPU hybrid tests exist and run them
if cargo test --test gpu_hybrid --quiet 2>&1; then
    report_success "GPU hybrid tests available and passing"
elif cargo test --list | grep -q gpu_hybrid; then
    report_warning "GPU hybrid tests exist but failed"
    cargo test --test gpu_hybrid 2>&1 | head -10
else
    report_warning "GPU hybrid tests not found"
fi

echo ""
echo "6. 📊 Summary"
echo "============="

echo "Critical blockers: $ERROR_COUNT"
echo "Warnings: $WARNING_COUNT"

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo -e "${RED}🚫 COMMIT BLOCKED: $ERROR_COUNT critical issues found${NC}"
    echo "Fix all errors before committing"
    exit 1
else
    echo ""
    echo -e "${GREEN}✅ QUALITY GATES PASSED${NC}"
    if [ $WARNING_COUNT -gt 0 ]; then
        echo -e "${YELLOW}⚠️  $WARNING_COUNT warnings found - review recommended${NC}"
    fi
    exit 0
fi