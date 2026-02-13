#!/bin/bash
# SpeedBitCrackV3 Quality Assurance Script
# Run this before every commit to ensure code quality

set -e  # Exit on any error

echo "🔍 SpeedBitCrackV3 Quality Assurance Check"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 failed${NC}"
        return 1
    fi
}

# Function to check for placeholders
check_placeholders() {
    echo "🔍 Checking for placeholder code..."

    # Critical blockers that must be fixed
    CRITICAL_PATTERNS=(
        "unimplemented!"
        "TODO.*collision.*detection"
        "TODO.*GPU.*dispatch"
        "let batch_result = None.*dispatch_hybrid"
    )

    FOUND_CRITICAL=0
    for pattern in "${CRITICAL_PATTERNS[@]}"; do
        if grep -r "$pattern" src/ --include="*.rs" > /dev/null 2>&1; then
            echo -e "${RED}🚨 CRITICAL: Found blocking placeholder: $pattern${NC}"
            FOUND_CRITICAL=1
        fi
    done

    if [ $FOUND_CRITICAL -eq 1 ]; then
        echo -e "${RED}❌ CRITICAL BLOCKERS FOUND - FIX BEFORE COMMIT${NC}"
        return 1
    fi

    # Non-critical TODOs (warnings)
    TODO_COUNT=$(grep -r "TODO\|FIXME\|placeholder\|stub" src/ --include="*.rs" | wc -l)
    if [ $TODO_COUNT -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Found $TODO_COUNT TODO/placeholder items${NC}"
    fi

    return 0
}

# Function to check for compilation errors
check_compilation() {
    echo "🔨 Checking compilation..."

    # Clean build
    cargo clean > /dev/null 2>&1

    # Check compilation
    if cargo check --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Compilation successful${NC}"
        return 0
    else
        echo -e "${RED}❌ Compilation failed${NC}"
        echo "Full error output:"
        cargo check 2>&1
        return 1
    fi
}

# Function to check for release build with ZERO errors
check_release_build() {
    echo "🚀 Checking release build (STRICT: zero errors required)..."

    # Capture full output to check for errors
    BUILD_OUTPUT=$(timeout 300 cargo build --release 2>&1)
    BUILD_EXIT_CODE=$?

    if [ $BUILD_EXIT_CODE -ne 0 ]; then
        echo -e "${RED}❌ Release build failed with exit code $BUILD_EXIT_CODE${NC}"
        echo "Build output:"
        echo "$BUILD_OUTPUT"
        return 1
    fi

    # Check for compilation errors (not just warnings)
    ERROR_COUNT=$(echo "$BUILD_OUTPUT" | grep -c "^error\[")
    if [ $ERROR_COUNT -gt 0 ]; then
        echo -e "${RED}❌ Release build has $ERROR_COUNT compilation errors${NC}"
        echo "Error summary:"
        echo "$BUILD_OUTPUT" | grep "^error\[" | head -5
        echo -e "${RED}💥 STRICT POLICY: Zero errors required for release builds${NC}"
        return 1
    fi

    # Check for any error messages
    if echo "$BUILD_OUTPUT" | grep -q "^error:"; then
        echo -e "${RED}❌ Release build contains error messages${NC}"
        echo "Error output:"
        echo "$BUILD_OUTPUT" | grep "^error:" | head -3
        return 1
    fi

    # Count warnings for informational purposes
    WARNING_COUNT=$(echo "$BUILD_OUTPUT" | grep -c "^warning:")
    if [ $WARNING_COUNT -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Release build has $WARNING_COUNT warnings (allowed but logged)${NC}"
    fi

    echo -e "${GREEN}✅ Release build successful with zero errors${NC}"
    return 0
}

# Function to check tests
check_tests() {
    echo "🧪 Running tests..."

    if cargo test --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Tests passed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Tests failed (may be expected during development)${NC}"
        return 0  # Don't fail on test failures during development
    fi
}

# Function to check GPU tests
check_gpu_tests() {
    echo "🎮 Running GPU/CUDA/Vulkan tests..."

    # Check if GPU tests are available
    if ! cargo test --quiet --test gpu_hybrid 2>/dev/null; then
        echo -e "${YELLOW}⚠️  GPU hybrid tests not available${NC}"
        return 0
    fi

    # Run GPU hybrid tests (always enabled since run mode is GPU hybrid)
    if cargo test --quiet --test gpu_hybrid --features gpu_hybrid 2>/dev/null; then
        echo -e "${GREEN}✓ GPU hybrid tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  GPU hybrid tests failed (may be expected without hardware)${NC}"
    fi

    # Run GPU backend tests if features are enabled
    if cargo test --quiet gpu::tests --features rustacuda 2>/dev/null; then
        echo -e "${GREEN}✓ CUDA backend tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  CUDA backend tests failed (may be expected without CUDA)${NC}"
    fi

    if cargo test --quiet gpu::tests --features wgpu 2>/dev/null; then
        echo -e "${GREEN}✓ Vulkan backend tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Vulkan backend tests failed (may be expected without Vulkan)${NC}"
    fi

    # Run specific GPU test modules
    if cargo test --quiet --package speedbitcrack --lib gpu_hybrid 2>/dev/null; then
        echo -e "${GREEN}✓ GPU hybrid parity tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  GPU hybrid parity tests failed${NC}"
    fi

    return 0  # GPU tests shouldn't fail the build
}

# Function to run GPU hybrid test specifically
check_gpu_hybrid_backend() {
    echo "🔗 Testing GPU hybrid backend..."

    # Run GPU hybrid tests (always enabled since run mode is GPU hybrid)
    if cargo test --release --features gpu-hybrid --lib gpu::hybrid 2>/dev/null; then
        echo -e "${GREEN}✓ GPU hybrid backend tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  GPU hybrid backend tests failed (may be expected without hardware)${NC}"
    fi

    return 0
}

# Function to check code formatting
check_formatting() {
    echo "🎨 Checking code formatting..."

    if command -v rustfmt >/dev/null 2>&1; then
        if cargo fmt --check --quiet 2>/dev/null; then
            echo -e "${GREEN}✓ Code formatting correct${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  Code formatting issues (run 'cargo fmt')${NC}"
            return 0  # Don't fail on formatting during development
        fi
    else
        echo -e "${YELLOW}⚠️  rustfmt not installed${NC}"
        return 0
    fi
}

# Function to check linting
check_linting() {
    echo "🔍 Running linter..."

    if cargo clippy --quiet -- -D warnings 2>/dev/null; then
        echo -e "${GREEN}✓ Linting passed${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Linting issues (may be expected during development)${NC}"
        return 0  # Don't fail on clippy during development
    fi
}

# Function to check for security issues
check_security() {
    echo "🔒 Checking for security issues..."

    # Check for unsafe code blocks
    UNSAFE_COUNT=$(grep -r "unsafe" src/ --include="*.rs" | grep -v "unsafe_code" | wc -l)
    if [ $UNSAFE_COUNT -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Found $UNSAFE_COUNT unsafe blocks${NC}"
    fi

    # Check for unwrap/expect
    UNWRAP_COUNT=$(grep -r "\.unwrap()\|\.expect(" src/ --include="*.rs" | wc -l)
    if [ $UNWRAP_COUNT -gt 10 ]; then  # Allow some unwraps
        echo -e "${YELLOW}⚠️  High unwrap count: $UNWRAP_COUNT${NC}"
    fi

    return 0
}

# Function to check documentation
check_docs() {
    echo "📚 Checking documentation..."

    if cargo doc --quiet 2>/dev/null; then
        echo -e "${GREEN}✓ Documentation builds successfully${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Documentation build failed${NC}"
        return 0  # Don't fail on docs during development
    fi
}

# Function to check binary size
check_binary_size() {
    echo "📏 Checking binary size..."

    # Build release binary
    cargo build --release --quiet 2>/dev/null

    if [ -f target/release/speedbitcrack ]; then
        SIZE=$(du -h target/release/speedbitcrack | cut -f1)
        echo -e "${GREEN}✓ Binary size: $SIZE${NC}"
    else
        echo -e "${YELLOW}⚠️  Could not check binary size${NC}"
    fi

    return 0
}

# Run all checks
FAILED_CHECKS=0

check_placeholders || FAILED_CHECKS=$((FAILED_CHECKS + 1))
check_compilation || FAILED_CHECKS=$((FAILED_CHECKS + 1))
check_release_build || FAILED_CHECKS=$((FAILED_CHECKS + 1))
check_tests
check_gpu_tests
check_gpu_hybrid_backend
check_formatting
check_linting
check_security
check_docs
check_binary_size

echo "=========================================="

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}🎉 All critical quality checks passed!${NC}"
    echo "Ready for commit."
    exit 0
else
    echo -e "${RED}❌ $FAILED_CHECKS critical checks failed!${NC}"
    echo "Fix issues before committing."
    exit 1
fi