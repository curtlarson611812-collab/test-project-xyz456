#!/bin/bash
# SpeedBitCrackV3 Development Environment Setup
# Installs quality assurance tools and pre-commit hooks

set -e

echo "🚀 Setting up SpeedBitCrackV3 Development Environment"
echo "====================================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1${NC}"
        return 0
    else
        echo -e "${RED}✗ $1 failed${NC}"
        return 1
    fi
}

# Install Rust tools
echo "🔧 Installing Rust development tools..."

if command -v rustc >/dev/null 2>&1; then
    echo "✓ Rust already installed"
else
    echo "❌ Rust not found. Please install Rust first: https://rustup.rs/"
    exit 1
fi

# Install rustfmt
if ! command -v rustfmt >/dev/null 2>&1; then
    echo "Installing rustfmt..."
    rustup component add rustfmt
    check_success "rustfmt installation"
else
    echo "✓ rustfmt already installed"
fi

# Install clippy
if ! command -v clippy >/dev/null 2>&1; then
    echo "Installing clippy..."
    rustup component add clippy
    check_success "clippy installation"
else
    echo "✓ clippy already installed"
fi

# Set up pre-commit hooks
echo "🎣 Setting up Git hooks..."

if [ -d .git ]; then
    # Install pre-commit hook
    cp scripts/pre-commit .git/hooks/pre-commit 2>/dev/null || {
        echo "Creating pre-commit hook..."
        mkdir -p .git/hooks
        cp scripts/pre-commit .git/hooks/pre-commit
    }
    chmod +x .git/hooks/pre-commit
    check_success "pre-commit hook installation"

    # Install commit-msg hook (optional)
    if [ -f scripts/commit-msg ]; then
        cp scripts/commit-msg .git/hooks/commit-msg
        chmod +x .git/hooks/commit-msg
        check_success "commit-msg hook installation"
    fi
else
    echo -e "${YELLOW}⚠️  Not a git repository - skipping hook installation${NC}"
fi

# Check scripts
echo "📜 Verifying scripts..."

if [ -x scripts/quality_check.sh ]; then
    echo "✓ quality_check.sh is executable"
else
    chmod +x scripts/quality_check.sh
    check_success "quality_check.sh permissions"
fi

if [ -x scripts/phase_tracker.sh ]; then
    echo "✓ phase_tracker.sh is executable"
else
    chmod +x scripts/phase_tracker.sh
    check_success "phase_tracker.sh permissions"
fi

# Initial quality check
echo "🔍 Running initial quality assessment..."

if ./scripts/quality_check.sh > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Initial quality check passed${NC}"
else
    echo -e "${YELLOW}⚠️  Initial quality check found issues${NC}"
    echo "Run './scripts/quality_check.sh' to see details"
fi

# Initial phase tracking
echo "📊 Running initial phase assessment..."

./scripts/phase_tracker.sh

echo ""
echo -e "${GREEN}🎉 Development environment setup complete!${NC}"
echo ""
echo "📖 Next steps:"
echo "1. Read DEVELOPMENT_WORKFLOW.md for detailed guidelines"
echo "2. Run './scripts/phase_tracker.sh' to see current status"
echo "3. Start with PHASE 1 critical blockers"
echo "4. Always run './scripts/quality_check.sh' before commits"
echo ""
echo -e "${BLUE}Happy coding! Remember: Quality over speed! ⚡${NC}"