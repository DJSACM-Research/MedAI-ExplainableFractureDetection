#!/bin/bash
# Test runner script for MedAI Explainable Fracture Detection
# Run this script to execute tests with various options

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Display usage
usage() {
    cat << EOF
Usage: ./run_tests.sh [OPTION]

Options:
    all         Run all tests (default)
    unit        Run only unit tests
    integration Run only integration tests
    diagnostic  Run diagnostic agent tests
    educational Run educational agent tests
    explain     Run explainability agent tests
    knowledge   Run knowledge agent tests
    ensemble    Run cross-validation/ensemble agent tests
    coverage    Run tests with coverage report
    quick       Run tests without verbose output
    watch       Run tests and watch for changes
    help        Display this help message

Examples:
    ./run_tests.sh all
    ./run_tests.sh unit
    ./run_tests.sh coverage
    ./run_tests.sh diagnostic
EOF
}

# Run all tests
run_all_tests() {
    print_info "Running all tests..."
    python -m pytest tests/ -v
}

# Run unit tests only
run_unit_tests() {
    print_info "Running unit tests..."
    python -m pytest tests/unit/ -v
}

# Run integration tests only
run_integration_tests() {
    print_info "Running integration tests..."
    python -m pytest tests/integration/ -v
}

# Run tests for specific agent
run_agent_tests() {
    local agent=$1
    print_info "Running tests for $agent agent..."
    python -m pytest tests/ -m "$agent" -v
}

# Run tests with coverage
run_coverage() {
    print_info "Running tests with coverage report..."
    python -m pytest tests/ -v --cov=src/agents --cov-report=html --cov-report=term-missing
    print_info "Coverage report generated in htmlcov/index.html"
}

# Run tests quietly
run_quick_tests() {
    print_info "Running tests (quiet mode)..."
    python -m pytest tests/ -q
}

# Watch mode (requires pytest-watch)
run_watch_mode() {
    print_info "Running tests in watch mode..."
    print_warn "This requires 'pytest-watch' to be installed"
    print_warn "Install with: pip install pytest-watch"
    ptw tests/ -- -v
}

# Check if pytest is installed
check_pytest() {
    if ! python -m pytest --version > /dev/null 2>&1; then
        print_error "pytest is not installed"
        print_info "Install with: pip install pytest"
        exit 1
    fi
}

# Main script logic
main() {
    check_pytest
    
    case "${1:-all}" in
        all)
            run_all_tests
            ;;
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        diagnostic)
            run_agent_tests "diagnostic"
            ;;
        educational)
            run_agent_tests "educational"
            ;;
        explain)
            run_agent_tests "explainability"
            ;;
        knowledge)
            run_agent_tests "knowledge"
            ;;
        ensemble)
            run_agent_tests "ensemble"
            ;;
        coverage)
            run_coverage
            ;;
        quick)
            run_quick_tests
            ;;
        watch)
            run_watch_mode
            ;;
        help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
