#!/bin/bash
#===============================================================================
# NCCL Automated Test Script
# 
# Test Configurations:
#   1. P2P=0, SHM=0 (Default: both P2P and SHM enabled)
#   2. P2P=0, SHM=1 (P2P only, SHM disabled)
#   3. P2P=1, SHM=0 (SHM only, P2P disabled)
#   4. P2P=1, SHM=1 (Both P2P and SHM disabled)
#
# Usage:
# git clone https://github.com/NVIDIA/nccl-tests.git
# bash run_nccl.sh
#===============================================================================

set -e

#===============================================================================
# Configuration
#===============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/nccl-tests//build"
RESULTS_DIR="${SCRIPT_DIR}/nccl-tests/results"

# NCCL library path
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}

# Test parameters
MIN_BYTES="8"
MAX_BYTES="128M"
STEP_FACTOR="2"
N_ITER="20"
WARMUP_ITER="5"

#===============================================================================
# Colors for output
#===============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#===============================================================================
# Functions
#===============================================================================

print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if nccl-tests directory exists, clone if not
check_nccl_tests() {
    if [ ! -d "${SCRIPT_DIR}/nccl-tests" ]; then
        print_info "nccl-tests directory not found. Cloning from GitHub..."
        git clone https://github.com/NVIDIA/nccl-tests.git
        if [ $? -ne 0 ]; then
            print_error "Failed to clone nccl-tests repository"
            exit 1
        fi
        print_info "Successfully cloned nccl-tests repository"
    else
        print_info "Found nccl-tests directory"
    fi
}

# Check if nccl-tests is built
check_build() {
    if [ ! -f "${BUILD_DIR}/all_reduce_perf" ] || [ ! -f "${BUILD_DIR}/alltoall_perf" ]; then
        print_error "Test binaries not found. Building nccl-tests..."
        
        # Try to build with make
        print_info "Build command: make -j CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.11/site-packages/nvidia/nccl"
        
        make -j CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/lib/python3.11/site-packages/nvidia/nccl
        
        # Check again after build
        if [ ! -f "${BUILD_DIR}/all_reduce_perf" ] || [ ! -f "${BUILD_DIR}/alltoall_perf" ]; then
            print_error "Build failed. Please check the build output above."
            exit 1
        fi
        
        print_info "Build successful!"
    fi
    print_info "Found all_reduce_perf at ${BUILD_DIR}/all_reduce_perf"
    print_info "Found alltoall_perf at ${BUILD_DIR}/alltoall_perf"
}

# Create results directory
create_results_dir() {
    mkdir -p "${RESULTS_DIR}"
    print_info "Results will be saved to: ${RESULTS_DIR}"
}

# Run a single test configuration
run_test() {
    local p2p_disable="$1"
    local shm_disable="$2"
    local ngpu="$3"
    local config_name="$4"
    local test_program="$5"  # New parameter: test program name
    
    # Reverse the naming: 0 -> 1 (enabled), 1 -> 0 (disabled) for file naming
    local p2p_enable_flag=$((1 - p2p_disable))
    local shm_enable_flag=$((1 - shm_disable))
    
    local output_file="${RESULTS_DIR}/${test_program}_p2p_${p2p_enable_flag}_shm_${shm_enable_flag}_${ngpu}gpu.txt"
    
    print_info "Running: ${config_name} (${ngpu}-GPU) with ${test_program}"
    print_info "  NCCL_P2P_DISABLE=${p2p_disable} NCCL_SHM_DISABLE=${shm_disable}"
    if [ "${p2p_disable}" -eq 0 ]; then
        export NCCL_P2P_LEVEL=SYS # Use system-level P2P
        print_info "  Using system-level P2P"
    fi
    
    NCCL_P2P_DISABLE=${p2p_disable} NCCL_SHM_DISABLE=${shm_disable} \
        "${BUILD_DIR}/${test_program}" \
        -b "${MIN_BYTES}" \
        -e "${MAX_BYTES}" \
        -f "${STEP_FACTOR}" \
        -g "${ngpu}" \
        -n "${N_ITER}" \
        -w "${WARMUP_ITER}" \
        2>&1 | tee "${output_file}"
    
    print_info "Results saved to: ${output_file}"
    echo ""
}

# Run all tests for all_reduce_perf
run_all_reduce_tests() {
    print_header "Starting NCCL all_reduce_perf Tests"
    
    # Define test matrix: P2P_DISABLE, SHM_DISABLE, GPU_COUNT, CONFIG_NAME, TEST_PROGRAM
    local tests=(
        "0:0:4:Default (P2P=0,SHM=0):all_reduce_perf"
        "0:1:4:P2P_only (P2P=0,SHM=1):all_reduce_perf"
        "1:0:4:SHM_only (P2P=1,SHM=0):all_reduce_perf"
        "1:1:4:Both_disabled (P2P=1,SHM=1):all_reduce_perf"
        "0:0:8:Default (P2P=0,SHM=0):all_reduce_perf"
        "0:1:8:P2P_only (P2P=0,SHM=1):all_reduce_perf"
        "1:0:8:SHM_only (P2P=1,SHM=0):all_reduce_perf"
        "1:1:8:Both_disabled (P2P=1,SHM=1):all_reduce_perf"
    )
    
    for test in "${tests[@]}"; do
        IFS=':' read -r p2p shm ngpu name test_program <<< "${test}"
        run_test "${p2p}" "${shm}" "${ngpu}" "${name}" "${test_program}"
    done
}

# Run all tests for alltoall_perf
run_alltoall_tests() {
    print_header "Starting NCCL alltoall_perf Tests"
    
    # Define test matrix: P2P_DISABLE, SHM_DISABLE, GPU_COUNT, CONFIG_NAME, TEST_PROGRAM
    local tests=(
        "0:0:4:Default (P2P=0,SHM=0):alltoall_perf"
        "0:1:4:P2P_only (P2P=0,SHM=1):alltoall_perf"
        "1:0:4:SHM_only (P2P=1,SHM=0):alltoall_perf"
        "1:1:4:Both_disabled (P2P=1,SHM=1):alltoall_perf"
        "0:0:8:Default (P2P=0,SHM=0):alltoall_perf"
        "0:1:8:P2P_only (P2P=0,SHM=1):alltoall_perf"
        "1:0:8:SHM_only (P2P=1,SHM=0):alltoall_perf"
        "1:1:8:Both_disabled (P2P=1,SHM=1):alltoall_perf"
    )
    
    for test in "${tests[@]}"; do
        IFS=':' read -r p2p shm ngpu name test_program <<< "${test}"
        run_test "${p2p}" "${shm}" "${ngpu}" "${name}" "${test_program}"
    done
}

# Extract peak busbw from result file
extract_peak_busbw() {
    local file="$1"
    local column="$2"  # 7 for out-of-place busbw, 12 for in-place busbw
    grep -E "^   " "${file}" | grep -v "OutOfBounds" | grep -v "Avg" | \
        awk -v col="${column}" '{print $col}' | sort -n | tail -1
}

# Helper: extract bandwidths from a result file
extract_bw() {
    local file="$1"
    if [ ! -f "${file}" ]; then
        echo "N/A N/A"
        return
    fi
    local algobw busbw
    algobw=$(extract_peak_busbw "${file}" 7)
    busbw=$(extract_peak_busbw "${file}" 12)
    [ -z "${algobw}" ] && algobw="N/A" || algobw=$(printf "%.2f" "${algobw}")
    [ -z "${busbw}" ] && busbw="N/A" || busbw=$(printf "%.2f" "${busbw}")
    echo "${algobw} ${busbw}"
}

# Generate summary table for all_reduce_perf, alltoall_perf
generate_summary() {
    local filename="$1"
    print_header "NCCL all_reduce_perf Test Results Summary (128MB, GB/s)"
    
    printf "%-30s %-15s %-15s %-15s %-15s\n" \
        "Configuration" "4-GPU (algobw)" "4-GPU (busbw)" "8-GPU (algobw)" "8-GPU (busbw)"
    printf "%-30s %-15s %-15s %-15s %-15s\n" \
        "$(printf '%.0s-' {1..30})" "$(printf '%.0s-' {1..15})" \
        "$(printf '%.0s-' {1..15})" "$(printf '%.0s-' {1..15})" "$(printf '%.0s-' {1..15})"
    
    # Define configurations
    local configs=(
        "P2P=1, SHM=1 (Default):1:1"
        "P2P=1, SHM=0 (P2P only):1:0"
        "P2P=0, SHM=1 (SHM only):0:1"
        "P2P=0, SHM=0 (Both disabled):0:0"
    )
    
    for config in "${configs[@]}"; do
        IFS=':' read -r name p2p shm <<< "${config}"
        
        read bw_4_algobw bw_4_busbw <<< "$(extract_bw "${RESULTS_DIR}/${filename}_p2p_${p2p}_shm_${shm}_4gpu.txt")"
        read bw_8_algobw bw_8_busbw <<< "$(extract_bw "${RESULTS_DIR}/${filename}_p2p_${p2p}_shm_${shm}_8gpu.txt")"
        
        printf "%-30s %-15s %-15s %-15s %-15s\n" \
            "${name}" "${bw_4_algobw}" "${bw_4_busbw}" "${bw_8_algobw}" "${bw_8_busbw}"
    done
    
    printf "%s\n" "$(printf '%.0s=' {1..95})"
}

#===============================================================================
# Main
#===============================================================================

main() {
    print_header "NCCL Automated Test Suite"
    
    # Pre-checks
    check_nccl_tests
    check_build
    create_results_dir
    
    run_all_reduce_tests
    run_alltoall_tests
    generate_summary "all_reduce_perf"
    generate_summary "alltoall_perf"
    
    print_header "All Tests Completed!"
    print_info "To view detailed results: cat ${RESULTS_DIR}/*.txt"
}

# Run main function
main "$@"
