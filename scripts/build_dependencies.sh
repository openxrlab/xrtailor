#!/bin/bash
# set -x

# Set project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INSTALL_DIR="${PROJECT_ROOT}/install"
THIRD_PARTY_DIR="${PROJECT_ROOT}/3rd_party"

# Create install directory if not exists
mkdir -p "${INSTALL_DIR}"

# Define dependencies and their build configurations in order
declare -A DEPS=(
    ["glfw"]="-DGLFW_BUILD_WAYLAND=OFF -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF"
    ["glm"]="-DGLM_BUILD_TESTS=OFF -DGLM_ENABLE_CXX_17=ON"
    ["imath"]="-DBUILD_DOCS=OFF"
    ["alembic"]="-DImath_DIR=../../../install/Imath/lib/cmake/Imath"
    ["jsoncpp"]=""
    ["zlib"]=""
    ["cnpy"]="-DCMAKE_PREFIX_PATH=../../install/zlib \
              -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    ["spdlog"]="-DCMAKE_CXX_STANDARD=17 -DSPDLOG_BUILD_EXAMPLE=OFF"
)

# Function to build dependency
build_dependency() {
    local dep_name=$1
    local extra_args=$2

    echo
    echo "Building ${dep_name}..."
    echo "--------------------------------------"

    if [ ! -d "${THIRD_PARTY_DIR}/${dep_name}" ]; then
        echo "Error: Directory ${dep_name} not found!"
        return 1
    fi

    cd "${THIRD_PARTY_DIR}/${dep_name}"

    rm -rf build
    mkdir -p build

    echo "Configuring ${dep_name}..."
    cmake -S . -B build \
          -DCMAKE_INSTALL_PREFIX="../../install/${dep_name}" \
          -DCMAKE_PREFIX_PATH="../../install" \
          -G "Ninja" ${extra_args}
    if [ $? -ne 0 ]; then
        echo "CMake configure failed for ${dep_name}"
        return 1
    fi

    echo "Building ${dep_name}..."
    cmake --build build --config Release
    if [ $? -ne 0 ]; then
        echo "Build failed for ${dep_name}"
        return 1
    fi

    echo "Installing ${dep_name}..."
    cmake --install build
    if [ $? -ne 0 ]; then
        echo "Install failed for ${dep_name}"
        return 1
    fi

    echo "${dep_name} built and installed successfully"
    echo
    cd "${PROJECT_ROOT}"
    return 0
}

# 清理函数
clean_builds() {
    echo "Cleaning build directories..."
    for dep_name in "${!DEPS[@]}"; do
        echo "Cleaning ${dep_name}..."
        rm -rf "${THIRD_PARTY_DIR}/${dep_name}/build"
        rm -rf "${INSTALL_DIR}/${dep_name}"
    done
    echo "Clean completed"
}
  
# 处理命令行参数
if [ "$1" = "clean" ]; then
    clean_builds
    exit 0
fi

# 检查必要的工具
command -v cmake >/dev/null 2>&1 || { echo "Error: cmake is required but not installed"; exit 1; }
command -v ninja >/dev/null 2>&1 || { echo "Error: ninja is required but not installed"; exit 1; }

# Main execution
echo "Starting dependency build process..."
echo "====================================="

# Build dependencies in specific order
declare -a BUILD_ORDER=(
    "glfw"
    "glm"
    "imath"
    "alembic"
    "jsoncpp"
    "zlib"
    "cnpy"
    "spdlog"
)

for dep_name in "${BUILD_ORDER[@]}"; do
    build_dependency "${dep_name}" "${DEPS[${dep_name}]}"
    if [ $? -ne 0 ]; then
        echo "Build process failed at ${dep_name}"
        exit 1
    fi
done

echo
echo "All dependencies built successfully!"
echo "====================================="

cd "${PROJECT_ROOT}"
