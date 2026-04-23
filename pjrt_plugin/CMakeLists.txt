cmake_minimum_required(VERSION 3.20)
project(pjrt_plugin_tt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- PJRT C API header ---
# We vendor the header from XLA. The version must match the installed jaxlib.
# To update: download pjrt_c_api.h from the XLA commit that jaxlib was built from.
set(PJRT_API_DIR "${CMAKE_SOURCE_DIR}/third_party/pjrt" CACHE PATH "Path to PJRT C API headers")

# --- MLIR / LLVM (for StableHLO parsing in Phase 3+) ---
# Built by scripts/setup_deps.sh into deps/install/
set(DEPS_PREFIX "${CMAKE_SOURCE_DIR}/deps/install" CACHE PATH "Prefix for MLIR/StableHLO/abseil")
# Not required for Phase 1 -- we stub Compile/Execute.
# When Phase 3 lands, uncomment:
# list(APPEND CMAKE_PREFIX_PATH "${DEPS_PREFIX}")
# find_package(MLIR REQUIRED CONFIG)
# find_package(LLVM REQUIRED CONFIG)

# --- Tenstorrent tt-metal / ttnn ---
# tt-metal is installed on the remote host. TT_METAL_HOME must be set.
if(NOT DEFINED ENV{TT_METAL_HOME})
    message(WARNING "TT_METAL_HOME not set. Trying /opt/tt-metal as default.")
    set(TT_METAL_HOME "/opt/tt-metal")
else()
    set(TT_METAL_HOME "$ENV{TT_METAL_HOME}")
endif()

# ttnn headers live under ttnn/cpp/ in the tt-metal tree
set(TTNN_INCLUDE_DIR "${TT_METAL_HOME}/ttnn/cpp" CACHE PATH "ttnn C++ headers")
set(TTNN_LIB_DIR "${TT_METAL_HOME}/build/lib" CACHE PATH "ttnn shared library dir")

# --- Build the plugin shared library ---
add_library(pjrt_plugin_tt SHARED
    src/plugin.cc
    src/client.cc
    src/buffer.cc
    src/executable.cc
)

target_include_directories(pjrt_plugin_tt PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${PJRT_API_DIR}
    ${TTNN_INCLUDE_DIR}
    ${TT_METAL_HOME}
    ${TT_METAL_HOME}/tt_metal
    # ttnn pulls in various tt-metal headers
    ${TT_METAL_HOME}/tt_metal/third_party/fmt/include
)

# Phase 1: link only ttnn. MLIR/StableHLO added in Phase 3.
find_library(TTNN_LIB NAMES _ttnn ttnn
    PATHS ${TTNN_LIB_DIR} ${TT_METAL_HOME}/build/lib
    NO_DEFAULT_PATH
)

if(TTNN_LIB)
    target_link_libraries(pjrt_plugin_tt PRIVATE ${TTNN_LIB})
else()
    message(WARNING "ttnn library not found at ${TTNN_LIB_DIR}. "
        "Build will succeed but the plugin won't link against ttnn. "
        "Set TTNN_LIB_DIR or ensure TT_METAL_HOME is correct.")
endif()

# Export GetPjrtApi as the plugin entry point
target_compile_definitions(pjrt_plugin_tt PRIVATE
    PJRT_PLUGIN_TT_EXPORT
)

# On Linux, ensure the entry point symbol is exported
if(UNIX AND NOT APPLE)
    target_link_options(pjrt_plugin_tt PRIVATE
        -Wl,--version-script=${CMAKE_SOURCE_DIR}/src/plugin.lds
    )
endif()

# Install
install(TARGETS pjrt_plugin_tt
    LIBRARY DESTINATION lib
)
