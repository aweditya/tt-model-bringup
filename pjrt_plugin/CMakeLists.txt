cmake_minimum_required(VERSION 3.20)
project(pjrt_plugin_tt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- PJRT C API header (real, from XLA matching jaxlib 0.6.2) ---
set(PJRT_API_DIR "${CMAKE_SOURCE_DIR}/third_party/pjrt")

# --- Build the plugin shared library ---
# Phase 1: no ttnn dependency. Just the PJRT skeleton.
add_library(pjrt_plugin_tt SHARED
    src/plugin.cc
    src/client.cc
)

target_include_directories(pjrt_plugin_tt PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${PJRT_API_DIR}
)

# Export only GetPjrtApi
if(UNIX AND NOT APPLE)
    target_link_options(pjrt_plugin_tt PRIVATE
        -Wl,--version-script=${CMAKE_SOURCE_DIR}/src/plugin.lds
    )
endif()

install(TARGETS pjrt_plugin_tt LIBRARY DESTINATION lib)
