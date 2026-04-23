cmake_minimum_required(VERSION 3.20)
project(pjrt_plugin_tt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- PJRT C API header (real, from XLA matching jaxlib 0.6.2) ---
set(PJRT_API_DIR "${CMAKE_SOURCE_DIR}/third_party/pjrt")

# --- Python (for calling engine.py from C++) ---
# Force Python 3.10 to match runtime (jax is installed in 3.10)
set(Python3_EXECUTABLE "/usr/bin/python3.10")
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Development)

# --- Build the plugin shared library ---
add_library(pjrt_plugin_tt SHARED
    src/plugin.cc
    src/client.cc
)

target_include_directories(pjrt_plugin_tt PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${PJRT_API_DIR}
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(pjrt_plugin_tt PRIVATE ${Python3_LIBRARIES})

# Export only GetPjrtApi
if(UNIX AND NOT APPLE)
    target_link_options(pjrt_plugin_tt PRIVATE
        -Wl,--version-script=${CMAKE_SOURCE_DIR}/src/plugin.lds
    )
endif()

install(TARGETS pjrt_plugin_tt LIBRARY DESTINATION lib)
