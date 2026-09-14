# WASM build of the regicide core engine + Embind bindings.
# Configure with the Emscripten toolchain, e.g.:
#   emcmake cmake -G Ninja -B build && cmake --build build
# Emits dist/regicore.mjs (ES6 module) + dist/regicore.wasm.
cmake_minimum_required(VERSION 3.20)
project(regicore_wasm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/regi_py/core)

# The shared engine: every core/*.cc except the pybind binding (exports.cc)
# and the native SOLO entry point (main.cc). embind.cc is the WASM binding.
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS ${CORE_DIR}/*.cc)
list(REMOVE_ITEM CORE_SOURCES ${CORE_DIR}/exports.cc ${CORE_DIR}/main.cc)

add_executable(regicore ${CORE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/embind.cc)
target_include_directories(regicore PRIVATE ${CORE_DIR})
target_compile_options(regicore PRIVATE -O3 -fwasm-exceptions)
target_link_options(regicore PRIVATE
    -fwasm-exceptions
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=RegiModule
    -sENVIRONMENT=web,node
    -sALLOW_MEMORY_GROWTH=1
)
set_target_properties(regicore PROPERTIES
    OUTPUT_NAME regicore
    SUFFIX ".mjs"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dist
)
