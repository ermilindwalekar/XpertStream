cmake_minimum_required(VERSION 3.22)
project(
    XpertStream
    VERSION 0.1.0
    DESCRIPTION "Model-Agnostic Agentic Static-Shape Inference Engine for Intel NPUs"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings and optimizations for low-level systems
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Werror=return-type)
endif()

# Public interface include paths
include_directories(include)

add_subdirectory(src)
