cmake_minimum_required(VERSION 3.18)
project(GaussianSplattingCUDA LANGUAGES CXX CUDA)

# ==============================================================================
#  CONFIGURATION & FLAGS
# ==============================================================================
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3")
endif()

# ==============================================================================
#  FIND PACKAGES
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# GLM (Header Only)
find_path(GLM_INCLUDE_DIR glm/glm.hpp
    HINTS /usr/include /usr/local/include
    DOC "The directory containing glm/glm.hpp"
)

if(NOT GLM_INCLUDE_DIR)
    message(FATAL_ERROR "GLM not found! Please install libglm-dev.")
endif()

# ==============================================================================
#  SOURCE FILES
# ==============================================================================
# 1. Include Directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include 
    ${GLM_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}
)

# 2. Glob Sources
file(GLOB_RECURSE SOURCES 
    "${CMAKE_SOURCE_DIR}/src/*.cpp" 
    "${CMAKE_SOURCE_DIR}/src/*.cu"
)

if(NOT SOURCES)
    message(FATAL_ERROR "No source files found in ${CMAKE_SOURCE_DIR}/src.")
endif()

# ==============================================================================
#  TARGETS
# ==============================================================================
add_executable(train_app 
    ${SOURCES}
)

target_link_libraries(train_app 
    PRIVATE 
    CUDA::cudart
)
