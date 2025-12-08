cmake_minimum_required(VERSION 3.18)

# Auto-detect CUDA Architecture using nvidia-smi if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    find_program(NVIDIA_SMI "nvidia-smi")
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE CUDA_COMPUTE_CAPABILITY
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(CUDA_COMPUTE_CAPABILITY)
            # Handle multiple GPUs (take the first one)
            string(REPLACE "\n" ";" CUDA_COMPUTE_CAPABILITY_LIST ${CUDA_COMPUTE_CAPABILITY})
            list(GET CUDA_COMPUTE_CAPABILITY_LIST 0 FIRST_GPU_ARCH)

            # Remove decimal point (e.g., 8.6 -> 86)
            string(REPLACE "." "" ARCH_CODE ${FIRST_GPU_ARCH})
            set(CMAKE_CUDA_ARCHITECTURES ${ARCH_CODE})
            message(STATUS "Auto-detected CUDA Architecture: ${ARCH_CODE} (from ${FIRST_GPU_ARCH})")
        else()
            message(WARNING "nvidia-smi found but failed to query compute capability. Defaulting to 'native'.")
            set(CMAKE_CUDA_ARCHITECTURES native)
        endif()
    else()
        message(WARNING "nvidia-smi not found. Defaulting CMAKE_CUDA_ARCHITECTURES to 'native'.")
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
else()
    message(STATUS "Using specified CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

project(GaussianSplattingCUDA LANGUAGES CXX CUDA)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

# ==============================================================================
#  CONFIGURATION & FLAGS
# ==============================================================================
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

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
