##
# 3D Gaussian Splatting Trainer
#
# @file Makefile
# @version 1.0
# --- Tools ---
NVCC = nvcc
CXX  = g++

# --- Architecture ---
# Change this to match your GPU (e.g., sm_86 for Ampere/RTX 30xx, sm_75 for Turing/RTX 20xx)
# You can run 'nvidia-smi' to check your GPU model.
GPU_ARCH = -arch=sm_86

# --- Flags ---
# NVCC Flags: 
# -g -G: Debug info (remove -G for release builds speed)
# --extended-lambda: Required for the functional memory allocators in rasterizer
# -std=c++17: Modern C++ features
NVCCFLAGS = -g -G $(GPU_ARCH) --extended-lambda -std=c++17 -Xcompiler "-O3"

# C++ Flags:
CXXFLAGS = -O3 -std=c++17 -g

# Includes (GLM, etc.)
# Assuming GLM is in standard paths. If not, add -I/path/to/glm here.
INCLUDES = -I.

# --- Targets ---
TARGET = train_app

# --- Source Files ---
# CUDA Sources (Kernels, Trainer, Main)
CUDA_SRCS = train.cu trainer.cu densification.cu rasterizer_impl.cu forward.cu backward.cu adam.cu ssim.cu

# C++ Sources (Loader, Utils)
CPP_SRCS = ColmapLoader.cpp ImageIO.cpp

# --- Object Files ---
# Automatically generate .o filenames from .cu and .cpp
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
CPP_OBJS  = $(CPP_SRCS:.cpp=.o)
ALL_OBJS  = $(CUDA_OBJS) $(CPP_OBJS)

# --- Rules ---

# Default target
all: $(TARGET)

# Link Step: Combine all objects into the final executable
$(TARGET): $(ALL_OBJS)
	$(NVCC) $(GPU_ARCH) $(ALL_OBJS) -o $@

# Compile CUDA files (.cu -> .o)
# We add separate compilation (-dc) if you strictly separate device code, 
# but standard compilation (-c) usually works for this project structure.
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ files (.cpp -> .o)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) $(ALL_OBJS) *.jpg

# Phony targets (not real files)
.PHONY: all clean
