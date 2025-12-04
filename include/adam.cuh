#ifndef CUDA_RASTERIZER_ADAM_H_INCLUDED
#define CUDA_RASTERIZER_ADAM_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace ADAM {

void adamUpdate(
    float* param,
    const float* param_grad,
    float* exp_avg,
    float* exp_avg_sq,
    const uint32_t* tiles_touched, // changed from bool, use explicit type
    const float lr,
    const float b1,
    const float b2,
    const float eps,
    const uint32_t N,
    const uint32_t M);
}

#endif
