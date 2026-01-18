#ifndef DEBUG_UTILS_CUH
#define DEBUG_UTILS_CUH

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include "auxiliary.cuh"

__global__ void checkNansKernel(
    int P,
    const float* dL_dmeans,
    const float* dL_dcov3Ds,
    const float* dL_dshs,
    const float* dL_dopacities,
    const glm::vec3* dL_dscales,
    const glm::vec4* dL_drots
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Check Means Gradients
    if (isnan(dL_dmeans[3*idx]) || isnan(dL_dmeans[3*idx+1]) || isnan(dL_dmeans[3*idx+2])) {
        printf("NaN detected in dL_dmeans at idx %d: %f %f %f\n", idx, dL_dmeans[3*idx], dL_dmeans[3*idx+1], dL_dmeans[3*idx+2]);
    }

    // Check Cov3D Gradients
    bool covNan = false;
    for (int k=0; k<6; k++) {
        if (isnan(dL_dcov3Ds[6*idx + k])) covNan = true;
    }
    if (covNan) {
        printf("NaN detected in dL_dcov3Ds at idx %d\n", idx);
    }

    // Check Scale Gradients
    if (isnan(dL_dscales[idx].x) || isnan(dL_dscales[idx].y) || isnan(dL_dscales[idx].z)) {
        printf("NaN detected in dL_dscales at idx %d\n", idx);
    }

    // Check Rotation Gradients
    if (isnan(dL_drots[idx].x) || isnan(dL_drots[idx].y) || isnan(dL_drots[idx].z) || isnan(dL_drots[idx].w)) {
        printf("NaN detected in dL_drots at idx %d\n", idx);
    }

    // Check Opacity Gradients
    if (isnan(dL_dopacities[idx])) {
        printf("NaN detected in dL_dopacities at idx %d\n", idx);
    }
}

void checkNans(
    int P,
    const float* dL_dmeans,
    const float* dL_dcov3Ds,
    const float* dL_dshs,
    const float* dL_dopacities,
    const glm::vec3* dL_dscales,
    const glm::vec4* dL_drots
) {
    checkNansKernel<<<(P + 255) / 256, 256>>>(P, dL_dmeans, dL_dcov3Ds, dL_dshs, dL_dopacities, dL_dscales, dL_drots);
    cudaDeviceSynchronize();
}

#endif // DEBUG_UTILS_CUH
