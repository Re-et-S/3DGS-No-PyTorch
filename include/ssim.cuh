#pragma once
#include "cuda_buffer.cuh"

void fusedssim_forward(
    int H, int W, int channels, float C1, float C2,
    const float* rendered_image,
    const float* gt_image,
    float* ssim_map,
    float* dm_dmu1,
    float* dm_dsigma1_sq,
    float* dm_dsigma12
);

void fusedssim_backward(
    int H, int W, int channels, float C1, float C2,
    const float* rendered_image,
    const float* gt_image,
    float  grad_scale,
    float* ssim_grads,
    float* dm_dmu1,
    float* dm_dsigma1_sq,
    float* dm_dsigma12
);

struct SSIMData {
    int W;
    int H;
    int channels;
    CudaBuffer<float> d_ssim_map;
    CudaBuffer<float> d_ssim_grads;
    CudaBuffer<float> d_dm_dmu1;
    CudaBuffer<float> d_dm_dsigma1_sq;
    CudaBuffer<float> d_dm_dsigma12;

    SSIMData(int width, int height, int channels):
        W(width),
        H(height),
        channels(channels),
        d_ssim_map(channels*width*height),
        d_ssim_grads(channels*width*height),
        d_dm_dmu1(channels*width*height),
        d_dm_dsigma1_sq(channels*width*height),
        d_dm_dsigma12(channels*width*height)
        {} 
};
