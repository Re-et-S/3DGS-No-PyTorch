#pragma once
#include "ColmapLoader.h"
#include "scene.cuh"
#include "cuda_buffer.cuh"
#include "optimizer.cuh"
#include "ssim.cuh"
#include "config.h"
#include <curand_kernel.h>

class Trainer {
public:
    GaussianScene& scene;
    GaussianGrads& grads;
    Optimizer& optimizer;
    SSIMData& ssim_data;

    GaussianSplatting::TrainingConfig config;

    // Image size
    int W, H;
    
    CudaBuffer<char> geomBuffer;
    CudaBuffer<char> binningBuffer;
    CudaBuffer<char> imgBuffer;

    CudaBuffer<float>  d_gt_image;
    CudaBuffer<float>  d_out_color;
    CudaBuffer<float>  d_bg_color;
    CudaBuffer<float>  d_viewmatrix;
    CudaBuffer<float>  d_projmatrix;
    CudaBuffer<double> d_loss;
    CudaBuffer<float>  dL_dpixels;
    

    Trainer(GaussianScene& scene_ref, GaussianGrads& grads_ref, Optimizer& opt_ref, SSIMData& ssim_ref, int width, int height, const GaussianSplatting::TrainingConfig& cfg):
          scene(scene_ref), grads(grads_ref), optimizer(opt_ref), ssim_data(ssim_ref),
          config(cfg), W(width), H(height),
          d_gt_image(width * height * 3), d_bg_color(3), d_viewmatrix(16), d_projmatrix(16),
          d_out_color(width * height * 3),
          // d_final_T(width * height),
          // d_n_contrib(width * height),
          d_loss(1),
          dL_dpixels(width * height * 3),
          // 512MB scratch space
          geomBuffer(1024 * 1024 * 512),
          binningBuffer(1024 * 1024 * 512),
          imgBuffer(1024 * 1024 * 512)
    {
        d_bg_color.to_device({0.0f,0.0f,0.0f});
    }

    double train_step(const TrainingView& view, const CudaBuffer<float>& d_gt_image, int active_sh_degree);
    void reset_opacity();
    void densify_and_prune(float scene_extent, curandState* rand_states, bool prune_only);
    void get_current_render(std::vector<float>& h_render);

private:
    void compute_loss(int W, int H, const float* rendered, const float* gt);
};
