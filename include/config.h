#pragma once

namespace GaussianSplatting {

struct TrainingConfig {
    // Training Iterations
    int total_iterations = 15000;
    int warmup_steps = 1000;

    // Intervals
    int sh_increase_interval = 1000;
    int densify_interval = 100;
    int opacity_reset_interval = 3000;
    int debug_image_interval = 200;
    int checkpoint_interval = 500;

    // Densification Thresholds
    float densify_grad_threshold_pixel = 170.0f;

    float densify_percent_dense = 0.01f;
    float densify_min_opacity = 0.01f;
    int densify_max_screen_size = 3000;

    // Loss Weights
    float lambda_dssim = 0.2f;

    // Buffer Limits
    size_t max_gaussian_count = 3900000;
};

} // namespace GaussianSplatting
