#include "trainer.cuh"
#include "rasterizer.cuh"
#include "densification.cuh"
#include "ssim.cuh"
#include <cub/cub.cuh>       // Required for DeviceScan
#include <glm/gtc/type_ptr.hpp>

static __global__ void compute_loss_and_gradient(
    const float* rendered_image, // Planar [C][H][W]
    const float* gt_image,       // Planar [C][H][W]
    double* d_loss_output,        // Scalar (size 1)
    float* dL_dpixels,           // Planar [C][H][W]
    int W, int H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = W * H;
    if (idx >= num_pixels) return;

    float pixel_loss = 0.0f;
    for (int c = 0; c < 3; c++) {
        int p_idx = c * num_pixels + idx;
        float diff = rendered_image[p_idx] - gt_image[p_idx];
        
        // dL/d(pixel) = (render - gt)
        dL_dpixels[p_idx] = diff;
        
        // L = 0.5 * (render - gt)^2
        pixel_loss += 0.5f * diff * diff;
    }
    atomicAdd(d_loss_output, (double)pixel_loss);
}

static __global__ void compute_combined_loss_and_gradient(
    const float* rendered_image, // Planar [C][H][W]
    const float* gt_image,       // Planar [C][H][W]
    const float* ssim_grads,     // Planar [C][H][W] (From SSIM Backward)
    const float* ssim_map,       // Planar [C][H][W] (From SSIM Forward)
    double* d_loss_output,       // Scalar Accumulator
    float* dL_dpixels,           // Output: Combined Gradient
    float lambda,                // 0.2
    int W, int H)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_pixels = W * H;
    if (idx >= num_pixels) return;

    float pixel_loss_sum = 0.0f;
    float ssim_loss_sum = 0.0f;

    for (int c = 0; c < 3; c++) {
        int p_idx = c * num_pixels + idx;
        
        // 1. Pixel Difference (L2 in your case, typically L1 in paper)
        float diff = rendered_image[p_idx] - gt_image[p_idx];
        
        // 2. SSIM Contribution
        // We want to minimize (1 - SSIM), so gradient is -d(SSIM)
        float ssim_g = ssim_grads[p_idx]; 
        
        // L1 Loss = |render - gt|
        pixel_loss_sum += fabsf(diff);
        
        // d(L1)/dx = sign(diff)
        // If diff > 0, grad is 1. If diff < 0, grad is -1.
        float l1_grad = (diff > 0.0f) ? 1.0f : -1.0f;
        
        // Combine gradients: (1-lambda)*L1 + lambda*(1-SSIM)
        dL_dpixels[p_idx] = (1.0f - lambda) * l1_grad - lambda * ssim_g;
        
        ssim_loss_sum += (1.0f - ssim_map[p_idx]);
    }

    // Weighted sum of scalar losses
    float total_local_loss = (1.0f - lambda) * pixel_loss_sum + lambda * ssim_loss_sum;
    
    atomicAdd(d_loss_output, (double)total_local_loss);
}

double Trainer::train_step(const TrainingView& view, const CudaBuffer<float>& d_gt_image, int active_sh_degree) {
        if (view.width > W || view.height > H) {
            throw std::runtime_error("View dimensions exceed Trainer's allocated buffers!");
        }
        // 1. Setup Allocators (Lambdas capture member pointers)
        auto geomAlloc = [&](size_t s) { return geomBuffer.get(); };
        auto binAlloc = [&](size_t s) { return binningBuffer.get(); };
        auto imgAlloc = [&](size_t s) { return imgBuffer.get(); };

        // 2. Zero Gradients
        grads.clear_all();

        // 3. Upload Camera (reuse member buffers)
        CUDA_CHECK(cudaMemcpy(d_viewmatrix.get(), glm::value_ptr(view.view_matrix), 16 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_projmatrix.get(), glm::value_ptr(view.view_proj_matrix), 16 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(scene.d_cam_pos.get(), &view.camera_center, sizeof(glm::vec3), cudaMemcpyHostToDevice));

        float tan_fovx = view.fxfy_tanfov[2];
        float tan_fovy = view.fxfy_tanfov[3];

        float f_x_pixels =  (view.projection_matrix)[0][0] * (W/2.0f);
        float f_y_pixels = -(view.projection_matrix)[1][1] * (H/2.0f);

        // 4. Forward
        d_out_color.clear();
        d_loss.clear();
        
        int num_rendered = CudaRasterizer::Rasterizer::forward(
            geomAlloc, binAlloc, imgAlloc,
            scene.count, active_sh_degree, scene.max_sh_coeffs,
            d_bg_color.get(),
            W, H,
            scene.d_points.get(),
            scene.d_dc.get(),
            scene.d_shs.get(),
            nullptr,
            scene.d_opacities.get(),
            (float*)scene.d_scales.get(),
            1.0f,
            (float*)scene.d_rotations.get(),
            nullptr,
            d_viewmatrix.get(),
            d_projmatrix.get(),
            (float*)scene.d_cam_pos.get(),
            tan_fovx, tan_fovy,
            f_x_pixels, f_y_pixels,
            false, // prefiltered
            d_out_color.get(),
            nullptr, // depth
            scene.d_tiles_touched.get(),
            false, // antialiasing
            scene.d_radii.get(),
            false // debug
        );

        float lambda_dssim = config.lambda_dssim;

        fusedssim_forward(H, W, 3, 0.01f * 0.01f, 0.03f * 0.03f,
                    d_out_color.get(), // Rendered Image
                    d_gt_image.get(),  // Ground Truth
                    ssim_data.d_ssim_map.get(), ssim_data.d_dm_dmu1.get(),
                    ssim_data.d_dm_dsigma1_sq.get(),
                    ssim_data.d_dm_dsigma12.get());

        // float grad_scale = 1.0f / (W * H * 3.0f); // Normalize by total elements
        float grad_scale = 1.0f;
        
        fusedssim_backward(
            H, W, 3, 0.01f*0.01f, 0.03f*0.03f,
            d_out_color.get(),
            d_gt_image.get(),
            grad_scale,
            ssim_data.d_ssim_grads.get(), // Output Gradients
            ssim_data.d_dm_dmu1.get(),
            ssim_data.d_dm_dsigma1_sq.get(),
            ssim_data.d_dm_dsigma12.get()
        );
        
        // 5. Loss
        // compute_loss_and_gradient<<<(W * H + 255) / 256, 256>>>(
        // d_out_color.get(), d_gt_image.get(), d_loss.get(), dL_dpixels.get(), W, H
        // );
        compute_combined_loss_and_gradient<<<(W * H + 255) / 256, 256>>>(
            d_out_color.get(), 
            d_gt_image.get(), 
            ssim_data.d_ssim_grads.get(), 
            ssim_data.d_ssim_map.get(),   
            d_loss.get(), 
            dL_dpixels.get(),             
            lambda_dssim, 
            W, H
        );
        
        double h_loss = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss.get(), sizeof(double), cudaMemcpyDeviceToHost));

        // 6. Backward
        CudaRasterizer::Rasterizer::backward(
            scene.count, active_sh_degree, scene.max_sh_coeffs, num_rendered,
            d_bg_color.get(),
            W,H,
            scene.d_points.get(),
            scene.d_dc.get(),
            scene.d_shs.get(),
            nullptr,
            scene.d_opacities.get(),
            (float*)scene.d_scales.get(),
            1.0f,
            (float*)scene.d_rotations.get(),
            nullptr,
            d_viewmatrix.get(), 
            d_projmatrix.get(),
            (float*)scene.d_cam_pos.get(),
            tan_fovx, tan_fovy,
            f_x_pixels, f_y_pixels,
            scene.d_radii.get(),
            geomBuffer.get(),
            binningBuffer.get(),
            imgBuffer.get(),
            dL_dpixels.get(),
            grads.d_dL_dmeans2D.get(),
            (float4*)grads.d_dL_dconic_opacity.get(),
            grads.d_dL_dopacities.get(),
            grads.d_dL_dcolors.get(),
            grads.d_dL_dpoints.get(),
            grads.d_dL_dcov3Ds.get(),
            grads.d_dL_ddc.get(),
            grads.d_dL_dshs.get(),    // dL_dsh
            (float*)grads.d_dL_dscales.get(),
            (float*)grads.d_dL_drotations.get(),
            false,
            false);

        accumulate_gradients(
            scene.count,
            grads.d_dL_dmeans2D.get(),
            scene.d_radii.get(),
            optimizer.accum_max_pos_grad.get(),
            optimizer.denom.get()
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        // 7. Optimize
        optimizer.step(scene, grads, scene.d_tiles_touched);
        
        return h_loss;
    }

void Trainer::reset_opacity() {
    int P = scene.count;
    reset_opacities(P, scene.d_opacities.get());
    optimizer.reset_opacity_state();
}

void Trainer::densify_and_prune(float scene_extent, curandState* rand_states, bool prune_only) {
    int P = scene.count;
    float grad_threshold = config.densify_grad_threshold_pixel;
    float min_opacity = config.densify_min_opacity;
    float percent_dense = config.densify_percent_dense;
    int max_screen_size_threshold = config.densify_max_screen_size;
    
    // 1. Mark Candidates
    CudaBuffer<int> d_scan_counts(P); 

    mark_densification_candidates(
        P,
        optimizer.accum_max_pos_grad.get(),
        optimizer.denom.get(),
        scene.d_scales.get(),
        scene.d_opacities.get(),
        scene.d_radii.get(),
        optimizer.clone_mask.get(), // Decisions output
        d_scan_counts.get(),        // Counts output (1, 0, 2, 2)
        grad_threshold,
        percent_dense,
        scene_extent,
        max_screen_size_threshold,
        min_opacity
    );

    if (prune_only) {
        // Overwrite d_scan_counts so that Split/Clone (2,3) count as 1 (Keep)
        // and Prune (1) counts as 0.
        compute_prune_only_counts(
            P, 
            optimizer.clone_mask.get(), 
            d_scan_counts.get()
        );
    }

    size_t temp_bytes = 0;
    compute_gradient_stats(P, optimizer.accum_max_pos_grad.get(), optimizer.denom.get(), nullptr, temp_bytes); // query size

    GradientStats stats = compute_gradient_stats(
        P, 
        optimizer.accum_max_pos_grad.get(), 
        optimizer.denom.get(), 
        binningBuffer.get(), 
        temp_bytes);

    printf("[Gradient Stats] Max: %.6f | Mean: %.6f | Target Threshold: %.6f\n", 
       stats.max_grad, stats.mean_grad, grad_threshold);
    
    // 2. Scan (Inclusive Sum) to find new positions
    CudaBuffer<int> d_scan_offsets(P);
    
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    // Determine temp storage size
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_scan_counts.get(),
        d_scan_offsets.get(),
        P
    );

    // Use binningBuffer as scratch space
    d_temp_storage = binningBuffer.get();
    
    // Execute Scan
    cub::DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes,
        d_scan_counts.get(),
        d_scan_offsets.get(),
        P
    );

    // 3. Get New Total Count
    int new_P = 0;
    CUDA_CHECK(cudaMemcpy(&new_P, d_scan_offsets.get() + (P - 1), sizeof(int), cudaMemcpyDeviceToHost));
    
    printf("Densification: %d -> %d Gaussians\n", P, new_P);

    // 4. Allocate New Buffers
    GaussianScene new_scene(new_P, scene.sh_degree);
    Optimizer new_optimizer(new_P, scene.max_sh_coeffs);

    // 5. Prepare Structs for Kernel
    GaussianData old_d = {
        scene.d_points.get(), scene.d_scales.get(), scene.d_rotations.get(),
        scene.d_opacities.get(), scene.d_dc.get(), scene.d_shs.get(),
        grads.d_dL_dpoints.get(), // Gradients for cloning logic
        // optimizer moments ...
        optimizer.m_points.get(),     optimizer.v_points.get(),
        optimizer.m_scales.get(),     optimizer.v_scales.get(),
        optimizer.m_rots.get(),       optimizer.v_rots.get(),
        optimizer.m_opacities.get(),  optimizer.v_opacities.get(),
        optimizer.m_dc.get(),         optimizer.v_dc.get(),
        optimizer.m_shs.get(),        optimizer.v_shs.get()
    };

    GaussianData new_d = {
        new_scene.d_points.get(), new_scene.d_scales.get(), new_scene.d_rotations.get(),
        new_scene.d_opacities.get(), new_scene.d_dc.get(), new_scene.d_shs.get(), 
        nullptr, // No gradients needed for new scene yet
        new_optimizer.m_points.get(),     new_optimizer.v_points.get(),
        new_optimizer.m_scales.get(),     new_optimizer.v_scales.get(),
        new_optimizer.m_rots.get(),       new_optimizer.v_rots.get(),
        new_optimizer.m_opacities.get(),  new_optimizer.v_opacities.get(),
        new_optimizer.m_dc.get(),         new_optimizer.v_dc.get(),
        new_optimizer.m_shs.get(),        new_optimizer.v_shs.get()
    };

    // 6. Run Densify Kernel
    if (prune_only) {
        prune(
            P,
            optimizer.clone_mask.get(), // Reuse decisions
            d_scan_offsets.get(),       // Using corrected offsets
            old_d,
            new_d,
            scene.sh_degree,
            scene.max_sh_coeffs
        );
    } else {
        densify(
            P,
            optimizer.clone_mask.get(),
            d_scan_offsets.get(),
            old_d,
            new_d,
            scene.sh_degree,
            scene.max_sh_coeffs,
            rand_states
        );
    }

    print_densification_stats(P, optimizer.clone_mask.get());

    // 7. Swap, resize and clean Up
    scene.replace_with(new_scene);
    optimizer.replace_with(new_optimizer);
    grads.resize(new_P);
    
    // Clear accumulation buffers for next cycle
    optimizer.accum_max_pos_grad.clear();
    optimizer.denom.clear();
    optimizer.clone_mask.clear();
}

void Trainer::get_current_render(std::vector<float>& h_render) {
    d_out_color.from_device(h_render);
}
