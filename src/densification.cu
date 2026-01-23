#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <glm/glm.hpp>
#include <cooperative_groups.h>
#include <cfloat>
#include "cuda_buffer.cuh"
#include "densification.cuh"
#include <cstdio> 
#include <iostream>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

__device__ float3 sample_standard_normal(curandState& state) {
    float3 z;
    // Box-Muller for x and y
    float u1 = curand_uniform(&state);
    float u2 = curand_uniform(&state);
    float radius = sqrtf(-2.0f * logf(u1 + 1e-6f));
    float theta = 2.0f * 3.14159265f * u2;
    z.x = radius * cosf(theta);
    z.y = radius * sinf(theta);
    
    // Box-Muller for z
    float u3 = curand_uniform(&state);
    float u4 = curand_uniform(&state);
    float radius2 = sqrtf(-2.0f * logf(u3 + 1e-6f));
    float theta2 = 2.0f * 3.14159265f * u4;
    z.z = radius2 * cosf(theta2);
    
    return z;
}

__global__ void init_curand_kernel(curandState* state, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

__global__ void accumulate_gradients_kernel(
    const int P,
    const float2* dL_dmean2D, // From backward pass
    const int* radii,            // From forward pass (visibility check)
    float* accum,                // The accumulation buffer
    int* denom)                  // The counter
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    // Only accumulate for visible Gaussians
    if (radii[idx] > 0) {
        // Calculate gradient magnitude
        float mag = sqrtf((dL_dmean2D[idx].x * dL_dmean2D[idx].x) + (dL_dmean2D[idx].y * dL_dmean2D[idx].y));

        // Accumulate
        accum[idx] += mag;
        denom[idx] += 1;
        
    }
}

__global__ void normalize_gradients_kernel(
    int P,
    const float* accum,
    const int* denom,
    float* out_normalized) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    int count = denom[idx];
    if (count > 0) {
        out_normalized[idx] = accum[idx] / count;
    } else {
        out_normalized[idx] = 0.0f;
    }
}

GradientStats compute_gradient_stats(
    int P,
    const float* accum_max_pos_grad,
    const int* denom,
    void* temp_storage,
    size_t& temp_storage_bytes) {
    GradientStats stats = {0.0f, 0.0f};
    if (P == 0) return stats;

    // 1. Allocate Temporary "Normalized Gradient" Buffer
    // Ideally this should be passed in or allocated from a scratch pool.
    // For now, we malloc it, but in production use Trainer::geomBuffer etc.
    float* d_norm_grads;
    CUDA_CHECK(cudaMalloc(&d_norm_grads, P * sizeof(float)));

    // 2. Run Normalization
    int block = 256;
    int grid = (P + block - 1) / block;
    normalize_gradients_kernel<<<grid, block>>>(P, accum_max_pos_grad, denom, d_norm_grads);
    CUDA_CHECK(cudaGetLastError());

    // 3. Compute MAX using CUB
    float* d_max_out;
    float* d_sum_out;
    CUDA_CHECK(cudaMalloc(&d_max_out, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_out, sizeof(float)));

    // Max Reduction
    // --- QUERY SIZE ---
    if (temp_storage == nullptr) {
        size_t size_max = 0;
        size_t size_sum = 0;
        
        cub::DeviceReduce::Max(nullptr, size_max, d_norm_grads, d_max_out, P);
        cub::DeviceReduce::Sum(nullptr, size_sum, d_norm_grads, d_sum_out, P);
        
        // Return the larger requirement
        temp_storage_bytes = (size_max > size_sum) ? size_max : size_sum;
        
        cudaFree(d_norm_grads); cudaFree(d_max_out); cudaFree(d_sum_out);
        return stats; 
    }
    
    // Run Max
    cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, d_norm_grads, d_max_out, P);
    
    // Run Sum (reuse temp storage if large enough, otherwise unsafe. 
    // CUB usually needs same size for Max and Sum. Let's assume passed buffer is big enough.)
    cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_norm_grads, d_sum_out, P);

    // 4. Retrieve Results
    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(&stats.max_grad, d_max_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum_out, sizeof(float), cudaMemcpyDeviceToHost));
    
    stats.mean_grad = h_sum / P;

    // 5. Cleanup
    cudaFree(d_norm_grads);
    cudaFree(d_max_out);
    cudaFree(d_sum_out);

    return stats;
}

__global__ void reset_opacity_kernel(
    const int P,
    float* opacities               
    )                 
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    opacities[idx] = -3.89182f; // logit(0.02f)
    
}

__global__ void mark_densification_candidates_kernel(
    const int P,
    const float* accum_max_pos_grad,
    const int* denom,
    const glm::vec3* scales,
    const float* opacities, // Need opacities for pruning
    int* radii, // Need radius for pruning
    int* decisions,         // Output: 0=Keep, 1=Prune, 2=Clone, 3=Split
    int* index_count,             // Output: Keep=1, Prune=0, Clone=2, Clone=2
    float grad_threshold,
    float percent_dense,
    float scene_extent,
    int max_screen_size_threshold,
    float min_opacity)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    // --- 1. PRUNING CHECK (Highest Priority) ---
    // Condition: Opacity too low OR Scale too huge (floater)
    
    float opacity = sigmoid(opacities[idx]); // Assuming pre-activation storage
    glm::vec3 s = scales[idx];
    float max_s = fmaxf(expf(s.x), fmaxf(expf(s.y), expf(s.z))); // World space scale

    if (opacity < min_opacity) {
        decisions[idx] = 1; // PRUNE
        index_count[idx] = 0;
        return;
    }
    // 1. Screen-Space size check and world-space size check
    // radii[idx] is the radius in pixels calculated by preprocessCUDA
    if (radii[idx] > max_screen_size_threshold || max_s > 2.0*scene_extent) {
        decisions[idx] = 1; // PRUNE
        index_count[idx] = 0;
        return;
    }
    
    // --- 2. DENSIFICATION CHECK ---
    
    // Calculate Average Gradient
    int count = denom[idx];
    float avg_grad = (count > 0) ? (accum_max_pos_grad[idx] / count) : 0.0f;

    if (avg_grad > grad_threshold) {
        
        if (max_s <= (percent_dense * scene_extent)) {
            decisions[idx] = 2; // CLONE (Small)
            index_count[idx] = 2;
        } else {
            decisions[idx] = 3; // SPLIT (Big)
            index_count[idx] = 2;
        }
    } else {
        decisions[idx] = 0; // KEEP
        index_count[idx] = 1;
    }
}

__global__ void count_decisions_kernel(int P, const int* decisions, int* global_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    int decision = decisions[idx];
    
    // Safety check for invalid decision codes
    if (decision >= 0 && decision <= 3) {
        atomicAdd(&global_counts[decision], 1);
    }
}

template <typename T>
__device__ void copy_param(
    const T* __restrict__ src, 
    T* __restrict__ dst, 
    int old_idx, 
    int new_idx) 
{
    dst[new_idx] = src[old_idx];
}

template <typename T>
__device__ void duplicate_param(
    const T* __restrict__ src, 
    T* __restrict__ dst, 
    int old_idx, 
    int new_idx_1, 
    int new_idx_2) 
{
    T val = src[old_idx];
    dst[new_idx_1] = val;
    dst[new_idx_2] = val;
}

template <typename T>
__device__ void copy_strided_param(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int old_idx,
    int new_idx,
    int stride)
{
    for (int k = 0; k < stride; ++k) {
        dst[new_idx * stride + k] = src[old_idx * stride + k];
    }
}

// Helper to duplicate strided arrays
template <typename T>
__device__ void duplicate_strided_param(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int old_idx,
    int new_idx_1,
    int new_idx_2,
    int stride)
{
    for (int k = 0; k < stride; ++k) {
        T val = src[old_idx * stride + k];
        dst[new_idx_1 * stride + k] = val;
        dst[new_idx_2 * stride + k] = val;
    }
}

__global__ void densify_kernel(
    const int P_old,
    const int* decisions,      // 0=Keep, 1=Prune, 2=Clone, 3=Split
    const int* scan_offsets,   // From InclusiveSum
    const GaussianData old_d,  
    GaussianData new_d,        
    curandState* rand_states,
    int sh_degree,
    int max_sh_coeffs
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P_old) return;

    int decision = decisions[idx];
    
    // 1. Prune: Do nothing
    if (decision == 1) return;

    // Calculate write index
    // The scan gives the cumulative count INCLUDING this one.
    // So start index is scan[idx] - count[idx]
    int count = (decision >= 2) ? 2 : 1; 
    int dest_idx = scan_offsets[idx] - count;

    // --- KEEP (Copy 1) ---
    if (decision == 0) {
        // Data
        copy_param(old_d.points, new_d.points, idx*3, dest_idx*3);     
        copy_param(old_d.points, new_d.points, idx*3+1, dest_idx*3+1);
        copy_param(old_d.points, new_d.points, idx*3+2, dest_idx*3+2);

        copy_param(old_d.scales, new_d.scales, idx, dest_idx);
        copy_param(old_d.rotations, new_d.rotations, idx, dest_idx);
        copy_param(old_d.opacities, new_d.opacities, idx, dest_idx);
        
        // Colors (DC is float* stride 3)
        copy_strided_param(old_d.dc, new_d.dc, idx, dest_idx, 3);
        
        // SHs (float* stride max_sh_coeffs * 3)
        copy_strided_param(old_d.shs, new_d.shs, idx, dest_idx, max_sh_coeffs * 3);

        // Optimizer State (Keep Momentum)
        copy_strided_param(old_d.m_points, new_d.m_points, idx, dest_idx, 3);
        copy_strided_param(old_d.v_points, new_d.v_points, idx, dest_idx, 3);
        
        copy_strided_param(old_d.m_scales, new_d.m_scales, idx, dest_idx, 3);
        copy_strided_param(old_d.v_scales, new_d.v_scales, idx, dest_idx, 3);

        copy_strided_param(old_d.m_rots, new_d.m_rots, idx, dest_idx, 4);
        copy_strided_param(old_d.v_rots, new_d.v_rots, idx, dest_idx, 4);

        copy_param(old_d.m_opacities, new_d.m_opacities, idx, dest_idx);
        copy_param(old_d.v_opacities, new_d.v_opacities, idx, dest_idx);

        copy_strided_param(old_d.m_dc, new_d.m_dc, idx, dest_idx, 3);
        copy_strided_param(old_d.v_dc, new_d.v_dc, idx, dest_idx, 3);
        
        copy_strided_param(old_d.m_shs, new_d.m_shs, idx, dest_idx, max_sh_coeffs * 3);
        copy_strided_param(old_d.v_shs, new_d.v_shs, idx, dest_idx, max_sh_coeffs * 3);

        return;
    }

    // --- CLONE or SPLIT (Copy 2) ---
    int dest1 = dest_idx;
    int dest2 = dest_idx + 1;

    // Duplicate basic properties first
    duplicate_strided_param(old_d.points, new_d.points, idx, dest1, dest2, 3); // Position
    duplicate_param(old_d.scales, new_d.scales, idx, dest1, dest2);
    duplicate_param(old_d.rotations, new_d.rotations, idx, dest1, dest2);
    duplicate_param(old_d.opacities, new_d.opacities, idx, dest1, dest2);
    duplicate_strided_param(old_d.dc, new_d.dc, idx, dest1, dest2, 3);
    duplicate_strided_param(old_d.shs, new_d.shs, idx, dest1, dest2, max_sh_coeffs * 3);

    // Reset Optimizer State
    auto zero_optimizer = [&](float* m, float* v, int d1, int d2, int stride) {
        for(int k=0; k<stride; ++k) {
            m[d1*stride + k] = 0.0f; v[d1*stride + k] = 0.0f;
            m[d2*stride + k] = 0.0f; v[d2*stride + k] = 0.0f;
        }
    };
    zero_optimizer(new_d.m_points, new_d.v_points, dest1, dest2, 3);
    zero_optimizer(new_d.m_scales, new_d.v_scales, dest1, dest2, 3);
    zero_optimizer(new_d.m_rots, new_d.v_rots, dest1, dest2, 4);
    new_d.m_opacities[dest1] = 0.0f; new_d.v_opacities[dest1] = 0.0f;
    new_d.m_opacities[dest2] = 0.0f; new_d.v_opacities[dest2] = 0.0f;
    zero_optimizer(new_d.m_dc, new_d.v_dc, dest1, dest2, 3);
    zero_optimizer(new_d.m_shs, new_d.v_shs, dest1, dest2, max_sh_coeffs * 3);

    // --- SPLIT LOGIC ---
    curandState local_state = rand_states[idx];
    if (decision == 3) { 
        float split_factor = logf(1.6f);
        new_d.scales[dest1].x -= split_factor;
        new_d.scales[dest2].x -= split_factor;
        new_d.scales[dest1].y -= split_factor;
        new_d.scales[dest2].y -= split_factor;
        new_d.scales[dest1].z -= split_factor;
        new_d.scales[dest2].z -= split_factor;

        
        // Sample from Gaussian
        // 1. Get Rotation Matrix (R) from Quaternion
        glm::vec4 q = old_d.rotations[idx];
        float r = q.x;
        float x = q.y;
        float y = q.z;
        float z = q.w;
    
        // Compute rotation matrix from quaternion
        glm::mat3 R = glm::mat3(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
            2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
            2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	    );

        
        // 2. Get Scale Vector (S)
        glm::vec3 s_log = old_d.scales[idx];
        glm::vec3 s = { expf(s_log.x), expf(s_log.y), expf(s_log.z) };
        
        // 3. Sample Standard Normal z ~ N(0, I)
        float3 nz = sample_standard_normal(local_state);
        glm::vec3 z_vec = {nz.x, nz.y, nz.z};

        // 4. Transform: v = R * S * z
        // S is diagonal, so S * z is element-wise multiplication
        glm::vec3 sz = s * z_vec; 
        glm::vec3 v = R * sz; // Matrix-vector multiply

        // 5. Apply Offsets
        // Dest1: +v
        new_d.points[3*dest1 + 0] += v.x;
        new_d.points[3*dest1 + 1] += v.y;
        new_d.points[3*dest1 + 2] += v.z;

        // Dest2: -v
        new_d.points[3*dest2 + 0] -= v.x;
        new_d.points[3*dest2 + 1] -= v.y;
        new_d.points[3*dest2 + 2] -= v.z;

    }
    
    // --- CLONE LOGIC (Using Gradient) ---
    if (decision == 2) {
        
        // Read gradient vector
        float grad_x = old_d.d_dL_dpoints[3*idx + 0];
        float grad_y = old_d.d_dL_dpoints[3*idx + 1];
        float grad_z = old_d.d_dL_dpoints[3*idx + 2];

        // Simple heuristic: move second copy slightly along gradient direction
        float scale_factor = 0.1f; // Move by 10% of the gaussian size
        glm::vec3 s = old_d.scales[idx];
        float size = expf(s.x); // Approximate size

        // Normalize gradient roughly to avoid flying away if grad is huge
        float grad_len = sqrtf(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
        if (grad_len > 1e-6f) {
             float dx = (grad_x / grad_len) * size * scale_factor;
             float dy = (grad_y / grad_len) * size * scale_factor;
             float dz = (grad_z / grad_len) * size * scale_factor;

             // Move the clone (dest2)
             new_d.points[3*dest2 + 0] += dx;
             new_d.points[3*dest2 + 1] += dy;
             new_d.points[3*dest2 + 2] += dz;
             
             // move original in opposite direction
             new_d.points[3*dest1 + 0] -= dx;
             new_d.points[3*dest1 + 1] -= dy;
             new_d.points[3*dest1 + 2] -= dz;
        }
    }

}

__global__ void get_prune_only_counts_kernel(
    int P, 
    const int* decisions, 
    int* counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;
    
    // Map decisions:
    // 1 (Prune) -> 0
    // 0 (Keep), 2 (Clone), 3 (Split) -> 1
    counts[idx] = (decisions[idx] == 1) ? 0 : 1;
}

__global__ void prune_only_kernel(
    const int P_old,
    const int* decisions,      // 0=Keep, 1=Prune, 2=Clone, 3=Split
    const int* scan_offsets,   // Calculated with (Dec!=1 ? 1 : 0)
    const GaussianData old_d,  
    GaussianData new_d,        
    int sh_degree,
    int max_sh_coeffs
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P_old) return;

    int decision = decisions[idx];
    
    // 1. Prune: Do nothing and exit
    if (decision == 1) return;

    // 2. Keep (Treating 0, 2, and 3 as Keep)
    // scan_offsets must have been calculated counting 0, 2, 3 as '1'
    int dest_idx = scan_offsets[idx] - 1;

    // Copy Data (Standard 1-to-1 copy)
    copy_param(old_d.points, new_d.points, idx*3, dest_idx*3);     
    copy_param(old_d.points, new_d.points, idx*3+1, dest_idx*3+1);
    copy_param(old_d.points, new_d.points, idx*3+2, dest_idx*3+2);

    copy_param(old_d.scales, new_d.scales, idx, dest_idx);
    copy_param(old_d.rotations, new_d.rotations, idx, dest_idx);
    copy_param(old_d.opacities, new_d.opacities, idx, dest_idx);
    
    copy_strided_param(old_d.dc, new_d.dc, idx, dest_idx, 3);
    copy_strided_param(old_d.shs, new_d.shs, idx, dest_idx, max_sh_coeffs * 3);

    // Optimizer State
    copy_strided_param(old_d.m_points, new_d.m_points, idx, dest_idx, 3);
    copy_strided_param(old_d.v_points, new_d.v_points, idx, dest_idx, 3);
    
    copy_strided_param(old_d.m_scales, new_d.m_scales, idx, dest_idx, 3);
    copy_strided_param(old_d.v_scales, new_d.v_scales, idx, dest_idx, 3);

    copy_strided_param(old_d.m_rots, new_d.m_rots, idx, dest_idx, 4);
    copy_strided_param(old_d.v_rots, new_d.v_rots, idx, dest_idx, 4);

    copy_param(old_d.m_opacities, new_d.m_opacities, idx, dest_idx);
    copy_param(old_d.v_opacities, new_d.v_opacities, idx, dest_idx);

    copy_strided_param(old_d.m_dc, new_d.m_dc, idx, dest_idx, 3);
    copy_strided_param(old_d.v_dc, new_d.v_dc, idx, dest_idx, 3);
    
    copy_strided_param(old_d.m_shs, new_d.m_shs, idx, dest_idx, max_sh_coeffs * 3);
    copy_strided_param(old_d.v_shs, new_d.v_shs, idx, dest_idx, max_sh_coeffs * 3);
}

// Host wrapper
void accumulate_gradients(
    int P,
    const float2* dL_dmean2D,
    const int* radii,
    float* accum,
    int* denom) 
{
    int block_size = 256;
    int grid_size = (P + block_size - 1) / block_size;
    accumulate_gradients_kernel<<<grid_size, block_size>>>(
        P, 
        dL_dmean2D, 
        radii, 
        accum, 
        denom
    );
}

void reset_opacities(
    const int P,
    float* opacities)                 
{
    int block_size = 256;
    int grid_size = (P + block_size - 1) / block_size;
    reset_opacity_kernel<<<grid_size, block_size>>>(P, opacities);
}

void mark_densification_candidates(
    int P,
    const float* accum_max_pos_grad,
    const int* denom,
    const glm::vec3* scales,
    const float* opacities,
    int* radii,
    int* decisions,
    int* index_count,
    float grad_threshold,
    float percent_dense,
    float scene_extent,
    int max_screen_size_threshold,
    float min_opacity) 
{
    int block_size = 256;
    int grid_size = (P + block_size - 1) / block_size;
    
    mark_densification_candidates_kernel<<<grid_size, block_size>>>(
        P, accum_max_pos_grad, denom, scales, opacities, radii, 
        decisions, index_count, grad_threshold, percent_dense, scene_extent, max_screen_size_threshold, min_opacity
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

// Host Wrapper
void densify(
    const int P_old,
    const int* decisions,
    const int* scan_offsets,
    GaussianData old_d,
    GaussianData new_d,
    int sh_degree,
    int max_sh_coeffs,
    curandState* rand_states 
) {
    int block_size = 256;
    int grid_size = (P_old + block_size - 1) / block_size;

    densify_kernel<<<grid_size, block_size>>>(
        P_old,
        decisions,
        scan_offsets,
        old_d,
        new_d,
        rand_states, // Pass the state buffer
        sh_degree,
        max_sh_coeffs
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

void compute_prune_only_counts(int P, const int* decisions, int* counts) {
    int block_size = 256;
    int grid_size = (P + block_size - 1) / block_size;
    get_prune_only_counts_kernel<<<grid_size, block_size>>>(P, decisions, counts);
    CUDA_CHECK(cudaDeviceSynchronize());
}

void prune(
    int P_old,
    const int* decisions,
    const int* scan_offsets,
    GaussianData old_d,
    GaussianData new_d,
    int sh_degree,
    int max_sh_coeffs
) {
    int block_size = 256;
    int grid_size = (P_old + block_size - 1) / block_size;
    
    prune_only_kernel<<<grid_size, block_size>>>(
        P_old, decisions, scan_offsets, old_d, new_d, sh_degree, max_sh_coeffs
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

void init_random_states(curandState* states, int n, unsigned long long seed) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    init_curand_kernel<<<grid_size, block_size>>>(states, n, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper: Calculate the diagonal length of the scene bounding box
// This is used to normalize thresholds (e.g. "don't clone if larger than 1% of scene")
float compute_scene_extent(const std::vector<glm::vec3>& points) {
    glm::vec3 min_pt(FLT_MAX);
    glm::vec3 max_pt(-FLT_MAX);

    for (const auto& p : points) {
        min_pt = glm::min(min_pt, p);
        max_pt = glm::max(max_pt, p);
    }

    return glm::length(max_pt - min_pt);
}

void print_densification_stats(int P, const int* decisions) {
    // 0: Keep, 1: Prune, 2: Clone, 3: Split
    int h_counts[4] = {0, 0, 0, 0};
    int* d_counts = nullptr;

    CUDA_CHECK(cudaMalloc(&d_counts, 4 * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counts, 0, 4 * sizeof(int)));

    int block_size = 256;
    int grid_size = (P + block_size - 1) / block_size;

    count_decisions_kernel<<<grid_size, block_size>>>(P, decisions, d_counts);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_counts, d_counts, 4 * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Clean up
    CUDA_CHECK(cudaFree(d_counts));

    // Print Report
    std::cout << "\n[Densification Statistics]" << std::endl;
    std::cout << "  Total Gaussians: " << P << std::endl;
    std::cout << "  KEEP  (0): " << h_counts[0] << std::endl;
    std::cout << "  PRUNE (1): " << h_counts[1] << std::endl;
    std::cout << "  CLONE (2): " << h_counts[2] << " (Under-reconstructed regions)" << std::endl;
    std::cout << "  SPLIT (3): " << h_counts[3] << " (Over-reconstructed regions)" << std::endl;
    
    int net_change = h_counts[2] + h_counts[3] - h_counts[1];
    std::cout << "  Net Change: " << (net_change > 0 ? "+" : "") << net_change << std::endl;
    
    if (h_counts[3] + h_counts[2] > P * 0.8) {
        std::cout << "  WARNING: >80% of Gaussians are densifying. Threshold might be too low." << std::endl;
    }
}
