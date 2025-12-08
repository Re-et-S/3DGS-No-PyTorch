#pragma once

#include <glm/glm.hpp>
#include <curand_kernel.h>

// A lightweight struct to hold pointers to Gaussian attributes.
// Passed by value to kernels to reduce argument list size.
struct GaussianData {
    float* points;
    glm::vec3* scales;
    glm::vec4* rotations;
    float* opacities;
    float* dc;
    float* shs;

    float*     d_dL_dpoints;
    // glm::vec3* d_dL_dscales;
    // glm::vec4* d_dL_drotations;
    // float*     d_dL_dopacities;
    // float*     d_dL_ddc;
    // float*     d_dL_dshs;
    // float*     d_dL_dcolors;
    // float*     d_dL_dcov3Ds;
    // float2*    d_dL_dmeans2D;
    // float4*    d_dL_dconic_opacity;
    // float*     d_dL_dinvdepths;

    float* m_points; float* v_points;
    float* m_scales; float* v_scales;
    float* m_rots; float* v_rots;
    float* m_opacities; float* v_opacities;
    float* m_dc; float* v_dc;
    float* m_shs; float* v_shs;

};

struct GradientStats {
    float max_grad;
    float mean_grad;
};

void accumulate_gradients(
    int P,
    const float2* dL_dmean2D,
    const int* radii,
    float* accum,
    int* denom);

void reset_opacities(
    const int P,
    float* opacities);

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
    float min_opacity);

void densify(
    const int P_old,
    const int* decisions,
    const int* scan_offsets,
    GaussianData old_d,
    GaussianData new_d,
    int sh_degree,
    int max_sh_coeffs,
    curandState* rand_states
);

void compute_prune_only_counts(int P, const int* decisions, int* counts);

void prune(
    int P_old,
    const int* decisions,
    const int* scan_offsets,
    GaussianData old_d,
    GaussianData new_d,
    int sh_degree,
    int max_sh_coeffs
);

void init_random_states(curandState* states, int n, unsigned long long seed); 
float compute_scene_extent(const std::vector<glm::vec3>& points);
void print_densification_stats(int P, const int* decisions);

GradientStats compute_gradient_stats(
    int P,
    const float* accum_max_pos_grad,
    const int* denom,
    void* temp_storage,        // Pointer to existing scratch buffer
    size_t& temp_storage_bytes // In/Out size management
);
