#pragma once

#include <iostream>
#include "ColmapLoader.h"
#include "cuda_buffer.cuh"
#include "nanoflann.hpp"

const float SH_C0_init = 0.28209479f; // 1 / (2 * sqrt(PI))
const float SAFE_EPS = 1e-10f;

class GaussianScene {
public:
    // Store the number of Gaussians in the scene
    size_t count;
    const int sh_degree;
    const int max_sh_coeffs;

    // --- GPU Buffers for Inputs to preprocess() ---
    CudaBuffer<float>     d_points;
    CudaBuffer<glm::vec3> d_scales;
    CudaBuffer<glm::vec4> d_rotations;
    CudaBuffer<float>     d_opacities;
    CudaBuffer<float>     d_dc;
    CudaBuffer<float>     d_shs;

    // --- GPU Buffers for Outputs of preprocess() ---
    CudaBuffer<bool>      d_clamped;
    CudaBuffer<glm::vec3> d_cam_pos;
    CudaBuffer<int>       d_radii;
    CudaBuffer<float2>    d_means2D;
    CudaBuffer<float>     d_depths;
    CudaBuffer<float>     d_cov3Ds;
    CudaBuffer<float>     d_rgb;
    CudaBuffer<float4>    d_conic_opacity;
    CudaBuffer<uint32_t>  d_tiles_touched;

    // Constructor handles all allocation and initial data transfer
    GaussianScene(
        int sh_degree,
        const std::vector<float>& h_points,
        const std::vector<glm::vec3>& h_scales,
        const std::vector<glm::vec4>& h_rotations,
        const std::vector<float>& h_opacities,
        const std::vector<float>& h_dc,
        const std::vector<float>& h_shs
    ) :
        count(h_scales.size()),
        sh_degree(sh_degree),
        max_sh_coeffs((sh_degree + 1) * (sh_degree + 1)),
        // Allocate input buffers
        d_points(count * 3),
        d_scales(count),
        d_rotations(count),
        d_opacities(count),
        d_dc(count*3),
        d_shs(count * (max_sh_coeffs) * 3),
        // Allocate output buffers
        d_clamped(count * 3),
        d_cam_pos(1),
        d_radii(count),
        d_means2D(count),
        d_depths(count),
        d_cov3Ds(count * 6),
        d_rgb(count * 3),
        d_conic_opacity(count),
        d_tiles_touched(count)
    {
        std::cout << "Allocating and uploading scene with " << count << " Gaussians to GPU..." << std::endl;

        // Copy all host data to the device
        d_points.to_device(h_points);
        d_scales.to_device(h_scales);
        d_rotations.to_device(h_rotations);
        d_opacities.to_device(h_opacities);
        d_dc.to_device(h_dc);
        d_shs.to_device(h_shs);

        std::cout << "Scene upload complete." << std::endl;
    }

    // Used for creating the 'new_scene' during densification
    GaussianScene(size_t new_count, int sh_degree) :
        count(new_count),
        sh_degree(sh_degree),
        max_sh_coeffs((sh_degree + 1) * (sh_degree + 1)),
        d_points(new_count * 3),
        d_scales(new_count),
        d_rotations(new_count),
        d_opacities(new_count),
        d_dc(new_count * 3),
        d_shs(new_count * max_sh_coeffs * 3),
        d_clamped(new_count * 3),
        d_cam_pos(1),
        d_radii(new_count),
        d_means2D(new_count),
        d_depths(new_count),
        d_cov3Ds(new_count * 6),
        d_rgb(new_count * 3),
        d_conic_opacity(new_count),
        d_tiles_touched(new_count)
    {
        // No uploads. Buffers are allocated but not initialized.
        // densify_kernel will write to them.
    }

    void replace_with(GaussianScene& other) {
        if (sh_degree != other.sh_degree) throw std::runtime_error("SH degree mismatch in swap");
        
        // Swap the count
        std::swap(count, other.count);

        // Swap all buffers
        d_points.swap(other.d_points);
        d_scales.swap(other.d_scales);
        d_rotations.swap(other.d_rotations);
        d_opacities.swap(other.d_opacities);
        d_dc.swap(other.d_dc);
        d_shs.swap(other.d_shs);
        
        d_clamped.swap(other.d_clamped);
        d_cam_pos.swap(other.d_cam_pos);
        d_radii.swap(other.d_radii);
        d_means2D.swap(other.d_means2D);
        d_depths.swap(other.d_depths);
        d_cov3Ds.swap(other.d_cov3Ds);
        d_rgb.swap(other.d_rgb);
        d_conic_opacity.swap(other.d_conic_opacity);
        d_tiles_touched.swap(other.d_tiles_touched);
    };

    void resize(size_t new_count, size_t sh_coeffs) {
        // 1. Create a temporary scene with the new size
        // This reuses all allocation logic in your constructor
        GaussianScene temp_scene(new_count, sh_coeffs);
        
        // 2. Swap the pointers
        // 'this' now owns the new buffers. 'temp_scene' owns the old buffers.
        this->replace_with(temp_scene);
    };

};

struct PointCloudAdapter {
    const std::vector<glm::vec3>& points;

    PointCloudAdapter(const std::vector<glm::vec3>& pts) : points(pts) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return points.size();
    }

    // Must return the L2_Simple distance between two points
    inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const {
        const float d0 = p1[0] - points[idx_p2].x;
        const float d1 = p1[1] - points[idx_p2].y;
        const float d2 = p1[2] - points[idx_p2].z;
        return d0*d0 + d1*d1 + d2*d2;
    }

    // Must return the dim'th component of the idx'th point
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        if (dim == 0) return points[idx].x;
        if (dim == 1) return points[idx].y;
        if (dim == 2) return points[idx].z;
        return 0;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; } // Not required
};

//  GaussianScene for all gradient buffers
struct GaussianGrads {
    size_t count;
    const int max_sh_coeffs;
    
    CudaBuffer<float>     d_dL_dpoints;
    CudaBuffer<glm::vec3> d_dL_dscales;
    CudaBuffer<glm::vec4> d_dL_drotations;
    CudaBuffer<float>     d_dL_dopacities;
    CudaBuffer<float>     d_dL_ddc;
    CudaBuffer<float>     d_dL_dshs;
    CudaBuffer<float>     d_dL_dcolors;
    CudaBuffer<float>     d_dL_dcov3Ds;
    CudaBuffer<float2>    d_dL_dmeans2D;
    CudaBuffer<float4>    d_dL_dconic_opacity;
    CudaBuffer<float>     d_dL_dinvdepths;

    GaussianGrads(size_t P, int D) :
        count(P),
        max_sh_coeffs((D + 1) * (D + 1)),
        d_dL_dpoints(P * 3),
        d_dL_dscales(P),
        d_dL_drotations(P),
        d_dL_dopacities(P),
        d_dL_ddc(P * 3),
        d_dL_dshs(P * max_sh_coeffs * 3),
        d_dL_dcolors(P * 3),
        d_dL_dcov3Ds(P * 6),
        d_dL_dmeans2D(P),
        d_dL_dconic_opacity(P),
        d_dL_dinvdepths(P)
    {}

    void resize(size_t new_P) {
        count = new_P;
        // Re-construct buffers with new size. 
        // Old buffers are destroyed (RAII), new ones allocated.
        d_dL_dpoints    = CudaBuffer<float>(new_P * 3);
        d_dL_dscales    = CudaBuffer<glm::vec3>(new_P);
        d_dL_drotations = CudaBuffer<glm::vec4>(new_P);
        d_dL_dopacities = CudaBuffer<float>(new_P);
        d_dL_ddc        = CudaBuffer<float>(new_P * 3);
        d_dL_dshs       = CudaBuffer<float>(new_P * max_sh_coeffs * 3);
        d_dL_dcolors    = CudaBuffer<float>(new_P * 3);
        d_dL_dcov3Ds    = CudaBuffer<float>(new_P * 6);
        d_dL_dmeans2D   = CudaBuffer<float2>(new_P);
        d_dL_dconic_opacity = CudaBuffer<float4>(new_P);
        d_dL_dinvdepths = CudaBuffer<float>(new_P);
    }

    void clear_all() {
        d_dL_dpoints.clear();
        d_dL_dscales.clear();
        d_dL_drotations.clear();
        d_dL_dopacities.clear();
        d_dL_dshs.clear();
        d_dL_dcolors.clear();
        d_dL_dcov3Ds.clear();
        d_dL_dmeans2D.clear();
        d_dL_dconic_opacity.clear();
        d_dL_dinvdepths.clear();
    }
};


// Helper: Inverse of sigmoid
inline float logit(float x) {
    x = std::max(SAFE_EPS, std::min(1.0f - SAFE_EPS, x));
    return logf(x / (1.0f - x));
}

inline void initGaussians(const std::vector<ColmapPoint3D>& colmap_points,
                   size_t N,
                   std::vector<glm::vec3>& h_points,
                   std::vector<float>& h_means,
                   std::vector<float>& h_opacities,
                   std::vector<glm::vec3>& h_scales,
                   std::vector<glm::vec4>& h_rotations,
                   std::vector<float>& h_dc) {

    
    for (size_t i=0; i<N; ++i) {
        const auto& point = colmap_points[i];

        // 1. Mean (Position)
        // Simple cast from double to float for feeding into the forward functions
        h_means[3*i] = (float)point.xyz[0];
        h_means[3*i+1] = (float)point.xyz[1];
        h_means[3*i+2] = (float)point.xyz[2];

        h_points[i] = {(float)point.xyz[0], (float)point.xyz[1], (float)point.xyz[2]};
        
        // 2. Rotation
        // Initialize to identity (w, x, y, z)
        h_rotations[i] = { 1.0f, 0.0f, 0.0f, 0.0f };

        // 3. Opacity
        // Initialize to a small value (e.g., 0.1)
        // The paper uses sigmoid, so we store the logit.
        h_opacities[i] = logit(0.05f);

        // 4. Spherical Harmonics (Color)
        float3 color_norm = { 
            point.rgb[0] / 255.0f, 
            point.rgb[1] / 255.0f, 
            point.rgb[2] / 255.0f 
        };

        // Store at the 0-th order (DC) coefficient location
        size_t sh_base_idx = i * 3;

        // Inverse of: Color = 0.5 + (DC * C0)
        // DC = (Color - 0.5) / C0
        h_dc[sh_base_idx + 0] = (color_norm.x - 0.5f) / SH_C0_init; // R
        h_dc[sh_base_idx + 1] = (color_norm.y - 0.5f) / SH_C0_init; // G
        h_dc[sh_base_idx + 2] = (color_norm.z - 0.5f) / SH_C0_init; // B    
        // All other 15 AC coefficients are already 0 from the vector initialization.
    }        
        // 5. Scale
    
        // a. Build a k-d tree from all h_point.
        // b. For each point i, find 3 nearest neighbors.
        // c. Compute mean_dist = average distance to neighbors.
        // d. h_scales[i] = { logf(mean_dist), logf(mean_dist), logf(mean_dist) };

        // 1. Create the adapter instance
    PointCloudAdapter adapter(h_points);

    // 2. Define the k-d tree index type
    //    We use L2_Simple (squared Euclidean distance) for efficiency
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdapter>,
            PointCloudAdapter,
            3, /* dimensions */
            size_t> KdTree;

    // 3. Construct the index
    std::cout << "Building k-d tree index..." << std::endl;
    KdTree index(3, adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();
    std::cout << "Index build complete." << std::endl;


    // --- Steps B, C, D: Query neighbors and compute scales ---
    
    const float SAFE_EPS = 1e-10f;
    
    // We must query for k=4.
    // The query point itself is the 1st neighbor (at distance 0).
    // The 3 nearest neighbors will be at indices 1, 2, and 3.
    const size_t num_neighbors = 4; 

    for (size_t i = 0; i < N; ++i) {
        // b. Find 4 nearest neighbors (self + 3 others)
        std::vector<size_t> ret_indices(num_neighbors);
        std::vector<float> out_dists_sq(num_neighbors); // nanoflann returns squared distances

        index.knnSearch(
            &h_points[i].x,               // Query point
            num_neighbors,               // Number of neighbors to find
            ret_indices.data(),          // Output indices
            out_dists_sq.data()          // Output squared distances
            );

            // c. Compute mean_dist
            //    ret_indices[0] is 'i' itself, out_dists_sq[0] is 0.0
            //    We use indices 1, 2, and 3.
            float mean_dist_sq = (out_dists_sq[1] + out_dists_sq[2] + out_dists_sq[3]) / 3.0f;
            float mean_dist = std::sqrt(mean_dist_sq);

            // d. h_scales[i] = { logf(mean_dist), ... }
            //    The paper uses log activation for scale.
            float initial_scale = std::max(mean_dist, SAFE_EPS);
            h_scales[i] = { std::log(initial_scale), std::log(initial_scale), std::log(initial_scale) };

    }
    
}
