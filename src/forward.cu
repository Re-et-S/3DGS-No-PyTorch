/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.cuh"
#include "auxiliary.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const float* means, glm::vec3 campos, const float* dc, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = { means[3 * idx + 0], means[3 * idx + 1], means[3 * idx + 2] };
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

    // Pointers to SH data
    const float* direct_color_ptr = dc + 3 * idx;
    const float* sh_ptr = shs + 3 * max_coeffs * idx;

    // 1. Compute Base Color (DC)

    glm::vec3 result = {
        SH_C0 * direct_color_ptr[0],
        SH_C0 * direct_color_ptr[1],
        SH_C0 * direct_color_ptr[2]
    };

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
        
		result = result - SH_C1 * y * glm::vec3(sh_ptr[3], sh_ptr[4], sh_ptr[5]) + SH_C1 * z * glm::vec3(sh_ptr[6], sh_ptr[7], sh_ptr[8]) - SH_C1 * x * glm::vec3(sh_ptr[9], sh_ptr[10], sh_ptr[11]);

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * glm::vec3(sh_ptr[12], sh_ptr[13], sh_ptr[14]) +
				SH_C2[1] * yz * glm::vec3(sh_ptr[15], sh_ptr[16], sh_ptr[17]) +
				SH_C2[2] * (2.0f * zz - xx - yy) * glm::vec3(sh_ptr[18], sh_ptr[19], sh_ptr[20]) +
				SH_C2[3] * xz * glm::vec3(sh_ptr[21], sh_ptr[22], sh_ptr[23]) +
				SH_C2[4] * (xx - yy) * glm::vec3(sh_ptr[24], sh_ptr[25], sh_ptr[26]);

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * glm::vec3(sh_ptr[27], sh_ptr[28], sh_ptr[29]) +
					SH_C3[1] * xy * z * glm::vec3(sh_ptr[30], sh_ptr[31], sh_ptr[32]) +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * glm::vec3(sh_ptr[33], sh_ptr[34], sh_ptr[35]) +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * glm::vec3(sh_ptr[36], sh_ptr[37], sh_ptr[38]) +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * glm::vec3(sh_ptr[39], sh_ptr[40], sh_ptr[41]) +
					SH_C3[5] * z * (xx - yy) * glm::vec3(sh_ptr[42], sh_ptr[43], sh_ptr[44]) +
					SH_C3[6] * x * (xx - 3.0f * yy) * glm::vec3(sh_ptr[45], sh_ptr[46], sh_ptr[47]);
			}
		}
	}
    
    // 2. Apply Standard Offset
	result += 0.5f;

	// We must track which channels were clamped (< 0) to zero out their gradients later.
	clamped[3 * idx + 0] = (result.x < 0.0f);
	clamped[3 * idx + 1] = (result.y < 0.0f);
	clamped[3 * idx + 2] = (result.z < 0.0f);

    // Note: We clamp to 0.0 (Lower Bound), but NOT to 1.0 (Upper Bound).
    // The loss function handles over-saturation.
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

    // Construct J to match original GLM code (which effectively created a Transposed Jacobian)
    // Original: glm::mat3 J(fx/z, 0, -(fx*x)/z^2, ...)
    // GLM constructor fills columns.
    // Col 0: fx/z, 0, -(fx*x)/z^2
    // Col 1: 0, fy/z, -(fy*y)/z^2
    // Col 2: 0, 0, 0
    float J[3][3] = {0};
    
    // Column 0
    J[0][0] = focal_x / t.z;
    J[1][0] = 0.0f;
    J[2][0] = -(focal_x * t.x) / (t.z * t.z);

    // Column 1
    J[0][1] = 0.0f;
    J[1][1] = focal_y / t.z;
    J[2][1] = -(focal_y * t.y) / (t.z * t.z);

    // Column 2 is all 0

    // Construct W to match original GLM code
    // Original: glm::mat3 W(view[0], view[4], view[8], ...)
    // Col 0: view[0], view[4], view[8]
    // Col 1: view[1], view[5], view[9]
    // Col 2: view[2], view[6], view[10]
    float W[3][3];
    
    W[0][0] = viewmatrix[0]; W[1][0] = viewmatrix[4]; W[2][0] = viewmatrix[8];
    W[0][1] = viewmatrix[1]; W[1][1] = viewmatrix[5]; W[2][1] = viewmatrix[9];
    W[0][2] = viewmatrix[2]; W[1][2] = viewmatrix[6]; W[2][2] = viewmatrix[10];

    // T = W * J
    float T[3][3] = {0};
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            T[i][j] = 0.0f;
            for(int k=0; k<3; ++k) {
                T[i][j] += W[i][k] * J[k][j];
            }
        }
    }

    // Vrk is 3D Covariance (symmetric).
    float Vrk[3][3];
    Vrk[0][0] = cov3D[0]; Vrk[0][1] = cov3D[1]; Vrk[0][2] = cov3D[2];
    Vrk[1][0] = cov3D[1]; Vrk[1][1] = cov3D[3]; Vrk[1][2] = cov3D[4];
    Vrk[2][0] = cov3D[2]; Vrk[2][1] = cov3D[4]; Vrk[2][2] = cov3D[5];

    // Cov = T^T * Vrk * T
    // Let M = T^T * Vrk
    // M[i][j] = sum_k (T^T)[i][k] * Vrk[k][j] = sum_k T[k][i] * Vrk[k][j]
    float M[3][3] = {0};
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            for(int k=0; k<3; ++k) {
                M[i][j] += T[k][i] * Vrk[k][j];
            }
        }
    }

    // Cov = M * T
    float cov[3][3] = {0};
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            for(int k=0; k<3; ++k) {
                cov[i][j] += M[i][k] * T[k][j];
            }
        }
    }

	return { cov[0][0], cov[0][1], cov[1][1] };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const float* scale, float mod, const float* rot, float* cov3D)
{
	// Create scaling matrix S
    // S is diagonal.
    float s_x = mod * expf(scale[0]);
    float s_y = mod * expf(scale[1]);
    float s_z = mod * expf(scale[2]);

	// Normalize quaternion to get valid rotation
    // q = (r, x, y, z)
	float r = rot[0];
	float x = rot[1];
	float y = rot[2];
	float z = rot[3];

	// Compute rotation matrix R from quaternion
    float R[3][3];
    R[0][0] = 1.f - 2.f * (y * y + z * z);
    R[0][1] = 2.f * (x * y - r * z);
    R[0][2] = 2.f * (x * z + r * y);
    
    R[1][0] = 2.f * (x * y + r * z);
    R[1][1] = 1.f - 2.f * (x * x + z * z);
    R[1][2] = 2.f * (y * z - r * x);

    R[2][0] = 2.f * (x * z - r * y);
    R[2][1] = 2.f * (y * z + r * x);
    R[2][2] = 1.f - 2.f * (x * x + y * y);

    // M = S * R
    // Since S is diagonal, this scales the rows of R.
    float M[3][3];
    M[0][0] = s_x * R[0][0]; M[0][1] = s_x * R[0][1]; M[0][2] = s_x * R[0][2];
    M[1][0] = s_y * R[1][0]; M[1][1] = s_y * R[1][1]; M[1][2] = s_y * R[1][2];
    M[2][0] = s_z * R[2][0]; M[2][1] = s_z * R[2][1]; M[2][2] = s_z * R[2][2];

	// Compute 3D world covariance matrix Sigma
    // Sigma = M^T * M
    // Sigma[i][j] = sum_k (M^T)[i][k] * M[k][j] = sum_k M[k][i] * M[k][j]
    float Sigma[3][3] = {0};
    for(int i=0; i<3; ++i) {
        for(int j=0; j<3; ++j) {
            for(int k=0; k<3; ++k) {
                Sigma[i][j] += M[k][i] * M[k][j];
            }
        }
    }

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* opacities,
    const float* dc,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales + 3 * idx, scale_modifier, rotations + 4 * idx, cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det <= 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
    
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);

    if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 cam_pos_vec = { cam_pos[0], cam_pos[1], cam_pos[2] };
		glm::vec3 result = computeColorFromSH(idx, D, M, orig_points, cam_pos_vec, dc, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	// float opacity = opacities[idx];
    float opacity = sigmoid(opacities[idx]);


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* opacities,
    const float* dc,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
    const float focal_x, float focal_y,
    int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
        dc,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
