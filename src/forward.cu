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
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* dc, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	if (idx == 0) {
		printf("[CTX] computeColorFromSH idx=0\n");
		printf("  pos: %f %f %f\n", pos.x, pos.y, pos.z);
		printf("  campos: %f %f %f\n", campos.x, campos.y, campos.z);
		printf("  dir: %f %f %f\n", dir.x, dir.y, dir.z);
	}

    // Pointers to SH data
    glm::vec3* direct_color = ((glm::vec3*)dc) + idx;
    glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

    // 1. Compute Base Color (DC)

    glm::vec3 result = {
        SH_C0 * direct_color[0].x,
        SH_C0 * direct_color[0].y,
        SH_C0 * direct_color[0].z
    };

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
        
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
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
	glm::vec3 res = glm::max(result, 0.0f);
	if (idx == 0) {
		printf("  result (before clamp): %f %f %f\n", result.x, result.y, result.z);
		printf("  result (final): %f %f %f\n", res.x, res.y, res.z);
	}
	return res;
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix, int idx)
{
    float3 t = transformPoint4x3(mean, viewmatrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    // J Matrix (Projection Derivative)
    float J00 = focal_x / t.z;
    float J02 = -(focal_x * t.x) / (t.z * t.z);
    float J11 = focal_y / t.z;
    float J12 = -(focal_y * t.y) / (t.z * t.z);

    // W Matrix (View Rotation)
    float W00 = viewmatrix[0]; float W01 = viewmatrix[4]; float W02 = viewmatrix[8];
    float W10 = viewmatrix[1]; float W11 = viewmatrix[5]; float W12 = viewmatrix[9];
    float W20 = viewmatrix[2]; float W21 = viewmatrix[6]; float W22 = viewmatrix[10];

    // FIX: T = J * W (Projection * View)
    // Row 0 of T = Row 0 of J * W
    float T00 = J00 * W00 + J02 * W20;
    float T01 = J00 * W01 + J02 * W21;
    float T02 = J00 * W02 + J02 * W22;

    // Row 1 of T = Row 1 of J * W
    float T10 = J11 * W10 + J12 * W20;
    float T11 = J11 * W11 + J12 * W21;
    float T12 = J11 * W12 + J12 * W22;

    // Load Vrk (3D Covariance)
    float V00 = cov3D[0]; float V01 = cov3D[1]; float V02 = cov3D[2];
                          float V11 = cov3D[3]; float V12 = cov3D[4];
                                                float V22 = cov3D[5];

    // Compute Cov2D = T * V * T^T
    auto compute_quadratic = [&](float a, float b, float c) {
        return a*a*V00 + b*b*V11 + c*c*V22 + 2.0f*(a*b*V01 + a*c*V02 + b*c*V12);
    };

    auto compute_bilinear = [&](float a1, float b1, float c1, float a2, float b2, float c2) {
        return a1*a2*V00 + b1*b2*V11 + c1*c2*V22 + 
               (a1*b2 + a2*b1)*V01 + (a1*c2 + a2*c1)*V02 + (b1*c2 + b2*c1)*V12;
    };

    float cov_x = compute_quadratic(T00, T01, T02);
    float cov_y = compute_bilinear(T00, T01, T02, T10, T11, T12);
    float cov_z = compute_quadratic(T10, T11, T12);

    return { cov_x, cov_y, cov_z };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
// 1. Unroll Scale Matrix (Diagonal)
    // S = diag(s.x, s.y, s.z)
    float sx = mod * expf(scale.x);
    float sy = mod * expf(scale.y);
    float sz = mod * expf(scale.z);

    // 2. Unroll Rotation Matrix from Quaternion
    // Normalize q to guarantee valid rotation
    glm::vec4 q = rot;
    float len = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (len > 0.0f) q /= len;
    
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // R matrix elements (Row-Major for mental model, but indices matter)
    // R00 R01 R02
    // R10 R11 R12
    // R20 R21 R22
    float R00 = 1.f - 2.f * (y * y + z * z);
    float R01 = 2.f * (x * y - r * z);
    float R02 = 2.f * (x * z + r * y);

    float R10 = 2.f * (x * y + r * z);
    float R11 = 1.f - 2.f * (x * x + z * z);
    float R12 = 2.f * (y * z - r * x);

    float R20 = 2.f * (x * z - r * y);
    float R21 = 2.f * (y * z + r * x);
    float R22 = 1.f - 2.f * (x * x + y * y);

    // 3. Compute M = R * S (Scale then Rotate) manually
    // Since S is diagonal, this is just scaling the columns of R
    float M00 = R00 * sx;
    float M01 = R01 * sy;
    float M02 = R02 * sz;

    float M10 = R10 * sx;
    float M11 = R11 * sy;
    float M12 = R12 * sz;

    float M20 = R20 * sx;
    float M21 = R21 * sy;
    float M22 = R22 * sz;

    // 4. Compute Covariance Sigma = M * M^T manually
    // Sigma is symmetric, so we only compute the upper triangle (and diagonal)
    
    // Cov00 = Row0 . Row0
    cov3D[0] = M00 * M00 + M01 * M01 + M02 * M02;
    
    // Cov01 = Row0 . Row1
    cov3D[1] = M00 * M10 + M01 * M11 + M02 * M12;
    
    // Cov02 = Row0 . Row2
    cov3D[2] = M00 * M20 + M01 * M21 + M02 * M22;
    
    // Cov11 = Row1 . Row1
    cov3D[3] = M10 * M10 + M11 * M11 + M12 * M12;
    
    // Cov12 = Row1 . Row2
    cov3D[4] = M10 * M20 + M11 * M21 + M12 * M22;
    
    // Cov22 = Row2 . Row2
    cov3D[5] = M20 * M20 + M21 * M21 + M22 * M22;
}

__global__ void verifyCovarianceKernel(int P, const glm::vec3* scales, const glm::vec4* rotations, const float scale_modifier)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    float cov3D[6];
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3D);

    // Reconstruct matrix
    // 0 1 2
    // 1 3 4
    // 2 4 5

    // Check diagonals (Must be > 0 for non-degenerate scaling)
    if (cov3D[0] <= 0 || cov3D[3] <= 0 || cov3D[5] <= 0) {
        if (idx < 5) printf("ERROR Gaussian %d: Invalid Diagonals. %f %f %f\n", idx, cov3D[0], cov3D[3], cov3D[5]);
    }

    // Determinant
    // a(ei - fh) - b(di - fg) + c(dh - eg)
    float a = cov3D[0], b = cov3D[1], c = cov3D[2];
    float d = cov3D[1], e = cov3D[3], f = cov3D[4];
    float g = cov3D[2], h = cov3D[4], i = cov3D[5];

    float det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);

    if (det <= 0) {
        if (idx < 5) printf("ERROR Gaussian %d: Invalid Determinant %f. Scales: %f %f %f. Rot: %f %f %f %f\n",
            idx, det, scales[idx].x, scales[idx].y, scales[idx].z, rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w);
    }
}

void FORWARD::verify_initial_covariances(int P, const glm::vec3* scales, const glm::vec4* rotations, const float scale_modifier) {
    verifyCovarianceKernel<<<(P + 255) / 256, 256>>>(P, scales, rotations, scale_modifier);
    cudaDeviceSynchronize();
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
    const float* dc,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
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
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	if (idx == 0) {
		printf("[CTX] preprocessCUDA idx=0\n");
		printf("  Scale: %f %f %f\n", scales[idx].x, scales[idx].y, scales[idx].z);
		printf("  Rot: %f %f %f %f\n", rotations[idx].x, rotations[idx].y, rotations[idx].z, rotations[idx].w);
		printf("  Cov3D: %f %f %f %f %f %f\n", cov3D[0], cov3D[1], cov3D[2], cov3D[3], cov3D[4], cov3D[5]);
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix, idx);

	if (idx == 0) {
		printf("  p_orig: %f %f %f\n", p_orig.x, p_orig.y, p_orig.z);
		printf("  Cov2D: %f %f %f\n", cov.x, cov.y, cov.z);
	}

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
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, dc, shs, clamped);
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

	if (idx == 0) {
		printf("  Opacity (raw): %f\n", opacities[idx]);
		printf("  Opacity (sigmoid): %f\n", opacity);
		printf("  h_convolution_scaling: %f\n", h_convolution_scaling);
		printf("  Conic: %f %f %f\n", conic.x, conic.y, conic.z);
		printf("  ConicOpacity: %f %f %f %f\n", conic_opacity[idx].x, conic_opacity[idx].y, conic_opacity[idx].z, conic_opacity[idx].w);
	}

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

	bool is_debug_pixel = (pix.x == W/2 && pix.y == H/2);
	if (is_debug_pixel) {
		printf("[CTX] renderCUDA Center Pixel (%d, %d)\n", pix.x, pix.y);
	}

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

			if (is_debug_pixel) {
				printf("  GausID: %d, Alpha: %f, T: %f, Feat[0]: %f\n", collected_id[j], alpha, T, features[collected_id[j] * CHANNELS + 0]);
			}

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
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
    const float* dc,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
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
