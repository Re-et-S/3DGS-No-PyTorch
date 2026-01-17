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

#include "backward.cuh"
#include "auxiliary.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const float* means, glm::vec3 campos, const float* dc, const float* shs, const bool* clamped, const float* dL_dcolor, float* dL_dmeans, float* dL_ddc, float* dL_dshs)
{
    // 1. Compute view direction (same as forward)
	float pos_x = means[3 * idx + 0];
    float pos_y = means[3 * idx + 1];
    float pos_z = means[3 * idx + 2];

	float dir_orig_x = pos_x - campos.x;
    float dir_orig_y = pos_y - campos.y;
    float dir_orig_z = pos_z - campos.z;

    float dir_len_sq = dir_orig_x*dir_orig_x + dir_orig_y*dir_orig_y + dir_orig_z*dir_orig_z;
    float dir_len = sqrt(dir_len_sq);

    float dir_x = dir_orig_x / dir_len;
    float dir_y = dir_orig_y / dir_len;
    float dir_z = dir_orig_z / dir_len;

	const float* sh_ptr = shs + 3 * max_coeffs * idx;

	// 2. Retrieve the upstream gradient (dL/dColor)
    float dL_dRGB_x = dL_dcolor[3 * idx + 0];
    float dL_dRGB_y = dL_dcolor[3 * idx + 1];
    float dL_dRGB_z = dL_dcolor[3 * idx + 2];

	// 3. Apply ReLU Derivative (Clamp)
	// If the forward pass clamped the value to 0, the gradient is blocked (0).
	dL_dRGB_x *= clamped[3 * idx + 0] ? 0.0f : 1.0f;
	dL_dRGB_y *= clamped[3 * idx + 1] ? 0.0f : 1.0f;
	dL_dRGB_z *= clamped[3 * idx + 2] ? 0.0f : 1.0f;

	// 4. Compute Gradient for DC (Base Color)
	// Forward: Color = 0.5 + SH_C0 * DC + ...
	// Derivative: dColor/dDC = SH_C0
	float* dL_ddirect_color_ptr = dL_ddc + 3 * idx;
	dL_ddirect_color_ptr[0] = dL_dRGB_x * SH_C0;
	dL_ddirect_color_ptr[1] = dL_dRGB_y * SH_C0;
	dL_ddirect_color_ptr[2] = dL_dRGB_z * SH_C0;

	// 5. Compute Gradients for SH coefficients (Rest) & View Direction
	float* dL_dsh_ptr = dL_dshs + 3 * max_coeffs * idx;
	
	// Accumulators for view direction gradient
    float dRGBdx_x = 0, dRGBdx_y = 0, dRGBdx_z = 0;
    float dRGBdy_x = 0, dRGBdy_y = 0, dRGBdy_z = 0;
    float dRGBdz_x = 0, dRGBdz_y = 0, dRGBdz_z = 0;
	
	float x = dir_x;
	float y = dir_y;
	float z = dir_z;

    // No tricks here, just high school-level calculus.
	// float dRGBdsh0 = SH_C0;
	// dL_ddirect_color[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;

        // sh[1]
        dL_dsh_ptr[3] = dRGBdsh1 * dL_dRGB_x;
        dL_dsh_ptr[4] = dRGBdsh1 * dL_dRGB_y;
        dL_dsh_ptr[5] = dRGBdsh1 * dL_dRGB_z;
        // sh[2]
        dL_dsh_ptr[6] = dRGBdsh2 * dL_dRGB_x;
        dL_dsh_ptr[7] = dRGBdsh2 * dL_dRGB_y;
        dL_dsh_ptr[8] = dRGBdsh2 * dL_dRGB_z;
        // sh[3]
        dL_dsh_ptr[9] = dRGBdsh3 * dL_dRGB_x;
        dL_dsh_ptr[10] = dRGBdsh3 * dL_dRGB_y;
        dL_dsh_ptr[11] = dRGBdsh3 * dL_dRGB_z;

        // dRGBdx = -SH_C1 * sh[3]
        dRGBdx_x = -SH_C1 * sh_ptr[9]; dRGBdx_y = -SH_C1 * sh_ptr[10]; dRGBdx_z = -SH_C1 * sh_ptr[11];
        // dRGBdy = -SH_C1 * sh[1]
        dRGBdy_x = -SH_C1 * sh_ptr[3]; dRGBdy_y = -SH_C1 * sh_ptr[4]; dRGBdy_z = -SH_C1 * sh_ptr[5];
        // dRGBdz = SH_C1 * sh[2]
        dRGBdz_x = SH_C1 * sh_ptr[6]; dRGBdz_y = SH_C1 * sh_ptr[7]; dRGBdz_z = SH_C1 * sh_ptr[8];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);

            // sh[4]..sh[8]
            // Unrolled loop for K=0..2
            dL_dsh_ptr[12+0] = dRGBdsh4 * dL_dRGB_x; dL_dsh_ptr[12+1] = dRGBdsh4 * dL_dRGB_y; dL_dsh_ptr[12+2] = dRGBdsh4 * dL_dRGB_z;
            dL_dsh_ptr[15+0] = dRGBdsh5 * dL_dRGB_x; dL_dsh_ptr[15+1] = dRGBdsh5 * dL_dRGB_y; dL_dsh_ptr[15+2] = dRGBdsh5 * dL_dRGB_z;
            dL_dsh_ptr[18+0] = dRGBdsh6 * dL_dRGB_x; dL_dsh_ptr[18+1] = dRGBdsh6 * dL_dRGB_y; dL_dsh_ptr[18+2] = dRGBdsh6 * dL_dRGB_z;
            dL_dsh_ptr[21+0] = dRGBdsh7 * dL_dRGB_x; dL_dsh_ptr[21+1] = dRGBdsh7 * dL_dRGB_y; dL_dsh_ptr[21+2] = dRGBdsh7 * dL_dRGB_z;
            dL_dsh_ptr[24+0] = dRGBdsh8 * dL_dRGB_x; dL_dsh_ptr[24+1] = dRGBdsh8 * dL_dRGB_y; dL_dsh_ptr[24+2] = dRGBdsh8 * dL_dRGB_z;

            // sh4..sh8 manual load
            float sh4_x = sh_ptr[12], sh4_y = sh_ptr[13], sh4_z = sh_ptr[14];
            float sh5_x = sh_ptr[15], sh5_y = sh_ptr[16], sh5_z = sh_ptr[17];
            float sh6_x = sh_ptr[18], sh6_y = sh_ptr[19], sh6_z = sh_ptr[20];
            float sh7_x = sh_ptr[21], sh7_y = sh_ptr[22], sh7_z = sh_ptr[23];
            float sh8_x = sh_ptr[24], sh8_y = sh_ptr[25], sh8_z = sh_ptr[26];

			// dRGBdx += SH_C2[0] * y * sh4 + SH_C2[2] * 2.f * -x * sh6 + SH_C2[3] * z * sh7 + SH_C2[4] * 2.f * x * sh8;
            dRGBdx_x += SH_C2[0] * y * sh4_x + SH_C2[2] * -2.f * x * sh6_x + SH_C2[3] * z * sh7_x + SH_C2[4] * 2.f * x * sh8_x;
            dRGBdx_y += SH_C2[0] * y * sh4_y + SH_C2[2] * -2.f * x * sh6_y + SH_C2[3] * z * sh7_y + SH_C2[4] * 2.f * x * sh8_y;
            dRGBdx_z += SH_C2[0] * y * sh4_z + SH_C2[2] * -2.f * x * sh6_z + SH_C2[3] * z * sh7_z + SH_C2[4] * 2.f * x * sh8_z;

			// dRGBdy += SH_C2[0] * x * sh4 + SH_C2[1] * z * sh5 + SH_C2[2] * 2.f * -y * sh6 + SH_C2[4] * 2.f * -y * sh8;
            dRGBdy_x += SH_C2[0] * x * sh4_x + SH_C2[1] * z * sh5_x + SH_C2[2] * -2.f * y * sh6_x + SH_C2[4] * -2.f * y * sh8_x;
            dRGBdy_y += SH_C2[0] * x * sh4_y + SH_C2[1] * z * sh5_y + SH_C2[2] * -2.f * y * sh6_y + SH_C2[4] * -2.f * y * sh8_y;
            dRGBdy_z += SH_C2[0] * x * sh4_z + SH_C2[1] * z * sh5_z + SH_C2[2] * -2.f * y * sh6_z + SH_C2[4] * -2.f * y * sh8_z;

			// dRGBdz += SH_C2[1] * y * sh5 + SH_C2[2] * 2.f * 2.f * z * sh6 + SH_C2[3] * x * sh7;
            dRGBdz_x += SH_C2[1] * y * sh5_x + SH_C2[2] * 4.f * z * sh6_x + SH_C2[3] * x * sh7_x;
            dRGBdz_y += SH_C2[1] * y * sh5_y + SH_C2[2] * 4.f * z * sh6_y + SH_C2[3] * x * sh7_y;
            dRGBdz_z += SH_C2[1] * y * sh5_z + SH_C2[2] * 4.f * z * sh6_z + SH_C2[3] * x * sh7_z;

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);

                dL_dsh_ptr[27+0] = dRGBdsh9 * dL_dRGB_x; dL_dsh_ptr[27+1] = dRGBdsh9 * dL_dRGB_y; dL_dsh_ptr[27+2] = dRGBdsh9 * dL_dRGB_z;
                dL_dsh_ptr[30+0] = dRGBdsh10 * dL_dRGB_x; dL_dsh_ptr[30+1] = dRGBdsh10 * dL_dRGB_y; dL_dsh_ptr[30+2] = dRGBdsh10 * dL_dRGB_z;
                dL_dsh_ptr[33+0] = dRGBdsh11 * dL_dRGB_x; dL_dsh_ptr[33+1] = dRGBdsh11 * dL_dRGB_y; dL_dsh_ptr[33+2] = dRGBdsh11 * dL_dRGB_z;
                dL_dsh_ptr[36+0] = dRGBdsh12 * dL_dRGB_x; dL_dsh_ptr[36+1] = dRGBdsh12 * dL_dRGB_y; dL_dsh_ptr[36+2] = dRGBdsh12 * dL_dRGB_z;
                dL_dsh_ptr[39+0] = dRGBdsh13 * dL_dRGB_x; dL_dsh_ptr[39+1] = dRGBdsh13 * dL_dRGB_y; dL_dsh_ptr[39+2] = dRGBdsh13 * dL_dRGB_z;
                dL_dsh_ptr[42+0] = dRGBdsh14 * dL_dRGB_x; dL_dsh_ptr[42+1] = dRGBdsh14 * dL_dRGB_y; dL_dsh_ptr[42+2] = dRGBdsh14 * dL_dRGB_z;
                dL_dsh_ptr[45+0] = dRGBdsh15 * dL_dRGB_x; dL_dsh_ptr[45+1] = dRGBdsh15 * dL_dRGB_y; dL_dsh_ptr[45+2] = dRGBdsh15 * dL_dRGB_z;

                float sh9_x = sh_ptr[27], sh9_y = sh_ptr[28], sh9_z = sh_ptr[29];
                float sh10_x = sh_ptr[30], sh10_y = sh_ptr[31], sh10_z = sh_ptr[32];
                float sh11_x = sh_ptr[33], sh11_y = sh_ptr[34], sh11_z = sh_ptr[35];
                float sh12_x = sh_ptr[36], sh12_y = sh_ptr[37], sh12_z = sh_ptr[38];
                float sh13_x = sh_ptr[39], sh13_y = sh_ptr[40], sh13_z = sh_ptr[41];
                float sh14_x = sh_ptr[42], sh14_y = sh_ptr[43], sh14_z = sh_ptr[44];
                float sh15_x = sh_ptr[45], sh15_y = sh_ptr[46], sh15_z = sh_ptr[47];

                // dRGBdx accumulation
                float term1 = SH_C3[0] * 3.f * 2.f * xy;
                float term2 = SH_C3[1] * yz;
                float term3 = SH_C3[2] * -2.f * xy;
                float term4 = SH_C3[3] * -6.f * xz;
                float term5 = SH_C3[4] * (-3.f * xx + 4.f * zz - yy);
                float term6 = SH_C3[5] * 2.f * xz;
                float term7 = SH_C3[6] * 3.f * (xx - yy);

                dRGBdx_x += term1 * sh9_x + term2 * sh10_x + term3 * sh11_x + term4 * sh12_x + term5 * sh13_x + term6 * sh14_x + term7 * sh15_x;
                dRGBdx_y += term1 * sh9_y + term2 * sh10_y + term3 * sh11_y + term4 * sh12_y + term5 * sh13_y + term6 * sh14_y + term7 * sh15_y;
                dRGBdx_z += term1 * sh9_z + term2 * sh10_z + term3 * sh11_z + term4 * sh12_z + term5 * sh13_z + term6 * sh14_z + term7 * sh15_z;

                // dRGBdy accumulation
                term1 = SH_C3[0] * 3.f * (xx - yy);
                term2 = SH_C3[1] * xz;
                term3 = SH_C3[2] * (-3.f * yy + 4.f * zz - xx);
                term4 = SH_C3[3] * -6.f * yz;
                term5 = SH_C3[4] * -2.f * xy;
                term6 = SH_C3[5] * -2.f * yz;
                term7 = SH_C3[6] * -6.f * xy;

                dRGBdy_x += term1 * sh9_x + term2 * sh10_x + term3 * sh11_x + term4 * sh12_x + term5 * sh13_x + term6 * sh14_x + term7 * sh15_x;
                dRGBdy_y += term1 * sh9_y + term2 * sh10_y + term3 * sh11_y + term4 * sh12_y + term5 * sh13_y + term6 * sh14_y + term7 * sh15_y;
                dRGBdy_z += term1 * sh9_z + term2 * sh10_z + term3 * sh11_z + term4 * sh12_z + term5 * sh13_z + term6 * sh14_z + term7 * sh15_z;

                // dRGBdz accumulation
                term1 = SH_C3[1] * xy;
                term2 = SH_C3[2] * 8.f * yz;
                term3 = SH_C3[3] * 3.f * (2.f * zz - xx - yy);
                term4 = SH_C3[4] * 8.f * xz;
                term5 = SH_C3[5] * (xx - yy);

                dRGBdz_x += term1 * sh10_x + term2 * sh11_x + term3 * sh12_x + term4 * sh13_x + term5 * sh14_x;
                dRGBdz_y += term1 * sh10_y + term2 * sh11_y + term3 * sh12_y + term4 * sh13_y + term5 * sh14_y;
                dRGBdz_z += term1 * sh10_z + term2 * sh11_z + term3 * sh12_z + term4 * sh13_z + term5 * sh14_z;
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
    // dL_ddir = dot(dRGBdx, dL_dRGB)
	float dL_ddir_x = dRGBdx_x * dL_dRGB_x + dRGBdx_y * dL_dRGB_y + dRGBdx_z * dL_dRGB_z;
    float dL_ddir_y = dRGBdy_x * dL_dRGB_x + dRGBdy_y * dL_dRGB_y + dRGBdy_z * dL_dRGB_z;
    float dL_ddir_z = dRGBdz_x * dL_dRGB_x + dRGBdz_y * dL_dRGB_y + dRGBdz_z * dL_dRGB_z;

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig_x, dir_orig_y, dir_orig_z }, float3{ dL_ddir_x, dL_ddir_y, dL_ddir_z });

    if (isnan(dL_dmean.x) || isnan(dL_dmean.y) || isnan(dL_dmean.z)) {
         printf("NaN detected in dL_dmean (SH contrib) at idx %d. DirOrig: %f %f %f. dL_ddir: %f %f %f\n",
            idx, dir_orig_x, dir_orig_y, dir_orig_z, dL_ddir_x, dL_ddir_y, dL_ddir_z);
    }

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
    atomicAdd(dL_dmeans + 3 * idx + 0, dL_dmean.x);
    atomicAdd(dL_dmeans + 3 * idx + 1, dL_dmean.y);
    atomicAdd(dL_dmeans + 3 * idx + 2, dL_dmean.z);
}

// Backward version of Covariance 2D Gradient calculation.
// This function mirrors the structure of computeCov2D in forward.cu
__device__ void computeCov2DGradient(
    const float3& mean,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    const float* cov3D,
    const float* viewmatrix,
    const float3& dL_dCov2D,
    float* dL_dmean3D,
    float* dL_dcov3D)
{
    // Recompute forward pass values necessary for gradient calculation
    float3 t = transformPoint4x3(mean, viewmatrix);

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
    const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

    // J Matrix
    float J00 = focal_x / t.z;
    float J02 = -(focal_x * t.x) / (t.z * t.z);
    float J11 = focal_y / t.z;
    float J12 = -(focal_y * t.y) / (t.z * t.z);

    // W Matrix
    float W00 = viewmatrix[0]; float W01 = viewmatrix[4]; float W02 = viewmatrix[8];
    float W10 = viewmatrix[1]; float W11 = viewmatrix[5]; float W12 = viewmatrix[9];
    float W20 = viewmatrix[2]; float W21 = viewmatrix[6]; float W22 = viewmatrix[10];

    // Vrk Matrix (3D Covariance)
    float V00 = cov3D[0]; float V01 = cov3D[1]; float V02 = cov3D[2];
                          float V11 = cov3D[3]; float V12 = cov3D[4];
                                                float V22 = cov3D[5];

    // T = W * J
    float T00 = W00 * J00;
    float T01 = W01 * J11;
    float T02 = W00 * J02 + W01 * J12;
    float T10 = W10 * J00;
    float T11 = W11 * J11;
    float T12 = W10 * J02 + W11 * J12;
    float T20 = W20 * J00;
    float T21 = W21 * J11;
    float T22 = W20 * J02 + W21 * J12;

    float dL_dc_xx = dL_dCov2D.x;
    float dL_dc_xy = dL_dCov2D.y;
    float dL_dc_yy = dL_dCov2D.z;

    // Gradients for Vrk (3D Covariance)
    // dL/dV = T * dL/dCov2D * T^T
    dL_dcov3D[0] = (T00 * T00 * dL_dc_xx + T00 * T01 * dL_dc_xy + T01 * T01 * dL_dc_yy);
    dL_dcov3D[3] = (T10 * T10 * dL_dc_xx + T10 * T11 * dL_dc_xy + T11 * T11 * dL_dc_yy);
    dL_dcov3D[5] = (T20 * T20 * dL_dc_xx + T20 * T21 * dL_dc_xy + T21 * T21 * dL_dc_yy);

    dL_dcov3D[1] = 2 * T00 * T10 * dL_dc_xx + (T00 * T11 + T01 * T10) * dL_dc_xy + 2 * T01 * T11 * dL_dc_yy;
    dL_dcov3D[2] = 2 * T00 * T20 * dL_dc_xx + (T00 * T21 + T01 * T20) * dL_dc_xy + 2 * T01 * T21 * dL_dc_yy;
    dL_dcov3D[4] = 2 * T20 * T10 * dL_dc_xx + (T10 * T21 + T20 * T11) * dL_dc_xy + 2 * T11 * T21 * dL_dc_yy;

    // Gradients w.r.t T
    // dL/dT = V * T * dL/dCov2D (approx, symmetric V)
    float dL_dT00 = 2 * (T00 * V00 + T10 * V01 + T20 * V02) * dL_dc_xx + (T01 * V00 + T11 * V01 + T21 * V02) * dL_dc_xy;
    float dL_dT01 = 2 * (T01 * V00 + T11 * V01 + T21 * V02) * dL_dc_yy + (T00 * V00 + T10 * V01 + T20 * V02) * dL_dc_xy;

    float dL_dT10 = 2 * (T10 * V11 + T00 * V01 + T20 * V12) * dL_dc_xx + (T11 * V11 + T01 * V01 + T21 * V12) * dL_dc_xy;
    float dL_dT11 = 2 * (T11 * V11 + T01 * V01 + T21 * V12) * dL_dc_yy + (T10 * V11 + T00 * V01 + T20 * V12) * dL_dc_xy;

    float dL_dT20 = 2 * (T20 * V22 + T00 * V02 + T10 * V12) * dL_dc_xx + (T21 * V22 + T01 * V02 + T11 * V12) * dL_dc_xy;
    float dL_dT21 = 2 * (T21 * V22 + T01 * V02 + T11 * V12) * dL_dc_yy + (T20 * V22 + T00 * V02 + T10 * V12) * dL_dc_xy;

    float dL_dT02 = 0; float dL_dT12 = 0; float dL_dT22 = 0;

    // Gradients w.r.t J
    float dL_dJ00 = dL_dT00 * W00 + dL_dT10 * W10 + dL_dT20 * W20;
    float dL_dJ02 = dL_dT02 * W00 + dL_dT12 * W10 + dL_dT22 * W20;
    float dL_dJ11 = dL_dT01 * W01 + dL_dT11 * W11 + dL_dT21 * W21;
    float dL_dJ12 = dL_dT02 * W01 + dL_dT12 * W11 + dL_dT22 * W21;

    float tz = 1.f / t.z;
    float tz2 = tz * tz;
    float tz3 = tz2 * tz;

    float dL_dtx = x_grad_mul * -focal_x * tz2 * dL_dJ02;
    float dL_dty = y_grad_mul * -focal_y * tz2 * dL_dJ12;
    float dL_dtz = -focal_x * tz2 * dL_dJ00 - focal_y * tz2 * dL_dJ11 + (2 * focal_x * t.x) * tz3 * dL_dJ02 + (2 * focal_y * t.y) * tz3 * dL_dJ12;

    float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, viewmatrix);

    dL_dmean3D[0] = dL_dmean.x;
    dL_dmean3D[1] = dL_dmean.y;
    dL_dmean3D[2] = dL_dmean.z;
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
    const float* opacities,
	const float4* dL_dconics,
    float* dL_dopacity,
	float* dL_dmeans,
	float* dL_dcov,
    bool antialiasing)
{
    // FIX 1: Use Standard Indexing (Avoids Cooperative Groups issues in CUDA 13)
	auto idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= P || !(radii[idx] > 0))
		return;

	const float* cov3D = cov3Ds + 6 * idx;

    // FIX 3: Load float* manually (Stride = 3 floats)
	float3 mean = { means[3*idx+0], means[3*idx+1], means[3*idx+2] };
	float3 dL_dconic = { dL_dconics[idx].x, dL_dconics[idx].y, dL_dconics[idx].w };
	
    float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
    // FIX 2: Unrolled Matrix Math (Bypasses broken GLM/Compiler ops)
    // J Matrix
	float J00 = h_x / t.z;
	float J02 = -(h_x * t.x) / (t.z * t.z);
	float J11 = h_y / t.z;
	float J12 = -(h_y * t.y) / (t.z * t.z);

    // W Matrix (View Rotation) - Explicit loading
    float W00 = view_matrix[0]; float W01 = view_matrix[4]; float W02 = view_matrix[8];
    float W10 = view_matrix[1]; float W11 = view_matrix[5]; float W12 = view_matrix[9];
    float W20 = view_matrix[2]; float W21 = view_matrix[6]; float W22 = view_matrix[10];

    // Vrk Matrix (3D Covariance) - Explicit loading
	float V00 = cov3D[0]; float V01 = cov3D[1]; float V02 = cov3D[2];
	                      float V11 = cov3D[3]; float V12 = cov3D[4];
	                                            float V22 = cov3D[5];

    // T = W * J (Explicit)
	float T00 = W00 * J00;
	float T01 = W01 * J11;
	float T02 = W00 * J02 + W01 * J12;
	float T10 = W10 * J00;
	float T11 = W11 * J11;
	float T12 = W10 * J02 + W11 * J12;
	float T20 = W20 * J00;
	float T21 = W21 * J11;
	float T22 = W20 * J02 + W21 * J12;

    // Cov2D = T^T * V * T (Explicit)
    auto compute_quadratic = [&](float a, float b, float c) {
        return a*a*V00 + b*b*V11 + c*c*V22 + 2.0f*(a*b*V01 + a*c*V02 + b*c*V12);
    };
    auto compute_bilinear = [&](float a1, float b1, float c1, float a2, float b2, float c2) {
        return a1*a2*V00 + b1*b2*V11 + c1*c2*V22 + 
               (a1*b2 + a2*b1)*V01 + (a1*c2 + a2*c1)*V02 + (b1*c2 + b2*c1)*V12;
    };

	float c_xx = compute_quadratic(T00, T10, T20);
	float c_xy = compute_bilinear(T00, T10, T20, T01, T11, T21);
	float c_yy = compute_quadratic(T01, T11, T21);

	// Antialiasing logic
	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;
	if(antialiasing)
	{
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); 
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	} 
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if(antialiasing)
	{
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}
	
	float denom = c_xx * c_yy - c_xy * c_xy;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

    if (denom2inv != 0) {
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
		
        // Use the new symmetric function to compute gradients for mean and 3D covariance
        float3 dL_dCov2D = { dL_dc_xx, dL_dc_xy, dL_dc_yy };

        if (isnan(dL_dconic.x) || isnan(dL_dconic.y) || isnan(dL_dconic.z) || isnan(dL_dconic.w)) {
             printf("NaN detected in dL_dconic at idx %d: %f %f %f\n", idx, dL_dconic.x, dL_dconic.y, dL_dconic.z);
        }

        if (isnan(dL_dCov2D.x) || isnan(dL_dCov2D.y) || isnan(dL_dCov2D.z)) {
             printf("NaN detected in dL_dCov2D at idx %d: %f %f %f\n", idx, dL_dCov2D.x, dL_dCov2D.y, dL_dCov2D.z);
        }

        computeCov2DGradient(mean, h_x, h_y, tan_fovx, tan_fovy, cov3D, view_matrix, dL_dCov2D, dL_dmeans + 3*idx, dL_dcov + 6*idx);

	} else {
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;

        dL_dmeans[3*idx+0] = 0;
        dL_dmeans[3*idx+1] = 0;
        dL_dmeans[3*idx+2] = 0;
	}
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, float* dL_dscales, float* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

    // R matrix elements (Row-Major indices)
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

    float sx = mod * expf(scale.x);
    float sy = mod * expf(scale.y);
    float sz = mod * expf(scale.z);

    // M = S * R (Scaling columns of R)
    // M00 M01 M02
    // M10 M11 M12
    // M20 M21 M22
	float M00 = R00 * sx;
    float M01 = R01 * sy;
    float M02 = R02 * sz;

    float M10 = R10 * sx;
    float M11 = R11 * sy;
    float M12 = R12 * sz;

    float M20 = R20 * sx;
    float M21 = R21 * sy;
    float M22 = R22 * sz;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

    // dL_dSigma Matrix (Symmetric)
    // S00 S01 S02
    // S10 S11 S12
    // S20 S21 S22
    float S00 = dL_dcov3D[0];
    float S01 = 0.5f * dL_dcov3D[1];
    float S02 = 0.5f * dL_dcov3D[2];

    float S10 = 0.5f * dL_dcov3D[1];
    float S11 = dL_dcov3D[3];
    float S12 = 0.5f * dL_dcov3D[4];

    float S20 = 0.5f * dL_dcov3D[2];
    float S21 = 0.5f * dL_dcov3D[4];
    float S22 = dL_dcov3D[5];

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
    // dL/dM = 2 * dL/dSigma * M
    // S is symmetric (dL/dSigma)
    // Row 0 (S00, S01, S02) * Columns of M
    float dL_dM00 = 2.0f * (S00 * M00 + S01 * M10 + S02 * M20);
    float dL_dM01 = 2.0f * (S00 * M01 + S01 * M11 + S02 * M21);
    float dL_dM02 = 2.0f * (S00 * M02 + S01 * M12 + S02 * M22);
    // Row 1 (S10, S11, S12) * Columns of M
    float dL_dM10 = 2.0f * (S10 * M00 + S11 * M10 + S12 * M20);
    float dL_dM11 = 2.0f * (S10 * M01 + S11 * M11 + S12 * M21);
    float dL_dM12 = 2.0f * (S10 * M02 + S11 * M12 + S12 * M22);
    // Row 2 (S20, S21, S22) * Columns of M
    float dL_dM20 = 2.0f * (S20 * M00 + S21 * M10 + S22 * M20);
    float dL_dM21 = 2.0f * (S20 * M01 + S21 * M11 + S22 * M21);
    float dL_dM22 = 2.0f * (S20 * M02 + S21 * M12 + S22 * M22);

	// Gradients of loss w.r.t. scale
	// Chain Rule: dL/dp = dL/ds * s
    // dL_dscales[0] = (R00 * dL_dM00 + R10 * dL_dM10 + R20 * dL_dM20) * sx;
    // dL_dscales[1] = (R01 * dL_dM01 + R11 * dL_dM11 + R21 * dL_dM21) * sy;
    // dL_dscales[2] = (R02 * dL_dM02 + R12 * dL_dM12 + R22 * dL_dM22) * sz;
	dL_dscales[0] = (R00 * dL_dM00 + R10 * dL_dM10 + R20 * dL_dM20) * sx;
	dL_dscales[1] = (R01 * dL_dM01 + R11 * dL_dM11 + R21 * dL_dM21) * sy;
	dL_dscales[2] = (R02 * dL_dM02 + R12 * dL_dM12 + R22 * dL_dM22) * sz;

    // Scale gradients for M^T
    // dL_dMt is Transpose(dL_dM) * S
    // dL_dMt00 = dL_dM00 * sx;
    // dL_dMt01 = dL_dM10 * sx; ...
    float dL_dMt00 = dL_dM00 * sx; float dL_dMt01 = dL_dM10 * sx; float dL_dMt02 = dL_dM20 * sx;
    float dL_dMt10 = dL_dM01 * sy; float dL_dMt11 = dL_dM11 * sy; float dL_dMt12 = dL_dM21 * sy;
    float dL_dMt20 = dL_dM02 * sz; float dL_dMt21 = dL_dM12 * sz; float dL_dMt22 = dL_dM22 * sz;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt01 - dL_dMt10) + 2 * y * (dL_dMt20 - dL_dMt02) + 2 * x * (dL_dMt12 - dL_dMt21);
	dL_dq.y = 2 * y * (dL_dMt10 + dL_dMt01) + 2 * z * (dL_dMt20 + dL_dMt02) + 2 * r * (dL_dMt12 - dL_dMt21) - 4 * x * (dL_dMt22 + dL_dMt11);
	dL_dq.z = 2 * x * (dL_dMt10 + dL_dMt01) + 2 * r * (dL_dMt20 - dL_dMt02) + 2 * z * (dL_dMt12 + dL_dMt21) - 4 * y * (dL_dMt22 + dL_dMt00);
	dL_dq.w = 2 * r * (dL_dMt01 - dL_dMt10) + 2 * x * (dL_dMt20 + dL_dMt02) + 2 * y * (dL_dMt12 + dL_dMt21) - 4 * z * (dL_dMt11 + dL_dMt00);

	// Gradients of loss w.r.t. unnormalized quaternion
	dL_drots[0] = dL_dq.x;
    dL_drots[1] = dL_dq.y;
    dL_drots[2] = dL_dq.z;
    dL_drots[3] = dL_dq.w;

    if (isnan(dL_dscales[0]) || isnan(dL_dscales[1]) || isnan(dL_dscales[2])) {
         printf("NaN detected in dL_dscales at idx %d\n", idx);
    }
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float* means,
	const int* radii,
    const float* dc,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float2* dL_dmean2D,
	float* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
    float* dL_ddc,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();

	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = { means[3 * idx + 0], means[3 * idx + 1], means[3 * idx + 2] };

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

    if (isnan(dL_dmean2D[idx].x) || isnan(dL_dmean2D[idx].y)) {
         printf("NaN detected in dL_dmean2D at idx %d: %f %f\n", idx, dL_dmean2D[idx].x, dL_dmean2D[idx].y);
    }

    if (isnan(dL_dmean.x) || isnan(dL_dmean.y) || isnan(dL_dmean.z)) {
         printf("NaN detected in dL_dmean (proj) at idx %d\n", idx);
    }

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
    atomicAdd(dL_dmeans + 3 * idx + 0, dL_dmean.x);
    atomicAdd(dL_dmeans + 3 * idx + 1, dL_dmean.y);
    atomicAdd(dL_dmeans + 3 * idx + 2, dL_dmean.z);

    glm::vec3 cam_pos_vec = *campos;

	// Compute gradient updates due to computing colors from SHs
	if (shs) {
		computeColorFromSH(idx, D, M, means, cam_pos_vec, dc, shs, clamped, dL_dcolor, dL_dmeans, dL_ddc, dL_dsh);
    }
	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales) {
        glm::vec3 scale = scales[idx];
        glm::vec4 rot = rotations[idx];

        computeCov3D(idx, scale, scale_modifier, rot, dL_dcov3D, (float*)(dL_dscale + idx), (float*)(dL_drot + idx));
    }

}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float2* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -1.0f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
            // Fix: Apply Sigmoid Chain Rule (dL/dp = dL/da * a * (1-a))
            // con_o.w holds the activated opacity 'a'
            const float a = con_o.w;
            const float d_sigmoid = a * (1.0f - a);
            
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha * d_sigmoid);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float* means3D,
	const int* radii,
    const float* dc,
	const float* shs,
	const bool* clamped,
    const float* opacities,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float2* dL_dmean2D,
	const float4* dL_dconic,
    float* dL_dopacity,
	float* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
    float* dL_ddc,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
    bool antialiasing)
{
	// Propagate gradients for the path of 2D conic matrix computation.
	// Somewhat long, thus it is its own kernel rather than being part of
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
        opacities,
		dL_dconic,
        dL_dopacity,
		dL_dmean3D,
		dL_dcov3D,
        antialiasing);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		radii,
        dc,
		shs,
		clamped,
		scales,
		rotations,
		scale_modifier,
		projmatrix,
		campos,
		dL_dmean2D,
		dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
        dL_ddc,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float2* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}
