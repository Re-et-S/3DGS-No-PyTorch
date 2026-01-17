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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <glm/glm.hpp>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
            const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const glm::vec3* scales,
			const float scale_modifier,
			const glm::vec4* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const glm::vec3* cam_pos,
			const float tan_fovx, float tan_fovy,
            const float focal_x, float focal_y,
			const bool prefiltered,
			float* out_color,
			float* depth,
            uint32_t* tiles_touched_out,
			bool antialiasing,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* means3D,
            const float* dc,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const glm::vec3* scales,
			const float scale_modifier,
			const glm::vec4* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const glm::vec3* campos,
			const float tan_fovx, float tan_fovy,
            const float focal_x, float focal_y,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float2* dL_dmean2D,
			float4* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
            float* dL_ddc,
			float* dL_dsh,
			glm::vec3* dL_dscale,
			glm::vec4* dL_drot,
			bool antialiasing,
			bool debug);
	};
};

#endif
