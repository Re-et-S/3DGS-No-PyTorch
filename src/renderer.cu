#include "renderer.cuh"
#include <iostream>

__global__ void RGB_to_RGBA_Kernel(
    const float* __restrict__ rgb_in, 
    cudaSurfaceObject_t surf_out,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_idx = y * width + x; 
    
    int num_pixels = width * height;

    float4 pixel;
    
    pixel.x = rgb_in[pixel_idx]; 
    pixel.y = rgb_in[pixel_idx + num_pixels]; 
    pixel.z = rgb_in[pixel_idx + 2 * num_pixels]; 
    
    pixel.w = 1.0f; // Alpha

    surf2Dwrite(pixel, surf_out, x * sizeof(float4), y); 
}

ViewerRenderer::ViewerRenderer():
      geomBuffer(1024*1024*256), 
      binningBuffer(1024*1024*256),
      imgBuffer(1024*1024*256),
      d_linear_rgb(width * height * 3),
      d_viewmatrix(16),
      d_projmatrix(16),
      d_bg_color(3),
      d_cam_pos(1){    
    resize(800, 600);
}

ViewerRenderer::~ViewerRenderer() {

}

void ViewerRenderer::resize(int w, int h) {
    if (w == width && h == height) return;
    width = w;
    height = h;

    geomBuffer.clear();
    binningBuffer.clear();
    imgBuffer.clear();
    d_linear_rgb.clear();

    // Resize the linear RGB buffer to the new dimensions
    d_linear_rgb.resize(width * height * 3);

    size_t MAX_GAUSSIANS = 2000000;
    d_radii.resize(MAX_GAUSSIANS); 
    d_tiles_touched.resize(MAX_GAUSSIANS);
}

void ViewerRenderer::render(
    const GaussianScene& scene,
    const glm::mat4& view,
    const glm::mat4& proj,
    const glm::vec3& cam_pos,
    float fov_x, float fov_y,
    int active_sh_degree,
    cudaGraphicsResource* cuda_gl_image
) {
    if (!d_linear_rgb.get()) return;

    float bg_color[3] = {0.0f, 0.0f, 0.0f};
    CUDA_CHECK(cudaMemcpy(d_bg_color.get(), bg_color, 3 * sizeof(float), cudaMemcpyHostToDevice));

    d_linear_rgb.clear();
    
    if (scene.count > d_radii.count) {
        d_radii.resize(scene.count);
        d_tiles_touched.resize(scene.count);
    }

    float tan_fovx = tan(fov_x * 0.5f);
    float tan_fovy = tan(fov_y * 0.5f);
    float focal_x = width / (2.0f * tan_fovx);
    float focal_y = height / (2.0f * tan_fovy);

    auto geomAlloc = [&](size_t s) { return geomBuffer.get(); };
    auto binAlloc = [&](size_t s) { return binningBuffer.get(); };
    auto imgAlloc = [&](size_t s) { return imgBuffer.get(); };

    glm::mat4 view_proj = proj*view;
    // glm::mat4 view_transpose = glm::transpose(view);
    // glm::mat4 proj_transpose = glm::transpose(view_proj);
    
    CUDA_CHECK(cudaMemcpy(d_viewmatrix.get(), glm::value_ptr(view), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_projmatrix.get(), glm::value_ptr(view_proj), 16 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cam_pos.get(), &cam_pos, sizeof(glm::vec3), cudaMemcpyHostToDevice));

    int num_rendered = CudaRasterizer::Rasterizer::forward(
        geomAlloc, binAlloc, imgAlloc,
        scene.count, active_sh_degree, scene.max_sh_coeffs,
        d_bg_color.get(),
        width, height,
        scene.d_points.get(),
        scene.d_dc.get(),
        scene.d_shs.get(),
        nullptr,
        scene.d_opacities.get(),
        scene.d_scales.get(),
        1.0f,
        scene.d_rotations.get(),
        nullptr,
        d_viewmatrix.get(),
        d_projmatrix.get(),
        d_cam_pos.get(),
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        false,
        d_linear_rgb.get(),
        nullptr, 
        d_tiles_touched.get(),
        false,
        d_radii.get(),
        false
    );

    cudaArray_t texture_ptr;
    cudaGraphicsMapResources(1, &cuda_gl_image, 0);
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_gl_image, 0, 0);

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texture_ptr;

    cudaSurfaceObject_t outputSurface;
    cudaCreateSurfaceObject(&outputSurface, &resDesc);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    RGB_to_RGBA_Kernel<<<gridSize, blockSize>>>(d_linear_rgb.get(), outputSurface, width, height);

    cudaDestroySurfaceObject(outputSurface);
    cudaGraphicsUnmapResources(1, &cuda_gl_image, 0);

}
