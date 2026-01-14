#pragma once
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <memory>
#include <functional>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "cuda_buffer.cuh"
#include "scene.cuh"
#include "rasterizer.cuh"

class ViewerRenderer {
public:
    ViewerRenderer();
    ~ViewerRenderer();

    void resize(int w, int h);

    void render(
        const GaussianScene& scene,
        const glm::mat4& view,
        const glm::mat4& proj,
        const glm::vec3& cam_pos,
        float fov_x, float fov_y,
        int active_sh_degree,
        cudaGraphicsResource* cuda_gl_image 
    );

private:
    int width = 800;
    int height = 600;

    CudaBuffer<char> geomBuffer;
    CudaBuffer<char> binningBuffer;
    CudaBuffer<char> imgBuffer;

    CudaBuffer<float> d_viewmatrix;
    CudaBuffer<float> d_projmatrix;
    CudaBuffer<float> d_bg_color;
    CudaBuffer<glm::vec3> d_cam_pos;

    // auxiliary needed for the forward pass function
    CudaBuffer<uint32_t> d_tiles_touched; 
    CudaBuffer<int> d_radii;
    
    CudaBuffer<float> d_linear_rgb;
};
