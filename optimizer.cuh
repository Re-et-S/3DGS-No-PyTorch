#pragma once
#include "cuda_buffer.cuh"
#include "adam.cuh"
 
class Optimizer {
public:
    struct Config {
        float lr_pos = 0.00016f;
        float lr_scale = 0.005f;
        float lr_rot = 0.001f;
        float lr_opacity = 0.05f;
        float lr_color = 0.0025f;
        float b1 = 0.9f;
        float b2 = 0.999f;
        float eps = 1e-8f;
    } config;

    // State buffers (1st and 2nd moments)
    CudaBuffer<float> m_points, v_points;
    CudaBuffer<float> m_scales, v_scales;
    CudaBuffer<float> m_rots, v_rots;
    CudaBuffer<float> m_opacities, v_opacities;
    CudaBuffer<float> m_dc, v_dc;
    CudaBuffer<float> m_shs, v_shs;

    // gradient accumulation buffer
    CudaBuffer<float> accum_max_pos_grad;
    CudaBuffer<int> denom;

    // masks
    CudaBuffer<int> clone_mask;
    
    Optimizer(size_t P, size_t sh_coeffs) :
        m_points(P*3), v_points(P*3),
        m_scales(P*3), v_scales(P*3),
        m_rots(P*4), v_rots(P*4),
        m_opacities(P), v_opacities(P),
        m_dc(P*3), v_dc(P*3),
        m_shs(P*sh_coeffs*3), v_shs(P*sh_coeffs*3),
        accum_max_pos_grad(P), denom(P),
        clone_mask(P)
    {
        // Initialize to 0
        m_points.clear(); v_points.clear();
        m_scales.clear(); v_scales.clear();
        m_rots.clear(); v_rots.clear();
        m_opacities.clear(); v_opacities.clear();
        m_dc.clear(); v_dc.clear();
        m_shs.clear(); v_shs.clear();
        accum_max_pos_grad.clear(); denom.clear();
        clone_mask.clear();
        
    }

    void step(GaussianScene& scene, const GaussianGrads& grads, const CudaBuffer<uint32_t>& tiles_touched) {
        size_t P = scene.count;
        
        // Helper lambda to keep code clean
        auto update = [&](auto& param, auto& grad, auto& m, auto& v, float lr, int M) {
            ADAM::adamUpdate(
                reinterpret_cast<float*>(param.get()),
                reinterpret_cast<const float*>(grad.get()),
                m.get(), v.get(),
                tiles_touched.get(),
                lr, config.b1, config.b2, config.eps,
                P, M
            );
        };

        update(scene.d_points,    grads.d_dL_dpoints,    m_points,    v_points,    config.lr_pos,     3);
        update(scene.d_scales,    grads.d_dL_dscales,    m_scales,    v_scales,    config.lr_scale,   3);
        update(scene.d_rotations, grads.d_dL_drotations, m_rots,      v_rots,      config.lr_rot,     4);
        update(scene.d_opacities, grads.d_dL_dopacities, m_opacities, v_opacities, config.lr_opacity, 1);
        update(scene.d_dc,        grads.d_dL_ddc,        m_dc,        v_dc,        config.lr_color,   3);
        update(scene.d_shs,       grads.d_dL_dshs,       m_shs,       v_shs,       config.lr_color,   scene.max_sh_coeffs * 3);
    }
    
    void replace_with(Optimizer& other) {
        m_points.swap(other.m_points); v_points.swap(other.v_points);
        m_scales.swap(other.m_scales); v_scales.swap(other.v_scales);
        m_rots.swap(other.m_rots); v_rots.swap(other.v_rots);
        m_opacities.swap(other.m_opacities); v_opacities.swap(other.v_opacities);
        m_dc.swap(other.m_dc); v_dc.swap(other.v_dc);
        m_shs.swap(other.m_shs); v_shs.swap(other.v_shs);
        accum_max_pos_grad.swap(other.accum_max_pos_grad); denom.swap(other.denom);
        clone_mask.swap(other.clone_mask);

        // clear buffers for densification
        accum_max_pos_grad.clear(); denom.clear();
        clone_mask.clear();        
    };

    void resize(size_t new_count, size_t sh_coeffs) {
        // 1. Create temp optimizer
        Optimizer temp_opt(new_count, sh_coeffs);
        
        // 2. Swap
        this->replace_with(temp_opt);
    };
    void reset_opacity_state() {
        // Reset the Adam moments for opacity to zero
        m_opacities.clear(); 
        v_opacities.clear();
    };
};
