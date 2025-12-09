#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "scene.cuh"

inline void save_ply(const std::string& filename, GaussianScene& scene) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Could not open " << filename << " for writing PLY." << std::endl;
        return;
    }

    // 1. Download Data from GPU to Host
    // We need CPU access to write to file
    std::vector<float> h_points(scene.count * 3);
    std::vector<float> h_dc(scene.count * 3);
    std::vector<float> h_shs(scene.count * scene.max_sh_coeffs * 3);
    std::vector<float> h_opacities(scene.count);
    std::vector<glm::vec3> h_scales(scene.count);
    std::vector<glm::vec4> h_rotations(scene.count);

    scene.d_points.from_device(h_points);
    scene.d_dc.from_device(h_dc);
    scene.d_shs.from_device(h_shs);
    scene.d_opacities.from_device(h_opacities);
    scene.d_scales.from_device(h_scales);
    scene.d_rotations.from_device(h_rotations);

    // 2. Write Header
    out << "ply\n";
    out << "format binary_little_endian 1.0\n";
    out << "element vertex " << scene.count << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property float nx\n";
    out << "property float ny\n";
    out << "property float nz\n";
    
    // DC (f_dc_0, f_dc_1, f_dc_2)
    out << "property float f_dc_0\n";
    out << "property float f_dc_1\n";
    out << "property float f_dc_2\n";

    // Rest of SH (f_rest_0 to f_rest_N)
    // Note: max_sh_coeffs includes DC (16 total for degree 3). 
    // We already wrote DC (3 floats), so we write the remaining (15 * 3 = 45 floats).
    int rest_coeffs = (scene.max_sh_coeffs - 1) * 3;
    for (int i = 0; i < rest_coeffs; ++i) {
        out << "property float f_rest_" << i << "\n";
    }

    out << "property float opacity\n";
    out << "property float scale_0\n";
    out << "property float scale_1\n";
    out << "property float scale_2\n";
    out << "property float rot_0\n";
    out << "property float rot_1\n";
    out << "property float rot_2\n";
    out << "property float rot_3\n";
    out << "end_header\n";

    // 3. Write Binary Data
    std::cout << "Writing PLY to " << filename << "..." << std::endl;

    for (size_t i = 0; i < scene.count; ++i) {
        // A. Position
        out.write(reinterpret_cast<const char*>(&h_points[i * 3 + 0]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_points[i * 3 + 1]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_points[i * 3 + 2]), sizeof(float));

        // B. Normals (Zero)
        float zero = 0.0f;
        out.write(reinterpret_cast<const char*>(&zero), sizeof(float));
        out.write(reinterpret_cast<const char*>(&zero), sizeof(float));
        out.write(reinterpret_cast<const char*>(&zero), sizeof(float));

        // C. DC Features (From d_dc)
        // R, G, B
        // 1. Load the optimizer state (Logit)
        out.write(reinterpret_cast<const char*>(&h_dc[i * 3 + 0]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_dc[i * 3 + 1]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_dc[i * 3 + 2]), sizeof(float));

        // D. Rest Features (From d_shs)
        int stride = scene.max_sh_coeffs * 3;
        int base_idx = i * stride;
        
        // Skip DC (3 floats) and write the rest
        for (int k = 3; k < stride; ++k) {
            out.write(reinterpret_cast<const char*>(&h_shs[base_idx + k]), sizeof(float));
        }

        // E. Opacity
        out.write(reinterpret_cast<const char*>(&h_opacities[i]), sizeof(float));

        // F. Scale
        out.write(reinterpret_cast<const char*>(&h_scales[i].x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_scales[i].y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_scales[i].z), sizeof(float));

        // G. Rotation (Quaternion)
        // normalize
        glm::vec4 q = h_rotations[i];
        float len = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
        if (len > 0) {
            q /= len; 
        } else {
            q = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); // Fallback
        }
        
        out.write(reinterpret_cast<const char*>(&h_rotations[i].w), sizeof(float)); // rot_0
        out.write(reinterpret_cast<const char*>(&h_rotations[i].x), sizeof(float)); // rot_1
        out.write(reinterpret_cast<const char*>(&h_rotations[i].y), sizeof(float)); // rot_2
        out.write(reinterpret_cast<const char*>(&h_rotations[i].z), sizeof(float)); // rot_3
    }

    out.close();
    std::cout << "PLY Saved." << std::endl;
}
