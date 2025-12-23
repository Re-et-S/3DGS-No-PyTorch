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
    out << "property float scale_0\n";
    out << "property float scale_1\n";
    out << "property float scale_2\n";    
    out << "property float opacity\n";
    out << "property float rot_0\n";
    out << "property float rot_1\n";
    out << "property float rot_2\n";
    out << "property float rot_3\n";
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


    out << "end_header\n";

    // 3. Write Binary Data
    std::cout << "Writing PLY to " << filename << "..." << std::endl;

    for (size_t i = 0; i < scene.count; ++i) {
        // A. Position
        out.write(reinterpret_cast<const char*>(&h_points[i * 3 + 0]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_points[i * 3 + 1]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_points[i * 3 + 2]), sizeof(float));

        // B. Scale
        out.write(reinterpret_cast<const char*>(&h_scales[i].x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_scales[i].y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_scales[i].z), sizeof(float));
      
        // C. Opacity
        out.write(reinterpret_cast<const char*>(&h_opacities[i]), sizeof(float));

        // D. Rotation (Quaternion)
        // normalize
        glm::vec4 q = h_rotations[i];
        float len = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
        if (len > 0) {
            q /= len; 
        } else {
            q = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); // Fallback
        }

        out.write(reinterpret_cast<const char *>(&q.x),
                  sizeof(float)); // rot_0 =  real
        float neg_i = -q.y;
        out.write(reinterpret_cast<const char *>(&neg_i),
                  sizeof(float)); // rot_1 = -i
        float neg_j = -q.z;
        out.write(reinterpret_cast<const char *>(&neg_j),
                  sizeof(float)); // rot_2 = -j
        float neg_k = -q.w;
        out.write(reinterpret_cast<const char *>(&neg_k),
                  sizeof(float)); // rot_3 = -k

        // E. DC Features (From d_dc)
        // R, G, B
        // 1. Load the optimizer state
        out.write(reinterpret_cast<const char*>(&h_dc[i * 3 + 0]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_dc[i * 3 + 1]), sizeof(float));
        out.write(reinterpret_cast<const char*>(&h_dc[i * 3 + 2]), sizeof(float));

        // F. Rest Features (From d_shs)
        int stride = scene.max_sh_coeffs * 3;
        int base_idx = i * stride;
        
        // Write SH coefficients in Planar order (RR...GG...BB...) to match Brush/Splat.
        // Skip DC (index 0) and write the rest (1 to max_sh_coeffs-1)
        for (int channel = 0; channel < 3; ++channel) {
          for (int j = 1; j < scene.max_sh_coeffs; ++j) {
            // Determine index in the interleaved h_shs array
            // coeff j for channel is at: j * 3 + channel
            int k = j * 3 + channel;
            float coeff = h_shs[base_idx + k];

            // Apply negation logic based on SH band index j
            // (Matches original logic: (k/3)%2 != 0  => j%2 != 0)
            if (j % 2 != 0) {
              coeff = -coeff;
            }
            out.write(reinterpret_cast<const char *>(&coeff), sizeof(float));
          }
        }

    }

    out.close();
    std::cout << "PLY Saved." << std::endl;
}

#include <sstream>

// Helper to determine SH degree from number of rest coefficients
// Total coeffs = 3 (DC) + num_rest
// coeffs_per_channel = Total / 3
// (degree + 1)^2 = coeffs_per_channel
inline int degree_from_rest_coeffs(int num_rest) {
    int total_coeffs_per_channel = (num_rest + 3) / 3;
    int degree = (int)sqrt(total_coeffs_per_channel) - 1;
    return degree;
}

inline GaussianScene load_ply(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open " + filename + " for reading PLY.");
    }

    std::cout << "Loading PLY from " << filename << "..." << std::endl;

    // Parse Header
    std::string line;
    size_t count = 0;
    int num_rest_coeffs = 0;

    while (std::getline(in, line)) {
        if (line.find("element vertex") != std::string::npos) {
            std::stringstream ss(line);
            std::string temp;
            ss >> temp >> temp >> count;
        } else if (line.find("property float f_rest_") != std::string::npos) {
            num_rest_coeffs++;
        } else if (line == "end_header") {
            break;
        }
    }

    int sh_degree = degree_from_rest_coeffs(num_rest_coeffs);
    int max_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);

    std::cout << "Detected " << count << " Gaussians." << std::endl;
    std::cout << "Detected " << num_rest_coeffs << " rest coefficients => SH Degree " << sh_degree << std::endl;

    // Allocate Host Buffers
    std::vector<float> h_points(count * 3);
    std::vector<float> h_dc(count * 3);
    std::vector<float> h_shs(count * max_sh_coeffs * 3);
    std::vector<float> h_opacities(count);
    std::vector<glm::vec3> h_scales(count);
    std::vector<glm::vec4> h_rotations(count);

    // Read Data
    for (size_t i = 0; i < count; ++i) {
        // A. Position
        in.read(reinterpret_cast<char*>(&h_points[i * 3 + 0]), sizeof(float));
        in.read(reinterpret_cast<char*>(&h_points[i * 3 + 1]), sizeof(float));
        in.read(reinterpret_cast<char*>(&h_points[i * 3 + 2]), sizeof(float));

        // B. Scale
        float sx, sy, sz;
        in.read(reinterpret_cast<char*>(&sx), sizeof(float));
        in.read(reinterpret_cast<char*>(&sy), sizeof(float));
        in.read(reinterpret_cast<char*>(&sz), sizeof(float));
        h_scales[i] = glm::vec3(sx, sy, sz);

        // C. Opacity
        in.read(reinterpret_cast<char*>(&h_opacities[i]), sizeof(float));

        // D. Rotation
        float r0, r1, r2, r3;
        in.read(reinterpret_cast<char*>(&r0), sizeof(float));
        in.read(reinterpret_cast<char*>(&r1), sizeof(float));
        in.read(reinterpret_cast<char*>(&r2), sizeof(float));
        in.read(reinterpret_cast<char*>(&r3), sizeof(float));

        // Apply Negation to imaginary parts to match internal representation
        h_rotations[i] = glm::vec4(r0, -r1, -r2, -r3);

        // E. DC
        in.read(reinterpret_cast<char*>(&h_dc[i * 3 + 0]), sizeof(float));
        in.read(reinterpret_cast<char*>(&h_dc[i * 3 + 1]), sizeof(float));
        in.read(reinterpret_cast<char*>(&h_dc[i * 3 + 2]), sizeof(float));

        // F. Rest Features (Planar -> Interleaved)
        // File: R1..Rn, G1..Gn, B1..Bn
        // Internal: R1,G1,B1, R2,G2,B2...

        std::vector<float> file_rest(num_rest_coeffs);
        in.read(reinterpret_cast<char*>(file_rest.data()), num_rest_coeffs * sizeof(float));

        int coeffs_per_channel = max_sh_coeffs; // Includes DC
        int rest_per_channel = coeffs_per_channel - 1;

        int base_idx = i * max_sh_coeffs * 3;

        for (int channel = 0; channel < 3; ++channel) {
            for (int j = 1; j < max_sh_coeffs; ++j) {
                // Input index (Planar): Channel offset + (coeff_index - 1)
                int file_idx = channel * rest_per_channel + (j - 1);

                float val = file_rest[file_idx];

                // Negation Logic for odd bands
                if (j % 2 != 0) {
                    val = -val;
                }

                // Output index (Interleaved): j * 3 + channel
                // Destination index in h_shs (relative to this Gaussian)
                int dest_idx = base_idx + j * 3 + channel;
                h_shs[dest_idx] = val;
            }
        }

        // Copy DC into h_shs[0,1,2] for completeness
        h_shs[base_idx + 0] = h_dc[i*3 + 0];
        h_shs[base_idx + 1] = h_dc[i*3 + 1];
        h_shs[base_idx + 2] = h_dc[i*3 + 2];
    }

    in.close();
    std::cout << "PLY Loaded." << std::endl;

    return GaussianScene(sh_degree, h_points, h_scales, h_rotations, h_opacities, h_dc, h_shs);
}
