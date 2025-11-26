#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "scene.cuh"
#include "optimizer.cuh"

// Magic number to verify file type
const uint32_t CHECKPOINT_MAGIC = 0xDEADBEEF;

struct CheckpointHeader {
    uint32_t magic;
    int iteration;
    size_t count;      // P
    int sh_degree;     // D
    int active_sh_degree;    
};

class CheckpointIO {
    public:
    static void save(const std::string& path, 
                     int iteration, 
                     GaussianScene& scene, 
                     Optimizer& opt,
                     int active_sh_degree) {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Failed to open " << path << " for writing." << std::endl;
            return;
        }

        // 1. Write Header
        CheckpointHeader header;
        header.magic = CHECKPOINT_MAGIC;
        header.iteration = iteration;
        header.count = scene.count;
        header.sh_degree = scene.sh_degree;
        header.active_sh_degree = active_sh_degree;

        out.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // 2. Helper Lambda to Write Buffer
        // We reuse a single host vector to save RAM
        auto write_buffer = [&](auto& buffer) {
            // Determine type of buffer by looking at pointer type (simplified for raw bytes)
            // We just calculate total bytes
            size_t bytes = buffer.size_bytes; 
            
            // Allocate temporary host memory
            std::vector<char> host_data(bytes);
            
            // Copy Device -> Host
            CUDA_CHECK(cudaMemcpy(host_data.data(), buffer.get(), bytes, cudaMemcpyDeviceToHost));
            
            // Write to file
            out.write(host_data.data(), bytes);
        };

        std::cout << "Saving checkpoint to " << path << " (" << scene.count << " gaussians)..." << std::endl;

        // 3. Write Scene Data
        // Must match Load order.
        write_buffer(scene.d_points);     // float3
        write_buffer(scene.d_scales);     // float3
        write_buffer(scene.d_rotations);  // float4
        write_buffer(scene.d_opacities);  // float
        write_buffer(scene.d_dc);         // float3
        write_buffer(scene.d_shs);        // float

        // 4. Write Optimizer Data
        // We must save both moments (m and v) for every parameter
        write_buffer(opt.m_points); write_buffer(opt.v_points);
        write_buffer(opt.m_scales); write_buffer(opt.v_scales);
        write_buffer(opt.m_rots);   write_buffer(opt.v_rots);
        write_buffer(opt.m_opacities); write_buffer(opt.v_opacities);
        write_buffer(opt.m_dc);     write_buffer(opt.v_dc);
        write_buffer(opt.m_shs);    write_buffer(opt.v_shs);

        out.close();
        std::cout << "Save complete." << std::endl;
    }
    // Returns the iteration number we resumed from
    static int load(const std::string& path, 
                    GaussianScene& scene,
                    GaussianGrads& grad,
                    Optimizer& opt,
                    int& active_sh_degree) 
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Could not open checkpoint file.");
        }

        // 1. Read Header
        CheckpointHeader header;
        in.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (header.magic != CHECKPOINT_MAGIC) {
            throw std::runtime_error("Invalid checkpoint file (Bad Magic Number).");
        }

        active_sh_degree = header.active_sh_degree;

        std::cout << "Loading checkpoint: " << header.count << " Gaussians from it " << header.iteration << std::endl;
        std::cout << "Loading checkpoint: " << header.sh_degree << " SH bands " << header.iteration << std::endl;
        std::cout << "Loading checkpoint: " << header.active_sh_degree << " SH bands active" << header.iteration << std::endl;
        
        // 2. RESIZE BUFFERS
        // Critical: The loaded scene might have more points (due to densification)
        // than the initialized scene.
        scene.resize(header.count, header.sh_degree);
        opt.resize(header.count, (header.sh_degree + 1)*(header.sh_degree + 1)); // Ensure you implemented resize() in Optimizer from previous step!
        grad.resize(header.count);
        
        // 3. Helper Lambda to Read Buffer
        auto read_buffer = [&](auto& buffer) {
            size_t bytes = buffer.size_bytes;
            std::vector<char> host_data(bytes);
            
            in.read(host_data.data(), bytes);
            
            CUDA_CHECK(cudaMemcpy(buffer.get(), host_data.data(), bytes, cudaMemcpyHostToDevice));
        };

        // 4. Read Scene Data (Exact same order as Save)
        read_buffer(scene.d_points);
        read_buffer(scene.d_scales);
        read_buffer(scene.d_rotations);
        read_buffer(scene.d_opacities);
        read_buffer(scene.d_dc);
        read_buffer(scene.d_shs);

        // 5. Read Optimizer Data
        read_buffer(opt.m_points); read_buffer(opt.v_points);
        read_buffer(opt.m_scales); read_buffer(opt.v_scales);
        read_buffer(opt.m_rots);   read_buffer(opt.v_rots);
        read_buffer(opt.m_opacities); read_buffer(opt.v_opacities);
        read_buffer(opt.m_dc);     read_buffer(opt.v_dc);
        read_buffer(opt.m_shs);    read_buffer(opt.v_shs);

        in.close();
        return header.iteration;
    }
};
