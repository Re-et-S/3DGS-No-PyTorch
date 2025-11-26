#include "Dataset.cuh"
#include "scene.cuh"
#include "trainer.cuh"
#include "optimizer.cuh"
#include "densification.cuh"
#include "checkpoint.cuh"
#include "ImageIO.h"
#include "plyIO.h"
#include <random>

int main(int argc, char** argv) {
    // 1. IO Phase (ColmapLoader)
    Dataset dataset("truck/sparse/0/", "truck/images/");
    // 2. Data Phase (Scene)
    const auto& scene_points = dataset.getPoints();
    
    const int P = scene_points.size();
    const int D = 3;
    const int M = (D + 1) * (D + 1); // M = 16



    // Host-side parameter buffers
    std::vector<glm::vec3>    h_points(P);
    std::vector<float>     h_means(P * 3);
    std::vector<glm::vec3> h_scales(P);
    std::vector<glm::vec4> h_rotations(P);
    std::vector<float>     h_opacities(P);
    std::vector<float>     h_dc(P * 3);          // P*1*3
    std::vector<float>     h_shs(P * M * 3, 0.0f); // P*16*3 (dirty buffer)
    initGaussians(scene_points,
                  scene_points.size(),
                  h_points,
                  h_means,
                  h_opacities,
                  h_scales,
                  h_rotations,
                  h_dc);
    GaussianScene scene(D, h_means, h_scales, h_rotations, h_opacities, h_dc, h_shs);
    GaussianGrads grads(P, D); 
    // Calculate scene extent for densification thresholds
    float scene_extent = compute_scene_extent(h_points);
    printf("Scene Extent: %f\n", scene_extent);
    
    // 3. Logic Phase (Optimizer)
    Optimizer optimizer(P, M);

    // Setup Random States for Densification
    // Allocate extra space because the number of Gaussians will increase during training.
    size_t max_capacity = 3900000; 
    CudaBuffer<curandState> d_rand_states(max_capacity);
    init_random_states(d_rand_states.get(), max_capacity, 42);
    printf("Finished init random state\n");
    
    // 4. Orchestration Phase (Trainer)
    // Retrieve max dimensions to configure the Trainer
    auto [max_w, max_h] = dataset.getMaxDimensions();
    const int num_pixels = max_w*max_h;

    std::vector<float> h_gt_image(num_pixels * 3);
    CudaBuffer<float> d_gt_image(num_pixels * 3);
    d_gt_image.to_device(h_gt_image);

    // Pass the objects and the config to the Trainer
    Trainer trainer(scene, grads, optimizer, max_w, max_h);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);

    std::vector<float> h_render(num_pixels * 3);

    // Training Constants
    const float densify_grad_threshold = 25.0f;
    const int total_iterations = 10000; // 
    const int warmup_steps = 1000;
    const int sh_increase_interval = 1000;
    const int densify_interval = 100;

    int active_sh_degree = 0;

    int start_iteration = 1;
    std::string checkpoint_path = "checkpoints/latest.ckpt";

    // CHECK FOR RESUME
    if (argc > 1) checkpoint_path = argv[1];
    // Simple file existence check
    std::ifstream f(checkpoint_path.c_str());
    bool resume = f.good();
    f.close();

    if (resume) {
        // This will Resize the scene and fill it with data
        start_iteration = CheckpointIO::load(checkpoint_path, scene, grads, optimizer, active_sh_degree);
                
        // TODO Also ensure Random States buffer is large enough
        // if (scene.count * 4 > d_rand_states.count) {
        //      // Reallocate RNG if needed
        // }
        
        printf("Resumed from step %d\n", start_iteration);
        printf("With active SH degree %d\n", active_sh_degree);

        if (start_iteration % 3000 == 0) {
            trainer.reset_opacity();
            printf("Force resetting opacity. Including momentum \n");
        }
        
        start_iteration++; // Start working on the next step
    } else {
        printf("No checkpoint found. Starting from scratch.\n");
        printf("Phase 1: Warmup (0-%d steps) - SH:0, No Densification\n", warmup_steps);
    }
    
    printf("Starting training loop.\n");
    
    // 5. Loop
    for (int i = start_iteration; i <= total_iterations; ++i) {
        auto item = dataset.get_item(dist(rng));
        
        // Upload GT
        d_gt_image.to_device(item.gt_image);

        if (i % sh_increase_interval == 0) {
            if (active_sh_degree < D) {
                active_sh_degree++;
                printf("[Step %d] UPGRADE: Increasing SH Degree to %d\n", i, active_sh_degree);
            }
        }

        // Train Step
        double loss = trainer.train_step(*item.view, d_gt_image, active_sh_degree);

        if (i % 5 == 0) {
            printf("Step %d | Loss: %f | Gaussians: %lu\n", i, loss, scene.count);
        }

        // Densification Logic
        // Usually applied after a warmup period (e.g. > 500) and stops before the end.
        // For this specific request, we run it strictly every 100 steps.
        if (i > warmup_steps && i % densify_interval == 0) {
            if (i < total_iterations - 1000) {
                if (scene.count * 2 > max_capacity) {
                    printf("WARNING: RNG buffer limit. Skipping densification.\n");
                    // TODO resize here
                } else {
                    trainer.densify_and_prune(
                        densify_grad_threshold, 
                        scene_extent, 
                        d_rand_states.get()
                    );
                }
            }            
        }

        // reset opacity
        if (i % 3000 == 0) {
            printf("Resetting opacities\n");
            trainer.reset_opacity();
        }
        // Save debug image occasionally
        if (i % 100 == 0 || i == total_iterations) {
            trainer.get_current_render(h_render);
            std::string filename = "train_step_" + std::to_string(i) + "_" + item.view->image_name + ".ppm";
            save_image_ppm(filename.c_str(), h_render, max_w, max_h);
        }

        // Save checkpoints
        if (i % 500 == 0) {
             std::string path = "checkpoints/step_" + std::to_string(i) + ".ckpt";
             CheckpointIO::save(path, i, scene, optimizer, active_sh_degree);
             
             // Also overwrite "latest"
             CheckpointIO::save("checkpoints/latest.ckpt", i, scene, optimizer, active_sh_degree);
             save_ply("point_cloud/iteration_" + std::to_string(i) + ".ply", scene);
        }
    }
    
    printf("Training complete.\n");
    return 0;
}
