#include "Dataset.cuh"
#include "scene.cuh"
#include "trainer.cuh"
#include "optimizer.cuh"
#include "densification.cuh"
#include "checkpoint.cuh"
#include "ssim.cuh"
#include "ImageIO.h"
#include "plyIO.h"

#include <filesystem>
#include <random>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    
    // 0. Argument Parsing & Setup
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_folder> <experiment_name> [checkpoint_step]" << std::endl;
        std::cerr << "       <data_folder>: Path to source data (containing images/ and sparse/0/)" << std::endl;
        std::cerr << "       <experiment_name>: Name of the run (e.g. truck_12072025)" << std::endl;
        std::cerr << "       [checkpoint_step]: Optional. Specific step to resume (e.g. step_500)" << std::endl;
        std::cerr << "Example New Run: " << argv[0] << " ./data/garden garden_experiment_01" << std::endl;
        std::cerr << "Example Resume:  " << argv[0] << " ./data/garden garden_experiment_01 step_5000" << std::endl;
        return -1;
    }

    // 1. Construct Input Paths
    fs::path data_root(argv[1]);
    fs::path sparse_fs_path = data_root / "sparse" / "0";
    fs::path images_fs_path = data_root / "images";

    std::string sparse_path = sparse_fs_path.string();
    std::string images_path = images_fs_path.string();

    // Validate Input Paths
    if (!fs::exists(sparse_fs_path)) {
        std::cerr << "Error: COLMAP sparse data not found at: " << sparse_path << std::endl;
        return -1;
    }
    if (!fs::exists(images_fs_path)) {
        std::cerr << "Error: Images directory not found at: " << images_path << std::endl;
        return -1;
    }

    // 2. Construct Output Directories
    std::string experiment_name = argv[2];
    fs::path output_root = "output";
    fs::path experiment_dir = output_root / experiment_name;
    
    fs::path checkpoints_dir = experiment_dir / "checkpoints";
    fs::path renders_dir = experiment_dir / "training_renders";
    fs::path clouds_dir = experiment_dir / "point_clouds";

    // Create directories
    try {
        if (!fs::exists(checkpoints_dir)) fs::create_directories(checkpoints_dir);
        if (!fs::exists(renders_dir)) fs::create_directories(renders_dir);
        if (!fs::exists(clouds_dir)) fs::create_directories(clouds_dir);
        
        std::cout << "Output Directory: " << experiment_dir.string() << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directories: " << e.what() << std::endl;
        return -1;
    }

    // 3. Resume Logic
    std::string checkpoint_path = "";
    bool resume = false;
    
    // Determine target checkpoint
    if (argc > 3) {
        // User specified a step (e.g., "step_500" or "step_500.ckpt")
        std::string step_arg = argv[3];
        fs::path target_ckpt = checkpoints_dir / step_arg;
        
        // Append extension if missing
        if (target_ckpt.extension() != ".ckpt") {
            target_ckpt += ".ckpt";
        }

        if (fs::exists(target_ckpt)) {
            checkpoint_path = target_ckpt.string();
            resume = true;
        } else {
            std::cerr << "Warning: Requested checkpoint " << target_ckpt.string() << " not found. Starting from scratch." << std::endl;
        }
    } else {
        // No specific step provided, check for 'latest.ckpt'
        fs::path latest_ckpt = checkpoints_dir / "latest.ckpt";
        if (fs::exists(latest_ckpt)) {
            checkpoint_path = latest_ckpt.string();
            resume = true;
            std::cout << "Auto-resume: Found latest.ckpt" << std::endl;
        }
    }


    // 1. Load Colmap data

    std::cout << "Loading data from root: " << data_root << std::endl;
    Dataset dataset(sparse_path, images_path);
    
    // 2. Setting up the scenes, Gaussians and gradients
    
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
    std::vector<float>     h_dc(P * 3);            
    std::vector<float>     h_shs(P * M * 3, 0.0f); 
    
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
    
    // 3. Initialize trainer and optimizer
    Optimizer optimizer(P, M);

    // Setup Random States for Densification
    size_t max_capacity = 3900000; 
    CudaBuffer<curandState> d_rand_states(max_capacity);
    init_random_states(d_rand_states.get(), max_capacity, 42);
    printf("Finished init random state\n");

    
    auto [max_w, max_h] = dataset.getMaxDimensions();
    const int num_pixels = max_w*max_h;

    SSIMData ssim_data(max_w, max_h, 3);
    
    std::vector<float> h_gt_image(num_pixels * 3);
    CudaBuffer<float> d_gt_image(num_pixels * 3);
    d_gt_image.to_device(h_gt_image);

    Trainer trainer(scene, grads, optimizer, ssim_data, max_w, max_h);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);

    std::vector<float> h_render(num_pixels * 3);

    // Training Constants
    const float densify_grad_threshold = 200.0f; // pixel unit
    const int total_iterations = 10000; 
    const int warmup_steps = 1000;
    const int sh_increase_interval = 1000;
    const int densify_interval = 100;
    const int opacitiy_reset_interval = 3000;
    const int debug_image_interval = 200;
    const int checkpoint_interval = 500;

    int active_sh_degree = 0;
    int start_iteration = 1;

    // Apply Resume
    if (resume) {
        start_iteration = CheckpointIO::load(checkpoint_path, scene, grads, optimizer, active_sh_degree);
                        
        printf("Resumed from step %d\n", start_iteration);
        printf("With active SH degree %d\n", active_sh_degree);

        if (start_iteration % 3000 == 0) {
            trainer.reset_opacity();
            printf("Force resetting opacity. Including momentum \n");
        }
        
        start_iteration++;
    } else {
        printf("Starting from scratch.\n");
        printf("Phase 1: Warmup (0-%d steps) - SH:0, No Densification\n", warmup_steps);
    }
    
    printf("Starting training loop.\n");
    
    
    // 4. Training Loop
    
    for (int i = start_iteration; i <= total_iterations; ++i) {
        auto item = dataset.get_item(dist(rng));
        
        // Upload ground truth image
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
        if (i > warmup_steps && i % densify_interval == 0) {
            if (i < total_iterations - 1000) {
                if (scene.count * 2 > max_capacity) {
                    printf("WARNING: RNG buffer limit. Skipping densification.\n");
                } else {
                    trainer.densify_and_prune(
                        densify_grad_threshold, 
                        scene_extent, 
                        d_rand_states.get()
                    );
                }
            }            
        }

        // Reset opacity
        if (i % opacitiy_reset_interval == 0) {
            printf("Resetting opacities\n");
            trainer.reset_opacity();
        }

        // Save debug image occasionally
        if (i % debug_image_interval == 0 || i == total_iterations) {
            trainer.get_current_render(h_render);
            
            // Construct filename: output/<experiment>/training_renders/train_step_X_imgname.jpg
            fs::path filename = renders_dir / ("train_step_" + std::to_string(i) + "_" + item.view->image_name + ".jpg");
            save_image_jpg(filename.string().c_str(), h_render, max_w, max_h, 90);
        }

        // Save checkpoints
        if (i % checkpoint_interval == 0) {
             
             fs::path ckpt_step_path = checkpoints_dir / ("step_" + std::to_string(i) + ".ckpt");
             fs::path ckpt_latest_path = checkpoints_dir / "latest.ckpt";
             fs::path ply_path = clouds_dir / ("iteration_" + std::to_string(i) + ".ply");

             CheckpointIO::save(ckpt_step_path.string(), i, scene, optimizer, active_sh_degree);
             CheckpointIO::save(ckpt_latest_path.string(), i, scene, optimizer, active_sh_degree);
             save_ply(ply_path.string(), scene);
        }
    }
    
    printf("Training complete.\n");
    return 0;
}
