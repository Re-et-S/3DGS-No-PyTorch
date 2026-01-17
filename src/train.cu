#include "Dataset.cuh"
#include "scene.cuh"
#include "trainer.cuh"
#include "optimizer.cuh"
#include "densification.cuh"
#include "checkpoint.cuh"
#include "ssim.cuh"
#include "ImageIO.h"
#include "plyIO.h"
#include "config.h"
#include "forward.cuh"

#include <filesystem>
#include <random>

namespace fs = std::filesystem;

__global__ void alignmentCheckKernel(float* float_ptr) {
    // Cast the float pointer to a vec3 pointer, just like your code does
    glm::vec3* vec3_ptr = reinterpret_cast<glm::vec3*>(float_ptr);

    printf("=== Alignment Verification ===\n");
    printf("Size of float:      %lu bytes\n", sizeof(float));
    printf("Size of glm::vec3:  %lu bytes\n", sizeof(glm::vec3));

    // Check offsets for index 0 and index 1
    // We expect index 1 to start at byte 12 (3 * 4 bytes)
    size_t addr_float_0 = (size_t)&float_ptr[0];
    size_t addr_float_3 = (size_t)&float_ptr[3]; // The start of the 2nd vector

    size_t addr_vec3_0  = (size_t)&vec3_ptr[0];
    size_t addr_vec3_1  = (size_t)&vec3_ptr[1]; // The start of the 2nd vector according to glm::vec3*

    printf("Address float[0]:   %llu\n", (unsigned long long)addr_float_0);
    printf("Address float[3]:   %llu (Expected start of 2nd vector)\n", (unsigned long long)addr_float_3);
    printf("Address vec3[1]:    %llu (Actual start of 2nd vector)\n", (unsigned long long)addr_vec3_1);

    long long stride_float = (long long)addr_float_3 - (long long)addr_float_0;
    long long stride_vec3  = (long long)addr_vec3_1 - (long long)addr_vec3_0;

    printf("Stride (float*3):   %lld bytes\n", stride_float);
    printf("Stride (vec3*):     %lld bytes\n", stride_vec3);

    if (stride_float != stride_vec3) {
        printf("\nCRITICAL FAILURE: Mismatch detected! Your pointers are drifting.\n");
        printf("Thread 1 reads from byte %lld, but data is at byte %lld.\n", stride_vec3, stride_float);
    } else {
        printf("\nAlignment is OK.\n");
    }
}

__global__ void debug_glm_ops() {
    printf("=== GLM Debug Unit Test ===\n");

    // 1. Setup Test Data (Identity Rotation, Simple Scale)
    glm::vec3 scale(1.0f, 2.0f, 3.0f); // exp(1)=2.71, exp(2)=7.38, exp(3)=20.08
    float mod = 1.0f;
    
    // 2. Construct Matrix S (Scale)
    glm::mat3 S(1.0f);
    S[0][0] = mod * expf(scale.x);
    S[1][1] = mod * expf(scale.y);
    S[2][2] = mod * expf(scale.z);

    // 3. Construct Matrix R (Identity for simplicity)
    glm::mat3 R(1.0f);

    // 4. Perform Multiplication (The suspect)
    glm::mat3 M = S * R;
    // Note: GLM is Column-Major. 
    // If S*R works, M[0][0] should be S[0][0].

    printf("Input Scale X: %f -> Matrix S[0][0]: %f\n", scale.x, S[0][0]);
    printf("Multiplication Result M[0][0]: %f\n", M[0][0]);

    // 5. Test Transpose & Covariance
    // Your original code did: transpose(M) * M
    glm::mat3 Sigma_Old = glm::transpose(M) * M;
    
    // Standard 3DGS Math: M * transpose(M) (assuming M=RS) or similar
    glm::mat3 Sigma_New = M * glm::transpose(M);

    printf("Sigma_Old [0][0]: %f\n", Sigma_Old[0][0]);
    printf("Sigma_New [0][0]: %f\n", Sigma_New[0][0]);

    // 6. Memory Layout Check (Did the compiler pad the struct?)
    printf("Size of glm::mat3: %lu bytes (Expected 36)\n", sizeof(glm::mat3));
    
    // 7. Pointer Arithmetic Check
    // Create an array and see if stride matches size
    glm::mat3 array[2];
    size_t addr0 = (size_t)&array[0];
    size_t addr1 = (size_t)&array[1];
    printf("Array Stride: %lu bytes\n", addr1 - addr0);

    if (sizeof(glm::mat3) != 36 || (addr1 - addr0) != 36) {
        printf("CRITICAL: Alignment mismatch detected!\n");
    } else {
        printf("Alignment matches packed floats.\n");
    }
}

int main(int argc, char** argv) {
    
    CudaBuffer<float> d_align_check(12);
    alignmentCheckKernel<<<1, 1>>>(d_align_check.get());
    debug_glm_ops<<<1, 1>>>();
    cudaDeviceSynchronize();

    // 0. Argument Parsing & Setup
    GaussianSplatting::TrainingConfig config;
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data_folder> <experiment_name> [checkpoint_step]" << std::endl;
        std::cerr << "       <data_folder>: Path to source data (containing images/ and sparse/0/)" << std::endl;
        std::cerr << "       <experiment_name>: Name of the run (e.g. truck_12072025)" << std::endl;
        std::cerr << "       [checkpoint_step]: Optional. Specific step to resume (e.g. step_500)" << std::endl;
        std::cerr << "Example New Run: " << argv[0] << " ./data/garden garden_experiment_01" << std::endl;
        std::cerr << "Example Resume:  " << argv[0] << " ./data/garden garden_experiment_01 step_5000" << std::endl;
        return -1;
    }

    fs::path data_root(argv[1]);
    fs::path sparse_fs_path = data_root / "sparse" / "0";
    fs::path images_fs_path = data_root / "images";

    std::string sparse_path = sparse_fs_path.string();
    std::string images_path = images_fs_path.string();

    if (!fs::exists(sparse_fs_path)) {
        std::cerr << "Error: COLMAP sparse data not found at: " << sparse_path << std::endl;
        return -1;
    }
    if (!fs::exists(images_fs_path)) {
        std::cerr << "Error: Images directory not found at: " << images_path << std::endl;
        return -1;
    }

    std::string experiment_name = argv[2];
    fs::path output_root = "output";
    fs::path experiment_dir = output_root / experiment_name;
    
    fs::path checkpoints_dir = experiment_dir / "checkpoints";
    fs::path renders_dir = experiment_dir / "training_renders";
    fs::path clouds_dir = experiment_dir / "point_clouds";

    try {
        if (!fs::exists(checkpoints_dir)) fs::create_directories(checkpoints_dir);
        if (!fs::exists(renders_dir)) fs::create_directories(renders_dir);
        if (!fs::exists(clouds_dir)) fs::create_directories(clouds_dir);
        
        std::cout << "Output Directory: " << experiment_dir.string() << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directories: " << e.what() << std::endl;
        return -1;
    }

    std::string checkpoint_path = "";
    bool resume = false;
    
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
        fs::path latest_ckpt = checkpoints_dir / "latest.ckpt";
        if (fs::exists(latest_ckpt)) {
            checkpoint_path = latest_ckpt.string();
            resume = true;
            std::cout << "Auto-resume: Found latest.ckpt" << std::endl;
        }
    }


    std::cout << "Loading data from root: " << data_root << std::endl;
    Dataset dataset(sparse_path, images_path);
    
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

    // Verify Covariance Initialization (Debug for CUDA 13.1 issue)
    printf("Verifying Initial Covariances...\n");
    FORWARD::verify_initial_covariances(
        scene.count,
        scene.d_scales.get(),
        scene.d_rotations.get(),
        1.0f // scale_modifier
    );
    printf("Verification complete.\n");
    
    // 3. Initialize trainer and optimizer
    Optimizer optimizer(P, M);

    // Setup Random States for Densification
    size_t max_capacity = config.max_gaussian_count;
    CudaBuffer<curandState> d_rand_states(max_capacity);
    init_random_states(d_rand_states.get(), max_capacity, 42);
    printf("Finished init random state\n");

    
    auto [max_w, max_h] = dataset.getMaxDimensions();
    const int num_pixels = max_w*max_h;

    SSIMData ssim_data(max_w, max_h, 3);
    
    std::vector<float> h_gt_image(num_pixels * 3);
    CudaBuffer<float> d_gt_image(num_pixels * 3);
    d_gt_image.to_device(h_gt_image);

    Trainer trainer(scene, grads, optimizer, ssim_data, max_w, max_h, config);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, dataset.size() - 1);

    std::vector<float> h_render(num_pixels * 3);

    int active_sh_degree = 0;
    int start_iteration = 1;

    // Apply Resume
    if (resume) {
        start_iteration = CheckpointIO::load(checkpoint_path, scene, grads, optimizer, active_sh_degree);
                        
        printf("Resumed from step %d\n", start_iteration);
        printf("With active SH degree %d\n", active_sh_degree);

        if (start_iteration % config.opacity_reset_interval == 0) {
            trainer.reset_opacity();
            printf("Force resetting opacity. Including momentum \n");
        }
        
        start_iteration++;
    } else {
        printf("Starting from scratch.\n");
        printf("Phase 1: Warmup (0-%d steps) - SH:0, No Densification\n", config.warmup_steps);
    }
    
    printf("Starting training loop.\n");
    
    
    // 4. Training Loop
    
    for (int i = start_iteration; i <= config.total_iterations; ++i) {
        auto item = dataset.get_item(dist(rng));
        
        // Upload ground truth image
        d_gt_image.to_device(item.gt_image);

        if (i % config.sh_increase_interval == 0) {
            if (active_sh_degree < D) {
                active_sh_degree++;
                printf("[Step %d] UPGRADE: Increasing SH Degree to %d\n", i, active_sh_degree);
            }
        }

        // Train Step
        double loss = trainer.train_step(*item.view, d_gt_image, active_sh_degree);

        if (i == start_iteration) {
            trainer.get_current_render(h_render);
            fs::path filename = renders_dir / ("debug_step_" + std::to_string(i) + ".jpg");
            save_image_jpg(filename.string().c_str(), h_render, max_w, max_h, 90);
            printf("Debug image saved to %s. \n", filename.string().c_str());
            break;
        }

        if (i % 5 == 0) {
            printf("Step %d | Loss: %f | Gaussians: %lu\n", i, loss, scene.count);
        }

        // Densification Logic
        if (i > config.warmup_steps && i % config.densify_interval == 0) {
            if (i < config.total_iterations - 1000) {
                if (scene.count * 2 > max_capacity) {
                    printf("WARNING: Gaussian count limit. Prune only.\n");
                    trainer.densify_and_prune(
                        scene_extent, 
                        d_rand_states.get(),
                        false
                    );
                } else {
                    trainer.densify_and_prune(
                        scene_extent, 
                        d_rand_states.get(),
                        false
                    );
                }
            }            
        }

        // Reset opacity
        if (i % config.opacity_reset_interval == 0) {
            printf("Resetting opacities\n");
            trainer.reset_opacity();
        }

        // Save debug image occasionally
        if (i % config.debug_image_interval == 0 || i == config.total_iterations) {
            trainer.get_current_render(h_render);
            
            // Construct filename: output/<experiment>/training_renders/train_step_X_imgname.jpg
            fs::path filename = renders_dir / ("train_step_" + std::to_string(i) + "_" + item.view->image_name + ".jpg");
            save_image_jpg(filename.string().c_str(), h_render, max_w, max_h, 90);
        }

        // Save checkpoints
        if (i % config.checkpoint_interval == 0) {
             
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
