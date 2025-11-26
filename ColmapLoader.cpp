#include "ColmapLoader.h"
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp> 
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

void ColmapLoader::loadCameras() {
  std::string file_path = sparse_path_ + "/cameras.bin";
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open " + file_path);
  }

  uint64_t num_cameras;
  file.read(reinterpret_cast<char *>(&num_cameras), sizeof(uint64_t));

  for (uint64_t i = 0; i < num_cameras; ++i) {
    ColmapCamera cam;
    cam.read(file);
    cameras_[cam.camera_id] = cam;
  }
}

void ColmapLoader::loadImages() {
  std::string file_path = sparse_path_ + "/images.bin";
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open " + file_path);
  }

  uint64_t num_images;
  file.read(reinterpret_cast<char *>(&num_images), sizeof(uint64_t));

  for (uint64_t i = 0; i < num_images; ++i) {
    ColmapImage img;
    img.read(file);
    images_[img.image_id] = img;
  }
}

void ColmapLoader::loadPoints3D() {
  std::string file_path = sparse_path_ + "/points3D.bin";
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Error: Could not open " + file_path);
  }

  uint64_t num_points;
  file.read(reinterpret_cast<char *>(&num_points), sizeof(uint64_t));

  points_.resize(num_points);
  for (uint64_t i = 0; i < num_points; ++i) {
    points_[i].read(file);
  }

  file.close();
}

std::pair<int, int> ColmapLoader::getMaxDimensions() const {
    // int max_w = 0;
    // int max_h = 0;

    // // Iterate over the unique cameras (O(C))
    // for (const auto& [id, cam] : cameras_) {
    //     if (static_cast<int>(cam.width) > max_w) max_w = static_cast<int>(cam.width);
    //     if (static_cast<int>(cam.height) > max_h) max_h = static_cast<int>(cam.height);
    // }
    
    // return {max_w, max_h};
    int max_w = 0;
    int max_h = 0;

    for (const auto& view : training_views_) {
        if (view.width > max_w) max_w = view.width;
        if (view.height > max_h) max_h = view.height;
    }
    
    return {max_w, max_h};
}

void ColmapLoader::buildTrainingViews(float znear, float zfar) {
  if (images_.empty() || cameras_.empty()) {
    std::cerr << "Warning: Cannot build training views. Call loadAll() first."
              << std::endl;
    return;
  }
  training_views_.clear();
  training_views_.reserve(images_.size());

  // Cache to avoid re-calculating scale for the same camera ID repeatedly
  std::map<uint32_t, float> camera_scale_map;

  for (const auto &[image_id, img] : images_) {
    if (cameras_.find(img.camera_id) == cameras_.end()) {
      continue; // Skip image if camera not found
    }
    // const ColmapCamera &cam = cameras_.at(img.camera_id);
    const ColmapCamera &raw_cam = cameras_.at(img.camera_id);

    // 1. Determine the scale factor if we haven't for this camera yet
    if (camera_scale_map.find(img.camera_id) == camera_scale_map.end()) {
        std::string full_path = image_path_ + img.name;
        // Read header only (IMREAD_UNCHANGED) to get dimensions quickly
        cv::Mat header = cv::imread(full_path, cv::IMREAD_UNCHANGED);

        if (!header.empty()) {
            float scale_x = (float)header.cols / (float)raw_cam.width;
            camera_scale_map[img.camera_id] = scale_x;
        } else {
            // Fallback if image missing (shouldn't happen in valid setup)
            camera_scale_map[img.camera_id] = 1.0f;
        }
    }
    
    float scale_factor = camera_scale_map[img.camera_id];
    
    // 2. Create a "Scaled" Camera struct to perform calculations
    ColmapCamera scaled_cam = raw_cam;
    
    // Scale Dimensions
    scaled_cam.width = (uint64_t)(std::round(raw_cam.width * scale_factor));
    scaled_cam.height = (uint64_t)(std::round(raw_cam.height * scale_factor));

    // Scale Intrinsics (Focal Length, Cx, Cy) based on Model ID
    // See ColmapCamera::read or buildProjectionMatrix for index mapping
    if (scaled_cam.model_id == 0 || scaled_cam.model_id == 2) { 
        // SIMPLE_PINHOLE / SIMPLE_RADIAL: [f, cx, cy, ...]
        scaled_cam.params[0] *= scale_factor; // f
        scaled_cam.params[1] *= scale_factor; // cx
        scaled_cam.params[2] *= scale_factor; // cy
    } 
    else if (scaled_cam.model_id == 1 || scaled_cam.model_id == 3) {
        // PINHOLE / RADIAL: [fx, fy, cx, cy, ...]
        scaled_cam.params[0] *= scale_factor; // fx
        scaled_cam.params[1] *= scale_factor; // fy
        scaled_cam.params[2] *= scale_factor; // cx
        scaled_cam.params[3] *= scale_factor; // cy
    }
    
    TrainingView view;
    view.image_name = img.name;
    view.width = (int)scaled_cam.width;
    view.height = (int)scaled_cam.height;

    // Build matrices
    view.view_matrix = buildViewMatrix(img);
    view.projection_matrix = buildProjectionMatrix(scaled_cam, znear, zfar);
    view.view_proj_matrix = view.projection_matrix * view.view_matrix;

    // Get camera center (world space)
    view.camera_center = getCameraCenter(img);
    
    view.fxfy_tanfov = getFxFyTanFov(scaled_cam);

    training_views_.push_back(view);
  }
  std::cout << "Built " << training_views_.size() << " training views from "
            << images_.size() << " images." << std::endl;
}

ColmapLoader::ColmapLoader(const std::string &sparse_path, const std::string &image_path) {
  // Sanitize sparse_path_
  sparse_path_ = sparse_path;
  if (!sparse_path_.empty() && sparse_path_.back() != '/') {
    sparse_path_ += '/';
  }

  // Sanitize image_path_
  image_path_ = image_path;
  if (!image_path_.empty() && image_path_.back() != '/') {
    image_path_ += '/';
  }
}

void ColmapLoader::loadAll() {
  ColmapLoader::loadCameras();
  ColmapLoader::loadImages();
  ColmapLoader::loadPoints3D();
  std::cout << "COLMAP data loaded." << std::endl;
}

void ColmapLoader::visualize(uint32_t image_id, const std::string& output_file) {
    if (images_.find(image_id) == images_.end()) {
        std::cerr << "Error: Image ID " << image_id << " not found." << std::endl;
        return;
    }

    const ColmapImage& image_to_vis = images_.at(image_id);
    const ColmapCamera& cam = cameras_.at(image_to_vis.camera_id);

    // 1. Load Image using OpenCV
    std::string full_path = image_path_ + image_to_vis.name;
    cv::Mat image_mat = cv::imread(full_path);
    
    if (image_mat.empty()) {
        std::cerr << "Error: Could not load image: " << full_path << std::endl;
        return;
    }
    
    // 2. Build Matrices (using private helpers)
    glm::mat4 view_mat = buildViewMatrix(image_to_vis);
    glm::mat4 proj_mat = buildProjectionMatrix(cam, 0.01f, 100.0f);
    glm::mat4 vp_mat = proj_mat * view_mat;

    int width = image_mat.cols;
    int height = image_mat.rows;

    // 3. Project and Draw
    for (const auto& pt : points_) {
        glm::vec4 p_world(pt.xyz[0], pt.xyz[1], pt.xyz[2], 1.0f);
        glm::vec4 p_clip = vp_mat * p_world;

        if (p_clip.w < 0.01f) continue;

        glm::vec3 p_ndc = glm::vec3(p_clip) / p_clip.w;
        float u = (p_ndc.x + 1.0f) * 0.5f * width;
        float v = (p_ndc.y + 1.0f) * 0.5f * height;

        if (u >= 0 && u < width && v >= 0 && v < height) {
            cv::circle(image_mat, cv::Point2f(u, v), 2, cv::Scalar(0, 0, 255), -1);
        }
    }

    cv::imwrite(output_file, image_mat);
    std::cout << "Visualization saved to " << output_file << std::endl;
}

glm::mat4 ColmapLoader::buildViewMatrix(const ColmapImage& image) {
        // 1. Get R_w2c (rotation) and t_w2c (translation)
        const glm::dmat3& R_w2c = glm::mat3_cast(image.qvec);
        const glm::dvec3& t_w2c = image.tvec;

        // 2. Construct 4x4 World-to-Camera matrix [R | t]
        //    We initialize the 4x4 matrix from the 3x3 rotation,
        //    which correctly sets the upper-left 3x3 block.
        glm::dmat4 view_d = glm::dmat4(R_w2c);
    
        //    Then, we set the 4th column (index 3) to the translation.
        view_d[3] = glm::dvec4(t_w2c, 1.0);

        // 3. Apply coordinate system flip.
        // (COLMAP: +Y Down, +Z In) -> (OpenGL: +Y Up, +Z Out)
        glm::mat4 C_flip = glm::mat4(1.0f);
        C_flip[1][1] = -1.0f; // Flip Y
        C_flip[2][2] = -1.0f; // Flip Z

        // 4. Cast to float and return
        //    Final transform is P_cam = C_flip * W_colmap * P_world
        return C_flip * glm::mat4(view_d);
}

glm::mat4 ColmapLoader::buildProjectionMatrix(const ColmapCamera& camera, float znear, float zfar) {
        float fx, fy, cx, cy;
        float W = (float)camera.width;
        float H = (float)camera.height;

        switch (camera.model_id) {
            case 0: // SIMPLE_PINHOLE (f, cx, cy)
                fx = (float)camera.params[0];
                fy = (float)camera.params[0]; // f (focal length) is shared
                cx = (float)camera.params[1];
                cy = (float)camera.params[2];
                break;
            case 1: // PINHOLE (fx, fy, cx, cy)
                fx = (float)camera.params[0];
                fy = (float)camera.params[1];
                cx = (float)camera.params[2];
                cy = (float)camera.params[3];
                break;
            case 2: // SIMPLE_RADIAL (f, cx, cy, k)
                fx = (float)camera.params[0];
                fy = (float)camera.params[0]; // f (focal length) is shared
                cx = (float)camera.params[1];
                cy = (float)camera.params[2];
                // Distortion param camera.params[3] (k) is ignored
                break;
            case 3: // RADIAL (f, cx, cy, k1, k2)
                fx = (float)camera.params[0];
                fy = (float)camera.params[0]; // f (focal length) is shared
                cx = (float)camera.params[1];
                cy = (float)camera.params[2];
                // Distortion params camera.params[3] (k1) and camera.params[4] (k2) are ignored
                break;
            default:
                // This case should be caught by the reader, but added for safety.
                throw std::runtime_error("Error: Unsupported camera model for projection: " + std::to_string(camera.model_id));
        }

        // Using the exact 3DGS projection matrix (Y-down in NDC)
        glm::mat4 proj_3dgs = glm::mat4(0.0f);

        proj_3dgs[0][0] = (2.0f * fx) / W;
        proj_3dgs[1][1] = -(2.0f * fy) / H; // Flip Y
        proj_3dgs[2][0] = (2.0f * cx - W) / W;
        proj_3dgs[2][1] = (H - 2.0f * cy) / H; // Flip Y
        proj_3dgs[2][2] = (zfar + znear) / (zfar - znear);
        proj_3dgs[3][2] = -(2.0f * zfar * znear) / (zfar - znear);
        proj_3dgs[2][3] = -1.0f;
        
        return proj_3dgs;
}

glm::vec3 ColmapLoader::getCameraCenter(const ColmapImage& image) {
  glm::dmat3 R_w2c = glm::mat3_cast(image.qvec);
  glm::dvec3 t_w2c = image.tvec;

  // Invert the transform: C_world = -R_w2c^T * t_w2c
  glm::dvec3 C_world = -glm::transpose(R_w2c) * t_w2c;

  return glm::vec3((float)C_world.x, (float)C_world.y, (float)C_world.z);
}

glm::vec4 ColmapLoader::getFxFyTanFov(const ColmapCamera& camera) {
  float fx, fy;

  // Extract focal lengths based on the camera model
  // This logic must be identical to your buildProjectionMatrix
  switch (camera.model_id) {
  case 0: // SIMPLE_PINHOLE (f, cx, cy)
    fx = (float)camera.params[0];
    fy = (float)camera.params[0]; // f (focal length) is shared
    break;
  case 1: // PINHOLE (fx, fy, cx, cy)
    fx = (float)camera.params[0];
    fy = (float)camera.params[1];
    break;
  case 2: // SIMPLE_RADIAL (f, cx, cy, k)
    fx = (float)camera.params[0];
    fy = (float)camera.params[0]; // f (focal length) is shared
    break;
  case 3: // RADIAL (f, cx, cy, k1, k2)
    fx = (float)camera.params[0];
    fy = (float)camera.params[0]; // f (focal length) is shared
    break;
  default:
    throw std::runtime_error("Error: Unsupported camera model for FOV: " +
                             std::to_string(camera.model_id));
  }

  // Calculate tan(FoV/2) using the pinhole camera model
  float tan_fovy = (0.5f * (float)camera.height) / fy;
  float tan_fovx = (0.5f * (float)camera.width) / fx;

  return glm::vec4(fx, fy, tan_fovx, tan_fovy);
}
