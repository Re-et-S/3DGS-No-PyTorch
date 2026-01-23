#include "ColmapLoader.h"
#include <iostream>
#include <stdexcept>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>
#include "stb_image.h"
#include "stb_image_write.h"

// --- Helper: scanline circle rasterizer ---
void draw_circle(unsigned char* img, int width, int height, int cx, int cy, int radius) {
    int r2 = radius * radius;
    
    int y_min = std::max(0, cy - radius);
    int y_max = std::min(height - 1, cy + radius);

    for (int y = y_min; y <= y_max; ++y) {
        int dy = y - cy;
        
        int half_width = static_cast<int>(std::sqrt(r2 - dy * dy));
        
        int x_min = std::max(0, cx - half_width);
        int x_max = std::min(width - 1, cx + half_width);

        unsigned char* ptr = img + (y * width + x_min) * 3;

        for (int x = x_min; x <= x_max; ++x) {
            ptr[0] = 255; // R
            ptr[1] = 0;   // G
            ptr[2] = 0;   // B
            ptr += 3;     // Move to next pixel
        }
    }
}


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

  std::map<uint32_t, float> camera_scale_map;

  for (const auto &[image_id, img] : images_) {
    if (cameras_.find(img.camera_id) == cameras_.end()) {
      continue; 
    }
    
    const ColmapCamera &raw_cam = cameras_.at(img.camera_id);

    if (camera_scale_map.find(img.camera_id) == camera_scale_map.end()) {
        std::string full_path = image_path_ + img.name;
        
        int w, h, comp;
        int ok = stbi_info(full_path.c_str(), &w, &h, &comp);

        if (ok) {
            float scale_x = (float)w / (float)raw_cam.width;
            camera_scale_map[img.camera_id] = scale_x;
        } else {
            std::cerr << "Warning: Could not read image header for " << full_path << ". Assuming scale 1.0." << std::endl;
            camera_scale_map[img.camera_id] = 1.0f;
        }
    }
    
    float scale_factor = camera_scale_map[img.camera_id];
    
    ColmapCamera scaled_cam = raw_cam;
    
    scaled_cam.width = (uint64_t)(std::round(raw_cam.width * scale_factor));
    scaled_cam.height = (uint64_t)(std::round(raw_cam.height * scale_factor));

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

    view.view_matrix = buildViewMatrix(img);
    view.projection_matrix = buildProjectionMatrix(scaled_cam, znear, zfar);
    view.view_proj_matrix = view.projection_matrix * view.view_matrix;

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

    std::string full_path = image_path_ + image_to_vis.name;
    int width, height, channels;
    
    unsigned char* img_data = stbi_load(full_path.c_str(), &width, &height, &channels, 3);
    
    if (!img_data) {
        std::cerr << "Error: Could not load image: " << full_path << " (" << stbi_failure_reason() << ")" << std::endl;
        return;
    }
  
    glm::mat4 view_mat = buildViewMatrix(image_to_vis);
    glm::mat4 proj_mat = buildProjectionMatrix(cam, 0.01f, 100.0f);
    glm::mat4 vp_mat = proj_mat * view_mat;

    for (const auto& pt : points_) {
        glm::vec4 p_world(pt.xyz[0], pt.xyz[1], pt.xyz[2], 1.0f);
        glm::vec4 p_clip = vp_mat * p_world;

        if (p_clip.w < 0.01f) continue;

        glm::vec3 p_ndc = glm::vec3(p_clip) / p_clip.w;
        
        float u = (p_ndc.x + 1.0f) * 0.5f * width;
        float v = (p_ndc.y + 1.0f) * 0.5f * height;

        if (u >= -5 && u < width+5 && v >= -5 && v < height+5) {
             draw_circle(img_data, width, height, (int)u, (int)v, 2);
        }
    }

    int success = stbi_write_jpg(output_file.c_str(), width, height, 3, img_data, 95);
    
    if (success) {
        std::cout << "Visualization saved to " << output_file << std::endl;
    } else {
        std::cerr << "Failed to write output image to " << output_file << std::endl;
    }

    stbi_image_free(img_data);
}

glm::mat4 ColmapLoader::buildViewMatrix(const ColmapImage& image) {

        const glm::dmat3& R_w2c = glm::mat3_cast(image.qvec);
        const glm::dvec3& t_w2c = image.tvec;

        glm::dmat4 view_d = glm::dmat4(R_w2c);
    
        view_d[3] = glm::dvec4(t_w2c, 1.0);

        glm::mat4 C_flip = glm::mat4(1.0f);
        C_flip[1][1] = -1.0f; // Flip Y
        C_flip[2][2] = -1.0f; // Flip Z

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
                break;
            case 3: // RADIAL (f, cx, cy, k1, k2)
                fx = (float)camera.params[0];
                fy = (float)camera.params[0]; // f (focal length) is shared
                cx = (float)camera.params[1];
                cy = (float)camera.params[2];
                break;
            default:
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
