#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <fstream> 

/**
 * @brief Represents a single camera from COLMAP's cameras.bin
 */
struct ColmapCamera {
    uint32_t camera_id;
    int32_t model_id; // 0=SIMPLE_PINHOLE, 1=PINHOLE, 2=SIMPLE_RADIAL, 3=RADIAL
    uint64_t width;
    uint64_t height;
    std::vector<double> params; // Parameter vector, size varies by model

    /**
     * @brief Reads camera data, handling variable parameter counts.
     */
    void read(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(&camera_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&model_id), sizeof(int32_t));
        file.read(reinterpret_cast<char*>(&width), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&height), sizeof(uint64_t));

        size_t num_params = 0;
        switch (model_id) {
            case 0: num_params = 3; break; // SIMPLE_PINHOLE (f, cx, cy)
            case 1: num_params = 4; break; // PINHOLE (fx, fy, cx, cy)
            case 2: num_params = 4; break; // SIMPLE_RADIAL (f, cx, cy, k)
            case 3: num_params = 5; break; // RADIAL (f, cx, cy, k1, k2)
            // Add cases for other models like OPENCV if needed
            default:
                throw std::runtime_error("Error: Unsupported COLMAP camera model_id: " + std::to_string(model_id));
        }

        params.resize(num_params);
        file.read(reinterpret_cast<char*>(params.data()), sizeof(double) * num_params);
    }
};

/**
 * @brief Represents a single image (pose) from COLMAP's images.bin
 *
 * NOTE: COLMAP's images.bin stores the World-to-Camera (w2c) transformation.
 * qvec (w,x,y,z) is the rotation R_w2c.
 * tvec (x,y,z) is the translation t_w2c.
 */
struct ColmapImage {
    uint32_t image_id;
    glm::dquat qvec; // w, x, y, z (R_w2c)
    glm::dvec3 tvec; // x, y, z (t_w2c)
    uint32_t camera_id;
    std::string name;

    void read(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(&image_id), sizeof(uint32_t));
        
        double q_arr[4];
        file.read(reinterpret_cast<char*>(&q_arr), sizeof(double) * 4);
        qvec = glm::dquat(q_arr[0], q_arr[1], q_arr[2], q_arr[3]); // w, x, y, z

        double t_arr[3];
        file.read(reinterpret_cast<char*>(&t_arr), sizeof(double) * 3);
        tvec = glm::dvec3(t_arr[0], t_arr[1], t_arr[2]);

        file.read(reinterpret_cast<char*>(&camera_id), sizeof(uint32_t));

        char c;
        name = "";
        while (file.get(c) && c != '\0') {
            name += c;
        }

        // Skip the 2D points track
        uint64_t num_points2D;
        file.read(reinterpret_cast<char*>(&num_points2D), sizeof(uint64_t));
        file.seekg( num_points2D * (2 * sizeof(double) + sizeof(uint64_t)) , std::ios::cur);
    }
};

struct TrainingView {
    std::string image_name;
    int width;
    int height;
    
    // World-to-Camera View Matrix (W)
    glm::mat4 view_matrix;
    
    // Camera Projection Matrix (K)
    glm::mat4 projection_matrix;
    
    // View-Projection Matrix (P = K * W)
    glm::mat4 view_proj_matrix;
    
    // Camera center in world space (needed for SH evaluation)
    glm::vec3 camera_center;

    // Camera field of view
    glm::vec4 fxfy_tanfov;
};

// Represents a single observation in the track
struct TrackEntry {
    uint32_t image_id;
    uint32_t point2D_idx;
};

// Represents the data stored in point3D.bin for *one* point
struct ColmapPoint3D {
    uint64_t point_id;
    double xyz[3];
    uint8_t rgb[3];
    double error;
    std::vector<TrackEntry> track;

    void read(std::ifstream& file) {
        file.read(reinterpret_cast<char*>(&point_id), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&xyz), sizeof(double) * 3);
        file.read(reinterpret_cast<char*>(&rgb), sizeof(uint8_t) * 3);
        file.read(reinterpret_cast<char*>(&error), sizeof(double));

        uint64_t track_len;
        file.read(reinterpret_cast<char*>(&track_len), sizeof(uint64_t));
        track.resize(track_len);
        file.read(reinterpret_cast<char*>(track.data()), sizeof(TrackEntry) * track_len);
    }
};

class ColmapLoader {
public:
    ColmapLoader(const std::string& sparse_path, const std::string& image_path = "");

    void loadAll();
    void buildTrainingViews(float znear, float zfar);
    
    // Returns max dimensions found in cameras
    std::pair<int, int> getMaxDimensions() const;

    // Getters
    const std::map<uint32_t, ColmapCamera>& getCameras() const { return cameras_; }
    const std::map<uint32_t, ColmapImage>& getImages() const { return images_; }
    const std::vector<ColmapPoint3D>& getPoints() const { return points_; }
    const std::vector<TrainingView>& getTrainingViews() const { return training_views_; }
    
    const TrainingView& getView(size_t index) const {
        return training_views_.at(index);
    }

    // Visualization (Implementation moved to .cpp)
    void visualize(uint32_t image_id, const std::string& output_file);

    // Explicit getter for image path
    const std::string& getImagePath() const { return image_path_; }

private:
    std::string sparse_path_;
    std::string image_path_;

    std::map<uint32_t, ColmapCamera> cameras_;
    std::map<uint32_t, ColmapImage> images_;
    std::vector<ColmapPoint3D> points_;
    std::vector<TrainingView> training_views_;

    void loadCameras();
    void loadImages();
    void loadPoints3D();

    static glm::mat4 buildViewMatrix(const ColmapImage& image);
    static glm::mat4 buildProjectionMatrix(const ColmapCamera& camera, float znear, float zfar);
    static glm::vec3 getCameraCenter(const ColmapImage& image);
    static glm::vec4 getFxFyTanFov(const ColmapCamera& camera);
};
