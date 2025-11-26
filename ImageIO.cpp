#include "ImageIO.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <iostream>

/**
 * @brief Loads the ground-truth image and converts it to planar float format.
 *
 * OpenCV loads images as interleaved (BGR, BGR, ...) uint8_t.
 * The 3DGS pipeline requires planar (RRR...GGG...BBB...) float [0, 1].
 * This function handles that conversion.
 */
std::vector<float> load_image_planar(const std::string& path, int& W, int& H)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Error: Could not load ground-truth image: " + path);
    }
    if (img.cols != W || img.rows != H) {
        std::cout << "Notice: Image resolution mismatch. Adapting to file dimensions." << std::endl;
        W = img.cols;
        H = img.rows;
    }

    std::vector<float> h_gt_image(W * H * 3);
    const int num_pixels = W * H;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            cv::Vec3b bgr = img.at<cv::Vec3b>(y, x); // BGR
            int px_idx = y * W + x;

            // Normalize and write to planar buffers
            h_gt_image[0 * num_pixels + px_idx] = (float)bgr[2] / 255.0f; // R
            h_gt_image[1 * num_pixels + px_idx] = (float)bgr[1] / 255.0f; // G
            h_gt_image[2 * num_pixels + px_idx] = (float)bgr[0] / 255.0f; // B
        }
    }
    return h_gt_image;
}

void save_image_ppm(const char* filename, const std::vector<float>& buffer, int width, int height) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s\n", filename);
        return;
    }
    fprintf(fp, "P3\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Note the planar format (RRR...GGG...BBB...)
            int r_idx = 0 * width * height + y * width + x;
            int g_idx = 1 * width * height + y * width + x;
            int b_idx = 2 * width * height + y * width + x;
            int r = (int)(fmax(0.0f, fmin(1.0f, buffer[r_idx])) * 255.0f);
            int g = (int)(fmax(0.0f, fmin(1.0f, buffer[g_idx])) * 255.0f);
            int b = (int)(fmax(0.0f, fmin(1.0f, buffer[b_idx])) * 255.0f);
            fprintf(fp, "%d %d %d ", r, g, b);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Saved image to %s\n", filename);
}

