#include "ImageIO.h"
#include <stdexcept>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * @brief Loads the ground-truth image and converts it to planar float format.
 *
 * stb_image loads as R, G, B (interleaved).
 * convert to Planar RRR...GGG...BBB...
 */
std::vector<float> load_image_planar(const std::string& path, int& W, int& H)
{
    int img_w, img_h, img_channels;
    
    unsigned char* data = stbi_load(path.c_str(), &img_w, &img_h, &img_channels, 3);

    if (!data) {
        throw std::runtime_error("Error: Could not load image: " + path + " \nReason: " + stbi_failure_reason());
    }

    if (W > 0 && H > 0) {
        if (img_w != W || img_h != H) {
            stbi_image_free(data);
            throw std::runtime_error("Error: Image dimensions mismatch. Expected " + 
                                     std::to_string(W) + "x" + std::to_string(H) + 
                                     " but got " + std::to_string(img_w) + "x" + std::to_string(img_h));
        }
    } else {
        W = img_w;
        H = img_h;
    }

    std::vector<float> h_gt_image(W * H * 3);
    const int num_pixels = W * H;

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int px_idx = y * W + x;
            int src_idx = px_idx * 3; 

            unsigned char r = data[src_idx + 0];
            unsigned char g = data[src_idx + 1];
            unsigned char b = data[src_idx + 2];

            h_gt_image[0 * num_pixels + px_idx] = (float)r / 255.0f; 
            h_gt_image[1 * num_pixels + px_idx] = (float)g / 255.0f; 
            h_gt_image[2 * num_pixels + px_idx] = (float)b / 255.0f; 
        }
    }

    stbi_image_free(data);
    return h_gt_image;
}

/**
 * @brief Saves a planar float buffer as a JPEG.
 * * 1. Converts Planar Float (RRR...) -> Interleaved Byte (RGB...)
 * 2. Writes using stb_image_write
 * * @param quality JPEG quality (1-100), default usually around 90
 */
void save_image_jpg(const char* filename, const std::vector<float>& buffer, int width, int height, int quality) {
    if (buffer.size() != width * height * 3) {
        std::cerr << "Error: Buffer size mismatch in save_image_jpg" << std::endl;
        return;
    }

    std::vector<unsigned char> interleaved(width * height * 3);
    int num_pixels = width * height;

    for (int i = 0; i < num_pixels; ++i) {

        float r_f = buffer[0 * num_pixels + i];
        float g_f = buffer[1 * num_pixels + i];
        float b_f = buffer[2 * num_pixels + i];

        unsigned char r = (unsigned char)(std::max(0.0f, std::min(1.0f, r_f)) * 255.0f);
        unsigned char g = (unsigned char)(std::max(0.0f, std::min(1.0f, g_f)) * 255.0f);
        unsigned char b = (unsigned char)(std::max(0.0f, std::min(1.0f, b_f)) * 255.0f);

        interleaved[i * 3 + 0] = r;
        interleaved[i * 3 + 1] = g;
        interleaved[i * 3 + 2] = b;
    }

    int success = stbi_write_jpg(filename, width, height, 3, interleaved.data(), quality);
    
    if (success) {
        std::cout << "Saved image to " << filename << std::endl;
    } else {
        std::cerr << "Failed to save image to " << filename << std::endl;
    }
}

// void save_image_ppm(const char* filename, const std::vector<float>& buffer, int width, int height) {
//     FILE* fp = fopen(filename, "w");
//     if (!fp) {
//         fprintf(stderr, "Error: Could not open %s\n", filename);
//         return;
//     }
//     fprintf(fp, "P3\n%d %d\n255\n", width, height);
//     for (int y = 0; y < height; y++) {
//         for (int x = 0; x < width; x++) {
//             // Note the planar format (RRR...GGG...BBB...)
//             int r_idx = 0 * width * height + y * width + x;
//             int g_idx = 1 * width * height + y * width + x;
//             int b_idx = 2 * width * height + y * width + x;
//             int r = (int)(fmax(0.0f, fmin(1.0f, buffer[r_idx])) * 255.0f);
//             int g = (int)(fmax(0.0f, fmin(1.0f, buffer[g_idx])) * 255.0f);
//             int b = (int)(fmax(0.0f, fmin(1.0f, buffer[b_idx])) * 255.0f);
//             fprintf(fp, "%d %d %d ", r, g, b);
//         }
//         fprintf(fp, "\n");
//     }
//     fclose(fp);
//     printf("Saved image to %s\n", filename);
// }

