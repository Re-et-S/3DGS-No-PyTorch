#pragma once
#include <vector>
#include <string>

std::vector<float> load_image_planar(const std::string& path, int& out_w, int& out_h);
// void save_image_ppm(const char* filename, const std::vector<float>& buffer, int width, int height);
void save_image_jpg(const char* filename, const std::vector<float>& buffer, int width, int height, int quality);

