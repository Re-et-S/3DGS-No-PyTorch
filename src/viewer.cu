#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "plyIO.h"
#include "camera.h"
#include "scene.cuh"
#include "renderer.cuh"

GLFWwindow* window = nullptr;
std::unique_ptr<GaussianScene> g_scene = nullptr;

char g_plyPath[256] = "iteration_9000.ply";
std::string g_statusMessage = "Ready to load.";
bool g_showDemoWindow = true;

Camera g_camera(glm::vec3(0.0f,0.0f,0.0f));

float deltaTime = 0.0f;
float lastFrame = 0.0f;

ViewerRenderer g_renderer{};
int active_sh_degree = 3;

GLuint g_renderTex = 0;
cudaGraphicsResource* g_cuda_gl_image = nullptr;
int g_texWidth = 1280;
int g_texHeight = 720;

void ResizeGLTexture(int newW, int newH) {
    if (g_cuda_gl_image) {
        cudaGraphicsUnregisterResource(g_cuda_gl_image);
    }
    
    glBindTexture(GL_TEXTURE_2D, g_renderTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, newW, newH, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsGLRegisterImage(&g_cuda_gl_image, g_renderTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    
    g_texWidth = newW;
    g_texHeight = newH;
}

glm::vec3 ComputeSceneCenter(GaussianScene& scene) {
    if (scene.count == 0) return glm::vec3(0.0f);

    // Copy points from GPU to CPU to calculate mean
    // (Ideally, do this in plyIO during load, but this works here)
    std::vector<float> h_points(scene.count * 3);
    scene.d_points.from_device(h_points);

    double sumX = 0, sumY = 0, sumZ = 0;
    for (size_t i = 0; i < scene.count; ++i) {
        sumX += h_points[i * 3 + 0];
        sumY += h_points[i * 3 + 1];
        sumZ += h_points[i * 3 + 2];
    }

    return glm::vec3(
        (float)(sumX / scene.count),
        (float)(sumY / scene.count),
        (float)(sumZ / scene.count)
    );
}

void HandleLoadPly() {
    try {
        g_statusMessage = "Loading...";
        std::cout << "Trying to load: " << g_plyPath << std::endl;

        GaussianScene newScene = load_ply(g_plyPath);
        glm::vec3 center = ComputeSceneCenter(newScene);
        g_camera.Position = center + glm::vec3(0.0f, 0.0f, -20.0f);
        g_camera.Front = glm::vec3(0.0f, 0.0f, 1.0f);
        
        g_scene = std::make_unique<GaussianScene>(std::move(newScene));
        g_statusMessage = "Success! Centered Camera at (" + 
                          std::to_string(center.x) + ", " + 
                          std::to_string(center.y) + ", " + 
                          std::to_string(center.z) + ")";
        
        // g_statusMessage = "Success! Loaded " + std::to_string(g_scene->count) + " splats.";
    } catch (const std::exception& e) {
        g_statusMessage = "Error: " + std::string(e.what());
        std::cerr << e.what() << std::endl;
    }
}

void RenderUI() {

    {
        ImGui::Begin("Gaussian Splatting Viewer");

        ImGui::Text("Scene Controls");
        ImGui::Separator();

        // File Input
        ImGui::InputText("PLY File", g_plyPath, IM_ARRAYSIZE(g_plyPath));
        
        // Load Button
        if (ImGui::Button("Load .ply")) {
            HandleLoadPly();
        }
        
        // Status Display
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s", g_statusMessage.c_str());

        ImGui::Separator();
        
        // Scene Info (Only if loaded)
        if (g_scene) {
            ImGui::Text("Scene Statistics:");
            ImGui::BulletText("Count: %d", g_scene->count);
            ImGui::BulletText("SH Degree: %d", g_scene->sh_degree);
            
            // Placeholder for future camera controls
            ImGui::Separator();
            ImGui::Text("Camera");

            ImGui::SliderFloat("FOV", &g_camera.Fov, 10.0f, 120.0f);
            ImGui::DragFloat("Move Speed", &g_camera.MovementSpeed, 0.1f, 0.1f, 100.0f);
        }
        ImVec2 avail = ImGui::GetContentRegionAvail();
        if (avail.x < 1) avail.x = 1;
        if (avail.y < 1) avail.y = 1;
        ImGui::Image((void*)(intptr_t)g_renderTex, avail);
        
        ImGui::End();
    }

    ImGui::Render();
}

int main(int argc, char** argv) {
    glfwSetErrorCallback([](int error, const char* description) {
        fprintf(stderr, "Glfw Error %d: %s\n", error, description);
    });

    if (!glfwInit())
        return 1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    int winWidth = 1280; 
    int winHeight = 720;
    window = glfwCreateWindow(winWidth, winHeight, "CUDA Gaussian Viewer", NULL, NULL);
    
    if (window == NULL)
        return 1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    glGenTextures(1, &g_renderTex);
    glBindTexture(GL_TEXTURE_2D, g_renderTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, winWidth, winHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    cudaGraphicsGLRegisterImage(&g_cuda_gl_image, g_renderTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

    while (!glfwWindowShouldClose(window)) {

        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        g_camera.ProcessInput(window, deltaTime);

        int currentW, currentH;
        glfwGetFramebufferSize(window, &currentW, &currentH);

        // RESIZE CHECK
        if (currentW != g_texWidth || currentH != g_texHeight) {
            if (currentW > 0 && currentH > 0) {
                ResizeGLTexture(currentW, currentH);
                g_renderer.resize(currentW, currentH);
            }
        }
        
        glViewport(0, 0, currentW, currentH);
        glClearColor(0.1f, 0.1f, 0.1f, 1.00f); 
        glClear(GL_COLOR_BUFFER_BIT);
        
        glm::mat4 view = g_camera.GetViewMatrix();
        glm::mat4 proj = g_camera.GetProjectionMatrix((float)currentW, (float)currentH);

        g_renderer.resize(currentW, currentH);

        if (g_scene) {

            float aspectRatio = (float)currentW / (float)currentH;
            float fovY = glm::radians(g_camera.Fov);
            float fovX = 2.0f * atan(tan(fovY / 2.0f) * aspectRatio);
            
            g_renderer.render(
                *g_scene,
                view,
                proj,
                g_camera.Position,
                fovX,
                fovY,
                active_sh_degree,
                g_cuda_gl_image
            );
        }

        RenderUI();
        
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
