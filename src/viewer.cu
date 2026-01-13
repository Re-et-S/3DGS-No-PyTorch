#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h> // Will include GL/gl.h

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "plyIO.h"
#include "scene.cuh"

GLFWwindow* window = nullptr;
std::unique_ptr<GaussianScene> g_scene = nullptr;

char g_plyPath[256] = "model.ply";
std::string g_statusMessage = "Ready to load.";
bool g_showDemoWindow = true;

void HandleLoadPly() {
    try {
        g_statusMessage = "Loading...";
        std::cout << "Trying to load: " << g_plyPath << std::endl;

        GaussianScene newScene = load_ply(g_plyPath);
        g_scene = std::make_unique<GaussianScene>(std::move(newScene));

        g_statusMessage = "Success! Loaded " + std::to_string(g_scene->count) + " splats.";
    } catch (const std::exception& e) {
        g_statusMessage = "Error: " + std::string(e.what());
        std::cerr << e.what() << std::endl;
    }
}

void RenderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
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
            ImGui::Text("Camera (TODO)");
            static float fov = 45.0f;
            ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f);
        }

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

    window = glfwCreateWindow(1280, 720, "CUDA Gaussian Viewer", NULL, NULL);
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

    while (!glfwWindowShouldClose(window)) {

        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f); // Background Color
        glClear(GL_COLOR_BUFFER_BIT);

        RenderUI();
        
        // TODO: Call CUDA Renderer here (passing g_scene and camera)
        // if (g_scene) { render_scene(*g_scene, ...); }

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
