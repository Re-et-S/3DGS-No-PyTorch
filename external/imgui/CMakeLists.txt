cmake_minimum_required(VERSION 3.10)
project(imgui)

set(IMGUI_SOURCES
    imgui.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
    backends/imgui_impl_glfw.cpp
    backends/imgui_impl_opengl3.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
)

# 4. Link dependencies (ImGui backends need GLFW and OpenGL)
# Assuming you have already found these packages in your root CMake
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

target_link_libraries(imgui PRIVATE glfw OpenGL::GL)
