#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <vector>

const float YAW         = -90.0f;
const float PITCH       =  0.0f;
const float SPEED       =  5.0f;
const float SENSITIVITY =  0.1f;
const float FOV         =  45.0f;

class Camera {
public:

    glm::vec3 Position;
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;

    float Yaw;
    float Pitch;

    float MovementSpeed;
    float MouseSensitivity;
    float Fov;

    bool isDragging = false;
    double lastX = 0.0;
    double lastY = 0.0;

    Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) 
        : Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Fov(FOV) {
        Position = position;
        WorldUp = up;
        Yaw = yaw;
        Pitch = pitch;
        updateCameraVectors();
    }

    glm::mat4 GetViewMatrix() const {
        return glm::lookAt(Position, Position + Front, Up);
    }

    glm::mat4 GetProjectionMatrix(float width, float height) const {
        return glm::perspective(glm::radians(Fov), width / height, 0.1f, 100.0f);
    }

    void ProcessInput(GLFWwindow* window, float deltaTime) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
            if (!isDragging) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                glfwGetCursorPos(window, &lastX, &lastY);
                isDragging = true;
            }

            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);

            float xoffset = static_cast<float>(xpos - lastX);
            float yoffset = static_cast<float>(lastY - ypos);

            lastX = xpos;
            lastY = ypos;

            ProcessMouseMovement(xoffset, yoffset);

            float velocity = MovementSpeed * deltaTime;
            
            if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
                velocity *= 3.0f;

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                Position += Front * velocity;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                Position -= Front * velocity;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                Position -= Right * velocity;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                Position += Right * velocity;
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                Position -= Up * velocity; 
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                Position += Up * velocity; 

        } else if (isDragging) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            isDragging = false;
        }
    }

private:
    void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true) {
        xoffset *= MouseSensitivity;
        yoffset *= MouseSensitivity;

        Yaw   += xoffset;
        Pitch += yoffset;

        if (constrainPitch) {
            if (Pitch > 89.0f)
                Pitch = 89.0f;
            if (Pitch < -89.0f)
                Pitch = -89.0f;
        }

        updateCameraVectors();
    }

    void updateCameraVectors() {
        
        glm::vec3 front;
        front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        front.y = sin(glm::radians(Pitch));
        front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
        Front = glm::normalize(front);
        
        Right = glm::normalize(glm::cross(Front, WorldUp));  
        Up    = glm::normalize(glm::cross(Right, Front));
    }
};
