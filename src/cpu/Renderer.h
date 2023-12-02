//
// Created by kient on 6/16/2023.
//

#ifndef MASS_RENDERER_H
#define MASS_RENDERER_H

#include "imgui.h" // version 1.78 and 1.60
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>
#include <stdlib.h>
#include "../gpu/GPUEntity.cuh"
#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>

class Renderer {
public:
    Renderer();
    ~Renderer();

    static Renderer& getInstance() // Singleton is accessed via getInstance()
    {
        static Renderer instance; // lazy singleton, instantiated on first use
        return instance;
    }

    void init(GPUEntity* gpu_entity, int width, int height);
    void render();
    void detach();

private:
    void renderGUI();
    void framebufferSizeCallbackImpl(GLFWwindow* window, int width, int height);
    void mouseCursorCallbackImpl(GLFWwindow* window, double x_pos_in, double y_pos_in);
    void mouseButtonCallbackImpl(GLFWwindow* window, int button, int action, int mods);
    void mouseScrollCallbackImpl(GLFWwindow* window, double x_offset, double y_offset);
    glm::dvec3 unProject( const glm::dvec3& win );
    glm::dvec2 unProjectPlane( const glm::dvec2& win );

public:
    GPUEntity *gpu_entity_;
    int window_width;
    int window_height;
    std::thread render_thread;

public:
    double width_scaled;
    double height_scaled;
    double aspect_ratio;
    bool is_drag_mode;
    double zoomFactor;
    glm::vec2 drag_speed;
    glm::vec2 mouse_position;
    glm::vec2 mouse_drag_center_start;
    glm::vec2 mouse_drag_center_current;
    glm::vec2 camera_view_center;

    double mouse_input_x;
    double mouse_input_y;
    double camera_center_x;
    double camera_center_y;
    GLFWwindow* renderer_window;
};


#endif //MASS_RENDERER_H
