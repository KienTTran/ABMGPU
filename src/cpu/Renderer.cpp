//
// Created by kient on 6/16/2023.
//

#include <random>
#include "Renderer.h"
#include "../utils/Thread.h"
#include "../cpu/Model.h"
#include "../cpu/Population/Population.cuh"
#include "../cpu/Population/Properties/PersonIndexGPU.h"

Renderer::Renderer(Model* model){
    model_ = model;
    window_width = 0;
    window_height = 0;
    width_scaled = 0.0;
    height_scaled = 0.0;
    aspect_ratio = 0.0;
    is_drag_mode = false;
    zoomFactor = 2.0;
    drag_speed = glm::vec2(1000.0f,1000.0f);
    mouse_position = glm::vec2(0.0f,0.0f);
    mouse_drag_center_start = glm::vec2(0.0f,0.0f);
    mouse_drag_center_current = glm::vec2(0.0f,0.0f);
    camera_view_center = glm::vec2(0.0f,0.0f);
    mouse_input_x = 0.0;
    mouse_input_y = 0.0;
    camera_center_x = 0.0;
    camera_center_y = 0.0;
}

Renderer::~Renderer() {
    glfwTerminate();
}

void Renderer::init(RenderEntity* gpu_entity) {
    window_width = Model::CONFIG->render_config().window_width;
    window_height = Model::CONFIG->render_config().window_height;
    width_scaled = Model::CONFIG->render_config().window_width;
    height_scaled = Model::CONFIG->render_config().window_height;
    aspect_ratio = (double)window_width / (double)window_height;
    mouse_position = glm::vec2(window_width/2.0f,window_height/2.0f);
    mouse_drag_center_start = glm::vec2(window_width/2.0f,window_height/2.0f);
    mouse_drag_center_current = glm::vec2(window_width/2.0f,window_height/2.0f);
    camera_view_center = glm::vec2(0.0f,0.0f);
    camera_center_x = (double)window_width/2.0;
    camera_center_y = (double)window_height/2.0;

    zoomFactor = Model::CONFIG->render_config().zoom_step;

    render_entity_ = gpu_entity;
}

void Renderer::start() {
    printf("[Renderer] Started\n");

    if (!glfwInit()) exit(EXIT_FAILURE);
    if (atexit(glfwTerminate)) { glfwTerminate(); exit(EXIT_FAILURE); }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // Core profile means no backward compatibility
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Allow forward compatibility
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    renderer_window = glfwCreateWindow(window_width, window_height, "MASS", NULL, NULL);

    glfwMakeContextCurrent(renderer_window);

    glfwSetWindowUserPointer(renderer_window, this);
    auto framebufferSizeCallback = [](GLFWwindow* w, int x, int y)
    {
        static_cast<Renderer*>(glfwGetWindowUserPointer(w))->framebufferSizeCallbackImpl(w,x,y);
    };
    glfwSetFramebufferSizeCallback(renderer_window, framebufferSizeCallback);
    auto mouseCursorCallback = [](GLFWwindow* w, double x, double y)
    {
        static_cast<Renderer*>(glfwGetWindowUserPointer(w))->mouseCursorCallbackImpl(w,x,y);
    };
    glfwSetCursorPosCallback(renderer_window, mouseCursorCallback);
    auto mouseButtonCallback = [](GLFWwindow* w, int x, int y, int z)
    {
        static_cast<Renderer*>(glfwGetWindowUserPointer(w))->mouseButtonCallbackImpl(w,x,y,z);
    };
    glfwSetMouseButtonCallback(renderer_window, mouseButtonCallback);
    auto mouseScrollCallback = [](GLFWwindow* w, double x, double y)
    {
        static_cast<Renderer*>(glfwGetWindowUserPointer(w))->mouseScrollCallbackImpl(w,x,y);
    };
    glfwSetScrollCallback(renderer_window, mouseScrollCallback);
    auto keyCallback = [](GLFWwindow *w, int k, int s, int a, int m)
    {
        static_cast<Renderer*>(glfwGetWindowUserPointer(w))->keyCallbackImpl(w,k,s,a,m);
    };
    glfwSetKeyCallback(renderer_window, keyCallback);

    glfwSetInputMode(renderer_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    glfwSetWindowAspectRatio(renderer_window, window_width, window_height);

    if (!renderer_window) exit(EXIT_FAILURE);

    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK) exit(EXIT_FAILURE);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();//ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL(renderer_window, true);
    const char* glsl_version = "#version 130";
    ImGui_ImplOpenGL3_Init(glsl_version);

    render_entity_->initRender(window_width,window_height);

    double start_time_all = glfwGetTime();
    int render_mode = GL_POINTS;
    int render_count = 1;
    if(Model::CONFIG->debug_config().enable_debug_render){
        render_mode = GL_TRIANGLES;
        render_count = 3;
    }
    while (!glfwWindowShouldClose(renderer_window))
    {
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        projection = glm::ortho(static_cast<float>(camera_center_x)-(static_cast<float>(width_scaled)/2.0f),
                                          static_cast<float>(camera_center_x)+(static_cast<float>(width_scaled)/2.0f),
                                          static_cast<float>(camera_center_y)-(static_cast<float>(height_scaled)/2.0f),
                                          static_cast<float>(camera_center_y)+(static_cast<float>(height_scaled)/2.0f), -1.0f, 1.0f);

        view = glm::mat4(1.0f);
        view = translate(view, glm::vec3(camera_view_center.x,camera_view_center.y, 0.0f));
        start_time_all = glfwGetTime();
        render_entity_->shader->use();
        render_entity_->shader->setMat4("view", view);
        render_entity_->shader->setMat4("projection", projection);
        glBindVertexArray(render_entity_->VAO);
        glBindBuffer(GL_ARRAY_BUFFER, render_entity_->VBO[0]);
        glBindBuffer(GL_ARRAY_BUFFER, render_entity_->VBO[1]);
        glBindBuffer(GL_ARRAY_BUFFER, render_entity_->EBO);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, render_entity_->CMD);
        auto *pi = Model::POPULATION->getPersonIndex<PersonIndexGPU>();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, render_entity_->SSBO[0]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, render_entity_->SSBO[0]);//bind to binding point 2 in shader.vert
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, render_entity_->SSBO[1]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, render_entity_->SSBO[1]);//bind to binding point 3 in shader.vert
        if(!Model::CONFIG->debug_config().enable_debug_render){
            if(height_scaled < 30.0f){
                render_mode = GL_TRIANGLES;
                render_count = 3;
            }
            else{
                render_mode = GL_POINTS;
                render_count = 1;
            }
        }
        glDrawElementsInstanced(render_mode, render_count, GL_UNSIGNED_INT, (void*)0, Model::RENDER_ENTITY->num_triangles_to_draw);
        if(Model::CONFIG->debug_config().enable_debug_render_text){
            printf("[Renderer] Render time: %f ms\n",glfwGetTime() - start_time_all);
        }

        renderGUI();

        glfwSwapBuffers(renderer_window);
        glBindVertexArray(0);

        glfwPollEvents();

        if(Model::CONFIG->render_config().close_window_on_finish && Model::MODEL->model_finished){
            glfwSetWindowShouldClose(renderer_window, true);
        }
    }
    glfwTerminate();
    return;
}

float Renderer::get_zoom_factor() {
    return window_height / height_scaled;
}

void Renderer::renderGUI() {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    //Put GUI windows here

    ImGui::SetNextWindowPos(ImVec2(0,0));
    ImGui::Begin("Zoom", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("Zoom: %.2f", get_zoom_factor());
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Renderer::framebufferSizeCallbackImpl(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
    window_width = width;
    window_height = height;
    width_scaled = width;
    height_scaled = height;
}

void Renderer::mouseCursorCallbackImpl(GLFWwindow* window, double x_pos_in, double y_pos_in)
{
    mouse_input_x = x_pos_in;
    mouse_input_y = y_pos_in;
    mouse_input_y = window_height - mouse_input_y;//OGL is from bottom left, GLFW is from top left
    mouse_position = glm::ivec2( mouse_input_x, mouse_input_y );

    if(is_drag_mode){
        mouse_drag_center_current = unProjectPlane( glm::dvec2( mouse_input_x, mouse_input_y ) );
        camera_view_center += ( mouse_drag_center_current - mouse_drag_center_start) * drag_speed;
        mouse_drag_center_start = unProjectPlane( glm::dvec2( mouse_input_x, mouse_input_y) );
    }
}

void Renderer::mouseButtonCallbackImpl(GLFWwindow* window, int button, int action, int mods)
{
    if (button != GLFW_MOUSE_BUTTON_LEFT){
        is_drag_mode = false;
        return;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        mouse_drag_center_start = unProjectPlane( glm::dvec2( mouse_input_x, mouse_input_y) );
        mouse_position = glm::ivec2( mouse_input_x, mouse_input_y );
        is_drag_mode = true;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
    {
        is_drag_mode = false;
        return;
    }
}

void Renderer::mouseScrollCallbackImpl(GLFWwindow* window, double x_offset, double y_offset)
{
    mouse_position = glm::ivec2( mouse_input_x, mouse_input_y );
    double x = mouse_input_x/window_width - 0.5f;
    double y = mouse_input_y/window_height - 0.5f;
    double preX = ( x * width_scaled );
    double preY = ( y * height_scaled );
    if( y_offset > 0 ) {//Zoom in
        width_scaled /= zoomFactor;
        height_scaled /= zoomFactor;
    }
    if( y_offset < 0 ) {//Zoom out
        width_scaled *= zoomFactor;
        height_scaled *= zoomFactor;
    }
    drag_speed = glm::vec2(1000.0f*width_scaled/window_width, 1000.0f*width_scaled/window_height);
    if(height_scaled <  10.0f){
        height_scaled = 10.0f;
        width_scaled = height_scaled*aspect_ratio;
    }
    if(height_scaled >  window_height) {
        height_scaled = window_height;
        width_scaled = window_width;
    }
    double postX = ( x * width_scaled );
    double postY = ( y * height_scaled );
    camera_center_x += ( preX - postX );
    camera_center_y += ( preY - postY );

}

void Renderer::keyCallbackImpl(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE){
        glfwSetWindowShouldClose(renderer_window, true);
        exit(0);
    }
    if (((key == GLFW_KEY_EQUAL && (mods & GLFW_MOD_SHIFT)) || key == GLFW_KEY_KP_ADD) && action == GLFW_PRESS) {
        // zoom in
        double x = mouse_input_x / window_width - 0.5f;
        double y = mouse_input_y / window_height - 0.5f;
        double preX = (x * width_scaled);
        double preY = (y * height_scaled);
        width_scaled /= zoomFactor;
        height_scaled /= zoomFactor;
        drag_speed = glm::vec2(1000.0f * width_scaled / window_width, 1000.0f * width_scaled / window_height);
        if (height_scaled < 10.0f) {
            height_scaled = 10.0f;
            width_scaled = height_scaled * aspect_ratio;
        }
        if (height_scaled > window_height) {
            height_scaled = window_height;
            width_scaled = window_width;
        }
        double postX = (x * width_scaled);
        double postY = (y * height_scaled);
        camera_center_x += (preX - postX);
        camera_center_y += (preY - postY);
    }
    if ((key == GLFW_KEY_MINUS || key == GLFW_KEY_KP_SUBTRACT) && action == GLFW_PRESS) {
        // zoom out
        double x = mouse_input_x / window_width - 0.5f;
        double y = mouse_input_y / window_height - 0.5f;
        double preX = (x * width_scaled);
        double preY = (y * height_scaled);
        width_scaled *= zoomFactor;
        height_scaled *= zoomFactor;
        drag_speed = glm::vec2(1000.0f * width_scaled / window_width, 1000.0f * width_scaled / window_height);
        if (height_scaled < 10.0f) {
            height_scaled = 10.0f;
            width_scaled = height_scaled * aspect_ratio;
        }
        if (height_scaled > window_height) {
            height_scaled = window_height;
            width_scaled = window_width;
        }
        double postX = (x * width_scaled);
        double postY = (y * height_scaled);
        camera_center_x += (preX - postX);
        camera_center_y += (preY - postY);
    }
    if (key == GLFW_KEY_0 && action == GLFW_PRESS) {
        // reset zoom
        width_scaled = window_width;
        height_scaled = window_height;
        camera_center_x = window_width / 2.0;
        camera_center_y = window_height / 2.0;
        camera_view_center = glm::vec2(0.0f, 0.0f);
        drag_speed = glm::vec2(1000.0f * width_scaled / window_width, 1000.0f * width_scaled / window_height);
    }
}

glm::dvec3 Renderer::unProject( const glm::dvec3& win )
{
    glm::ivec4 view;
    glm::dmat4 proj, model;
    glGetDoublev( GL_MODELVIEW_MATRIX, &model[0][0] );
    glGetDoublev( GL_PROJECTION_MATRIX, &proj[0][0] );
    glGetIntegerv( GL_VIEWPORT, &view[0] );
    glm::dvec3 world = glm::unProject( win, model, proj, view );
    return world;
}

glm::dvec2 Renderer::unProjectPlane( const glm::dvec2& win )
{
    glm::dvec3 world1 = unProject( glm::dvec3( win, 0.01 ) );
    glm::dvec3 world2 = unProject( glm::dvec3( win, 0.99 ) );
    double u = -world1.z / ( world2.z - world1.z );
    if( u < 0 ) u = 0;
    if( u > 1 ) u = 1;

    return glm::dvec2( world1 + u * ( world2 - world1 ) );
}
