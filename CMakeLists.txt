if(WIN32)
    cmake_minimum_required(VERSION 3.25)
else()
    cmake_minimum_required(VERSION 3.22)
endif()

project(MASS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

if(WIN32)
    set(CMAKE_CUDA_COMPILER "D:/SDK/CUDA/12.2.1/Toolkit/bin/nvcc.exe")
    set(CMAKE_TOOLCHAIN_FILE "D:/SDK/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

add_executable(${PROJECT_NAME}
        src/main.cpp
        src/cpu/Renderer.cpp
        src/cpu/Renderer.h
        src/utils/Shader.h
        src/gpu/GPUPerson.cu
        src/gpu/GPUPerson.cuh
        src/gpu/GPUBuffer.cu
        src/gpu/GPUBuffer.cuh
        src/gpu/GPUEntity.cu
        src/gpu/GPUEntity.cuh
        src/gpu/Population.cu
        src/gpu/Population.cuh
        src/utils/GPURandom.cu
        src/utils/GPURandom.cuh
        src/utils/GPUUtils.cu
        src/utils/GPUUtils.cuh
        src/cpu/Model.cpp
        src/cpu/Model.h
        src/cpu/Config.cpp src/cpu/Config.h src/utils/AscFile.h src/utils/AscFile.cpp)

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")

if(WIN32)
    target_include_directories(${PROJECT_NAME} PUBLIC "D:/SDK/vcpkg/installed/x64-windows/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "D:/SDK/vcpkg/installed/x64-windows/lib")
    #CUDA - NVIDIA
    target_include_directories(${PROJECT_NAME} PUBLIC "D:/SDK/CUDA/12.2.1/Toolkit/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "D:/SDK/CUDA/12.2.1/Toolkit/lib/x64")
    target_link_libraries(${PROJECT_NAME} PRIVATE cuda)

    #GLFW3 glfw3:x64-windows
    find_package(glfw3 CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

    #OpenGL opengl:x64-windows
    find_package(OpenGL REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE opengl32)

    #GLEW glew:x64-windows
    find_package(GLEW REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE GLEW::GLEW)

    #GLM glm:x64-windows
    find_package(glm CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

    #IMGUI imgui[core,glfw-binding,opengl3-binding]:x64-windows
    find_package(imgui CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE imgui::imgui)

    #FreeGLUT freeglut:x64-windows
    find_package(FreeGLUT CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>)
    find_package(GLUT REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE GLUT::GLUT)

    #assimp assimp:x64-windows
    find_package(assimp CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)

    #GLAD glad:x64-windows
    find_package(glad CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE glad::glad)

#    #Easyloggingpp easyloggingpp:x64-windows
#    target_link_libraries(${PROJECT_NAME} PRIVATE easyloggingpp)

    find_package(date CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE date::date date::date-tz)

    find_package(yaml-cpp CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)

    #    #OPENCV opencv:x64-windows
    #    set(OpenCV_DIR "D:/SDK/OpenCV470/build/install")
    #    find_package(OpenCV CONFIG REQUIRED)
    #    include_directories("${OpenCV_INCLUDE_DIRS}")
    #    target_link_directories(${PROJECT_NAME} PUBLIC "D:/SDK/OpenCV470/build/install/x64/vc17/bin")
    #    target_link_libraries(${PROJECT_NAME} PRIVATE "${OpenCV_LIBS}")
else()
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    include_directories("/usr/local/cuda/include")
    include_directories("/usr/local/include")
    include_directories("/usr/include")
    target_link_libraries(${PROJECT_NAME} glfw GL)
endif()
