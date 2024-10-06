if(WIN32)
    cmake_minimum_required(VERSION 3.25)
else()
    cmake_minimum_required(VERSION 3.22)
endif()

if(WIN32)
    set(CMAKE_CUDA_COMPILER "D:/SDK/CUDA/12.2.1/Toolkit/bin/nvcc.exe")
    set(CMAKE_TOOLCHAIN_FILE "D:/SDK/vcpkg/scripts/buildsystems/vcpkg.cmake")
else()
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    set(CMAKE_TOOLCHAIN_FILE "/mnt/d/SDK/vcpkg/scripts/buildsystems/vcpkg.cmake")
endif()

#vcpkg install glfw3 opengl glew glm imgui[core,glfw-binding,opengl3-binding] easyloggingpp date yaml-cpp

project(MASS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
set(EXT_PROJECTS_DIR ${PROJECT_SOURCE_DIR}/ext)
include_directories(${EXT_PROJECTS_DIR}/easyloggingpp.9.96.4)

add_executable(${PROJECT_NAME}
        src/cpu/Core/Config/CustomConfigItem.h
        src/cpu/Core/Config/CustomConfigItem.cpp
        src/cpu/Core/Config/Config.cpp
        src/cpu/Core/PropertyMacro.h
        src/cpu/Core/Config/Config.cpp
        src/cpu/Core/Config/Config.h
        src/cpu/Core/Scheduler.cpp
        src/cpu/Core/Scheduler.h
        src/cpu/Core/Dispatcher.h
        src/cpu/Core/Dispatcher.cpp
        src/cpu/Core/Config/IConfigItem.h
        src/cpu/Core/Config/IConfigItem.cpp
        src/cpu/Core/Config/ConfigItem.hxx
        src/cpu/Core/Config/YamlConverter.hxx
        src/cpu/Core/Random.h
        src/cpu/Core/Random.cpp
        src/cpu/Core/TypeDef.h
        src/cpu/Population/Person.cpp
        src/cpu/Population/Person.h
        src/cpu/Population/Population.cu
        src/cpu/Population/Population.cuh
        src/main.cpp
        src/cpu/Renderer.cpp
        src/cpu/Renderer.h
        src/utils/Shader.h
        src/gpu/RenderEntity.cu
        src/gpu/RenderEntity.cuh
        src/utils/GPURandom.cu
        src/utils/GPURandom.cuh
        src/utils/GPUUtils.cu
        src/utils/GPUUtils.cuh
        src/cpu/Model.cpp
        src/cpu/Model.h
        src/utils/Thread.cpp
        src/utils/Thread.h
        src/cpu/Helpers/TimeHelpers.h
        src/cpu/Spatial/Coordinate.h
        src/cpu/Spatial/Coordinate.cpp
        src/cpu/Spatial/Location.cpp
        src/cpu/GIS/SpatialData.h
        src/cpu/GIS/SpatialData.cpp
        src/cpu/GIS/AscFile.cpp
        ${EXT_PROJECTS_DIR}/easyloggingpp.9.96.4/easylogging++.h
        ${EXT_PROJECTS_DIR}/easyloggingpp.9.96.4/easylogging++.cc
        src/cpu/Environment/SeasonalEquation.cpp
        src/cpu/Environment/SeasonalRainfall.cpp
        src/cpu/Environment/SeasonalInfo.h
        src/cpu/Spatial/SpatialModelBuilder.hxx
        src/cpu/Spatial/SpatialModel.hxx
        src/cpu/Spatial/BurkinaFaso.hxx
        src/cpu/Helpers/NumberHelpers.h
        src/cpu/Spatial/BarabasiSM.cpp
        src/cpu/Spatial/GeneralGravitySM.cpp
        src/cpu/Spatial/WesolowskiSM.hxx
        src/cpu/Constants.h
        src/cpu/Helpers/ObjectHelpers.h
        src/cpu/Events/Event.cpp
#        src/cpu/Events/PersonUpdateEvent.cpp
#        src/cpu/Events/PersonUpdateEvent.h
#        src/cpu/Events/PersonUpdateRenderEvent.cpp
#        src/cpu/Events/PersonUpdateRenderEvent.h
        src/cpu/version.h
        src/cpu/Helpers/OSHelpers.h
        src/cpu/Helpers/OSHelpers.cpp
        src/cpu/Reporter/Reporter.h
        src/cpu/Reporter/Reporter.cpp
        src/cpu/MDC/ModelDataCollector.h
        src/cpu/MDC/ModelDataCollector.cpp
        src/cpu/Reporter/MonthlyReporter.cpp
        src/cpu/Helpers/StringHelpers.h
        src/cpu/Population/Properties/IndexHandler.h
        src/cpu/Population/Properties/IndexHandler.cpp
        src/cpu/Population/Properties/PersonIndex.cpp
        src/cpu/Population/Properties/PersonIndex.h
        src/cpu/Population/Properties/PersonIndexAll.cpp
        src/cpu/Population/Properties/PersonIndexAll.h
        src/cpu/Population/Properties/PersonIndexAllHandler.cpp
        src/cpu/Population/Properties/PersonIndexAllHandler.h
        src/cpu/Population/Properties/PersonIndexByLocationStateAgeClassHandler.cpp
        src/cpu/Population/Properties/PersonIndexByLocationStateAgeClassHandler.h
        src/cpu/Population/Properties/PersonIndexByLocationStateAgeClass.h
        src/cpu/Population/Properties/PersonIndexByLocationStateAgeClass.cpp
        src/cpu/Population/Properties/PersonIndexGPU.cpp
        src/cpu/Population/Properties/PersonIndexGPU.h
        src/cpu/Population/Properties/PersonIndexToRenderHandler.cpp
        src/cpu/Population/Properties/PersonIndexToRenderHandler.h
        src/cpu/Parasite/Genotype.cpp
        src/cpu/Parasite/Genotype.h
        src/cpu/Parasite/Parasite.h
        src/cpu/Parasite/Parasite.cpp
)

set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")

#Easyloggingpp easyloggingpp
#target_link_libraries(${PROJECT_NAME} PRIVATE easyloggingpp)
#target_link_libraries(${PROJECT_NAME} PRIVATE ${EXT_PROJECTS_DIR}/easyloggingpp.9.96.4/easyloggingpp.cc)

#GLFW3 glfw3
find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

#GLEW glew
find_package(GLEW REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE GLEW::GLEW)

#GLM glm
find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

#IMGUI imgui[core,glfw-binding,opengl3-binding]
find_package(imgui CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE imgui::imgui)

#Date date
find_package(date CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE date::date date::date-tz)

#Yaml yaml-cpp
find_package(yaml-cpp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)

#fmt fmt
find_package(fmt CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt fmt::fmt-header-only)

#gsl gsl
find_package(GSL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE GSL::gsl GSL::gslcblas)

#args
find_package(args CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE taywee::args)

if(WIN32)
    target_include_directories(${PROJECT_NAME} PUBLIC "D:/SDK/vcpkg/installed/x64-windows/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "D:/SDK/vcpkg/installed/x64-windows/lib")
    #CUDA - NVIDIA
    target_include_directories(${PROJECT_NAME} PUBLIC "D:/SDK/CUDA/12.2.1/Toolkit/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "D:/SDK/CUDA/12.2.1/Toolkit/lib/x64")
    target_link_libraries(${PROJECT_NAME} PRIVATE cuda)

    #OpenGL opengl
    find_package(OpenGL REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE opengl32)

else()
    target_include_directories(${PROJECT_NAME} PUBLIC "/mnt/d/SDK/vcpkg/installed/x64-linux/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "/mnt/d/SDK/vcpkg/installed/x64-linux/lib")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    include_directories("/usr/local/cuda/include")
    include_directories("/usr/local/include")
    include_directories("/usr/include")

    #OpenGL opengl
    find_package(OpenGL REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)

endif()

# Extra libs, just in case
##FreeGLUT freeglut
#find_package(FreeGLUT CONFIG REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>)
#find_package(GLUT REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE GLUT::GLUT)
##assimp assimp
#find_package(assimp CONFIG REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE assimp::assimp)
##GLAD glad
#find_package(glad CONFIG REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE glad::glad)
#find_package(SFML COMPONENTS system window graphics CONFIG REQUIRED)
#target_link_libraries(${PROJECT_NAME} PRIVATE sfml-system sfml-network sfml-graphics sfml-window)
##OPENCV opencv
#set(OpenCV_DIR "D:/SDK/OpenCV470/build/install")
#find_package(OpenCV CONFIG REQUIRED)
#include_directories("${OpenCV_INCLUDE_DIRS}")
#target_link_directories(${PROJECT_NAME} PUBLIC "D:/SDK/OpenCV470/build/install/x64/vc17/bin")
#target_link_libraries(${PROJECT_NAME} PRIVATE "${OpenCV_LIBS}")
#Easyloggingpp easyloggingpp
#target_link_libraries(${PROJECT_NAME} PRIVATE easyloggingpp)