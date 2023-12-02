# ABMGPU
Demo of agent based model on GPU using CUDA 12.2.1 and OpenGL Windows/Linux.

Features:
1. Agent instances on GPU memory
2. Render with OpenGL with GLFW3 window manager
3. OpenGL CUDA interop
4. Camera view (pan,zoom with mouse)
5. Configration via YAML file
6. Libraries installed using vcpkg
7. Code is dirty and buggy
   
Libraries:
vcpkg install glfw3 opengl glew glm imgui[core,glfw-binding,opengl3-binding] easyloggingpp date yaml-cpp

![](https://github.com/KienTTran/ABMGPU/blob/master/ABMGPU.gif)

