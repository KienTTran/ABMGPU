# ABMGPU
Demo of agent based model on GPU using CUDA 12.2.1 and OpenGL 4.5 Windows/Linux.

Each triangle is an agent with different color and trajectory. 
On GTX 3060 the software can render 5-10M agents without problem. 
The image below illustrates a demo of 12000 agents for easier observation.

![](https://github.com/KienTTran/ABMGPU/blob/master/ABMGPU.gif)

Features:
1. Agent instances on GPU memory
2. Renders with OpenGL with GLFW3 window manager
3. Uses SSBO in OpenGL 4.5
4. OpenGL CUDA interop
5. Dynamic camera views (pan,zoom with mouse)
6. Configration via YAML file
7. Libraries installed using vcpkg
8. Code is dirty and buggy
   
Libraries:
vcpkg install glfw3 opengl glew glm imgui[core,glfw-binding,opengl3-binding] easyloggingpp date yaml-cpp


