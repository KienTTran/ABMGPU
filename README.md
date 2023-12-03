# ABMGPU
:tada: Demo of agent based model on GPU using CUDA 12.2.1 and OpenGL 4.5 Windows/Linux :tada:\

:dart: Each triangle is an agent with different color and trajectory.\
:dart: On GTX 3060 the software can render 5-10M agents without problem.\

:flower_playing_cards: The image below illustrates a demo of 12000 agents for easier observation.\

![](https://github.com/KienTTran/ABMGPU/blob/master/ABMGPU.gif)

:gem: Features :gem:\
:white_check_mark: Agent instances on GPU memory\
:white_check_mark: Renders with OpenGL with GLFW3 window manager\
:white_check_mark: Uses SSBO in OpenGL 4.5\
:white_check_mark: OpenGL CUDA interop\
:white_check_mark: Dynamic camera views (pan,zoom with mouse)\
:white_check_mark: Configration via YAML file\
:white_check_mark: Libraries installed using vcpkg\
:warning: Code is dirty and buggy
   
:books: Libraries :books:
vcpkg install glfw3 opengl glew glm imgui[core,glfw-binding,opengl3-binding] easyloggingpp date yaml-cpp


