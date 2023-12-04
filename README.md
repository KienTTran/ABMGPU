# ABMGPU
| :tada: Demo of agent based model on GPU using CUDA 12.2.1 and OpenGL 4.5 (Windows/Linux) :tada: |

A demo of using CUDA and OpenGL to render different populations with multiple agents with their own properties.\
This is a useful begin place for those want to illustrate or experiment how an agent based model works.\
You can update the `adjust_person_entity` function to change the moving and color pathway of each agent.

:dart: Each triangle is an agent with different color and trajectory.\
:dart: On GTX 3060 the software can render 5-10M agents without problem.

:flower_playing_cards: The image below illustrates a demo of 12000 agents for easier observation.

In this demo, each agent is a triangle and it has two properties: color and moving path.\
The color is assigned by the population so all agents in the same population will have the same color.\
The moving trajectory is randomized for each agent.

![](https://github.com/KienTTran/ABMGPU/blob/master/ABMGPU.gif)

| :gem: Features :gem: |

:white_check_mark: Agent instances on GPU memory\
:white_check_mark: Uses SSBO for instanced objects (with GLSL 450 shaders)\
:white_check_mark: CUDA OpenGL interops\
:white_check_mark: Renders with GLFW3 window manager\
:white_check_mark: Dynamic camera views in OpenGL (pan,zoom with mouse)\
:white_check_mark: Libraries installed using vcpkg\
:warning: Code is dirty and buggy
   
| :books: Libraries :books: |

vcpkg install glfw3 opengl glew glm imgui[core,glfw-binding,opengl3-binding] easyloggingpp date yaml-cpp


