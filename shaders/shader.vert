#version 460 core
layout (location = 0) in vec4 in_pos;
layout (location = 1) in vec4 in_color;
layout(std430, binding = 2) buffer BufferModel
{
    mat4 in_models[];
} buff_model;
layout(std430, binding = 3) buffer BufferColor
{
    vec4 in_colors[];//glsl layout accept vec4 only
} buff_color;

uniform mat4 view;
uniform mat4 projection;

out vec4 out_color;

void main()
{
    out_color = buff_color.in_colors[gl_BaseInstance+gl_InstanceID];
    gl_Position = projection * view * buff_model.in_models[gl_BaseInstance+gl_InstanceID] * in_pos;
//    out_color = in_color;
//    gl_Position = projection * view * in_pos;

}