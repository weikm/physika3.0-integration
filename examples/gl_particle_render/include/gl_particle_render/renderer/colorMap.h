#pragma once
#ifndef _COLOR_MAP_H
#define _COLOR_MAP_H
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <gl_particle_render/renderer/shader.h>
#include <cuda_gl_interop.h>
#include <string>
    class ColorMap
    {
    public:
        ColorMap(std::string map_file);
        void create1DTexture();
        void mallocMap(const int width);
        void setTextureUnit(const Shader& shader, std::string location, GLuint tex_id);
        void activeTexture(GLuint tex_id);

    private:
        GLuint      textureID;
        std::string map_path;
        glm::vec4*  color_map;
    };
    #endif