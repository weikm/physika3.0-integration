#include <gl_particle_render/renderer/cuParticleRenderer.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <iostream>
#include <stdio.h>
#include "device_launch_parameters.h"

#ifndef STYLE_HOME
#define STYLE_HOME
#endif

std::string CUParticleRenderer::style_dir = STYLE_HOME "/gl_render_style/";

CUParticleRenderer::CUParticleRenderer(float* particles_pos, void* color_addr, unsigned int instance_num, float particle_radius, 
    int defaultColorAttr, int defaultPosDim, 
    std::string color_name, std::string vertex_name, std::string fragment_name)
    : m_particles_pos(particles_pos), pos_dim(defaultPosDim), m_particle_radius(particle_radius),
    // shaders path
    colorRamp(style_dir + color_name), particleShader((style_dir + "shaders/" + vertex_name).c_str(), (style_dir + "shaders/" + fragment_name).c_str()),
    m_instance_num(instance_num), posAttr(instance_num * sizeof(glm::vec3)),  colorAttr(instance_num * sizeof(float)), color_ptr(color_addr)
{
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    unsigned int idx[] = { 0, 1, 2, 2, 3, 0 };
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

    float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.5f, 0.5f, 0.0f,
        -0.5f, 0.5f, 0.0f
    };
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    float uvs[] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
    glGenBuffers(1, &VUV);
    glBindBuffer(GL_ARRAY_BUFFER, VUV);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    this->preparePos();
    posAttr.bind();
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
    glVertexAttribDivisor(3, 1);
    posAttr.unBind();

    particleShader.use();
    colorRamp.create1DTexture();
    colorRamp.setTextureUnit(particleShader, "colorRamp", 0);
    
    color_type.attr_id = defaultColorAttr;
    // Init your color config here. See 'ColorConfig'
    switch (color_type.attr_id){
    case attr_for_color::VEL: {
        color_type.fn = 2; // sqrt
        color_type.data_length = 3;
        this->particleShader.setFloat("min_value", 0.0f);
        this->particleShader.setFloat("max_value", 5.0f);
    }
        break;
    case attr_for_color::DENSITY: {
        color_type.fn = 3; // sqr
        color_type.data_length = 1;
        this->particleShader.setFloat("min_value", 0.0f);
        this->particleShader.setFloat("max_value", 100.0f);
    }
        break;
    case attr_for_color::TEMPERATURE: {
        color_type.fn = 1; // linear
        color_type.data_length = 1;
        this->particleShader.setFloat("min_value", 0.0f);
        this->particleShader.setFloat("max_value", 100.0f);
    }
        break;
    case attr_for_color::TYPE: {
        color_type.fn = 7;
        color_type.data_length = 1;
        color_type.data_type = 0;
        this->particleShader.setFloat("min_value", 0.05f);
        this->particleShader.setFloat("max_value", 0.95f);
    }
        break;
    case attr_for_color::PSCALE: {
        color_type.fn = 8;
        color_type.data_length = 4;
        color_type.data_type = 1;
        this->particleShader.setFloat("min_value", 0.0f);
        this->particleShader.setFloat("max_value", 1.0f);
    }
        break;
    case attr_for_color::COUPLINGPHASE: {
        color_type.fn          = 7;
        color_type.data_length = 1;
        color_type.data_type   = 0;
        this->particleShader.setFloat("min_value", 0.05f);
        this->particleShader.setFloat("max_value", 3.95f);
    }
    break;
    /*
    * Init NewType 
    */
    default:
        color_type.fn = 1;
        this->particleShader.setFloat("min_value", 0.0f);
        this->particleShader.setFloat("max_value", 1.0f);
    }
    this->particleShader.setInt("mapping_fn", color_type.fn);

    //this->prepareColor();
    colorAttr.bind();
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
    glVertexAttribDivisor(6, 1);
    colorAttr.unBind();
   
    glBindVertexArray(0);
}
void CUParticleRenderer::render(const glm::mat4& view, const glm::mat4& projection) {
    particleShader.use();
    particleShader.setFloat("radius", m_particle_radius);
    particleShader.setMat4("view", view);
    particleShader.setMat4("projection", projection);
    // Rendering
    glBindVertexArray(VAO);

    // cuda vbo bind
    this->preparePos();
    this->prepareColor();

    colorRamp.activeTexture(0);

    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, m_instance_num);
}

__global__ void generatePos4ToVBO(float3* des_ptr, float4* pos_ptr, unsigned int nums) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nums) return;
    des_ptr[idx].x = pos_ptr[idx].x;
    des_ptr[idx].y = pos_ptr[idx].y;
    des_ptr[idx].z = pos_ptr[idx].z;
}

__global__ void generateColorToVBO(float* des_ptr, float3* pos_ptr, unsigned int nums) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nums) return;
    des_ptr[idx] = length(pos_ptr[idx]);
}

__global__ void generateColor4ToVBO(float* des_ptr, float4* pos_ptr, unsigned int nums) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nums) return;
    des_ptr[idx] = pos_ptr[idx].w;
}

__global__ void generateIntColorToFloatVBO(float* des_ptr, int* pos_ptr, unsigned int nums) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nums) return;
    des_ptr[idx] = static_cast<float>(pos_ptr[idx]);
}

void CUParticleRenderer::preparePos() {
    float* positions = nullptr;
    size_t num_bytes = 0;
    // Map
    posAttr.cuMapPointer((void**) & positions, &num_bytes);
    // Transport positions from cuda to GL
    if (pos_dim == 4) 
        generatePos4ToVBO<<<(m_instance_num - 1) / 256 + 1, 256>>>((float3*)positions, (float4*)m_particles_pos, this->m_instance_num);
    else if (pos_dim == 3)
        checkCudaErrors(cudaMemcpy(positions, m_particles_pos, sizeof(float3) * m_instance_num, cudaMemcpyDeviceToDevice));
    // Unmap
    posAttr.cuUnMap();
}

void CUParticleRenderer::prepareColor() {
    void* color = nullptr;
    size_t num_bytes = 0;
    // Map
    colorAttr.cuMapPointer((void**)&color, &num_bytes);
    // transport attr
    if (color_type.data_type == 1) {
        if (color_type.data_length == 1)
            checkCudaErrors(cudaMemcpy(color, this->color_ptr, sizeof(float) * this->m_instance_num, cudaMemcpyDeviceToDevice));
        else if (color_type.data_length == 3) {
            generateColorToVBO << <(m_instance_num - 1) / 256 + 1, 256 >> > ((float*)color, (float3*)this->color_ptr, this->m_instance_num);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else if (color_type.data_length == 4) {
            generateColor4ToVBO << <(m_instance_num - 1) / 256 + 1, 256 >> > ((float*)color, (float4*)this->color_ptr, this->m_instance_num);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
    else if(color_type.data_type == 0) {
        if (color_type.data_length == 1) {
            generateIntColorToFloatVBO << <(m_instance_num - 1) / 256 + 1, 256 >> > ((float*)color, (int*)this->color_ptr, this->m_instance_num);
            checkCudaErrors(cudaDeviceSynchronize());
        }
    }
    // Unmap
    colorAttr.cuUnMap();
}

bool CUParticleRenderer::valid() { return true; }
void CUParticleRenderer::update() {}
void CUParticleRenderer::toDelete() {
    posAttr.cuUnregister();
    colorAttr.cuUnregister();
    glDeleteBuffers(1, &this->EBO);
    glDeleteBuffers(1, &this->VUV);
    glDeleteBuffers(1, &this->VBO);
    glDeleteVertexArrays(1, &this->VAO);
}