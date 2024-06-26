#pragma once
#ifndef CUPARRENDERER_H
#define CUPARRENDERER_H
#include "basicRenderer.h"
#include <vector>
#include <gl_particle_render/renderer/vbo.h>
#include <gl_particle_render/renderer/colorMap.h>
#include <functional>
enum attr_for_color
{
    Norm,
    VEL,
    TEMPERATURE,
    DENSITY,
    TYPE,
    PSCALE,
    COUPLINGPHASE
};
class CUParticleRenderer : public Renderer
{
public:
    struct ColorConfig
    {
        // Init in consturctor
        int   attr_id     = 0;  // attr_for _color 
        int   fn          = 0;  // color interpolation type (see vertex shader)
        int   data_length = 1;  // length  
        int   data_type   = 1;  // 0: 'int', 1: 'float'
        float min         = 0.0; // min value of your attr
        float max         = 1.0; // max value of your attr 
    };
    explicit CUParticleRenderer(float* particles_pos, void* color_addr, unsigned int instance_num, float particle_radius, 
        int defaultColorAttr = 0, int defaultPosDim = 3, 
        std::string color_name = std::string("Blues.png"), 
        std::string vertex_name = std::string("shader1.vs"), 
        std::string fragment_name = std::string("shader1.fs")
        );  //
    CUParticleRenderer(const CUParticleRenderer&)            = delete;
    CUParticleRenderer& operator=(const CUParticleRenderer&) = delete;
    virtual void        update() override;
    virtual void        render(const glm::mat4& view, const glm::mat4& projection);
    virtual bool        valid() override;
    virtual void        toDelete() override;

private:
    float* m_particles_pos;
    int    pos_dim;
    float  m_particle_radius;

    CudaVBO      posAttr;  // A buffer of particles' positions and color
    CudaVBO      colorAttr;
    GLuint       VAO, VBO, EBO, VUV;
    unsigned int m_instance_num;
    Shader       particleShader;
    // texture color_map
    ColorMap    colorRamp;  // texture_id: 0
    ColorConfig color_type;
    void*       pos_ptr;
    void*       color_ptr;

    static std::string style_dir;
    // function
    void preparePos();
    void prepareColor();
};
#endif