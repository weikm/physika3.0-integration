#pragma once
#ifndef BASERENDERER_H
#define BASERENDERER_H
#include "shader.h"
#include <cuda_gl_interop.h>
class Renderer
{
public:
    virtual void update()                                                   = 0;
    virtual bool valid()                                                    = 0;
    virtual void render(const glm::mat4& view, const glm::mat4& projection) = 0;
    virtual void toDelete()                                                 = 0;
};
  // namespace render_bku 
#endif