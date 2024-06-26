#pragma once
#ifndef VBO_MINATO_H
#define VBO_MINATO_H
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
class VBO {
public:
	VBO();
	void bind()const;
	void unBind();
	GLuint getID() const;
protected:
	GLuint ID;
};
class CudaVBO : public VBO {
public:
	CudaVBO(size_t mem_size);
	void cuRegister(size_t size);
	void cuUnMap();
	void cuMapPointer(void** pos_ptr, size_t* num_bytes);
	void cuUnregister();
private:
	struct cudaGraphicsResource* VBO_CUDA;
	size_t m_size;
};
#endif