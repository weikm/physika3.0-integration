#include <gl_particle_render/renderer/vbo.h>
#include <helper_cuda.h>
#include <iostream>
VBO::VBO() {
	glGenBuffers(1, &ID);
}
void VBO::bind() const {
	glBindBuffer(GL_ARRAY_BUFFER, ID);
}
void VBO::unBind() {
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
GLuint VBO::getID() const {
	return this->ID;
}
CudaVBO::CudaVBO(size_t mem_size):m_size(mem_size) {
	this->bind();
	glBufferData(GL_ARRAY_BUFFER, m_size, 0, GL_DYNAMIC_DRAW);
	this->unBind();
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&this->VBO_CUDA,
		ID,
		cudaGraphicsMapFlagsWriteDiscard));
}
void CudaVBO::cuRegister(size_t size) {
	this->bind();
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	this->unBind();
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&this->VBO_CUDA,
		ID,
		cudaGraphicsMapFlagsWriteDiscard));
}
void CudaVBO::cuUnMap() {
	checkCudaErrors(cudaGraphicsUnmapResources(1, &VBO_CUDA, 0));
}
void CudaVBO::cuUnregister() {
	glDeleteBuffers(1, &ID);
	checkCudaErrors(cudaGraphicsUnregisterResource(this->VBO_CUDA));
}
void CudaVBO::cuMapPointer(void** pos_ptr, size_t* num_bytes) {
	checkCudaErrors(cudaGraphicsMapResources(1, &this->VBO_CUDA));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
		pos_ptr,
		num_bytes,
		this->VBO_CUDA
	));
}