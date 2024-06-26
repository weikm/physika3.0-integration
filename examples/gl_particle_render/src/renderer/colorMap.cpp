#include <gl_particle_render/renderer/colorMap.h>
#include <iostream>
#include <gl_particle_render/renderer/stb_image.h>
ColorMap::ColorMap(std::string map_file):map_path(map_file), textureID(-1), color_map(nullptr) {

}
void ColorMap::create1DTexture() {
	int width, height, nrChannels;
	unsigned char* data = stbi_load(this->map_path.c_str(), &width, &height, &nrChannels, 0);

	if (data)
	{
		mallocMap(width);
		// transform 2D map to 1D
		for (int it = 0; it < width; it += 1) {
			int index = it * nrChannels;
			color_map[it] = glm::vec4((float)data[index] / 256.0f,
									(float)data[index + 1] / 256.0f,
									(float)data[index + 2] / 256.0f, 1.0f);
		}
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
		stbi_image_free(data);
		return;
	}
	stbi_image_free(data);

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_1D, textureID);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage1D(
		GL_TEXTURE_1D,
		0,
		GL_RGBA32F, width,
		0,
		GL_RGBA, GL_FLOAT, color_map);

	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_1D, 0);

	// ending create
	delete[] color_map;
}

void ColorMap::mallocMap(const int width) {
	color_map = new glm::vec4[width];
}

void ColorMap::setTextureUnit(const Shader& shader, std::string location, GLuint tex_id) {
	glUniform1i(glGetUniformLocation(shader.ID, location.c_str()), tex_id);
}

void ColorMap::activeTexture(GLuint tex_id) {
	glActiveTexture(GL_TEXTURE0 + tex_id);
	glBindTexture(GL_TEXTURE_1D, textureID);
}