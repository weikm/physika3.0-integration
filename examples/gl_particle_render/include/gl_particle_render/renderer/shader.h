#ifndef SHADER_H
#define SHADER_H
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
class Shader
{
public:
    unsigned int ID;
    // constructor
    Shader(const char* vertexPath, const char* fragmentPath);
    // use/activate program
    void use();
    // uniform tool functions
    void setBool(const std::string& name, bool value) const;
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setMat4(const std::string& name, const glm::mat4& mat) const;

private:
    void checkCompileErrors(GLuint shader, std::string type);
};
#endif