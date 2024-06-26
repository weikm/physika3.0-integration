#pragma once
#ifndef GLWIDGET_H
#define GLWIDGET_H
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<gl_particle_render/renderer/cuParticleRenderer.h>
#include "Camera.h"
#include <string>
#include <memory>
#include <iostream>

class GLWidGet {
private:
	/*
	* For Camera
	*/
	static float deltaTime;
	static float lastFrame;

	static float lastX, lastY;
	static bool firstMouse;

	static Camera myCamera;

	/*
	* For window
	*/
	unsigned int screenWidth;
	unsigned int screenHeight;
	std::string widName;
	GLFWwindow* window;

public:
	explicit GLWidGet(unsigned int width, unsigned int height, std::string name, 
		unsigned int majorVersion, unsigned int minorVersion, int& status);
	GLWidGet(const GLWidGet&) = delete;
	GLWidGet& operator=(const GLWidGet&) = delete;
	~GLWidGet();
	void windowUpdate(std::shared_ptr<Renderer>& drawParticles);
	void windowUpdate(std::shared_ptr<Renderer>& drawParticles, std::shared_ptr<Renderer>& drawBound);
    void windowUpdate(std::vector<std::shared_ptr<Renderer>>& darwParticleList);
	bool windowCheck();
	void windowEnd();

	friend void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	friend void processInput(GLFWwindow* window);
	friend void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	friend void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
};
#endif