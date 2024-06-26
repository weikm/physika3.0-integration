#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>

float GLWidGet::deltaTime = 0.0f;
float GLWidGet::lastFrame = 0.0f;

float GLWidGet::lastX = 400.0f;
float GLWidGet::lastY = 300.0f;
bool GLWidGet::firstMouse = true;
Camera GLWidGet::myCamera(glm::vec3(20.0f, 20.0f, 100.0f));

/*
* Initialize glfw and some basic settings.
*/
GLWidGet::GLWidGet(unsigned int width, unsigned int height, std::string name, 
	unsigned int majorVersion, unsigned int minorVersion, int& status): screenWidth(width), screenHeight(height), widName(name) {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, majorVersion);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minorVersion);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	this->window = glfwCreateWindow(screenWidth, screenHeight, name.c_str(), nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		status = -1;
		return ;
	}
    // callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    // capture mouse
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        status = -1;
        return;
    }
    glEnable(GL_DEPTH_TEST); // z-buffer
	status = 0;
}
GLWidGet::~GLWidGet() {
    glfwTerminate();
}
/*
* 
*/
void GLWidGet::windowUpdate(std::shared_ptr<Renderer>& drawParticles) {
    /*
    * Camera speed
    */
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    /*
    * End
    */
    processInput(window);
    glClearColor(0.3f, 0.3f, 0.3f, 0.4f);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear z-buffer

    // translate matrix
    glm::mat4 view = myCamera.GetViewMatrix();
    glm::mat4 projection = glm::mat4(1.0);
    projection = glm::perspective(glm::radians(myCamera.Zoom),
        static_cast<float>(screenWidth) / static_cast<float>(screenHeight),
        0.1f, 100.0f);
    // render
    drawParticles->render(view, projection);

    glfwSwapBuffers(window);
    glfwPollEvents();
}
void GLWidGet::windowUpdate(std::shared_ptr<Renderer>& drawParticles, std::shared_ptr<Renderer>& drawBound) {
    /*
    * Camera speed
    */
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    /*
    * End
    */
    processInput(window);
    glClearColor(0.3f, 0.3f, 0.3f, 0.4f);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear z-buffer

    // translate matrix
    glm::mat4 view = myCamera.GetViewMatrix();
    glm::mat4 projection = glm::mat4(1.0);
    projection = glm::perspective(glm::radians(myCamera.Zoom),
        static_cast<float>(screenWidth) / static_cast<float>(screenHeight),
        0.3f, 100.0f);
    // render
    drawParticles->render(view, projection);
    drawBound->render(view, projection);
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void GLWidGet::windowUpdate(std::vector<std::shared_ptr<Renderer>>& drawParticleList)
{
    /*
     * Camera speed
     */
    float currentFrame = glfwGetTime();
    deltaTime          = currentFrame - lastFrame;
    lastFrame          = currentFrame;
    /*
     * End
     */
    processInput(window);
    glClearColor(0.3f, 0.3f, 0.3f, 0.4f);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // clear z-buffer

    // translate matrix
    glm::mat4 view       = myCamera.GetViewMatrix();
    glm::mat4 projection = glm::mat4(1.0);
    projection           = glm::perspective(glm::radians(myCamera.Zoom),
                                  static_cast<float>(screenWidth) / static_cast<float>(screenHeight),
                                  0.1f,
                                  100.0f);
    // render
    for (auto draw : drawParticleList)
    {
        draw->render(view, projection);
    }
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool GLWidGet::windowCheck() {
    return !glfwWindowShouldClose(window);
}
void GLWidGet::windowEnd() {
    glfwTerminate();
}
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (GLWidGet::firstMouse) {
        GLWidGet::lastX = xpos;
        GLWidGet::lastY = ypos;
        GLWidGet::firstMouse = false;
    }
    float xoffset = xpos - GLWidGet::lastX;
    float yoffset = GLWidGet::lastY - ypos;
    GLWidGet::lastX = xpos;
    GLWidGet::lastY = ypos;

    GLWidGet::myCamera.ProcessMouseMovement(xoffset, yoffset);
}

void processInput(GLFWwindow* window) {
    float cameraSpeed = 2.5f * GLWidGet::deltaTime;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        GLWidGet::myCamera.ProcessKeyBoard(FORWARD, GLWidGet::deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        GLWidGet::myCamera.ProcessKeyBoard(BACKWARD, GLWidGet::deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        GLWidGet::myCamera.ProcessKeyBoard(LEFT, GLWidGet::deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        GLWidGet::myCamera.ProcessKeyBoard(RIGHT, GLWidGet::deltaTime);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    GLWidGet::myCamera.ProcessMouseScroll(static_cast<float>(yoffset));
}