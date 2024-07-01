/**
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-06-07
 * @description: A sample of using Physika
 * @version    : 1.0
 */
//#define STB_IMAGE_IMPLEMENTATION
//#include "framework/world.hpp"
//#include "framework/scene.hpp"
//#include "framework/object.hpp"
//#include "pbd_granular/interface/granular_particle_solver.hpp"
//#include <gl_particle_render/glWindow/glWidGet.h>
//#include <gl_particle_render/renderer/cuParticleRenderer.h>
//#include <iostream>
//#include <random>
//#include <vector_types.h>
//#include <vector_functions.hpp>
//#include "stb_image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "pbd_granular/interface/granular_particle_solver_integration.hpp"
#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>
#include <iostream>
#include <random>
//#include <vector_types.h>
#include <vector_functions.hpp>
#include "stb_image.h"
using namespace Physika;


// 双线性插值函数
//求点(x,y)处的插值计算得到的value
//data数组每个元素的值，是一个字节，0~255整数
float bilinearInterpolate(const unsigned char* data, int width, int height, float x, float y)
{
    //x取整得x1，进一得x2
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    //fx1和fx2，是x1到x的距离，和x到x2的距离
    float fx1 = x - x1;
    float fy1 = y - y1;
    float fx2 = 1.0f - fx1;
    float fy2 = 1.0f - fy1;

    float value = 0.0f;

    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
        value += data[y1 * width + x1] * fx2 * fy2;

    if (x2 >= 0 && x2 < width && y1 >= 0 && y1 < height)
        value += data[y1 * width + x2] * fx1 * fy2;

    if (x1 >= 0 && x1 < width && y2 >= 0 && y2 < height)
        value += data[y2 * width + x1] * fx2 * fy1;

    if (x2 >= 0 && x2 < width && y2 >= 0 && y2 < height)
        value += data[y2 * width + x2] * fx1 * fy1;

    return value;
}

// 从灰度图加载高度场
std::vector<float> loadHeightFieldFromImage(const std::string& image_path, int height_x_num, int height_z_num, float scale_height, float basic_height)
{
    int width, height, channels;

    //读取灰度图，存到data
    unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, STBI_grey);
    if (!data)
    {
        throw std::runtime_error("Failed to load image");
    }
    int                a = 10;
    std::vector<float> heightfield(height_x_num * height_z_num);

    // 将像素值转换为高度值，并进行双线性插值
    for (int j = 0; j < height_z_num; ++j)
    {
        for (int i = 0; i < height_x_num; ++i)
        {
            float x                           = (static_cast<float>(i) / height_x_num) * width;
            float y                           = (static_cast<float>(j) / height_z_num) * height;
            float value                       = bilinearInterpolate(data, width, height, x, y);
            heightfield[i + j * height_x_num] = value / 255.0f * scale_height + basic_height;
        }
    }

    stbi_image_free(data);
    return heightfield;
}


int main()
{
    //auto scene_a = World::instance().createScene();
    //auto obj_a    = World::instance().createObject();
    //obj_a->addComponent<GranularParticleComponent>();
    //auto  component       = obj_a->getComponent<GranularParticleComponent>();
    //std::vector<float3> init_pos;
    //component->addInstance(0.3, init_pos);
    //component->addInstance(0.3,   { -30.f, -20.f, -15.f }, { 45.f, 3.f, 30.f },  { 0.f, 0.f, 0.f });
    //component->addInstance(0.24,  { -8.f, -12.2f, -8.f },  { 16.f, 4.8f, 16.f }, { 0.f, 0.f, 0.f });
    //component->addInstance(0.1875,{ -8.f, -7.4f, -8.f },   { 8.f, 9.6f, 16.f },  { 0.f, 0.f, 0.f });
    //component->addInstance(0.15,  { 0.f, -7.4f, -8.f },    { 8.f, 9.6f, 16.f },  { 0.f, 0.f, 0.f });
    //component->addInstance(0.375, { 15.f, -20.f, -15.f },  { 12.f, 7.8f, 30.f }, { 0.f, 0.f, 0.f });
    //component->addInstance(0.6,   { -11.f, -17.f, -15.f }, { 22.f, 4.8f, 30.f }, { 0.f, 0.f, 0.f });
    //
    //

    //auto granular_solver = World::instance().createSolver<GranularParticleSolver>();
    //GranularParticleSolver::SolverConfig granular_config;
    //granular_config.m_dt         = 1.f / 60.f;
    //granular_config.m_total_time = 20.f;
    //granular_config.m_write2ply    = true;
    //granular_solver->config(granular_config);
    //scene_a->addObject(obj_a);
    //granular_solver->attachObject(obj_a);

    //// set solver boundary
    //granular_solver->setSleepThreshold(0.032);
    //std::vector<float> world_boundaries = {-60, -20, -20, 40, 20, 20};
    //granular_solver->setWorldBoundary(-60, -20, -20, 40, 20, 20);
    //float unit_height  = 0.3;
    //int   height_x_num = static_cast<int>((world_boundaries[3] - world_boundaries[0]) / unit_height);
    //int   height_z_num = static_cast<int>((world_boundaries[5] - world_boundaries[2]) / unit_height);
    //
    ////The absolute path to the cost location needs to be modified
    //std::vector<float> height = loadHeightFieldFromImage("E:/project_code/physika3.0_release0.1.0_20240613/Physika/examples/granular_sample/heightmap.png", height_x_num, height_z_num, 5,-25);

    ////set HF in granular solver
    //granular_solver->setHeightField(height, unit_height, height_x_num, height_z_num);
    //scene_a->addSolver(granular_solver);

    //std::cout << "numParticles: " << obj_a->getComponent<GranularParticleComponent>()->m_num_particles << std::endl;

    //// if GUI
    //bool gui_test = true;
    //if (gui_test)
    //{
    //    unsigned int screenWidth  = 1440;
    //    unsigned int screenHeight = 900;
    //    int          status;
    //    GLWidGet     test_window(screenWidth, screenHeight, "Granular", 3, 3, status);

    //    if (status != 0)
    //    {
    //        return -1;
    //    }

    //    /*
    //    * 1. Create a renderer or more.
    //    */
    //    // Positions ptr, color_attrs ptr, num, particle_radius, color type
    //    float*       pos_ptr         = obj_a->getComponent<GranularParticleComponent>()->m_device_pos;
    //    void*        color_ptr       = obj_a->getComponent<GranularParticleComponent>()->m_device_pos;
    //    unsigned int num_particles   = obj_a->getComponent<GranularParticleComponent>()->m_num_particles;
    //    float        particle_radius ;
    //    int          color_type      = attr_for_color::PSCALE;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
    //    int          pos_dim         = 4;                    // set 4 if a freaky 'float4' type is used for position_ptr.

    //    particle_radius = 0.2;
    //    // Renderer.
    //    std::shared_ptr<Renderer> drawElasticParticles =
    //        std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, color_type, pos_dim, "rainbow.png", "shader1.vs", "shader1.fs");

    //    /*
    //    * 2. Put renderer into renderer list
    //    */
    //    // Renderer List
    //    std::vector<std::shared_ptr<Renderer>> drawList;
    //    drawList.push_back(drawElasticParticles);

    //    /*
    //    * 3. Run simulation
    //    */
    //    double total_time = granular_config.m_total_time;
    //    double cur_time   = 0.0;
    //    double dt         = granular_config.m_dt;
    //    while (cur_time < total_time && test_window.windowCheck())
    //    {
    //        double dt = (cur_time + granular_config.m_dt <= total_time) ? granular_config.m_dt : total_time - cur_time;
    //        cur_time += dt;
    //        granular_config.m_dt = dt;
    //        granular_solver->step();
    //        // Draw particles every time step
    //        test_window.windowUpdate(drawList);
    //    }
    //    test_window.windowEnd();
    //}
    //else
    //{
    //    World::instance().run();
    //}
    //return 0;

   //auto scene_a = World::instance().createScene();
    //auto obj_a    = World::instance().createObject();
    //obj_a->addComponent<GranularParticleComponentIntegration>();
    //auto  component       = obj_a->getComponent<GranularParticleComponentIntegration>();
    GranularParticleComponentIntegration* component = new GranularParticleComponentIntegration;
    std::vector<glm::vec3>                init_pos;
    //component->addInstance(0.3, init_pos);
    component->addInstance(0.3, { -30.f, -20.f, -15.f }, { 45.f, 3.f, 30.f }, { 0.f, 0.f, 0.f });
    component->addInstance(0.24, { -8.f, -12.2f, -8.f }, { 16.f, 4.8f, 16.f }, { 0.f, 0.f, 0.f });
    component->addInstance(0.1875, { -8.f, -7.4f, -8.f }, { 8.f, 9.6f, 16.f }, { 0.f, 0.f, 0.f });
    component->addInstance(0.15, { 0.f, -7.4f, -8.f }, { 8.f, 9.6f, 16.f }, { 0.f, 0.f, 0.f });
    component->addInstance(0.375, { 15.f, -20.f, -15.f }, { 12.f, 7.8f, 30.f }, { 0.f, 0.f, 0.f });
    component->addInstance(0.6, { -11.f, -17.f, -15.f }, { 22.f, 4.8f, 30.f }, { 0.f, 0.f, 0.f });
    //component->initialize();

    GranularParticleSolverIntegration* granular_solver = new GranularParticleSolverIntegration;

    //granular_solver->initialize();
    GranularParticleSolverIntegration::SolverConfig granular_config;
    granular_config.m_dt         = 1.f / 60.f;
    granular_config.m_total_time = 2000.f;
    granular_config.m_write2ply  = true;
    granular_solver->config(granular_config);
    //scene_a->addObject(obj_a);
    //granular_solver->attachObject(obj_a);
    granular_solver->setComponent(component);

    granular_solver->setInitialize();

    // set solver boundary
    granular_solver->setSleepThreshold(0.032);
    std::vector<float> world_boundaries = { -60, -20, -20, 40, 20, 20 };
    granular_solver->setWorldBoundary(-60, -20, -20, 40, 20, 20);
    float unit_height  = 0.3;
    int   height_x_num = static_cast<int>((world_boundaries[3] - world_boundaries[0]) / unit_height);
    int   height_z_num = static_cast<int>((world_boundaries[5] - world_boundaries[2]) / unit_height);

    //The absolute path to the cost location needs to be modified
    std::vector<float> height = loadHeightFieldFromImage("E:/project_code/physika3.0_release0.1.0_20240613/Physika/examples/granular_integration_sample/heightmap.png", height_x_num, height_z_num, 5, -25);

    //set HF in granular solver
    granular_solver->setHeightField(height, unit_height, height_x_num, height_z_num);
    //scene_a->addSolver(granular_solver);

    std::cout << "numParticles: " << component->m_num_particles << std::endl;

    // if GUI
    bool gui_test = true;
    if (gui_test)
    {
        unsigned int screenWidth  = 1440;
        unsigned int screenHeight = 900;
        int          status;
        GLWidGet     test_window(screenWidth, screenHeight, "Granular", 3, 3, status);

        if (status != 0)
        {
            return -1;
        }

        /*
        * 1. Create a renderer or more.
        */
        // Positions ptr, color_attrs ptr, num, particle_radius, color type
        float*       pos_ptr       = component->m_device_pos;
        void*        color_ptr     = component->m_device_pos;
        unsigned int num_particles = component->m_num_particles;
        float        particle_radius;
        int          color_type = attr_for_color::PSCALE;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int          pos_dim    = 4;                       // set 4 if a freaky 'float4' type is used for position_ptr.

        particle_radius = 0.2;
        // Renderer.
        std::shared_ptr<Renderer> drawElasticParticles =
            std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, color_type, pos_dim, "rainbow.png", "shader1.vs", "shader1.fs");

        /*
        * 2. Put renderer into renderer list
        */
        // Renderer List
        std::vector<std::shared_ptr<Renderer>> drawList;
        drawList.push_back(drawElasticParticles);

        /*
        * 3. Run simulation
        */
        double total_time = granular_config.m_total_time;
        double cur_time   = 0.0;
        double dt         = granular_config.m_dt;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = (cur_time + granular_config.m_dt <= total_time) ? granular_config.m_dt : total_time - cur_time;
            cur_time += dt;
            granular_config.m_dt = dt;
            granular_solver->step();  //step
            // Draw particles every time step
            test_window.windowUpdate(drawList);
            // Or set some conditions:
            /*
            if (···){
                test_window.windowUpdate(drawList);
            }
            */
        }
        test_window.windowEnd();
    }
    //else
    //{
    //    World::instance().run();
    //}
    return 0;


}