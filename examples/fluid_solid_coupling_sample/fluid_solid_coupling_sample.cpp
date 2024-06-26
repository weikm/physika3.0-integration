/**
 * @author     : Yuhang Xu (mr.xuyh@qq.com)
 * @date       : 2023-10-07
 * @description: A sample of using Physika
 * @version    : 1.0
 */
#include <chrono>
#include <iostream>
#include <vector_types.h>
#include <vector_functions.hpp>
#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "utils/interface/model_helper.hpp"
#include "pbd_fluid_solid_coupling/interface/fluid_solid_coupling_solver.hpp"

using namespace Physika;

int main()
{
    auto scene_a = World::instance().createScene();
    auto obj_a   = World::instance().createObject();
    obj_a->addComponent<CouplingParticleComponent>();
    auto component = obj_a->getComponent<CouplingParticleComponent>();

    float  particle_radius = 0.3;

    ParticleModelConfig solid_config;
    solid_config.shape           = ObjectShape::Cube;
    solid_config.particle_radius = 0.3;
    solid_config.lb              = { 5.f, -20.f, 7.f };
    solid_config.size            = { 3.f, 20.f, 10.f };
    float3 solid_vel                  = { 0.f, 0.f, 0.f };
    float  solid_particle_inmass        = 10.0f;
    // add by shape
    component->addInstance(
        CouplingParticlePhase::SOLID,
        solid_config,
        solid_vel,
        solid_particle_inmass);

    ParticleModelConfig fluid_config;
    fluid_config.shape           = ObjectShape::Cube;
    fluid_config.particle_radius = 0.3;
    fluid_config.lb              = { -40.f, -20.f, -20.f };
    fluid_config.size            = { 40.f, 10.f, 31.f };
    auto   fluid_init_pos             = ModelHelper::create3DParticleModel(fluid_config);
    float3 fluid_vel                  = { 10.f, 0.f, 0.f };
    float  fluid_particle_inmass      = 1.f;
    // add by init pos
    component->addInstance(
        CouplingParticlePhase::FLUID,
        fluid_init_pos,
        fluid_vel,
        fluid_particle_inmass);
    
    auto fluid_solid_coupling_solver = World::instance().createSolver<ParticleFluidSolidCouplingSolver>();
    
    ParticleFluidSolidCouplingSolver::SolverConfig fluid_solid_coupling_config;
    fluid_solid_coupling_config.m_dt         = 1.f / 60.f;
    fluid_solid_coupling_config.m_total_time = 60.f;
    fluid_solid_coupling_solver->config(fluid_solid_coupling_config);
    scene_a->addObject(obj_a);
    fluid_solid_coupling_solver->attachObject(obj_a);
    // set solver boundary
    fluid_solid_coupling_solver->setWorldBoundary(-40, -20, -20, 40, 20, 20);
    scene_a->addSolver(fluid_solid_coupling_solver);

    std::cout << "numParticles: " << obj_a->getComponent<CouplingParticleComponent>()->m_num_particles << std::endl;
    std::cout << "solidNumParticles: " << obj_a->getComponent<CouplingParticleComponent>()->m_num_solid << std::endl;
    std::cout << "fluidNumParticles: " << obj_a->getComponent<CouplingParticleComponent>()->m_num_fluid << std::endl;

        // if GUI
    bool gui_test = true;
    if (gui_test)
    {
        unsigned int screenWidth  = 1440;
        unsigned int screenHeight = 900;
        int          status;
        GLWidGet     test_window(screenWidth, screenHeight, "fluid solid coupling", 3, 3, status);

        if (status != 0)
        {
            return -1;
        }

        auto lastTime   = std::chrono::high_resolution_clock::now();
        int  frameCount = 0;

        /*
        * 1. Create a renderer or more.
        */
        // Positions ptr, color_attrs ptr, num, particle_radius, color type
        float*       pos_ptr         = obj_a->getComponent<CouplingParticleComponent>()->m_device_pos;
        void*        color_ptr       = obj_a->getComponent<CouplingParticleComponent>()->m_device_phase;
        //void*        color_ptr       = obj_a->getComponent<CouplingParticleComponent>()->m_device_vel;
        unsigned int num_particles   = obj_a->getComponent<CouplingParticleComponent>()->m_num_particles;
        float        particle_radius;
        fluid_solid_coupling_solver->getParticleRadius(particle_radius);
        int          color_type      = attr_for_color::COUPLINGPHASE;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        //int          color_type      = attr_for_color::VEL;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int          pos_dim         = 4;                    // set 4 if a freaky 'float4' type is used for position_ptr.

        // Renderer.
        std::shared_ptr<Renderer> drawParticles =
            std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, color_type, pos_dim);
        
        /*
        * 2. Put renderer into renderer list
        */
        // Renderer List
        std::vector<std::shared_ptr<Renderer>> drawList;
        drawList.push_back(drawParticles);

        /*
        * 3. Run simulation
        */
        double total_time = fluid_solid_coupling_config.m_total_time;
        double cur_time   = 0.0;
        double dt         = fluid_solid_coupling_config.m_dt;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = (cur_time + fluid_solid_coupling_config.m_dt <= total_time) ? fluid_solid_coupling_config.m_dt : total_time - cur_time;
            cur_time += dt;
            fluid_solid_coupling_config.m_dt = dt;
            World::instance().step();
            // Draw particles every time step
            test_window.windowUpdate(drawList);
            // Or set some conditions:
            /*
            if (···){
                test_window.windowUpdate(drawList);
            }
            */

            auto   currentTime = std::chrono::high_resolution_clock::now();
            double deltaTime   = std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - lastTime).count();

            if (deltaTime >= 0.1)  // 更新间隔
            {
                double fps = frameCount / deltaTime;
                std::cout << "FPS: " << fps << std::endl;  // 在控制台上显示 FPS

                frameCount = 0;
                lastTime   = currentTime;
            }

            frameCount++;
        }
        test_window.windowEnd();
    }
    else
    {
        World::instance().run();
    }

    return 0;
}