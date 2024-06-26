/**
 * @author     : Yuege Xiong (candybear0714@163.com)
 * @date       : 2023-11-23
 * @description: A sample of using Physika
 * @version    : 1.0
 */
#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "pbd_fluid/interface/fluid_particle_solver.hpp"

#include <iostream>

#include <vector_types.h>
#include <vector_functions.hpp>
#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>

using namespace Physika;

int main()
{
    auto scene_a = World::instance().createScene();
    auto obj_a    = World::instance().createObject();
    float particle_radius = 0.3;
    float solid_cube_x    = 80;
    float solid_cube_y    = 5;
    float solid_cube_z    = 40;
    obj_a->addComponent<PBDFluidComponent>(particle_radius, solid_cube_x, solid_cube_y, solid_cube_z);
    auto fluid_solver = World::instance().createSolver<PBDFluidParticleSolver>();
    PBDFluidParticleSolver::SolverConfig fluid_config;
    fluid_config.m_dt            = 1.f / 60.f;
    fluid_config.m_total_time = 10.f;
    fluid_solver->config(fluid_config);
    scene_a->addObject(obj_a);
    fluid_solver->attachObject(obj_a);
    // set solver boundary
    fluid_solver->setWorldBoundary(-40, -20, -20, 40, 20, 20);
    fluid_solver->setSFCoeff(1.1f);
    fluid_solver->setAdhesionCoeff(1.2f);
    scene_a->addSolver(fluid_solver);
    std::cout << "numParticles: " << obj_a->getComponent<PBDFluidComponent>()->m_num_particles << std::endl;

    // if GUI
    bool gui_test = true;
    if (gui_test)
    {
        unsigned int screenWidth  = 1440;
        unsigned int screenHeight = 900;
        int          status;
        GLWidGet     test_window(screenWidth, screenHeight, "fluid", 3, 3, status);

        if (status != 0)
        {
            return -1;
        }

        /*
         * 1. Create a renderer or more.
         */
        // Positions ptr, color_attrs ptr, num, particle_radius, color type
        float*       pos_ptr         = obj_a->getComponent<PBDFluidComponent>()->m_device_pos;
        void*        color_ptr       = obj_a->getComponent<PBDFluidComponent>()->m_device_phase;
        unsigned int num_particles   = obj_a->getComponent<PBDFluidComponent>()->m_num_particles;
        float        particle_radius = 0.3f;
        int          color_type      = attr_for_color::TYPE;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int          pos_dim         = 4;                    // set 4 if a freaky 'float4' type is used for position_ptr.

        // Renderer.
        std::shared_ptr<Renderer> drawElasticParticles =
            std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, color_type, pos_dim, "Blues.png", "shader1.vs", "shader1.fs");

        /*
         * 2. Put renderer into renderer list
         */
        // Renderer List
        std::vector<std::shared_ptr<Renderer>> drawList;
        drawList.push_back(drawElasticParticles);

        /*
         * 3. Run simulation
         */
        double total_time = 10.0;
        double cur_time   = 0.0;
        double dt         = fluid_config.m_dt;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = (cur_time + fluid_config.m_dt <= total_time) ? fluid_config.m_dt : total_time - cur_time;
            cur_time += dt;
            fluid_config.m_dt = dt;
            World::instance().step();
            // Draw particles every time step
            test_window.windowUpdate(drawList);
            // Or set some conditions:
            /*
            if (¡¤¡¤¡¤){
                test_window.windowUpdate(drawList);
            }
            */
        }
        test_window.windowEnd();
    }
    else
    {
        World::instance().run();
    }

    return 0;
}