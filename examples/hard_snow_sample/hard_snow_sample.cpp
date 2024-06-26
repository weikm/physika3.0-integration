/**
 * @author     : Yuanmu Xu (xyuan1517@gmail.com)
 * @date       : 2023-06-10
 * @description: A sample of using snow solver
 * @version    : 1.0
 */

#include "mpm_snow/Interface/mpm_snow.hpp"

#include <iostream>

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>
#include <vector_types.h>

using namespace Physika;

int main()
{
    auto scene_a = World::instance().createScene();
    auto obj_a   = World::instance().createObject();
    obj_a->addComponent<MPMSnowComponent>();
    auto component = obj_a->getComponent<MPMSnowComponent>();

    ParticleModelConfig snow_component_config;
    snow_component_config.shape           = ObjectShape::Cube;
    snow_component_config.particle_radius = 0.03;
    snow_component_config.lb              = { 0.3, 0.3, 0.3 };
    snow_component_config.size            = { 1.f, 1.f, 1.f };
    auto                snow_init_pos = ModelHelper::create3DParticleModel(snow_component_config);
    std::vector<float3> snow_vel;
    snow_vel.push_back({ 0, 0, 0 });
    // add by init pos
    component->addInstance(
        snow_init_pos,
        snow_vel);
    
    std::vector<float>          world_boundaries = { 0, 0, 0, 10, 10, 10 };
    auto                        snow_solver      = World::instance().createSolver<MPMSnowSolver>(world_boundaries);
    MPMSnowSolver::SolverConfig snow_config;
    snow_config.m_total_time = 10.f;
    snow_config.m_dt         = 1.0 / 60.0;
    snow_config.m_write2ply  = false;
    snow_config.m_showGUI    = !snow_config.m_write2ply;
    snow_solver->config(snow_config);
    scene_a->addObject(obj_a);
    snow_solver->attachObject(obj_a);
    snow_solver->setStick(false);
    snow_solver->setYoungsModulus(1.4e5);
    snow_solver->setPoissonRatio(0.2);
    snow_solver->setHardeningCeoff(10);
    snow_solver->setStretch(0.0025f);
    snow_solver->setCompressionCoeff(0.001f);
    scene_a->addSolver(snow_solver);
    std::cout << obj_a->getComponent<MPMSnowComponent>()->m_numParticles << std::endl;

    // if GUI
    bool gui_test = snow_config.m_showGUI;
    if (gui_test)
    {
        unsigned int screenWidth  = 1440;
        unsigned int screenHeight = 900;
        int          status;
        GLWidGet     test_window(screenWidth, screenHeight, "Elastic", 3, 3, status);

        if (status != 0)
        {
            return -1;
        }

        /*
         * 1. Create a renderer or more.
         */
        // Positions ptr, color_attrs ptr, num, particle_radius, color type
        float*       pos_ptr         = obj_a->getComponent<MPMSnowComponent>()->m_devicePos;
        void*        color_ptr       = obj_a->getComponent<MPMSnowComponent>()->m_deviceVel;
        unsigned int num_particles   = obj_a->getComponent<MPMSnowComponent>()->m_numParticles;
        float        particle_radius = 0.01f;
        int          color_type      = attr_for_color::VEL;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int          pos_dim         = 3;                    // set 4 if a freaky 'float4' type is used for position_ptr.

        // Renderer.
        std::shared_ptr<Renderer> drawElasticParticles =
            std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, color_type, pos_dim);

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
        double dt         = snow_config.m_dt;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = (cur_time + snow_config.m_dt <= total_time) ? snow_config.m_dt : total_time - cur_time;
            cur_time += dt;
            snow_config.m_dt = dt;
            for (int i = 0; i < 10; i++)
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