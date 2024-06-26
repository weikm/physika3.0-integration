/**
 * @author     : Ruolan Li (3230137958@qq.com)
 * @date       : 2023-11-17
 * @description: Radiate Sample of Solid-liquid phase change simulation
 * @version    : 1.0
 */
#include "pbd_solidLiquidPhaseChange/interface/solid_liquid_phase_change_solver.hpp"

#include <iostream>

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>
#include <vector_types.h>
#include <vector_functions.hpp>
#include "pbd_solidLiquidPhaseChange/source/default_model_tool.cuh"

using namespace Physika;

int main()
{
    std::cout << "start" << std::endl;
    auto                                       scene_a = World::instance().createScene();
    auto                                       obj_a   = World::instance().createObject();
    SolidLiquidPhaseChangeSolver::SolverConfig solid_liquid_phase_change_config;

    std::cout << "addComponent" << std::endl;
    obj_a->addComponent<SolidLiquidPhaseChangeComponent>();
    auto                component = obj_a->getComponent<SolidLiquidPhaseChangeComponent>();
    std::vector<float3> init_pos;

    init_pos = generate_cube({ 0, 0, 0 }, { 30, 21, 30 }, 0.3);
    component->addInstance(0.3, init_pos, static_cast<int>(SLPCPhase::SOLID), -20.f);

    std::cout << "createSolver" << std::endl;
    auto solid_liquid_phase_change_solver = World::instance().createSolver<SolidLiquidPhaseChangeSolver>();
    scene_a->addObject(obj_a);
    solid_liquid_phase_change_solver->attachObject(obj_a);
    solid_liquid_phase_change_solver->setWorldBoundary(-6, 0, -6, 30, 28, 30);
    auto is_radiate = true;
    solid_liquid_phase_change_solver->setParam("m_radiate", &is_radiate);

    std::cout << "numParticles: " << obj_a->getComponent<SolidLiquidPhaseChangeComponent>()->m_num_particles << std::endl;
    std::cout << "run" << std::endl;
    // if GUI
    bool gui_test = true;
    if (gui_test)
    {
        unsigned int screenWidth  = 1440;
        unsigned int screenHeight = 900;
        int          status;
        GLWidGet     test_window(screenWidth, screenHeight, "solid_liquid_phase_change", 3, 3, status);

        if (status != 0)
        {
            return -1;
        }

        /*
         * 1. Create a renderer or more.
         */
        // Positions ptr, color_attrs ptr, num, particle_radius, color type
        float*       pos_ptr       = obj_a->getComponent<SolidLiquidPhaseChangeComponent>()->m_device_pos;
        void*        color_ptr     = obj_a->getComponent<SolidLiquidPhaseChangeComponent>()->m_device_tem;
        unsigned int num_particles = obj_a->getComponent<SolidLiquidPhaseChangeComponent>()->m_num_particles;
        float        particle_radius;
        solid_liquid_phase_change_solver->getParticleRadius(particle_radius);
        int color_type = attr_for_color::TEMPERATURE;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int pos_dim    = 4;                            // set 4 if a freaky 'float4' type is used for position_ptr.

        // Renderer.
        std::shared_ptr<Renderer> drawSolidLiquidPhaseChangeParticles =
            std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, color_type, pos_dim, "rainbow.png", "shader1.vs", "shader1.fs");

        /*
         * 2. Put renderer into renderer list
         */
        // Renderer List
        std::vector<std::shared_ptr<Renderer>> drawList;
        drawList.push_back(drawSolidLiquidPhaseChangeParticles);

        /*
         * 3. Run simulation
         */
        double total_time = solid_liquid_phase_change_config.m_total_time;
        double cur_time   = 0.0;
        double dt         = solid_liquid_phase_change_config.m_dt;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = (cur_time + solid_liquid_phase_change_config.m_dt <= total_time) ? solid_liquid_phase_change_config.m_dt : total_time - cur_time;
            cur_time += dt;
            solid_liquid_phase_change_config.m_dt = dt;
            //World::instance().step();
            solid_liquid_phase_change_solver->step();
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
    else
    {
        World::instance().run();
    }

    return 0;
}