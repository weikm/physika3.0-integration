/**
 * @author     : Ruolan Li (3230137958@qq.com)
 * @date       : 2023-11-17
 * @description: Melt Sample of Solid-liquid phase change simulation
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

using namespace Physika;

int main()
{
    std::cout << "start" << std::endl;
    auto                                       scene_a = World::instance().createScene();
    auto                                       obj_a   = World::instance().createObject();
    SolidLiquidPhaseChangeSolver::SolverConfig solid_liquid_phase_change_config;
    // set scene parameters
    solid_liquid_phase_change_config.m_dt               = 1.f / 60.f;
    solid_liquid_phase_change_config.m_is_convect       = true;
    //solid_liquid_phase_change_config.m_world_size       = make_float3(14.0f, 25.0f, 14.0f);
    solid_liquid_phase_change_config.m_fluid_size       = make_float3(28.0f, 21.0f, 28.0f);
    solid_liquid_phase_change_config.m_fluid_locate     = make_float3(28.0f, 50.0f, 28.0f);
    solid_liquid_phase_change_config.m_fluid_tem        = 75.0f;
    solid_liquid_phase_change_config.m_solid_size       = make_float3(14.0f, 14.0f, 14.0f);
    solid_liquid_phase_change_config.m_solid_locate     = make_float3(14.0f, 0.0f, 14.0f);
    solid_liquid_phase_change_config.m_solid_tem        = 0.0f;
    solid_liquid_phase_change_config.m_boundary_size    = make_float3(0.0f, 0.0f, 0.0f);
    solid_liquid_phase_change_config.m_boundary_locate  = make_float3(0.0f, 0.0f, 0.0f);
    solid_liquid_phase_change_config.m_boundary_tem     = 22.0f;
    solid_liquid_phase_change_config.m_total_time       = 100;
    solid_liquid_phase_change_config.m_write_ply        = false;
    solid_liquid_phase_change_config.m_write_statistics = false;
    solid_liquid_phase_change_config.m_radiate          = false;

    std::cout << "config" << std::endl;
    std::cout << "createSolver" << std::endl;
    auto solid_liquid_phase_change_solver = World::instance().createSolver<SolidLiquidPhaseChangeSolver>();

    solid_liquid_phase_change_solver->config(solid_liquid_phase_change_config);
    std::cout << "addComponent" << std::endl;
    obj_a->addComponent<SolidLiquidPhaseChangeComponent>(static_cast<void*>(solid_liquid_phase_change_solver->m_params));
    scene_a->addObject(obj_a);
    solid_liquid_phase_change_solver->attachObject(obj_a);
    solid_liquid_phase_change_solver->setWorldBoundary(-14, -25, -14, 14, 25, 14);
    scene_a->addSolver(solid_liquid_phase_change_solver);

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
            // World::instance().step();
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