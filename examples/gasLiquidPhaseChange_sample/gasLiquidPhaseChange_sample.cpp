/**
 * @author     : Wang Qianwei (729596003@qq.com)
 * @date       : 2023-11-17
 * @description: sample of gas-liquid phase change
 * @version    : 1.0
 */
#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "pbd_gasLiquidPhaseChange/interface/gas_liquid_phase_change_solver.hpp"
#include <iostream>
#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>
#include <vector_types.h>
#include <vector_functions.hpp>

using namespace Physika;
void loadCube(std::vector<float3>& pos, std::vector<float3>& vel, float3 lb, float size_x, float size_y, float size_z, float spacing);
int main()
{
    std::cout << "start" << std::endl;
    auto                                       scene_a = World::instance().createScene();
    auto                                       obj_a   = World::instance().createObject();

    std::cout << "config" << std::endl;
    std::cout << "createSolver" << std::endl;
    auto gas_liquid_phase_change_solver = World::instance().createSolver<GasLiquidPhaseChangeSolver>();

    //gas_liquid_phase_change_solver->config(solid_liquid_phase_change_config);
    std::cout << "create to component" << std::endl;
    obj_a->addComponent<GasLiquidPhaseChangeComponent>();

    auto obj_gasLiquidPC = obj_a->getComponent<GasLiquidPhaseChangeComponent>();

    // Set particle positions
    std::vector<float3> init_pos;
    std::vector<float3> init_vel;
    float               sampling_radius = 0.01f;
    loadCube(init_pos, init_vel, make_float3(0.1f, 0.02f, 0.1f), 0.90f, 0.84f, 0.84f, sampling_radius * 2.0f);
    obj_gasLiquidPC->initializeParticleRadius(sampling_radius);
    obj_gasLiquidPC->initializeLiquidInstance(make_float3(0.1f, 0.02f, 0.1f), make_float3(0.90f, 0.84f, 0.84f), sampling_radius * 2.0f);
    obj_gasLiquidPC->initializePartilcePosition(init_pos);
    // Init world boundary
    obj_gasLiquidPC->initializeWorldBoundary(make_float3(1.1f, 1.0f, 1.1f));
    // Initialize GasLiquidPhaseChangeComponent
    obj_gasLiquidPC->initialize();

    std::cout << "addComponent" << std::endl;
    scene_a->addObject(obj_a);

    std::cout << "attach component to solver" << std::endl;
    gas_liquid_phase_change_solver->attachObject(obj_a);

    std::cout << "add solver" << std::endl;
    scene_a->addSolver(gas_liquid_phase_change_solver);

    std::cout << "numCapacity: " << obj_a->getComponent<GasLiquidPhaseChangeComponent>()->m_num_particles << std::endl;
    std::cout << "run" << std::endl;
    // if GUI
    bool gui_test = true;
    if (gui_test)
    {
        unsigned int screenWidth  = 1440;
        unsigned int screenHeight = 900;
        int          status;
        GLWidGet     test_window(screenWidth, screenHeight, "gas_liquid_phase_change", 3, 3, status);

        if (status != 0)
        {
            return -1;
        }

        /*
         * 1. Create a renderer or more.
         */
        // Positions ptr, color_attrs ptr, num, particle_radius, color type
        float*       pos_ptr       = obj_a->getComponent<GasLiquidPhaseChangeComponent>()->getFluidParticlePositionPtr();
        void*        color_ptr     = obj_a->getComponent<GasLiquidPhaseChangeComponent>()->getFluidParticleTypePtr();
        unsigned int num_particles = obj_a->getComponent<GasLiquidPhaseChangeComponent>()->m_fluid_capacity;
        float        particle_radius;
        gas_liquid_phase_change_solver->getParticleRadius(particle_radius);
        int color_type = attr_for_color::TYPE;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int pos_dim    = 3;                           

        // Renderer.
        std::shared_ptr<Renderer> drawGasLiquidPhaseChangeParticles =
            std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, color_type, pos_dim, "rainbow.png", "shader1.vs", "shader1.fs");

        /*
         * 2. Put renderer into renderer list
         */
        // Renderer List
        std::vector<std::shared_ptr<Renderer>> drawList;
        drawList.push_back(drawGasLiquidPhaseChangeParticles);

        /*
         * 3. Run simulation
         */
        double total_time = 200.0f;
        double cur_time   = 0.0;
        double dt         = 0.003;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = 0.003;
            cur_time += dt;
            World::instance().step();
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
        obj_a->getComponent<GasLiquidPhaseChangeComponent>()->freeMemory();
    }
    else
    {
        World::instance().run();
    }

    return 0;
}

void loadCube(std::vector<float3>& pos, std::vector<float3>& vel, float3 lb, float size_x, float size_y, float size_z, float spacing)
{
    int num_x = size_x / spacing;
    int num_y = size_y / spacing;
    int num_z = size_z / spacing;
    for (auto i = 0; i < num_y; ++i)
    {
        for (auto j = 0; j < num_x; ++j)
        {
            for (auto k = 0; k < num_z; ++k)
            {
                auto x = make_float3(lb.x + spacing * j,
                                     lb.y + spacing * i,
                                     lb.z + spacing * k);
                pos.push_back(x);
                vel.push_back(make_float3(0.0f, 0.0f, 0.0f));
            }
        }
    }
}