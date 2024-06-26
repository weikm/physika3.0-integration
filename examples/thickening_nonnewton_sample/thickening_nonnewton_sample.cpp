//@author        : Long Shen
//@date          : 2023/10/7
//@description   :
//@version       : 1.0

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"

/** NOTE
 * define CROSS_N depends on your requirement
 * CROSS_N: [2,5] simulate shear-thinning fluid
 * CROSS_N: [-5,-2] simulate shear-thickening fluid
 * not defined will lead to viscous newTon fluid
 */
#define CROSS_N -5

#include "pbf_nonnewton/interface/pbf_nonnewton_solver.hpp"

#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>

#include "pbf_nonnewton/source/default_model_tool.cuh"


using namespace Physika;

int main()
{
    auto scene_a = World::instance().createScene();

    float particle_radius = 0.05f;
    auto  obj_a           = World::instance().createObject();
    obj_a->addComponent<NonNewtonFluidComponent>(SimMaterial::FLUID);
    auto cube_particles = generate_cube({ -1, -3, -1 }, { 2, 6.5, 2 }, particle_radius);
    obj_a->getComponent<NonNewtonFluidComponent>()->addParticles(cube_particles);

    auto obj_b = World::instance().createObject();
    obj_b->addComponent<NonNewtonFluidComponent>(SimMaterial::RIGID);
    auto box_particles   = generate_box({ -4, -4, -4 }, { 8, 8, 8 }, particle_radius);
    auto plane_particles = generate_plane_X({ -9, -3.5, -9 }, { 18, 6, 18 }, particle_radius);
    obj_b->getComponent<NonNewtonFluidComponent>()->addParticles(plane_particles);

    scene_a->addObject(obj_a);
    scene_a->addObject(obj_b);

    auto solver = World::instance().createSolver<PBFNonNewtonSolver>();
    solver->setUnifiedParticleRadius(particle_radius);

    PBFNonNewtonSolver::SolverConfig solverConfig{};
    solverConfig.m_dt       = 0.002;
    solverConfig.m_iter_num = 2;
    solver->config(solverConfig);

    scene_a->addSolver(solver);

    // if GUI
    bool gui_test = true;
    if (gui_test)
    {
        // for qw render, otherwise delete
        solver->attachObject(obj_a);
        solver->attachObject(obj_b);
        solverConfig.m_use_qwUI = true;
        solver->config(solverConfig);
        solver->initialize();

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
        auto         pbfSolver     = static_cast<PBFNonNewtonSolver*>(solver);
        float*       pos_ptr       = reinterpret_cast<float*>(pbfSolver->getPosDevPtr());
        void*        color_ptr     = pbfSolver->getVisDevPtr();
        unsigned int num_particles = obj_a->getComponent<NonNewtonFluidComponent>()->getParticleNum();
        //        float        particle_radius = obj_a->getComponent<NNComponent>()->m_params->m_particle_radius;
        int color_type = attr_for_color::VEL;  // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int pos_dim    = 3;                    // set 4 if a freaky 'float4' type is used for position_ptr.

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
        double total_time = 6.0;
        double cur_time   = 0.0;
        double dt         = solverConfig.m_dt;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = (cur_time + solverConfig.m_dt <= total_time) ? solverConfig.m_dt : total_time - cur_time;
            cur_time += dt;
            solverConfig.m_dt = dt;
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
    }
    else
    {
        World::instance().run();
    }

    World::instance().destroyAllSolvers();

    return 0;
}