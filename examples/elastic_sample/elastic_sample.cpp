/*
 * @Author: pgpgwhp 1213388412@qq.com
 * @Date: 2023-09-21 11:02:56
 * @LastEditors: pgpgwhp 1213388412@qq.com
 * @LastEditTime: 2023-09-21 16:16:15
 * @FilePath: \physika\examples\elastic_sample\elastic_sample.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */

#include <iostream>
#include <vector_types.h>
#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "pbd_elastic/pbd_elastic_solver.hpp"
#include "pbd_elastic/pbd_elastic_params.hpp"
#include <gl_particle_render/glWindow/glWidGet.h>
#include <gl_particle_render/renderer/cuParticleRenderer.h>


using namespace Physika;

int main() {
    float3 lb       = make_float3(-10.0f, -10.0f, -10.0f);
    float3 rt       = make_float3(10.0f, 10.0f, 10.0f);

    auto scene_a = World::instance().createScene();
    auto obj_a    = World::instance().createObject();
    float radius  = 0.1;
    obj_a->addComponent<ElasticComponent>(radius, lb, rt);
    // set demo
    float3 lb0 = make_float3(0.f, 0.0f, 0.f);
    float3 rt0 = make_float3(6.f, 8.f, 6.f);
    float3 lb1 = make_float3(-6.f, -7.0f, -6.f);
    float3 rt1 = make_float3(0.f, -1.f, 0.f);
    float3 lb2 = make_float3(-7.f, 0.0f, -7.f);
    float3 rt2 = make_float3(-1.f, 6.f, -1.f);
    float3 lb3 = make_float3(1.f, -9.0f, 1.f);
    float3 rt3 = make_float3(9.f, -1.f, 9.f);

    //obj_a->getComponent<ElasticComponent>()->demo1();
    //obj_a->getComponent<ElasticComponent>()->addInstance();
    obj_a->getComponent<ElasticComponent>()->addInstance(PARTICLE_PHASE::SOLID, radius, lb0, rt0);
    obj_a->getComponent<ElasticComponent>()->addInstance(PARTICLE_PHASE::SOLID, radius, lb1, rt1);
    obj_a->getComponent<ElasticComponent>()->addInstance(PARTICLE_PHASE::SOLID, radius, lb2, rt2);
    obj_a->getComponent<ElasticComponent>()->addInstance(PARTICLE_PHASE::SOLID, radius, lb3, rt3);
    
    auto elastic_solver = World::instance().createSolver<ElasticParticleSovler>();
    ElasticParticleSovler::SolverConfig elastic_config;
    elastic_config.m_dt         = 1.f / 180.f;
    elastic_config.m_solver_iteration = 5;
    elastic_config.m_total_time = 5.f;
    elastic_solver->config(elastic_config);
    scene_a->addObject(obj_a);
    elastic_solver->attachObject(obj_a);
    // set young_modules and possion_ratio
    elastic_solver->setYoungModules(3000);
    elastic_solver->setPossionRatio(0.2);
    elastic_solver->setWriteToPly(false);

    scene_a->addSolver(elastic_solver);

    std::cout << "numParticles: " <<  obj_a->getComponent<ElasticComponent>()->m_num_particle << std::endl;
    
    // if GUI
    bool         gui_test     = true;
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
        float*       pos_ptr       = obj_a->getComponent<ElasticComponent>()->m_device_pos;
        void*        color_ptr     = obj_a->getComponent<ElasticComponent>()->m_device_vel;
        unsigned int num_particles = obj_a->getComponent<ElasticComponent>()->m_num_particle;
        float        particle_radius = obj_a->getComponent<ElasticComponent>()->m_params->m_particle_radius;
        int          color_type    = attr_for_color::VEL; // Alternative. See enum 'attr_for_color' and 'ColorConfig'
        int          pos_dim       = 3;  // set 4 if a freaky 'float4' type is used for position_ptr.
        
        // Renderer. 
        std::shared_ptr<Renderer> drawElasticParticles =
            std::make_shared<CUParticleRenderer>(pos_ptr, color_ptr, num_particles, particle_radius, 
                 color_type, pos_dim, "Blues.png", "shader1.vs", "shader1.fs");
        
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
        double dt         = elastic_config.m_dt;
        while (cur_time < total_time && test_window.windowCheck())
        {
            double dt = (cur_time + elastic_config.m_dt <= total_time) ? elastic_config.m_dt : total_time - cur_time;
            cur_time += dt;
            elastic_config.m_dt = dt;
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
    return 0;
}