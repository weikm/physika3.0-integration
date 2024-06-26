/**
 * @author     : Zhu Fei (zhufei@simversus.com)
 * @date       : 2023-03-23
 * @description: sample code to demonstrate how the (world, scene, object, solver) framework of Physika works
 * @version    : 1.0
 */

#include <iostream>

#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "single_obj_solver.hpp"
#include "dummy_sfi_solver.hpp"
#include "dummy_cuda_solver.hpp"

using namespace Physika;
/**
 * Description of the World setup:
 * - 2 scenes (Scene A and Scene B)
 * - 3 solvers (SingleObjSolver in both scenes, DummyCudaSolver in Scene A, and DummySFISolver in Scene B)
 * - 6 objects
 *   - Object A: with DummyComponent in Scene A
 *   - Object B: without DummyComponent in Scene A
 *   - Object C: with SolidComponent in Scene B
 *   - Object D: with FluidComponent in Scene B
 *   - Object E: with DummyComponent in Scene B
 *   - Object F: with ThrustComponent in Scene A
 * The world runs for 2 seconds.
 */
int main()
{
    auto scene_a = World::instance().createScene();
    auto obj_a   = World::instance().createObject();
    auto obj_b   = World::instance().createObject();
    auto obj_f   = World::instance().createObject();
    obj_a->addComponent<DummyComponent>();
    obj_f->addComponent<ThrustComponent>();
    obj_f->getComponent<ThrustComponent>()->resize(5, 0.0);
    scene_a->addObject(obj_a);
    scene_a->addObject(obj_b);
    scene_a->addObject(obj_f);
    auto                          single_obj_solver = World::instance().createSolver<SingleObjSolver>();
    SingleObjSolver::SolverConfig so_config;
    so_config.m_dt         = 0.1;
    so_config.m_total_time = 2.0;
    single_obj_solver->config(so_config);
    scene_a->addSolver(single_obj_solver);
    auto                          dummy_cuda_solver = World::instance().createSolver<DummyCudaSolver>();
    DummyCudaSolver::SolverConfig cu_config;
    cu_config.m_dt         = 0.1;
    cu_config.m_total_time = 2.0;
    cu_config.m_gravity    = 9.8;
    cu_config.m_use_gpu    = true;
    dummy_cuda_solver->config(cu_config);
    scene_a->addSolver(dummy_cuda_solver);
    std::cout << "Scene A: id(" << scene_a->id() << ").\n";
    std::cout << "    Object A: id(" << obj_a->id() << "), with DummyComponent.\n";
    std::cout << "    Object B: id(" << obj_b->id() << "), without DummyComponent.\n";
    std::cout << "    Object F: id(" << obj_f->id() << "), with ThrustComponent.\n";
    std::cout << "    SingleObjSolver: id(" << single_obj_solver->id() << ").\n";
    std::cout << "    DummyCudaSolver: id(" << dummy_cuda_solver->id() << ").\n";
    std::cout << "End\n";

    auto scene_b = World::instance().createScene();
    auto obj_c   = World::instance().createObject();
    obj_c->addComponent<SolidComponent>();
    scene_b->addObject(obj_c);
    auto obj_d = World::instance().createObject();
    obj_d->addComponent<FluidComponent>();
    scene_b->addObject(obj_d);
    auto obj_e = World::instance().createObject();
    obj_e->addComponent<DummyComponent>();
    scene_b->addObject(obj_e);
    scene_b->addSolver(single_obj_solver);
    auto                         sfi_solver = World::instance().createSolver<DummySFISolver>();
    DummySFISolver::SolverConfig sfi_config;
    sfi_config.m_dt         = 0.1;
    sfi_config.m_total_time = 2.0;
    sfi_solver->config(sfi_config);
    scene_b->addSolver(sfi_solver);
    std::cout << "Scene B: id(" << scene_b->id() << ").\n";
    std::cout << "    Object C: id(" << obj_c->id() << "), with SolidComponent.\n";
    std::cout << "    Object D: id(" << obj_d->id() << "), with FluidComponent.\n";
    std::cout << "    Object E: id(" << obj_e->id() << "), with DummyComponent.\n";
    std::cout << "    SingleObjSolver: id(" << single_obj_solver->id() << ").\n";
    std::cout << "    DummySFISolver: id(" << sfi_solver->id() << ").\n";
    std::cout << "End\n";

    std::cout << "\nWorld::run() is called.\n";
    World::instance().run();  // update the world

    World::instance().clear();

    return 0;
}