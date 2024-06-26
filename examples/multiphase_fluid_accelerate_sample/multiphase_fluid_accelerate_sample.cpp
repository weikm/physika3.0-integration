/**
 * @author     : Yan Xiao xiao
 * @date       : 2023-07-09
 * @description: sample code to demonstrate how the (world, scene, object, solver) framework of Physika works
 * @version    : 1.0
 */

#include <iostream>
#include <fstream>
#include "framework/world.hpp"
#include "framework/scene.hpp"
#include "framework/object.hpp"
#include "multiphase_fluids_accelerate/interface/multiphase_fluid_solver.hpp"
using namespace Physika;
/**
 * Description of the World setup:
 * -1 scene scene_p: this scence is for testing porous multiphase solver
 * -1 solver porous_solver: multiphase solver
 * -1 object obj_p: one object with three components
 *      -PorousRigidComponent: rigid paticles
 *      -PorousSandComponent: sand particles
 *      -porousWaterComponent: water particles
 * 
 * other description: this scene need boundary component which is initialized with coordinate, it is initialized internally 
 */
//load particles from txt file
bool loadParticleFromOutside(const char* filename,float*mpos,int &pnum)
{
    std::fstream f;
    f.open(filename, std::ios::in);
    int num = 0;
    f >> num;
    if (num == 0||mpos==nullptr)
    {
        std::cout << "file " << filename << " not open\n";
        pnum = 0;
        return false;
    }
    else
    {
        std::cout << "file open, sand particles num: " << num << std::endl;
        pnum = num;
    }
    float x, y, z;
    int   n = 0;
    for (int i = 0; i < num; i++)
    {

        f >> x >> y >> z;
        mpos[i*3] = x;
        mpos[i*3 + 1] = y;
        mpos[i*3 + 2] = z;
        if (i == 1000)
            std::cout << "BUG File position test: x " << x << " y " << y << " z " << z << std::endl;
    }
    f.close();
    return true;
}
//save particles to txt file
bool saveFile(const char* filename,float* mpos,float* mclr,int pnum) {
    std::fstream f;
    f.open(filename, std::fstream::out);
    if (!f.is_open())
    {
        std::cout << "file not open" << std::endl;
        return false;
    }
    for (int i = 0; i < pnum; i++)
    {
        float x, y, z;
        float r, g, b, a;
        x = mpos[i * 3 + 0];
        y = mpos[i * 3 + 1];
        z = mpos[i * 3 + 2];
        r = mclr[i * 4 + 0];
        g = mclr[i * 4 + 1];
        b = mclr[i * 4 + 2];
        a = mclr[i * 4 + 3];
        f << x << ' ' << y << ' ' << z << ' ' << r << ' ' << g << ' ' << b << ' ' << a << '\n';
        if (i % (pnum/10)==0)
        {
            std::cout << "* ";
        }
    }
    std::cout << '\n';
    return true;
}
int  main(int argc, char* argv[])
{
    int totalnum = 300000;
    cudaInit(argc,argv);
    auto scene_p = World::instance().createScene();
    auto obj_p   = World::instance().createObject();
    auto porous_solver = World::instance().createSolver<MultiphaseFluidSolver>();
    MultiphaseFluidSolver::SolverConfig p_config;
    p_config.m_dt = 0.0005;
    p_config.m_total_time = 0.005;
    obj_p->addComponent<MultiphaseFluidComponent>();
    porous_solver->config(p_config);
    scene_p->addSolver(porous_solver);
    scene_p->addObject(obj_p);
    
    std::cout << "Scene P: id(" << scene_p->id() << ").\n";
    std::cout << "    Object P: id(" << obj_p->id() << "), with Components.\n";
    std::cout << "    PorousSolver: id(" << porous_solver->id() << ").\n";
    std::cout << "End\n";
    std::cout << "\nWorld::run() is called.\n";
    World::instance().run();  // update the world
    return 0;
}