//
// Created by ADMIN on 2024/3/23.
//

#include "utils/interface/model_helper.hpp"

using namespace Physika;

int main()
{
    // cube
    ParticleModelConfig cube_config;
    cube_config.particle_radius = 0.05;
    cube_config.shape           = ObjectShape::Cube;
    cube_config.lb              = { -3, -3, -3 };
    cube_config.size            = { 6, 6, 6 };
    auto cube                   = ModelHelper::create3DParticleModel(cube_config);

    // box
    ParticleModelConfig box_config;
    box_config.particle_radius = 0.05;
    box_config.shape           = ObjectShape::Box;
    box_config.lb              = { -3, -3, -3 };
    box_config.size            = { 6, 6, 6 };
    box_config.layer           = 1;
    auto box                   = ModelHelper::create3DParticleModel(box_config);

    // plane
    ParticleModelConfig plane_config;
    plane_config.particle_radius = 0.05;
    plane_config.shape           = ObjectShape::Plane;
    plane_config.lb              = { -1, 0, -1 };
    plane_config.size            = { 2, 0, 2 };
    plane_config.layer           = 3;
    auto plane                   = ModelHelper::create3DParticleModel(plane_config);

    // cylinder
    ParticleModelConfig cylinder_config;
    cylinder_config.particle_radius    = 0.05;
    cylinder_config.shape              = ObjectShape::Cylinder;
    cylinder_config.center             = { 0, 0, 0 };
    cylinder_config.bottom_area_radius = 0.8;
    cylinder_config.height             = 5;
    auto cylinder                      = ModelHelper::create3DParticleModel(cylinder_config);

    // sphere
    ParticleModelConfig sphere_config;
    sphere_config.particle_radius = 0.05;
    sphere_config.shape           = ObjectShape::Sphere;
    sphere_config.center          = { 0, 0, 0 };
    sphere_config.volume_radius   = 1;
    auto sphere                   = ModelHelper::create3DParticleModel(sphere_config);

    // export particles as ply
    //    ModelHelper::export3DModelAsPly(box, "D:/Projects/ProjectSet.1001/Physika/examples/model_helper_sample/ply_export", "box");

    // get vector<float> from vector<float3>
    auto cube_float = ModelHelper::transformVec3fSetToFloatSet(cube);
}