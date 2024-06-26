//
// Created by ADMIN on 2024/3/23.
//

#ifndef PHYSIKA_MODEL_HELPER_HPP
#define PHYSIKA_MODEL_HELPER_HPP

#include <vector_types.h>
#include <iostream>
#include <vector>
#include <string>
#include <optional>

namespace Physika {

enum ObjectShape : uint8_t
{
    Cube,
    Box,
    Plane,
    Cylinder,
    Sphere
};

struct ParticleModelConfig
{
    std::optional<float> particle_radius;

    /* default building */
    std::optional<ObjectShape> shape;
    // cube/box/plane
    float3 lb{ -1, -1, -1 };
    float3 size{ 2, 2, 2 };
    // box/plane
    float layer{ 2 };
    // cylinder/sphere
    float3 center{ 0, 0, 0 };
    // cylinder
    float bottom_area_radius{ 1 };
    float height{ 1 };
    // sphere
    float volume_radius{ 1 };

    /* load 3D model */
    std::optional<std::string> ply_file;
};

class ModelHelper
{
public:
    static std::vector<float3> create3DParticleModel(ParticleModelConfig& config);

    static std::vector<float> transformVec3fSetToFloatSet(std::vector<float3>& vec3f_set);

    static void export3DModelAsPly(const std::vector<float3>& particles, const std::string& dir, const std::string& file_name);

private:
    static std::vector<float3> create3DParticleCube(float particle_radius, float3 lb, float3 size);

    static std::vector<float3> create3DParticleBox(float particle_radius, float3 lb, float3 size, float layer);

    static std::vector<float3> create3DParticlePlane(float particle_radius, float3 lb, float3 size, float layer);

    static std::vector<float3> create3DParticleCylinder(float particleRadius, float3 center, float height, float bottom_area_radius);

    static std::vector<float3> create3DParticleSphere(float particle_radius, float3 center, float volume_radius);

    //    static std::vector<float3> loadParticle3DModel(std::string ply_file);
};

}  // namespace Physika

#endif  // PHYSIKA_MODEL_HELPER_HPP
