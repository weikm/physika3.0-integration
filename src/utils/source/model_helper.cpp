//
// Created by ADMIN on 2024/3/23.
//

#include <fstream>
#include <filesystem>

#include "model_helper.hpp"

// #include "assimp/Importer.hpp"
// #include "assimp/postprocess.h"
// #include "assimp/scene.h"

namespace Physika {

std::vector<float3> ModelHelper::create3DParticleModel(ParticleModelConfig& config)
{
    static std::vector<float3> particles;

    if (!config.particle_radius.has_value())
    {
        config.particle_radius = 0.1;
        std::cout << "Warning: Not specified particle radius, use default 0.1.\n";
    }

    if (!config.shape.has_value() && !config.ply_file.has_value())
    {
        config.shape = ObjectShape::Cube;
        std::cout << "Warning: Not specified object shape and ply model file, use default Cube shape.\n";
    }

    if (config.shape.has_value())
    {

        switch (config.shape.value())
        {
            case Cube:
                particles = create3DParticleCube(config.particle_radius.value(),
                                                 config.lb,
                                                 config.size);
                break;
            case Box:
                particles = create3DParticleBox(config.particle_radius.value(),
                                                config.lb,
                                                config.size,
                                                config.layer);
                break;
            case Plane:
                particles = create3DParticlePlane(config.particle_radius.value(),
                                                  config.lb,
                                                  config.size,
                                                  config.layer);
                break;
            case Cylinder:
                particles = create3DParticleCylinder(config.particle_radius.value(),
                                                     config.center,
                                                     config.height,
                                                     config.bottom_area_radius);
                break;
            case Sphere:
                particles = create3DParticleSphere(config.particle_radius.value(),
                                                   config.center,
                                                   config.volume_radius);
                break;
            default:
                std::cout << "Ops! No matching shape.\n";
                break;
        }
    }

    //    if (m_particleObjectConfig->model_file.has_value())
    //    {
    //        m_particles = ModelHelper::loadParticle3DModel(m_particleObjectConfig->model_file.value());
    //    }

    return particles;
}

void ModelHelper::export3DModelAsPly(const std::vector<float3>& particles, const std::string& dir, const std::string& file_name)
{
    auto dir_ = dir;
#if defined(WIN32)
    size_t pos = 0;
    while ((pos = dir_.find('/', pos)) != std::string::npos)
    {
        dir_.replace(pos, 1, "\\");
        pos += 1;
    }
#endif

    if (!std::filesystem::exists(dir_))
        std::filesystem::create_directories(dir_);

    std::ofstream ofs(dir_ + "\\" + file_name + ".ply");

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << particles.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";

    for (const auto& particle : particles)
    {
        ofs << particle.x << " " << particle.y << " " << particle.z << "\n";
    }

    ofs.close();
}

std::vector<float3> ModelHelper::create3DParticleCube(float particle_radius, float3 lb, float3 size)
{
    std::vector<float3> particles;
    auto                diameter = 2 * particle_radius;

    for (float z = particle_radius + lb.z; z < lb.z + size.z; z += diameter)
    {
        for (float y = particle_radius + lb.y; y < lb.y + size.y; y += diameter)
        {
            for (float x = particle_radius + lb.x; x < lb.x + size.x; x += diameter)
            {
                float3 _particles = { x, y, z };

                particles.push_back(_particles);
            }
        }
    }

    return particles;
}

std::vector<float3> ModelHelper::create3DParticleBox(float particle_radius, float3 lb, float3 size, float layer)
{
    std::vector<float3> particles;

    int numParticles[] = {
        static_cast<int>(size.x / (2.0 * particle_radius)),
        static_cast<int>(size.y / (2.0 * particle_radius)),
        static_cast<int>(size.z / (2.0 * particle_radius))
    };

    for (int i = 0; i < numParticles[0]; ++i)
    {
        for (int j = 0; j < numParticles[1]; ++j)
        {
            for (int k = 0; k < numParticles[2]; ++k)
            {
                // If this particle is in the first two or last two layers in any dimension...
                if (i < layer || i >= numParticles[0] - layer || j < layer || j >= numParticles[1] - layer || k < layer || k >= numParticles[2] - layer)
                {
                    float3 p;
                    p.x = static_cast<float>(lb.x + particle_radius + 2.0 * particle_radius * i);
                    p.y = static_cast<float>(lb.y + particle_radius + 2.0 * particle_radius * j);
                    p.z = static_cast<float>(lb.z + particle_radius + 2.0 * particle_radius * k);
                    particles.push_back(p);
                }
            }
        }
    }

    return particles;
}

std::vector<float3> ModelHelper::create3DParticlePlane(float particle_radius, float3 lb, float3 size, float layer)
{
    std::vector<float3> particles;
    auto                diameter = 2 * particle_radius;

    for (float z = particle_radius + lb.z; z < lb.z + size.z; z += diameter)
    {
        for (float y = particle_radius + lb.y, cnt = 0; cnt < layer; y += diameter, cnt += 1)
        {
            for (float x = particle_radius + lb.x; x < lb.x + size.x; x += diameter)
            {
                float3 _particles = { x, y, z };

                particles.push_back(_particles);
            }
        }
    }

    return particles;
}

std::vector<float3> ModelHelper::create3DParticleCylinder(float  particleRadius,
                                                          float3 center,
                                                          float  height,
                                                          float  bottom_area_radius)
{
    std::vector<float3> particles;
    float               diameter  = 2 * particleRadius;
    float3              topCenter = { center.x, center.y + height / 2, center.z };
    float               y0        = topCenter.y;

    for (float y = y0 - particleRadius; y >= y0 - height; y -= diameter)
    {
        float x0 = topCenter.x - bottom_area_radius;

        for (float x = x0 + particleRadius; x <= topCenter.x + bottom_area_radius; x += diameter)
        {

            float m_cos  = fabs(topCenter.x - x) / bottom_area_radius;
            float length = bottom_area_radius * sqrt(1 - m_cos * m_cos);
            float z0     = topCenter.z - length;
            for (float z = z0 + particleRadius; z <= topCenter.z + length; z += diameter)
            {
                float3 particle = { x, y, z };
                particles.push_back(particle);
            }
        }
    }

    return particles;
}

std::vector<float3> ModelHelper::create3DParticleSphere(float particle_radius, float3 center, float volume_radius)
{
    std::vector<float3> particles;
    // 计算粒子间的间距，假设粒子紧密排列
    float gap = particle_radius * 2.0f;

    // 计算球体内的立方体边界
    int num_particles_per_side = std::ceil(volume_radius / gap);
    for (int i = -num_particles_per_side; i <= num_particles_per_side; ++i)
    {
        for (int j = -num_particles_per_side; j <= num_particles_per_side; ++j)
        {
            for (int k = -num_particles_per_side; k <= num_particles_per_side; ++k)
            {
                float3 particle = { float(i) * gap + center.x, float(j) * gap + center.y, float(k) * gap + center.z };
                // 检查粒子是否在球体内
                if ((particle.x - center.x) * (particle.x - center.x) + (particle.y - center.y) * (particle.y - center.y) + (particle.z - center.z) * (particle.z - center.z) <= volume_radius * volume_radius)
                {
                    particles.push_back(particle);
                }
            }
        }
    }

    return particles;
}
std::vector<float> ModelHelper::transformVec3fSetToFloatSet(std::vector<float3>& vec3f_set)
{
    std::vector<float> particles;
    particles.resize(3 * vec3f_set.size());
    memcpy(particles.data(), vec3f_set.data(), vec3f_set.size() * sizeof(float3));
    return particles;
}

// std::vector<float3> ModelHelper::loadParticle3DModel(std::string ply_file)
//{
//     std::vector<float3> particles;
//
//     Assimp::Importer importer;
//     const aiScene*   scene = importer.ReadFile(ply_file, aiProcess_Triangulate);
//
//     if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
//     {
//         std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
//         return {};
//     }
//
//     for (unsigned int m = 0; m < scene->mNumMeshes; m++)
//     {
//         const aiMesh* mesh = scene->mMeshes[m];
//         for (unsigned int v = 0; v < mesh->mNumVertices; v++)
//         {
//             const aiVector3D& vertex     = mesh->mVertices[v];
//             float3            _particles = { vertex.x, vertex.y, vertex.z };
//             particles.emplace_back(_particles);
//         }
//     }
//
//     return particles;
// }

}  // namespace Physika
