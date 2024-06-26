//@author        : Long Shen
//@date          : 2023/9/30
//@description   :
//@version       : 1.0

#ifndef PHYSIKA_DEFAULT_MODEL_TOOL_H
#define PHYSIKA_DEFAULT_MODEL_TOOL_H

#include <vector_types.h>

namespace Physika {

/**
 * @brief  : generate default cylinder shape
 *
 * @param[in]  : topCenter  center pos of top plane
 * @param[in]  : height  height of the cylinder
 * @param[in]  : areaRadius    area of bottom plane
 * @param[in]  : particleRadius    particle radius
 *
 * @return    a set of particle pos
 */
std::vector<float3>
generate_default_cylinder(std::vector<float> topCenter, float height, float areaRadius, float particleRadius)
{

    std::vector<float3> pos;
    float               diameter = 2 * particleRadius;
    float               y0       = topCenter[1];

    for (float y = y0 - particleRadius; y >= y0 - height; y -= diameter)
    {
        float x0 = topCenter[0] - areaRadius;

        for (float x = x0 + particleRadius; x <= topCenter[0] + areaRadius; x += diameter)
        {

            float m_cos  = fabs(topCenter[0] - x) / areaRadius;
            float length = areaRadius * sqrt(1 - m_cos * m_cos);
            float z0     = topCenter[2] - length;
            for (float z = z0 + particleRadius; z <= topCenter[2] + length; z += diameter)
            {
                float3 point{ x, y, z };

                pos.push_back(point);
            }
        }
    }

    return pos;
}

/**
 * @brief  : generate default cube shape
 *
 * @param[in]  : cubeLB  left-bottom coordinate of cube
 * @param[in]  : cubeSize  size of cube
 * @param[in]  : particleRadius    particle radius
 *
 * @return    a set of particle pos
 */
std::vector<float3>
generate_cube(float3 cubeLB, float3 cubeSize, float particleRadius) {
    std::vector<float3> pos;
    auto diameter = 2 * particleRadius;

    for (float z = particleRadius + cubeLB.z; z < cubeLB.z + cubeSize.z; z += diameter) {
        for (float y = particleRadius + cubeLB.y; y < cubeLB.y + cubeSize.y; y += diameter) {
            for (float x = particleRadius + cubeLB.x; x < cubeLB.x + cubeSize.x; x += diameter) {
                float3 _pos = make_float3(x, y, z);

                pos.push_back(_pos);
            }
        }
    }

    return pos;
}

/**
 * @brief  : generate default box shape
 *
 * @param[in]  : boxLB  left-bottom coordinate of box
 * @param[in]  : boxSize  size of box
 * @param[in]  : particleRadius    particle radius
 *
 * @return    a set of particle pos
 */
std::vector<float3>
generate_box(float3 boxLB, float3 boxSize, float particleRadius) {
    std::vector<float3> pos;

    int numParticles[] = {
        static_cast<int>(boxSize.x / (2.0 * particleRadius)),
        static_cast<int>(boxSize.y / (2.0 * particleRadius)),
        static_cast<int>(boxSize.z / (2.0 * particleRadius))
    };

    for (int i = 0; i < numParticles[0]; ++i) {
        for (int j = 0; j < numParticles[1]; ++j) {
            for (int k = 0; k < numParticles[2]; ++k) {
                // If this particle is in the first two or last two layers in any dimension...
                if (i < 2 || i >= numParticles[0] - 2 || j < 2 || j >= numParticles[1] - 2 || k < 2 ||
                    k >= numParticles[2] - 2) {
                    float3 p;
                    p.x = boxLB.x + particleRadius + 2.0 * particleRadius * i;
                    p.y = boxLB.y + particleRadius + 2.0 * particleRadius * j;
                    p.z = boxLB.z + particleRadius + 2.0 * particleRadius * k;
                    pos.push_back(p);
                }
            }
        }
    }

    return pos;
}

/**
 * @brief  : generate default plane along the X-axis
 *
 * @param[in]  : planeLB  left-bottom coordinate of plane
 * @param[in]  : planeSize  size of plane
 * @param[in]  : particleRadius    particle radius
 *
 * @return    a set of particle pos
 */
std::vector<float3>
generate_plane_X(float3 planeLB, float3 planeSize, float particleRadius) {
    std::vector<float3> pos;
    auto diameter = 2 * particleRadius;

    for (float z = particleRadius + planeLB.z; z < planeLB.z + planeSize.z; z += diameter) {
        for (float y = particleRadius + planeLB.y, cnt = 0; cnt < 2; y += diameter, cnt += 1) {
            for (float x = particleRadius + planeLB.x; x < planeLB.x + planeSize.x; x += diameter) {
                float3 _pos = make_float3(x, y, z);

                pos.push_back(_pos);
            }
        }
    }

    return pos;
}

/**
 * @brief  : generate default plane along the Z-axis
 *
 * @param[in]  : planeLB  left-bottom coordinate of plane
 * @param[in]  : planeSize  size of plane
 * @param[in]  : particleRadius    particle radius
 *
 * @return    a set of particle pos
 */
std::vector<float3>
generate_plane_Z(float3 planeLB, float3 planeSize, float particleRadius) {
    std::vector<float3> pos;
    auto diameter = 2 * particleRadius;

    for (float z = particleRadius + planeLB.z, cnt = 0; cnt < 2; z += diameter, cnt += 1) {
        for (float y = particleRadius + planeLB.y; y < planeLB.y + planeSize.y; y += diameter) {
            for (float x = particleRadius + planeLB.x; x < planeLB.x + planeSize.x; x += diameter) {
                float3 _pos = make_float3(x, y, z);

                pos.push_back(_pos);
            }
        }
    }

    return pos;
}

/**
 * @brief  : [deprecated] generate default cylinder shape
 *
 * @param[in]  : topCenter  center pos of top plane
 * @param[in]  : height  height of the cylinder
 * @param[in]  : areaRadius    area of bottom plane
 * @param[in]  : particleRadius    particle radius
 *
 * @return    a set of particle pos
 */
std::vector<float3>
generate_cylinder(float3 topCenter, float height, float areaRadius, float particleRadius) {

    std::vector<float3> pos;
    float diameter = 2 * particleRadius;
    float y0 = topCenter.y;

    for (float y = y0 - particleRadius; y >= y0 - height; y -= diameter) {
        float x0 = topCenter.x - areaRadius;

        for (float x = x0 + particleRadius; x <= topCenter.x + areaRadius; x += diameter) {

            float m_cos = fabs(topCenter.x - x) / areaRadius;
            float length = areaRadius * sqrt(1 - m_cos * m_cos);
            float z0 = topCenter.z - length;
            for (float z = z0 + particleRadius; z <= topCenter.z + length; z += diameter) {
                float3 _pos = {x, y, z};
                pos.push_back(_pos);
            }
        }
    }

    return pos;
}


}  // namespace Physika

#endif  // PHYSIKA_DEFAULT_MODEL_TOOL_H
