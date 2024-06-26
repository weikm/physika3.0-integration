/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: collideable triangleMesh component in Physika.
 *               if an object has this componet, then it is collidable
 *               it use data of triangleComponent to detect collision
 * @version    : 1.0
 */

/**
 * a component fot triangleMesh to enable collision detect
 */
#pragma once
#include "collision/internal/trimesh.hpp"
namespace Physika {
struct CollidableTriangleMeshComponent
{
    void reset() 
    {
        m_mesh = nullptr;
    }

    triMesh* m_mesh;
};
}  // namespace Physika
