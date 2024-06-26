/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-11-15
 * @description: collision detect of mesh and ray
 * @version    : 1.0
 */
#pragma once
#include "collision/internal/trimesh.hpp"
#include "collision/internal/ray.hpp"
namespace Physika {
// * MeshRayCollisionDetect detect the collision between ray and trimesh
// * and return the first intersect point
class MeshRayCollisionDetect
{
public:
    MeshRayCollisionDetect(Ray& ray, triMesh& mesh);
    ~MeshRayCollisionDetect() {}

    /**
     * @brief function to execute mesh mesh collision detect
     */
    void execute();

    /**
     * @brief function to execute mesh mesh collision detect
     *
     * @param[out] intersectPoint   the intersectPoint of mesh and ray, if not collide, it will be vec3f(0, 0, 0)
     */
    void getIntersectPoint(vec3f& intersectPoint) const;

    /**
     * @brief function to execute mesh mesh collision detect
     *
     * @param[out] isIntersect   the status of mesh ray collision detect. true: collide, false: not collide
     */
    void getIntersectState(bool& isIntersect) const;

private:
    void executeInternal();

    // input
    Ray*     m_ray;                 // ray
    triMesh* m_mesh;                // collision meshes


    // output
    vec3f m_intersectPoint = vec3f(0, 0, 0);  // the first intersect point of mesh and ray
    bool  m_isIntersect    = false;           // whether the ray intersect with mesh
};
}  // namespace Physika