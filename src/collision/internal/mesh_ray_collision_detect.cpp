#include "collision/interface/mesh_ray_collision_detect.hpp"

#include <queue>
namespace Physika {
static bool rayTriCollision(const Ray& ray, const triMesh& mesh, int triId, float& hit_t)
{
    auto  vtxs   = mesh.getVtxs();
    auto  tri    = mesh._tris[triId];
    vec3f p1     = vtxs[tri.id0()];
    vec3f p2     = vtxs[tri.id1()];
    vec3f p3     = vtxs[tri.id2()];
    vec3f origin = ray.m_origin;
    vec3f dir    = ray.m_dir;

    // Moller-Trumbore
    vec3f e1 = p2 - p1;
    vec3f e2 = p3 - p1;
    vec3f s  = origin - p1;
    vec3f s1 = dir.cross(e2);
    vec3f s2 = s.cross(e1);

    float div = s1.dot(e1);

    if (div == 0)
    {
        return false;
    }

    float t     = s2.dot(e2) / div;
    float beta  = s1.dot(s) / div;
    float gamma = s2.dot(dir) / div;
    float alpha = 1.0 - beta - gamma;

    // Judge if the hitpoint is in the triangle
    auto isBarycentricCoord = [](float alpha) -> bool { return alpha >= 0.0 && alpha <= 1.0; };

    hit_t = t;
    if (t >= 0)
    {
        return isBarycentricCoord(alpha) && isBarycentricCoord(beta) && isBarycentricCoord(gamma);
    }
    return false;
}

static bool rayBoxCollision(const Ray& ray, const BOX<float>& aabb)
{
    vec3f min = aabb._min;
    vec3f max = aabb._max;

    vec3f origin = ray.m_origin;
    vec3f dir    = ray.m_dir;

    float rayT = 0;

    auto rangeContains = [](float min, float max, float param) -> bool {
        return min <= param && param <= max;
    };

    // The pickray's origin is in the bbox
    if (aabb.inside(origin))
    {
        return true;
    }

    // The pickray's origin is not in the bbox, try to hit the nearest plane
    if (origin.x <= min.x && dir.x > 0)
    {
        rayT = (min.x - origin.x) / dir.x;

        vec3f hitpoint = ray.getPoint(rayT);

        if (rangeContains(min.y, max.y, hitpoint.y) && rangeContains(min.z, max.z, hitpoint.z))
        {
            return true;
        }
    }

    if (origin.x >= max.x && dir.x < 0)
    {
        rayT = (max.x - origin.x) / dir.x;

        vec3f hitpoint = ray.getPoint(rayT);

        if (rangeContains(min.y, max.y, hitpoint.y) && rangeContains(min.z, max.z, hitpoint.z))
        {
            return true;
        }
    }

    if (origin.y <= min.y && dir.y > 0)
    {
        rayT = (min.y - origin.y) / dir.y;

        vec3f hitpoint = ray.getPoint(rayT);

        if (rangeContains(min.x, max.x, hitpoint.x) && rangeContains(min.z, max.z, hitpoint.z))
        {
            return true;
        }
    }

    if (origin.y >= max.y && dir.y < 0)
    {
        rayT = (max.y - origin.y) / dir.y;

        vec3f hitpoint = ray.getPoint(rayT);

        if (rangeContains(min.x, max.x, hitpoint.x) && rangeContains(min.z, max.z, hitpoint.z))
        {
            return true;
        }
    }

    if (origin.z <= min.z && dir.z > 0)
    {
        rayT = (min.z - origin.z) / dir.z;

        vec3f hitpoint = ray.getPoint(rayT);

        if (rangeContains(min.x, max.x, hitpoint.x) && rangeContains(min.y, max.y, hitpoint.y))
        {
            return true;
        }
    }

    if (origin.z >= max.z && dir.z < 0)
    {
        rayT = (max.z - origin.z) / dir.z;

        vec3f hitpoint = ray.getPoint(rayT);

        if (rangeContains(min.x, max.x, hitpoint.x) && rangeContains(min.y, max.y, hitpoint.y))
        {
            return true;
        }
    }

    return false;
}

typedef std::vector<int> rayBVHHitRes;
static rayBVHHitRes      rayBVHCollision(const Ray& ray, bvh& bvhTree)
{
    rayBVHHitRes          res;
    BOX<float>            bbox;
    bvh_node*             node;
    std::queue<bvh_node*> q;
    if (bvhTree.root())
    {
        q.emplace(bvhTree.root());
    }

    while (!q.empty())
    {
        node = q.front();
        q.pop();

        bbox = node->box();
        if (rayBoxCollision(ray, bbox))
        {
            if (node->isLeaf())
            {
                res.emplace_back(node->triID());
            }
            else
            {
                q.emplace(node->left());
                q.emplace(node->right());
            }
        }
    }

    return res;
}

MeshRayCollisionDetect::MeshRayCollisionDetect(Ray& ray, triMesh& mesh)
{
	m_ray  = &ray;
	m_mesh = &mesh;
}

void MeshRayCollisionDetect::execute()
{
    executeInternal();
}

void MeshRayCollisionDetect::executeInternal()
{
    auto  ray  = m_ray;
    auto  mesh = m_mesh;
    float ray_T{ 0 };
    float min_T{ -1 };
    bvh  BVHTree(mesh);
    auto bvhRes = rayBVHCollision(*ray, BVHTree);

    for (auto triId : bvhRes)
    {
        if (rayTriCollision(*ray, *mesh, triId, ray_T))
        {
            min_T = min_T < 0 ? ray_T : std::min(min_T, ray_T);
        }
    }

    m_intersectPoint = ray->getPoint(min_T);
    // if (m_transformToLCS)
    //{
    //transform();
    //}
    m_isIntersect = (min_T >= 0) ? true : false;
    return;
}

void MeshRayCollisionDetect::getIntersectPoint(vec3f& intersectPoint) const
{
    intersectPoint = m_intersectPoint;
}

void MeshRayCollisionDetect::getIntersectState(bool& isIntersect) const
{
    isIntersect = m_isIntersect;
}
};  // namespace Physika