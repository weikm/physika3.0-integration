#include "collision/interface/mesh_hf_collision_detect.hpp"
namespace Physika {
MeshHfCollisionDetect::MeshHfCollisionDetect(triMesh& mesh, heightField1d& heightField)
{
	m_trimesh = &mesh;
	m_heightField = &heightField;
}

void MeshHfCollisionDetect::setHeightField(heightField1d& heightField)
{
    m_heightField  = &heightField;
}

void MeshHfCollisionDetect::setMesh(triMesh& trimesh)
{
    m_trimesh = &trimesh;
}

void MeshHfCollisionDetect::execute()
{
    executeInternal();
}

void MeshHfCollisionDetect::executeInternal() 
{
    vec3f* vertices     = m_trimesh->_vtxs;
    int    vertices_num = m_trimesh->_num_vtx;
    m_counter           = 0;
    auto grid           = m_heightField;
    for (int i = 0; i < vertices_num; i++)
    {
        vec3f vertex = vertices[i];
        float h      = grid->get(vertex[0], vertex[2]);
        float depth  = h - vertex[1];
        if (depth >= 0)
        {

            if (m_counter > m_maxCollision)
                break;
            vec3f pointnormal             = grid->heightFieldNormal(vertex[0], vertex[2]);
            m_collision_id[m_counter]     = i;
            m_collision_normal[m_counter] = pointnormal;
            m_counter += 1;
        }
    }
}

void MeshHfCollisionDetect::getResult(int* collisionID, vec3f* collisionNormal, int& count) const
{
    count              = m_counter;
    collisionID        = m_collision_id;
    collisionNormal    = m_collision_normal;
}

} // namespace Physika