#pragma once

#include "collision/internal/trimesh.hpp"
#include "collision/internal/collision_pair.hpp"

namespace Physika {
/**
 * MeshMeshCollisionDetect is detect the collision between meshes by using dcd
 * it will use gpu function to do collision detect
 */
class MeshMeshDiscreteCollisionDetect
{
public:
    /**
     * @brief construct for collision detection between two meshes
     * @param[in] meshed   a vector of meshes
     */
    MeshMeshDiscreteCollisionDetect(std::vector<triMesh*> m_meshes);
    MeshMeshDiscreteCollisionDetect();
    ~MeshMeshDiscreteCollisionDetect();

    /**
     * @brief collision detect using gpu with standard bvh
     */
    bool execute();

    /**
     * @brief get the collided meshes indices and corresponding triangles
     *
     * @return the collided meshes indices and corresponding triangles
     */
    void getContactPairs(std::vector<std::vector<id_pair>>& contact_pairs) const
    {
        contact_pairs = m_contact_pairs;
    }

    /**
     * @brief get number of collision pairs
     *
     * @return number of collision pairs
     */
    size_t getNumContacts() const
    {
        return m_contact_pairs.size();
    }


    /**
     * @brief push mesh data to gpu
     */
    void pushMesh2GPU();

    /**
     * @brief update mesh data to gpu
     */
    void updateMesh2GPU();

    /**
	 * @brief set whether to update bvh
	 */
    void setUpdate(bool update)
	{
		m_update = update;
	}


private:
    bool                                   m_is_init;	   //!< is init flag
    bool                                   m_update;       //!< update bvh flag
    bvh*                                   m_bvh;          //!< collision bvh
    tri3f*                                 m_faces;        //!< collision faces use for gpu
    vec3f*                                 m_nodes;        //!< collision nodes usr for gpu
    int                                    m_numFace = 0;  //!< number of faces
    int                                    m_numVert = 0;  //!< number of vertex
    // input
    std::vector<triMesh*>                  m_mesh;           //!< collision meshes
    // output
    std::vector<id_pair>                   m_mesh_pairs;     //!< collision mesh pairs
    std::vector<std::vector<id_pair>>      m_contact_pairs;  //!< collision results

};
}  // namespace Physika