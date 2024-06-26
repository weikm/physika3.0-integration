#pragma once

#include "collision/internal/trimesh.hpp"
#include "collision/internal/collision_pair.hpp"

namespace Physika
{
/**
 * MeshMeshCollisionDetect is detect the collision between meshes by using ccd
 * it will use gpu function to do collision detect
 */
	class MeshMeshCollisionDetect
	{
    public:
        /**
         * @brief construct for collision detection between two meshes
         * @param[in] meshed   a vector of meshes
         */
        MeshMeshCollisionDetect(std::vector<triMesh*> m_meshes);
        MeshMeshCollisionDetect();
        ~MeshMeshCollisionDetect();

        /**
         * @brief set the meshes for collision detection
         * @param[in] meshed   a vector of meshes
         */
        void setMeshes(std::vector<triMesh*> m_meshes)
        {
			m_mesh = m_meshes;
		}

        /**
         * @brief collision detect using gpu with standard bvh
         */
        bool execute();

        /**
         * @brief get the collided meshes indices and corresponding triangles
         *
         * @return the collided meshes indices and corresponding triangles
         */
        void getContactPairs(std::vector<std::vector<TrianglePair>>& contact_pairs) const
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
         * @brief return ccd results
         *
         * @return ccd time
         * @retval 1 collision appears
         * @retval 0 no collision
         */
        int getCCD_res() const
        {
            return m_CCDtime;
        }

        /**
         * @brief set thickness, the min thickness should be set greater than 1e-3
         *
         * @param[in]     thickness     thickness of the face
         */
        void setThickness(const float tt)
        {
            m_thickness = tt;
        }

        /**
         * @brief get collision info
         *
         * @return array of impact info
         */
        void getImpactInfo(std::vector<ImpactInfo>& contact_info) const
        {
            contact_info = m_contact_info;
        }

        /**
         * @brief push mesh data to gpu
         */
        void pushMesh2GPU();

        /**
         * @brief update mesh data to gpu
         */
        void updateMesh2GPU();

    private:
        bool                                   m_is_init;	   //!< is init flag
        bool                                   m_update;		 //!< update bvh flag
        int                                    m_CCDtime;        //!< return 1 when collision appear else return 0
        float                                  m_thickness;      //!< thickness of the face
        tri3f*                                 m_faces;          //!< collision faces use for gpu
        vec3f*                                 m_nodes;          //!< collision nodes use for gpu
        int                                    m_numFace = 0;    //!< number of faces
        int                                    m_numVert = 0;    //!< number of vertex
        // input
        std::vector<triMesh*>                  m_mesh;           //!< collision meshes
        std::vector<std::vector<TrianglePair>> m_contact_pairs;  //!< collision results
        //output
        std::vector<ImpactInfo>                m_contact_info;   //!< collision impact info
        bvh*                                   m_bvh;            //!< collision bvh

	};
}