/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2024-04-09
 * @description: solver for mesh-mesh discrete collision
 * @version    : 1.0
 */
#pragma once

#include <assert.h>
#include <vector>

#include "collision/internal/trimesh.hpp"
#include "collision/internal/collision_pair.hpp"
#include "framework/solver.hpp"


namespace Physika {
// collision between meshes' result component
struct MeshDiscreteCollisionComponent
{
    using MeshPair = std::pair<int, int>;  //!< pair of mesh indices, stores the id of triangle mesh

    MeshDiscreteCollisionComponent(){};
    ~MeshDiscreteCollisionComponent(){};

    bool   m_update;       //!< update bvh flag
    bvh*   m_bvh;          //!< collision bvh
    tri3f* m_faces;        //!< collision faces use for gpu
    vec3f* m_nodes;        //!< collision nodes usr for gpu
    int    m_numFace = 0;  //!< number of faces
    int    m_numVert = 0;  //!< number of vertex
    // input
    std::vector<triMesh*> m_mesh;  //!< collision meshes
    // output
    std::vector<id_pair>              m_mesh_pairs;     //!< collision mesh pairs
    std::vector<std::vector<id_pair>> m_contact_pairs;  //!< collision results

};

/**
 * MeshMeshCollisionSolver is a sample solver for collision detect
 * it detect collision between two triangle meshes using CCD
 * now it can only use gpu to detect collision
 */
class MeshMeshDiscreteCollisionSolver : public Solver
{
public:
    struct SolverConfig
    {
        float m_dt;
        float m_total_time;
    };

public:
    using MeshPair = std::pair<int, int>;  //!< pair of mesh indices, stores the id of triangle mesh

    MeshMeshDiscreteCollisionSolver();
    ~MeshMeshDiscreteCollisionSolver();

    /**
     * @brief initialize the solver to get it ready for execution.
     *        The behavior of duplicate calls is up to the developers of subclasses, yet it is
     *        recommended that duplicate calls should be ignored to avoid redundant computation.
     *
     * @return  true if initialization succeeds, otherwise return false
     *
     */
    bool initialize() override;

    /**
     * @brief get the initialization state of the solver.
     *
     * @return   true if solver has been properly initialized, otherwise return false
     */
    bool isInitialized() const override;

    /**
     * @brief reset the solver to newly constructed state
     *
     * @return    true if reset succeeds, otherwise return false
     */
    bool reset() override;

    /**
     * @brief  run the solver in a time step, in this solver, step() is the same as run()
     *
     * @return    true if reset succeeds, otherwise return false
     */
    bool step() override;

    /**
     * @brief run the solver to get the collision results
     *
     * @return true if procedure successfully completes, otherwise return false
     */
    bool run() override;

    /**
     * @brief check whether the solver is applicable to given object
     *        in this case, the solver is applicable to objects that have the CollidableTriangleMesh component
     *
     * @param[in] object    the object to check
     *
     * @return    true if the solver is applicable to the given object, otherwise return false
     */
    bool isApplicable(const Object* object) const override;

    /**
     * @brief attach an object to the solver
     *
     * @param[in] object    the object to attach
     *
     * @return    true if the object is successfully attached, otherwise return false
     */
    bool attachObject(Object* object) override;

    /**
     * @brief detach an object from the solver
     *
     * @param[in] object    the object to detach
     *
     * @return    true if the object is successfully detached, otherwise return false
     */
    bool detachObject(Object* object) override;

    /**
     * @brief clear the attachment of the solver
     */
    void clearAttachment() override;

    /**
     * @brief the main function of run, this function update data to gpu and use doCollsionGPu() to detect collision
     */
    bool doCollision();

    /**
     * @brief collision detect using gpu with standard bvh
     */
    bool doCollisionGPU();

public:
    /**
     * @brief get the collided meshes indices and corresponding triangles
     *
     * @return the collided meshes indices and corresponding triangles
     */
    void getContactPairs(std::vector<std::vector<id_pair>>& contact_pairs) const;

    /**
     * @brief get number of collision pairs
     *
     * @return number of collision pairs
     */
    size_t getNumContacts() const;


    /**
     * @brief set triMesh, the mesh to detect collision
     *
     * @param[in]     triMesh     vector of triMesh
     */
    void setMesh(const std::vector<triMesh*> m_mesh);

    /**
     * @brief get collision info
     *
     * @return pairs which record the id of mesh
     */
    void getMeshPairs(std::vector<id_pair>& mesh_pairs) const;

    /**
     * @brief push mesh data to gpu
     */
    void pushMesh2GPU();

    /**
     * @brief update mesh data to gpu
     */
    void updateMesh2GPU();

private:
    bool  m_is_init;   // the flag of initialization
    float m_cur_time;  // current time

    SolverConfig m_config;		  // solver configuration
    Object* m_dcd_object;           //!< collision object
};
}  // namespace Physika