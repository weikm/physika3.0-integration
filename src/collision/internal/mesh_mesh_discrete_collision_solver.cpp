#include "collision/interface/mesh_mesh_discrete_collision_solver.hpp"
#include "framework/object.hpp"
typedef unsigned int uint;
extern void          initDCDGPU();
extern void          pushMesh2GPUDCD(int numFace, int numVert, void* faces, void* nodes);
extern void          updateMesh2GPUDCD(void* nodes);
extern int           getCollisionsDCDGPU(int* rets);

namespace Physika {

MeshMeshDiscreteCollisionSolver::MeshMeshDiscreteCollisionSolver()
    : Solver(), m_is_init(false)
{
}

MeshMeshDiscreteCollisionSolver::~MeshMeshDiscreteCollisionSolver()
{
    this->reset();
}

bool MeshMeshDiscreteCollisionSolver::initialize()
{
    std::cout << "MeshMeshDiscreteCollisionSolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool MeshMeshDiscreteCollisionSolver::isInitialized() const
{
    std::cout << "MeshMeshDiscreteCollisionSolverr::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

bool MeshMeshDiscreteCollisionSolver::reset()
{
    std::cout << "MeshMeshDiscreteCollisionSolver::reset() sets the solver to newly constructed state.\n";
    m_is_init   = false;
    auto component      = m_dcd_object->getComponent<MeshDiscreteCollisionComponent>();
    component->m_bvh       = nullptr;
    component->m_mesh.clear();

    component->m_mesh_pairs.clear();
    component->m_contact_pairs.clear();

    return true;
}

bool MeshMeshDiscreteCollisionSolver::run()
{
    return step();
}

bool MeshMeshDiscreteCollisionSolver::step()
{
    std::cout << "MeshMeshDiscreteCollisionSolver::run() updates the solver till termination criteria are met.\n";
    if (!m_is_init)
    {
        std::cout << "error: solver not initialized.\n";
        return false;
    }
    if (!m_dcd_object)
    {
        std::cout << "error: dcd object not attached to the solver.\n";
        return false;
    }
    std::cout << "MeshMeshDiscreteCollisionSolver::run() applied to ccd object " << m_dcd_object->id() << ".\n";
    doCollision();
    return true;
}

bool MeshMeshDiscreteCollisionSolver::isApplicable(const Object* object) const
{
    std::cout << "MeshMeshDiscreteCollisionSolver::isApplicable() checks if object has CollidableMeshComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<MeshDiscreteCollisionComponent>();
}

bool MeshMeshDiscreteCollisionSolver::attachObject(Object* object)
{
    std::cout << "MeshMeshDiscreteCollisionSolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "error: object is not applicable.\n";
        return false;
    }
    if (object->hasComponent<MeshDiscreteCollisionComponent>())
    {
        std::cout << "    object attached .\n";
        m_dcd_object = object;
    }
    
    return true;
}

bool MeshMeshDiscreteCollisionSolver::detachObject(Object* object)
{

    std::cout << "MeshMeshDiscreteCollisionSolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_dcd_object == object)
    {
        m_dcd_object = nullptr;
        std::cout << "    dcd object detached.\n";
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void MeshMeshDiscreteCollisionSolver::clearAttachment()
{
    std::cout << "MeshMeshDiscreteCollisionSolver::clearAttachment() clears the target list of the solver.\n";
    m_dcd_object = nullptr;
}

bool MeshMeshDiscreteCollisionSolver::doCollision()
{
    auto component = m_dcd_object->getComponent<MeshDiscreteCollisionComponent>();
    component->m_contact_pairs.clear();
    doCollisionGPU();
    return true;
}

void MeshMeshDiscreteCollisionSolver::getContactPairs(std::vector<std::vector<id_pair>>& contact_pairs) const
{
    contact_pairs = m_dcd_object->getComponent<MeshDiscreteCollisionComponent>()->m_contact_pairs;
}

size_t MeshMeshDiscreteCollisionSolver::getNumContacts() const
{
    return m_dcd_object->getComponent<MeshDiscreteCollisionComponent>()->m_contact_pairs.size();
}


void MeshMeshDiscreteCollisionSolver::getMeshPairs(std::vector<id_pair>& mesh_pairs) const
{
    mesh_pairs = m_dcd_object->getComponent<MeshDiscreteCollisionComponent>()->m_mesh_pairs;
}

void MeshMeshDiscreteCollisionSolver::setMesh(const std::vector<triMesh*> m_mesh)
{
    m_dcd_object->getComponent<MeshDiscreteCollisionComponent>()->m_mesh = m_mesh;
}

void MeshMeshDiscreteCollisionSolver::pushMesh2GPU()
{
    auto component = m_dcd_object->getComponent<MeshDiscreteCollisionComponent>();
    auto mesh = component->m_mesh;
    auto numFace   = component->m_numFace;
    auto numVert   = component->m_numVert;
    auto faces   = component->m_faces;
    auto nodes   = component->m_nodes;
    for (int i = 0; i < mesh.size(); i++)
    {
        numFace += mesh[i]->_num_tri;
        numVert += mesh[i]->_num_vtx;
    }

    faces = new tri3f[numFace];
    nodes = new vec3f[numVert];

    int    curFace   = 0;
    int    vertCount = 0;
    vec3f* curVert   = nodes;
    for (int i = 0; i < mesh.size(); i++)
    {
        auto m = mesh[i];
        for (int j = 0; j < m->_num_tri; j++)
        {
            tri3f& t           = m->_tris[j];
            faces[curFace++] = tri3f(t.id0() + vertCount, t.id1() + vertCount, t.id2() + vertCount);
        }
        vertCount += m->_num_vtx;

        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }
    ::pushMesh2GPUDCD(numFace, numVert, faces, nodes);
}

void MeshMeshDiscreteCollisionSolver::updateMesh2GPU()
{
    auto   component = m_dcd_object->getComponent<MeshDiscreteCollisionComponent>();
    auto   mesh      = component->m_mesh;
    auto   numFace   = component->m_numFace;
    auto   numVert   = component->m_numVert;
    auto   faces     = component->m_faces;
    auto   nodes     = component->m_nodes;
    vec3f* curVert   = nodes;

    for (int i = 0; i < mesh.size(); i++)
    {
        auto m = mesh[i];
        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }

    ::updateMesh2GPUDCD(nodes);
}

bool MeshMeshDiscreteCollisionSolver::doCollisionGPU()
{
    auto                         component = m_dcd_object->getComponent<MeshDiscreteCollisionComponent>();
    auto                         mesh      = component->m_mesh;
    auto                         numFace   = component->m_numFace;
    auto                         numVert   = component->m_numVert;
    auto                         faces     = component->m_faces;
    auto                         nodes     = component->m_nodes;
    auto                         BVH       = component->m_bvh;
    component->m_contact_pairs.clear();

    static front_list            fIntra;
    static std::vector<triMesh*> meshes;
    static std::vector<int>      _tri_offset;
    _tri_offset.push_back(0);
#define MAX_CD_PAIRS 14096
    int* buffer = new int[MAX_CD_PAIRS * 2];

    int count = 0;

    if (component->m_update)
    {
        if (BVH != NULL)
        {
            delete BVH;
            BVH = NULL;
        }
    }

    if (BVH == NULL)
    {
        for (int i = 0; i < mesh.size(); i++)
        {
            meshes.push_back(mesh[i]);
            _tri_offset.push_back(i == 0 ? mesh[i]->_num_tri : (_tri_offset[i] + mesh[i]->_num_tri));
        }
        BVH = new bvh(meshes);

        BVH->self_collide(fIntra, meshes);

        ::initDCDGPU();
        pushMesh2GPU();
        BVH->push2GPU(true);
        fIntra.push2GPU(BVH->root());
    }

    updateMesh2GPU();

    count = ::getCollisionsDCDGPU(buffer);

    TrianglePair*             pairs = ( TrianglePair* )buffer;
    std::vector<TrianglePair> ret(pairs, pairs + count);
    // Find mesh id and face id
    for (int i = 0; i < count; i++)
    {
        std::vector<id_pair> tem;
        unsigned int         mid1, mid2;
        unsigned int         fid1, fid2;
        ret[i].get(fid1, fid2);

        for (int j = 0; j < _tri_offset.size() - 1; j++)
        {
            if (fid1 < _tri_offset[j + 1] && fid1 >= _tri_offset[j])
            {
                mid1 = j;
                break;
            }
        }

        tem.push_back(id_pair(mid1, fid1 - _tri_offset[mid1], false));

        for (int j = 0; j < _tri_offset.size() - 1; j++)
        {
            if (fid2 < _tri_offset[j + 1] && fid2 >= _tri_offset[j])
            {
                mid2 = j;
                break;
            }
        }
        tem.push_back(id_pair(mid2, fid2 - _tri_offset[mid2], false));

        component->m_contact_pairs.push_back(tem);
    }

    delete[] buffer;
    return true;
}

}  // namespace Physika

