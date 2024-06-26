#include <iostream>

#include "collision/interface/mesh_mesh_collision_solver.hpp"
#include "collision/interface/collidable_trianglemesh.hpp"
#include "framework/object.hpp"

typedef unsigned int uint;
extern void          initGPU();
extern void          pushMesh2GPU(int numFace, int numVert, void* faces, void* nodes);
extern void          updateMesh2GPU(void* nodes, void* prenodes, float thickness);
extern int           getCollisionsGPU(int* rets, int* vf_ee, int* vertex_id, float* dist, int* time, int* CCDres, float* thickness);

namespace Physika {

MeshMeshCollisionSolver::MeshMeshCollisionSolver()
    : Solver(), m_is_init(false)
{
}

MeshMeshCollisionSolver::~MeshMeshCollisionSolver()
{
    this->reset();
}

bool MeshMeshCollisionSolver::initialize()
{
    std::cout << "MeshMeshCollisionSolver::initialize() initializes the solver.\n";
    m_is_init = true;
    return true;
}

bool MeshMeshCollisionSolver::isInitialized() const
{
    std::cout << "MeshMeshCollisionSolverr::isInitialized() gets the initialization status of the  solver.\n";
    return m_is_init;
}

bool MeshMeshCollisionSolver::reset()
{
    std::cout << "MeshMeshCollisionSolver::reset() sets the solver to newly constructed state.\n";
    m_is_init   = false;
    auto component = m_ccd_object->getComponent<MeshCollisionComponent>();
    component->m_CCDtime = 0.0;
    component->m_thickness = 0.0;
    component->m_mesh.clear();

    component->m_mesh_pairs.clear();
    component->m_contact_pairs.clear();
    component->m_contact_info.clear();

    return true;
}

bool MeshMeshCollisionSolver::run()
{
    return step();
}

bool MeshMeshCollisionSolver::step()
{
    std::cout << "MeshMeshCollisionSolver::run() updates the solver till termination criteria are met.\n";
    if (!m_is_init)
    {
        std::cout << "error: solver not initialized.\n";
        return false;
    }
    if (!m_ccd_object)
    {
        std::cout << "error: ccd object not attached to the solver.\n";
        return false;
    }
    std::cout << "MeshMeshCollisionSolver::run() applied to ccd object " << m_ccd_object->id() << ".\n";
    doCollision();
    return true;
}

bool MeshMeshCollisionSolver::isApplicable(const Object* object) const
{
    std::cout << "MeshMeshCollisionSolver::isApplicable() checks if object has CollidableMeshComponent.\n";
    if (!object)
        return false;

    return object->hasComponent<MeshCollisionComponent>();
}

bool MeshMeshCollisionSolver::attachObject(Object* object)
{
    std::cout << "MeshMeshCollisionSolver::attachObject() set the target of the solver.\n";
    if (!object)
        return false;

    if (!this->isApplicable(object))
    {
        std::cout << "error: object is not applicable.\n";
        return false;
    }
    if (object->hasComponent<MeshCollisionComponent>())
    {
        std::cout << "    object attached .\n";
        m_ccd_object = object;
    }
    
    return true;
}

bool MeshMeshCollisionSolver::detachObject(Object* object)
{

    std::cout << "MeshMeshCollisionSolver::detachObject() remove the object from target list of the solver.\n";
    if (!object)
        return false;

    if (m_ccd_object == object)
    {
        m_ccd_object = nullptr;
        std::cout << "    ccd object detached.\n";
        return true;
    }
    std::cout << "    error: object is not attached to the solver.\n";
    return false;
}

void MeshMeshCollisionSolver::clearAttachment()
{
    std::cout << "MeshMeshCollisionSolver::clearAttachment() clears the target list of the solver.\n";
    m_ccd_object = nullptr;
}

bool MeshMeshCollisionSolver::doCollision()
{
    auto component = m_ccd_object->getComponent<MeshCollisionComponent>();
    component->m_contact_pairs.clear();
    component->m_contact_info.clear();
    component->m_CCDtime = 0;
    doCollisionGPU();
    return true;
}

void MeshMeshCollisionSolver::getContactPairs(std::vector<std::vector<TrianglePair>>& contact_pairs) const
{
    contact_pairs = m_ccd_object->getComponent<MeshCollisionComponent>()->m_contact_pairs;
}

size_t MeshMeshCollisionSolver::getNumContacts() const
{
    return m_ccd_object->getComponent<MeshCollisionComponent>()->m_contact_pairs.size();
}

int MeshMeshCollisionSolver::getCCD_res() const
{
    return m_ccd_object->getComponent<MeshCollisionComponent>()->m_CCDtime;
}

void MeshMeshCollisionSolver::setThickness(const float tt)
{
    m_ccd_object->getComponent<MeshCollisionComponent>()->m_thickness = tt;
}

void MeshMeshCollisionSolver::getImpactInfo(std::vector<ImpactInfo>& contact_info) const
{
    contact_info = m_ccd_object->getComponent<MeshCollisionComponent>()->m_contact_info;
}

void MeshMeshCollisionSolver::setMesh(const std::vector<triMesh*> m_mesh)
{
    m_ccd_object->getComponent<MeshCollisionComponent>()->m_mesh = m_mesh;
}

void MeshMeshCollisionSolver::pushMesh2GPU()
{
    auto component = m_ccd_object->getComponent<MeshCollisionComponent>();
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
            tri3f& t = m->_tris[j];
            faces[curFace++]              = tri3f(t.id0() + vertCount, t.id1() + vertCount, t.id2() + vertCount);
        }
        vertCount += m->_num_vtx;

        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }
    ::pushMesh2GPU(numFace, numVert, faces, nodes);
}

void MeshMeshCollisionSolver::updateMesh2GPU()
{
    auto   component = m_ccd_object->getComponent<MeshCollisionComponent>();
    auto   mesh      = component->m_mesh;
    auto   numFace   = component->m_numFace;
    auto   numVert   = component->m_numVert;
    auto   faces     = component->m_faces;
    auto   nodes     = component->m_nodes;
    auto   thickness = component->m_thickness;
    vec3f* curVert = nodes;

    // rky
    vec3f*             preVert = new vec3f[numVert];
    std::vector<vec3f> tem;
    vec3f*             oldcurVert = preVert;
    for (int i = 0; i < mesh.size(); i++)
    {
        auto m = mesh[i];
        memcpy(oldcurVert, m->_ovtxs, sizeof(vec3f) * m->_num_vtx);
        oldcurVert += m->_num_vtx;
    }

    for (int i = 0; i < mesh.size(); i++)
    {
        auto m = mesh[i];
        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }

    for (int i = 0; i < mesh.size(); i++)
    {
        for (int j = 0; j < mesh[i]->_num_vtx; j++)
        {
            tem.push_back(mesh[i]->_vtxs[j]);
            tem.push_back(mesh[i]->_ovtxs[j]);
        }
    }

    ::updateMesh2GPU(nodes, preVert, thickness);
}

bool MeshMeshCollisionSolver::doCollisionGPU()
{
    auto                         component = m_ccd_object->getComponent<MeshCollisionComponent>();
    auto                         mesh      = component->m_mesh;
    auto                         numFace   = component->m_numFace;
    auto                         numVert   = component->m_numVert;
    auto                         faces     = component->m_faces;
    auto                         nodes     = component->m_nodes;
    auto                         thickness = component->m_thickness;

    static bvh*                  bvhC = NULL;
    static front_list            fIntra;
    static std::vector<triMesh*> meshes;
    static std::vector<int>      _tri_offset;

#define MAX_CD_PAIRS 14096
    int* buffer      = new int[MAX_CD_PAIRS * 2];
    int* time_buffer = new int[1];

    int*   buffer_vf_ee     = new int[MAX_CD_PAIRS];
    int*   buffer_vertex_id = new int[MAX_CD_PAIRS * 4];
    float* buffer_dist      = new float[MAX_CD_PAIRS];

    int* buffer_CCD = new int[MAX_CD_PAIRS];

    int count = 0;

    if (bvhC == NULL)
    {
        for (int i = 0; i < mesh.size(); i++)
        {
            meshes.push_back(mesh[i]);
            _tri_offset.push_back(i == 0 ? mesh[i]->_num_tri : (_tri_offset[i - 1] + mesh[i]->_num_tri));
        }
        bvhC = new bvh(meshes);

        bvhC->self_collide(fIntra, meshes);
        ::initGPU(); 
        pushMesh2GPU();  
        bvhC->push2GPU(true);
        fIntra.push2GPU(bvhC->root());
    }
    updateMesh2GPU();  
    printf("thickness is %f\n", thickness);
    
    count = ::getCollisionsGPU(buffer, buffer_vf_ee, buffer_vertex_id, buffer_dist, time_buffer, buffer_CCD, &thickness);

    TrianglePair*             pairs = ( TrianglePair* )buffer;
    std::vector<TrianglePair> ret(pairs, pairs + count);

    for (int i = 0; i < count; i++)
    {
        ImpactInfo tem = ImpactInfo(buffer[i * 2], buffer[i * 2 + 1], buffer_vf_ee[i], buffer_vertex_id[i * 4], buffer_vertex_id[i * 4 + 1], buffer_vertex_id[i * 4 + 2], buffer_vertex_id[i * 4 + 3], buffer_dist[i], time_buffer[0], buffer_CCD[i]);

        component->m_contact_info.push_back(tem);
    }

    component->m_CCDtime = time_buffer[0];

    // Find mesh id and face id
    for (int i = 0; i < count; i++)
    {
        std::vector<TrianglePair> tem;
        int                       mid1, mid2;
        unsigned int              fid1, fid2;
        ret[i].get(fid1, fid2);

        for (int j = 0; j < _tri_offset.size(); j++)
        {
            if (fid1 <= _tri_offset[j])
            {
                mid1 = j == 0 ? 0 : j;
                break;
            }
        }

        tem.push_back(TrianglePair(mid1, fid1 == 0 ? 0 : fid1 - (mid1 == 0 ? 0 : _tri_offset[mid1 - 1])));

        int temtt = fid1 - 1 - (mid1 == 0 ? 0 : _tri_offset[mid1 - 1]);

        for (int j = 0; j < _tri_offset.size(); j++)
        {
            if (fid2 <= _tri_offset[j])
            {
                mid2 = j == 0 ? 0 : j;
                break;
            }
        }

        tem.push_back(TrianglePair(mid2, fid2 == 0 ? 0 : fid2 - (mid2 == 0 ? 0 : _tri_offset[mid2 - 1])));

        component->m_contact_pairs.push_back(tem);
    }
    delete[] buffer;
    delete[] time_buffer;
    delete[] buffer_vf_ee;
    delete[] buffer_vertex_id;
    delete[] buffer_dist;
    delete[] buffer_CCD;
    return true;
}
std::vector<triMesh*> CollisionManager::Meshes;

}  // namespace Physika

