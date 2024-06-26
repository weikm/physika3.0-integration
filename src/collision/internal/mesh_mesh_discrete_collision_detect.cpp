#include "collision/interface/mesh_mesh_discrete_collision_detect.hpp"
typedef unsigned int uint;
extern void          initDCDGPU();
extern void          pushMesh2GPUDCD(int numFace, int numVert, void* faces, void* nodes);
extern void          updateMesh2GPUDCD(void* nodes);
extern int           getCollisionsDCDGPU(int* rets);

namespace Physika {

MeshMeshDiscreteCollisionDetect::MeshMeshDiscreteCollisionDetect(std::vector<triMesh*> m_meshes)
    : m_is_init(false), m_mesh(), m_contact_pairs(), m_bvh(NULL), m_numFace(0), m_numVert(0)
{
    m_mesh = m_meshes;
}

MeshMeshDiscreteCollisionDetect::MeshMeshDiscreteCollisionDetect()
    : m_is_init(false), m_mesh(), m_contact_pairs(),m_bvh(NULL), m_numFace(0), m_numVert(0) {}

MeshMeshDiscreteCollisionDetect::~MeshMeshDiscreteCollisionDetect() {}

bool MeshMeshDiscreteCollisionDetect::execute()
{
    m_contact_pairs.clear();

    static front_list            fIntra;
    static std::vector<triMesh*> meshes;
    static std::vector<int> _tri_offset;
    _tri_offset.push_back(0);
#define MAX_CD_PAIRS 14096
    int* buffer      = new int[MAX_CD_PAIRS * 2];

    int count = 0;

    if (m_update)
    {
        if (m_bvh != NULL)
		{
			delete m_bvh;
			m_bvh = NULL;
		}
    }

    if (m_bvh == NULL)
    {
        for (int i = 0; i < m_mesh.size(); i++)
        {
            meshes.push_back(m_mesh[i]);
            _tri_offset.push_back(i == 0 ? m_mesh[i]->_num_tri : (_tri_offset[i] + m_mesh[i]->_num_tri));
        }
        m_bvh = new bvh(meshes);

        m_bvh->self_collide(fIntra, meshes);

        ::initDCDGPU();
        pushMesh2GPU();
        m_bvh->push2GPU(true);
        fIntra.push2GPU(m_bvh->root());
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
        unsigned int       fid1, fid2;
        ret[i].get(fid1, fid2);

        for (int j = 0; j < _tri_offset.size()-1; j++)
        {
            if (fid1 < _tri_offset[j + 1] && fid1 >= _tri_offset[j])
            {
                mid1 = j ;
                break;
            }
        }

        tem.push_back(id_pair(mid1, fid1 - _tri_offset[mid1],false));


        for (int j = 0; j < _tri_offset.size() - 1; j++)
        {
            if (fid2 < _tri_offset[j + 1] && fid2 >= _tri_offset[j])
            {
                mid2 = j;
                break;
            }
        }
        tem.push_back(id_pair(mid2, fid2 - _tri_offset[mid2], false));

        m_contact_pairs.push_back(tem);
    }

    delete[] buffer;
    return true;
}

void MeshMeshDiscreteCollisionDetect::updateMesh2GPU()
{
    vec3f* curVert = m_nodes;

    for (int i = 0; i < m_mesh.size(); i++)
    {
        auto m = m_mesh[i];
        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }

    ::updateMesh2GPUDCD(m_nodes);
}

void MeshMeshDiscreteCollisionDetect::pushMesh2GPU()
{
    for (int i = 0; i < m_mesh.size(); i++)
    {
        m_numFace += m_mesh[i]->_num_tri;
        m_numVert += m_mesh[i]->_num_vtx;
    }

    m_faces = new tri3f[m_numFace];
    m_nodes = new vec3f[m_numVert];

    int    curFace   = 0;
    int    vertCount = 0;
    vec3f* curVert   = m_nodes;
    for (int i = 0; i < m_mesh.size(); i++)
    {
        auto m = m_mesh[i];
        for (int j = 0; j < m->_num_tri; j++)
        {
            tri3f& t           = m->_tris[j];
            m_faces[curFace++] = tri3f(t.id0() + vertCount, t.id1() + vertCount, t.id2() + vertCount);
        }
        vertCount += m->_num_vtx;

        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }
    ::pushMesh2GPUDCD(m_numFace, m_numVert, m_faces, m_nodes);
}
}  // namespace Physika