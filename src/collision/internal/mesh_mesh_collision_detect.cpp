#include "collision/interface/mesh_mesh_collision_detect.hpp"
typedef unsigned int uint;
extern void          initGPU();
extern void          pushMesh2GPU(int numFace, int numVert, void* faces, void* nodes);
extern void          updateMesh2GPU(void* nodes, void* prenodes, float thickness);
extern int           getCollisionsGPU(int* rets, int* vf_ee, int* vertex_id, float* dist, int* time, int* CCDres, float* thickness);

namespace Physika {

MeshMeshCollisionDetect::MeshMeshCollisionDetect(std::vector<triMesh*> m_meshes)
: m_is_init(false), m_CCDtime(0.0), m_mesh(), m_contact_pairs(), m_contact_info(), m_thickness(0.0), m_bvh(NULL), m_numFace(0), m_numVert(0)
{
    m_mesh = m_meshes;
}

MeshMeshCollisionDetect::MeshMeshCollisionDetect()
    : m_is_init(false), m_CCDtime(0.0), m_mesh(), m_contact_pairs(), m_contact_info(), m_thickness(0.0), m_bvh(NULL), m_numFace(0), m_numVert(0) {}

MeshMeshCollisionDetect::~MeshMeshCollisionDetect() {}

bool MeshMeshCollisionDetect::execute()
{
    m_contact_pairs.clear();
    m_contact_info.clear();
    m_CCDtime                         = 0;
    
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
        for (int i = 0; i < m_mesh.size(); i++)
        {
            meshes.push_back(m_mesh[i]);
            _tri_offset.push_back(i == 0 ? m_mesh[i]->_num_tri : (_tri_offset[i - 1] + m_mesh[i]->_num_tri));
        }

        bvhC = new bvh(meshes);

        bvhC->self_collide(fIntra, meshes);
        
        ::initGPU();
        pushMesh2GPU();
        bvhC->push2GPU(true);
        fIntra.push2GPU(bvhC->root());
    }
    
    updateMesh2GPU();
    printf("thickness is %f\n", m_thickness);

    count = ::getCollisionsGPU(buffer, buffer_vf_ee, buffer_vertex_id, buffer_dist, time_buffer, buffer_CCD, &m_thickness);

    TrianglePair*             pairs = ( TrianglePair* )buffer;
    std::vector<TrianglePair> ret(pairs, pairs + count);

    for (int i = 0; i < count; i++)
    {
        ImpactInfo tem = ImpactInfo(buffer[i * 2], buffer[i * 2 + 1], buffer_vf_ee[i], buffer_vertex_id[i * 4], buffer_vertex_id[i * 4 + 1], buffer_vertex_id[i * 4 + 2], buffer_vertex_id[i * 4 + 3], buffer_dist[i], time_buffer[0], buffer_CCD[i]);
        //printf("%d %d\n", buffer[i * 2], buffer[i * 2 + 1]);
        m_contact_info.push_back(tem);
    }

    m_CCDtime = time_buffer[0];

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

        m_contact_pairs.push_back(tem);
    }
    delete[] buffer;
    delete[] time_buffer;
    delete[] buffer_vf_ee;
    delete[] buffer_vertex_id;
    delete[] buffer_dist;
    delete[] buffer_CCD;
    return true;
}

void MeshMeshCollisionDetect::updateMesh2GPU()
{
    vec3f* curVert = m_nodes;
    // rky
    vec3f*             preVert = new vec3f[m_numVert];
    std::vector<vec3f> tem;
    vec3f*             oldcurVert = preVert;

    for (int i = 0; i < m_mesh.size(); i++)
    {
        auto m = m_mesh[i];
        memcpy(oldcurVert, m->_ovtxs, sizeof(vec3f) * m->_num_vtx);
        oldcurVert += m->_num_vtx;
    }

    for (int i = 0; i < m_mesh.size(); i++)
    {
        auto m = m_mesh[i];
        memcpy(curVert, m->_vtxs, sizeof(vec3f) * m->_num_vtx);
        curVert += m->_num_vtx;
    }

    for (int i = 0; i < m_mesh.size(); i++)
    {
        for (int j = 0; j < m_mesh[i]->_num_vtx; j++)
        {
            tem.push_back(m_mesh[i]->_vtxs[j]);
            tem.push_back(m_mesh[i]->_ovtxs[j]);
        }
    }
    
    ::updateMesh2GPU(m_nodes, preVert, m_thickness);
}

void MeshMeshCollisionDetect::pushMesh2GPU()
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
    ::pushMesh2GPU(m_numFace, m_numVert, m_faces, m_nodes);
}
}  // namespace Physika