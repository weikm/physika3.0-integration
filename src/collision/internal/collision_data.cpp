#include "collision/interface/collision_data.hpp"

namespace Physika
{
ImpactInfo::ImpactInfo(const int fid1, const int fid2, const int vf_ee, const int v, const int v2, const int v3, const int v4, const float d, const float t, const int CCD)
{
    m_faceId[0] = fid1;
    m_faceId[1] = fid2;

    m_IsVF_OR_EE = vf_ee;

    m_vertexId[0] = v;
    m_vertexId[1] = v2;
    m_vertexId[2] = v3;
    m_vertexId[3] = v4;

    m_dist = d;
    m_time = t;

    m_ccdres = CCD;
}

unsigned int TrianglePair::id0() const
{
    return m_id[0];
}

unsigned int TrianglePair::id1() const
{
    return m_id[1];
}

TrianglePair::TrianglePair(unsigned int id1, unsigned int id2)
{
    if (id1 < id2)
    {
        m_id[0] = id1;
        m_id[1] = id2;
    }
    else
    {
        m_id[0] = id2;
        m_id[1] = id1;
    }
}

void TrianglePair::get(unsigned int& id1, unsigned int& id2)
{
    id1 = m_id[0];
    id2 = m_id[1];
}

bool TrianglePair::operator<(const TrianglePair& other) const
{
    if (m_id[0] == other.m_id[0])
        return m_id[1] < other.m_id[1];
    else
        return m_id[0] < other.m_id[0];
}
}  // namespace Physika