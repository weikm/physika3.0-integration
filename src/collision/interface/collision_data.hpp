/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: collsion data of CCD detection for mesh mesh
 * @version    : 1.0
 */
#pragma once

namespace Physika
{ 
/**
 * Data structure to store collision results.
 *
 * Sample usage:
 * ImpactInfo result = ...
 * auto faceID1 = result.f_id[0];
 * auto faceID2 = result.f_id[1];
 */
struct ImpactInfo
{
    /*
    * constructor, used internally
    */
    ImpactInfo() = delete;
    ImpactInfo(const int fid1, const int fid2, const int vf_ee, const int v, const int v2, const int v3, const int v4, const float d, const float t, const int CCD);
    ImpactInfo(const ImpactInfo& other) = default;
    ~ImpactInfo() = default;

    int m_faceId[2];  //<! face id

    int m_IsVF_OR_EE;  //<! 0:vf 1:ee

    int m_vertexId[4];  //<! vertices ids

    float m_dist;  //<! distance
    float m_time;  //<! time

    int m_ccdres;  //<! ccd results
};

/**
 * Data structure to store collision results.
 *
 * Sample usage:
 * TrianglePair result = ...
 * auto meshID = result.id0();
 * auto triangleID = result.id1();
 */
struct TrianglePair
{
public:

    /**
     * constructor
     *
     * @param[in] id1 mesh id
     * @param[in] id2 triangle id
     */
    TrianglePair(unsigned int id1, unsigned int id2);

    /**
     * get mesh index
     *
     * @return mesh index
     */
    unsigned int id0() const;

    /**
     * get triangle index
     *
     * @return triangle index
     */
    unsigned int id1() const;

    /**
     * get mesh index and triangle index
     *
     * @param[out] id1 mesh index
     * @param[out] id2 triangle index
     */
    void get(unsigned int& id1, unsigned int& id2);

    /**
     * operator < to define partial order of TrianglePair
     *
     * @param[in] other the TrianglePair to be compared with
     */
    bool operator<(const TrianglePair& other) const;
    
    unsigned int m_id[2];  //< ! mesh index - triangle index pair
};
} // namespace Physika
