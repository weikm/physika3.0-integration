#pragma once

#include "collision/internal/collision_tri3f.hpp"
#include "collision/internal/collision_bvh.hpp"
#include "collision/internal/collision_qbvh.hpp"
#include "collision/internal/collision_pair.hpp"
#include "collision/internal/distance_field3d.hpp"
namespace Physika {
class bvh;
class qbvh;
class sbvh;

__forceinline vec3f getPointInetia(const vec3f& point, float mass)
{
    float x2 = point[0] * point[0];
    float y2 = point[1] * point[1];
    float z2 = point[2] * point[2];
    return vec3f(mass * (y2 + z2), mass * (x2 + z2), mass * (x2 + y2));
}
class triMesh
{
public:
    unsigned int _num_vtx;
    unsigned int _num_tri;
    tri3f*       _tris;
    vec3f*       _vtxs;
    vec3f*     _ovtxs;
    vec3f*     _fnrms;
    vec3f*     _nrms;
    BOX<REAL>* _bxs;
    BOX<REAL>  _bx;
    bvh*       _bvh;
    qbvh*      _qbvh;
    sbvh*      _sbvh;

    int _dl;

    DistanceField3D* _sdf;
public:
    triMesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs, bool cyl);

    triMesh(){}

    ~triMesh();

    unsigned int getNbVertices() const;
    unsigned int getNbFaces();
    vec3f*       getVtxs() const;
    vec3f*       getNrms() const;
    vec3f*       getFNrms() const;
    tri3f*       getTris() const;

    // calc norms, and prepare for display ...
    void updateNrms();

    // really displaying ...
    void display(bool cyl, int);
    void displayStatic(bool cyl, int);
    void displayDynamic(bool cyl, int);

    // prepare for display
    void updateDL(bool cyl, int);
    // finish for display
    void destroyDL();

    // prepare for bvh
    void updateBVH(bool cyl);
    // finish bvh
    void destroyBVH();

    void updateBxs();

    BOX<REAL> bound();

    void rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret);
    void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f& dirR, REAL& ret);
    void query(const BOX<REAL>& bx, std::vector<int>& rets);
    void query2(const BOX<REAL>& bx, std::vector<int>& rets);

    void collide(const triMesh* other, const cbox2boxTrfCache& trf, std::vector<id_pair>& rets);

#ifdef GPU
    void push2G(int i);
#endif

	void calculateInertia(float mass, vec3f& inertia)
    {
        inertia = vec3f();

        float pointmass = mass / _num_vtx;

        // for (int i = 0; i < _num_vtx; i++)
        int i = _num_vtx;
        while (i--)
        {
            vec3f& v         = _vtxs[i];
            vec3f  ptInteria = getPointInetia(v, pointmass);
            inertia += ptInteria;
        }
    }
};
}  // namespace Physika