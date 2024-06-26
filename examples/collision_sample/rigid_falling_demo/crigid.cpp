#include "crigid.h"
#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>

// #include "mesh_defs.h"
#include "crigid.h"
#include "collision/internal/collision_bvh.hpp"

#include <set>
using namespace std;

extern std::vector<int> fset;
extern std::set<int>    vset;
extern vec3f            projDir;
extern REAL             maxDist;

// #define FOR_VOLMESH
// #define FOR_SDF

// for fopen
#pragma warning(disable : 4996)

inline vec3f update(vec3f& v1, vec3f& v2, vec3f& v3)
{
    vec3f s = (v2 - v1);
    return s.cross(v3 - v1);
}

inline vec3f
update(tri3f& tri, vec3f* vtxs)
{
    vec3f& v1 = vtxs[tri.id0()];
    vec3f& v2 = vtxs[tri.id1()];
    vec3f& v3 = vtxs[tri.id2()];

    return update(v1, v2, v3);
}

void triMesh::updateNrms()
{
    if (_fnrms == nullptr)
        _fnrms = new vec3f[_num_tri];

    if (_nrms == nullptr)
        _nrms = new vec3f[_num_vtx];

    for (unsigned int i = 0; i < _num_tri; i++)
    {
        vec3f n = ::update(_tris[i], _vtxs);
        n.normalize();
        _fnrms[i] = n;
    }

    for (unsigned int i = 0; i < _num_vtx; i++)
        _nrms[i] = vec3f::zero();

    for (unsigned int i = 0; i < _num_tri; i++)
    {
        vec3f& n = _fnrms[i];
        _nrms[_tris[i].id0()] += n;
        _nrms[_tris[i].id1()] += n;
        _nrms[_tris[i].id2()] += n;
    }

    for (unsigned int i = 0; i < _num_vtx; i++)
        _nrms[i].normalize();
}

void triMesh::display(bool cyl, int level)
{
    if (_dl == -1)
        updateDL(cyl, level);

    glCallList(_dl);
    displayDynamic(cyl, level);
}

void triMesh::displayStatic(bool cyl, int level)
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
#ifdef USE_DOUBLE
    glVertexPointer(3, GL_DOUBLE, sizeof(REAL) * 3, _vtxs);
    glNormalPointer(GL_DOUBLE, sizeof(REAL) * 3, _nrms);
#else
    glVertexPointer(3, GL_FLOAT, sizeof(REAL) * 3, _vtxs);
    glNormalPointer(GL_FLOAT, sizeof(REAL) * 3, _nrms);
#endif

    glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

}

void triMesh::displayDynamic(bool cyl, int level)
{
    if (level >= 0)
        _bvh->visualize(level);
}

void triMesh::destroyDL()
{
    if (_dl != -1)
        glDeleteLists(_dl, 1);
}

void triMesh::updateDL(bool cyl, int level)
{
    if (_dl != -1)
        destroyDL();

    _dl = glGenLists(1);
    glNewList(_dl, GL_COMPILE);
    displayStatic(cyl, level);
    glEndList();
}

//void triMesh::destroySDF()
//{
//    if (_sdf != nullptr)
//        delete _sdf;
//}

void triMesh::destroyBVH()
{
    if (_bvh != nullptr)
        delete _bvh;

    if (_qbvh != nullptr)
        delete _qbvh;

    if (_sbvh != nullptr)
        delete _sbvh;
}

void triMesh::updateBVH(bool cyl)
{
    if (_bvh != nullptr)
        delete _bvh;
    if (_qbvh != nullptr)
        delete _qbvh;
    if (_sbvh != nullptr)
        delete _sbvh;

    std::vector<triMesh*> ms;
    ms.push_back(this);
    _bvh  = new bvh(ms);
    _qbvh = new qbvh(_bvh);
    _sbvh = new sbvh(_bvh, 4096);
}

//extern bool triContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3);

//int triMesh::collide(const triMesh* other, const transf& trf0, const transf& trf1, std::vector<id_pair>& rets)
//{
//    cbox2boxTrfCache trf;
//    trf.calc_from_homogenic(trf0, trf1);
//
//    std::vector<id_pair> broadRet;
//
//#if 1
//    _bvh->collideWithStack(other->_bvh, trf, broadRet);
//#else
//    _bvh->collide(other->_bvh, trf, rets);
//#endif
//
//    int num = 0;
//    for (auto t : broadRet)
//    {
//        unsigned int t0, t1;
//        t.get(t0, t1);
//        vec3f v0, v1, v2;
//        other->getTriangleVtxs(t1, v0, v1, v2);
//        vec3f p0 = trf.transform(v0);
//        vec3f p1 = trf.transform(v1);
//        vec3f p2 = trf.transform(v2);
//
//        vec3f q0, q1, q2;
//        this->getTriangleVtxs(t0, q0, q1, q2);
//
//        if (triContact(p0, p1, p2, q0, q1, q2))
//        {
//            rets.push_back(id_pair(t0, t1, false));
//            num++;
//        }
//    }
//
//#if 0
//	const transf trfA2B = trf1.inverse() * trf0;
//	int num2 = 0;
//	for (auto t : broadRet) {
//		unsigned int t0, t1;
//		t.get(t0, t1);
//		vec3f v0, v1, v2;
//		this->getTriangleVtxs(t0, v0, v1, v2);
//		vec3f p0 = trfA2B.getVertex(v0);
//		vec3f p1 = trfA2B.getVertex(v1);
//		vec3f p2 = trfA2B.getVertex(v2);
//
//		vec3f q0, q1, q2;
//		other->getTriangleVtxs(t1, q0, q1, q2);
//
//		if (triContact(p0, p1, p2, q0, q1, q2)) {
//			num2++;
//		}
//	}
//
//	if (num != num2)
//		printf("ERE!\n");
//#endif
//
//    return broadRet.size();
//}
//
//void triMesh::query2(const BOX& bx, std::vector<int>& rets)
//{
//    unsigned short minPt[3], maxPt[3];
//    _qbvh->quantizePoint(minPt, bx.getMin());
//    _qbvh->quantizePoint(maxPt, bx.getMax());
//
//    _sbvh->query(minPt, maxPt, rets);
//}
//
//void triMesh::query(const BOX& bx, std::vector<int>& rets)
//{
//#if 0
//	_bvh->query(bx, rets);
//#else
//    unsigned short minPt[3], maxPt[3];
//    _qbvh->quantizePoint(minPt, bx.getMin());
//    _qbvh->quantizePoint(maxPt, bx.getMax());
//
//    _qbvh->query(minPt, maxPt, rets);
//#endif
//}

void triMesh::rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret)
{
    _bvh->rayCasting(pt, dir, this, ret);
}

void triMesh::rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f& dirR, REAL& ret)
{
#if 1
    _bvh->rayCasting(pt, dir, this, ret);
    //_bvh->rayCasting2(pt, dir, dirR, this, ret);
    //_bvh->rayCasting3(pt, dir, dirR, this, ret);
#else
    double r1 = ret;
    _bvh->rayCasting3(pt, dir, dirR, this, r1);

    double r2 = ret;
    _bvh->rayCasting2(pt, dir, dirR, this, r2);

    if (r1 != r2)
        printf("Error: something is wrong");

    ret = r1;
#endif
}

extern std::vector<vec3f> vtxset;

void drawOther()
{
}

#ifdef GPU
extern void pushMesh2(int, int, int, void*, void*, int, void*, void*, void*, void*, int, int);

void triMesh::push2G(int i)
{
    pushMesh2(i, _num_vtx, _num_tri, _tris, _vtxs, _bvh->num(), _bvh->root(), _qbvh->root(), _sbvh->upperNodes(), _sbvh->lowerNodes(), _sbvh->upperNum(), _sbvh->lowerNum());
}

#if 0
extern void pushRigid2(int, void*, REAL*);

void
crigid::push2G(int i)
{
	pushRigid2(i, &_trf, _off.v);
}
#endif
#endif

float getLargestVelocityNorm(crigid* body1, crigid* body2)
{
    vec3f v1 = body1->getLinearVelocity();
    vec3f v2 = body2->getLinearVelocity();
    return (v1 - v2).length2();
}

int crigid::checkCollision(crigid* rb, std::vector<id_pair>& pairs)
{
    //crigid*       ra   = this;
    //const transf& trfA = ra->getTrf();
    //const transf& trfB = rb->getTrf();
    //return ra->getMesh()->collide(rb->getMesh(), trfA, trfB, pairs);
    return 0;
}