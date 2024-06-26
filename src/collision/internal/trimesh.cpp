#pragma once
#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/gl.h>
#include "collision/internal/trimesh.hpp"

namespace Physika {
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

triMesh::triMesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs, bool cyl)
{
    _num_vtx = numVtx;
    _num_tri = numTri;
    _tris    = tris;
    _vtxs    = vtxs;

    _fnrms = nullptr;
    _nrms  = nullptr;
    _bxs   = nullptr;
    _dl    = -1;
    _bvh   = nullptr;
    _qbvh  = nullptr;
    _sbvh  = nullptr;

    updateNrms();
    updateBxs();
    updateBVH(cyl);
    updateDL(cyl, -1);
}

triMesh::~triMesh()
{
    delete[] _tris;
    delete[] _vtxs;

    if (_fnrms != nullptr)
        delete[] _fnrms;
    if (_nrms != nullptr)
        delete[] _nrms;
    if (_bxs != nullptr)
        delete[] _bxs;

    destroyDL();
    destroyBVH();
}

unsigned int triMesh::getNbVertices() const
{
    return _num_vtx;
}

unsigned int triMesh::getNbFaces()
{
    return _num_tri;
}

vec3f* triMesh::getVtxs() const
{
    return _vtxs;
}

vec3f* triMesh::getNrms() const
{
    return _nrms;
}

vec3f* triMesh::getFNrms() const
{
    return _fnrms;
}

tri3f* triMesh::getTris() const
{
    return _tris;
}

void triMesh::updateNrms()
{
    if (_fnrms == nullptr)
        _fnrms = new vec3f[_num_tri];

    if (_nrms == nullptr)
        _nrms = new vec3f[_num_vtx];

    for (unsigned int i = 0; i < _num_tri; i++)
    {
        vec3f n = update(_tris[i], _vtxs);
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
//    if (_dl == -1)
//        updateDL(cyl, level);
//
//    glCallList(_dl);
//    displayDynamic(cyl, level);
}

//void triMesh::displayStatic(bool cyl, int level)
//{
////    glEnableClientState(GL_VERTEX_ARRAY);
////    glEnableClientState(GL_NORMAL_ARRAY);
////#ifdef USE_DOUBLE
////    glVertexPointer(3, GL_DOUBLE, sizeof(REAL) * 3, _vtxs);
////    glNormalPointer(GL_DOUBLE, sizeof(REAL) * 3, _nrms);
////#else
////    glVertexPointer(3, GL_FLOAT, sizeof(REAL) * 3, _vtxs);
////    glNormalPointer(GL_FLOAT, sizeof(REAL) * 3, _nrms);
////#endif
////
////    glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);
////
////    glDisableClientState(GL_VERTEX_ARRAY);
////    glDisableClientState(GL_NORMAL_ARRAY);
//}

void triMesh::displayDynamic(bool cyl, int level)
{
    //if (level >= 0)
    //    _bvh->visualize(level);
}

void triMesh::updateDL(bool cyl, int level)
{
    //if (_dl != -1)
    //    destroyDL();

    //_dl = glGenLists(1);
    //glNewList(_dl, GL_COMPILE);
    //displayStatic(cyl, level);
    //glEndList();
}

void triMesh::destroyDL()
{
    //if (_dl != -1)
    //    glDeleteLists(_dl, 1);
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

void triMesh::destroyBVH()
{
    if (_bvh != nullptr)
        delete _bvh;

    if (_qbvh != nullptr)
        delete _qbvh;

    if (_sbvh != nullptr)
        delete _sbvh;
}

void triMesh::updateBxs()
{
    if (_bxs == nullptr)
        _bxs = new BOX<REAL>[_num_tri];

    _bx.init();

    for (int i = 0; i < _num_tri; i++)
    {
        tri3f& a  = _tris[i];
        vec3f  p0 = _vtxs[a.id0()];
        vec3f  p1 = _vtxs[a.id1()];
        vec3f  p2 = _vtxs[a.id2()];

        BOX<REAL> bx(p0, p1);
        bx += p2;
        _bxs[i] = bx;

        _bx += bx;
    }
}

BOX<REAL> triMesh::bound()
{
    return _bx;
}

void triMesh::rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret)
{
    //_bvh->rayCasting(pt, dir, this, ret);
}

void triMesh::rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f& dirR, REAL& ret)
{
    //_bvh->rayCasting(pt, dir, this, ret);
}

void triMesh::query(const BOX<REAL>& bx, std::vector<int>& rets)
{
    unsigned short minPt[3], maxPt[3];
    _qbvh->quantizePoint(minPt, bx.getMin());
    _qbvh->quantizePoint(maxPt, bx.getMax());

    _qbvh->query(minPt, maxPt, rets);
}

void triMesh::query2(const BOX<REAL>& bx, std::vector<int>& rets)
{
    unsigned short minPt[3], maxPt[3];
    _qbvh->quantizePoint(minPt, bx.getMin());
    _qbvh->quantizePoint(maxPt, bx.getMax());

    _sbvh->query(minPt, maxPt, rets);
}

void triMesh::collide(const triMesh* other, const cbox2boxTrfCache& trf, std::vector<id_pair>& rets)
{
}
}  // namespace Physika