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
    // draw nodes
    glDisable(GL_LIGHTING);
    glPointSize(2.0);
    glColor3f(0, 1, 0);
    glBegin(GL_POINTS);
    for (int i = 0; i < _num_vtx; i++)
    {
        glVertex3fv(_vtxs[i].v);
    }
    glEnd();
    for (int i = 0; i < _num_tri; i++)
    {
        tri3f& t = _tris[i];
        glBegin(GL_LINE_LOOP);
        glVertex3fv(_vtxs[t.id0()].v);
        glVertex3fv(_vtxs[t.id1()].v);
        glVertex3fv(_vtxs[t.id2()].v);
        glEnd();
    }
    glEnable(GL_LIGHTING);
}

extern vec3f dPt0, dPt1, dPtw;

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

void triMesh::rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret)
{
    _bvh->rayCasting(pt, dir, this, ret);
}

void triMesh::rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f& dirR, REAL& ret)
{
    _bvh->rayCasting(pt, dir, this, ret);
}

extern std::vector<vec3f> vtxset;

void drawOther()
{
}

int crigid::checkCollision(crigid* rb, std::vector<id_pair>& pairs)
{
    //crigid*       ra   = this;
    //const transf& trfA = ra->getTrf();
    //const transf& trfB = rb->getTrf();
    //return ra->getMesh()->collide(rb->getMesh(), trfA, trfB, pairs);
    return 0;
}