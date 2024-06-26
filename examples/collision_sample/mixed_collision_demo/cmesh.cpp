#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <GL/gl.h>

#include "crigid.h"
#include "collision/internal/collision_bvh.hpp"
#include <stdio.h>
#include <string.h>

extern aabb<REAL> g_projBx;

#include <set>
using Physika::bvh;
using Physika::bvh_node;

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

#pragma warning(disable : 4996)

void beginDraw(aabb<REAL>& bx)
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    vec3f pt  = bx.center();
    REAL  len = bx.height() + bx.depth() + bx.width();
    REAL  sc  = 6.0 / len;

    // glRotatef(-90, 0, 0, 1);
    glScalef(sc, sc, sc);
    glTranslatef(-pt.x, -pt.y, -pt.z);
}

void endDraw()
{
    glPopMatrix();
}

#include "GL/glut.h"

#include "crigid.h"

void drawRigid(crigid* r, bool cyl, int level, vec3f& off)
{
    glPushMatrix();
    glTranslated(off.x, off.y, off.z);

    // glLoadIdentity();
    matrix3f              R   = r->getRot();
    vec3f                 T   = r->getOffset();
    std::vector<GLdouble> Twc = { R(0, 0), R(1, 0), R(2, 0), 0., R(0, 1), R(1, 1), R(2, 1), 0., R(0, 2), R(1, 2), R(2, 2), 0., T[0], T[1], T[2], 1. };

    glMultMatrixd(Twc.data());
    r->getMesh()->display(cyl, level);
    glPopMatrix();
}

void getTriangleVtxs(triMesh* km, int fid, vec3f& v0, vec3f& v1, vec3f& v2)
{
    tri3f& f = km->_tris[fid];
    v0       = km->_vtxs[f.id0()];
    v1       = km->_vtxs[f.id1()];
    v2       = km->_vtxs[f.id2()];
}

void drawCDPair(crigid* r0, crigid* r1, std::vector<id_pair>& pairs)
{
    if (pairs.size() == 0)
        return;

    transf& t0  = r0->getWorldTransform();
    transf& t1  = r1->getWorldTransform();
    triMesh*  km0 = r0->getMesh();
    triMesh*  km1 = r1->getMesh();

    glDisable(GL_LIGHTING);
    {
        glColor3f(1, 0, 0);
        glBegin(GL_TRIANGLES);
        for (auto t : pairs)
        {
            unsigned int fid0, fid1;
            t.get(fid0, fid1);

            vec3f v0, v1, v2;
            getTriangleVtxs(km0, fid0, v0, v1, v2);
            vec3f p0 = t0.getVertex(v0);
            vec3f p1 = t0.getVertex(v1);
            vec3f p2 = t0.getVertex(v2);

#ifdef USE_DOUBLE
            glVertex3dv(p0.v);
            glVertex3dv(p1.v);
            glVertex3dv(p2.v);
#else
            glVertex3fv(p0.v);
            glVertex3fv(p1.v);
            glVertex3fv(p2.v);
#endif

            getTriangleVtxs(km1, fid1, v0, v1, v2);
            p0 = t1.getVertex(v0);
            p1 = t1.getVertex(v1);
            p2 = t1.getVertex(v2);
#ifdef USE_DOUBLE
            glVertex3dv(p0.v);
            glVertex3dv(p1.v);
            glVertex3dv(p2.v);
#else
            glVertex3fv(p0.v);
            glVertex3fv(p1.v);
            glVertex3fv(p2.v);
#endif
        }
        glEnd();
    }

    glEnable(GL_LIGHTING);
}

void drawSdfPair(crigid* r0, crigid* r1, std::vector<vec3f>& pairs)
{
    if (pairs.size() == 0)
        return;

    transf& t0 = r0->getWorldTransform();
    transf& t1 = r1->getWorldTransform();
    REAL    dt = 0.2;

    glDisable(GL_LIGHTING);
    {
        glColor3f(1, 0, 0);
        glPointSize(5.0);
        int num = pairs.size() / 2;
        glBegin(GL_POINTS);
        for (int i = 0; i < num; i++)
        {
            vec3f rp0 = pairs[i * 2];
            vec3f rp1 = pairs[i * 2 + 1];

            vec3f p0 = t1.getVertex(rp0);
            // vec3f p1 = t1.getVertex(rp0+rp1*dt);

#ifdef USE_DOUBLE
            glVertex3dv(p0.v);
            // glVertex3dv(p1.v);
#else
            glVertex3fv(p0.v);
            //glVertex3fv(p1.v);
#endif
        }
        glEnd();
        glEnd();
    }
    if (1)
    {
        glColor3f(0, 1, 1);
        glLineWidth(2.0);
        int num = pairs.size() / 3;
        glBegin(GL_LINES);
        for (int i = 0; i < num; i++)
        {
            vec3f rp0 = pairs[i * 2];
            vec3f rp1 = pairs[i * 2 + 1];

            vec3f p0 = t1.getVertex(rp0);
            vec3f p1 = t1.getVertex(rp0 + rp1 * dt);

#ifdef USE_DOUBLE
            glVertex3dv(p0.v);
            glVertex3dv(p1.v);
#else
            glVertex3fv(p0.v);
            glVertex3fv(p1.v);
#endif
        }
        glEnd();
    }
    glEnable(GL_LIGHTING);
}