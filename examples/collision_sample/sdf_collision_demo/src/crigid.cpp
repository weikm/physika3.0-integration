#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>

//#include "mesh_defs.h"
#include "crigid.h"
#include "sdf.h"

#include <set>
using namespace std;

extern std::vector<int> fset;
extern std::set<int> vset;
extern vec3f projDir;
extern REAL maxDist;

//#define FOR_VOLMESH
//#define FOR_SDF

// for fopen
#pragma warning(disable: 4996)

inline vec3f update(vec3f &v1, vec3f &v2, vec3f &v3)
{
	vec3f s = (v2-v1);
	return s.cross(v3-v1);
}

inline vec3f
update(tri3f &tri, vec3f *vtxs)
{
	vec3f &v1 = vtxs[tri.id0()];
	vec3f &v2 = vtxs[tri.id1()];
	vec3f &v3 = vtxs[tri.id2()];

	return update(v1, v2, v3);
}

void kmesh::updateNrms()
{
	if (_fnrms == nullptr)
		_fnrms = new vec3f[_num_tri];

	if (_nrms == nullptr)
		_nrms = new vec3f[_num_vtx];

	for (unsigned int i = 0; i < _num_tri; i++) {
		vec3f n = ::update(_tris[i], _vtxs);
		n.normalize();
		_fnrms[i] = n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i] = vec3f::zero();

	for (unsigned int i=0; i<_num_tri; i++) {
		vec3f& n = _fnrms[i];
		_nrms[_tris[i].id0()] += n;
		_nrms[_tris[i].id1()] += n;
		_nrms[_tris[i].id2()] += n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i].normalize();
}

void kmesh::display(bool cyl, int level)
{
	if (_dl == -1)
		updateDL(cyl, level);

	glCallList(_dl);
	displayDynamic(cyl, level);
}

#if defined(FOR_VOLMESH) //|| defined(FOR_SDF)
void kmesh::displayStatic(bool cyl, int level)
{
	//draw nodes
	glDisable(GL_LIGHTING);
	glPointSize(2.0);
	glColor3f(0, 1, 0);
	glBegin(GL_POINTS);
	for (int i = 0; i < _num_vtx; i++) {
		glVertex3dv(_vtxs[i].v);
	}
	glEnd();
	for (int i = 0; i < _num_tri; i++) {
		tri3f& t = _tris[i];
		glBegin(GL_LINE_LOOP);
		glVertex3dv(_vtxs[t.id0()].v);
		glVertex3dv(_vtxs[t.id1()].v);
		glVertex3dv(_vtxs[t.id2()].v);
		glEnd();
	}
	glEnable(GL_LIGHTING);
}
#else
void kmesh::displayStatic(bool cyl, int level)
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
#endif

extern vec3f dPt0, dPt1, dPtw;

void kmesh::displayDynamic(bool cyl, int level)
{

}

void kmesh::destroyDL()
{
	if (_dl != -1)
		glDeleteLists(_dl, 1);
}

void kmesh::updateDL(bool cyl, int level)
{
	if (_dl != -1)
		destroyDL();

	_dl = glGenLists(1);
	glNewList(_dl, GL_COMPILE);
	displayStatic(cyl, level);
	glEndList();
}

void kmesh::destroySDF()
{
	if (_sdf != nullptr)
		delete _sdf;
}

void kmesh::destroyBVH()
{
	if (_bvh != nullptr)
		delete _bvh;

	if (_qbvh != nullptr)
		delete _qbvh;

	if (_sbvh != nullptr)
		delete _sbvh;
}

void kmesh::updateBVH(bool cyl)
{

}

extern bool triContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3);

int kmesh::collide(const kmesh* other,  const transf &trf0, const transf &trf1, std::vector<id_pair>& rets)
{
    return 0;
}

void kmesh::query2(const aabb<REAL>& bx, std::vector<int>& rets)
{

}

void kmesh::query(const aabb<REAL>& bx, std::vector<int> &rets)
{

}

void kmesh::rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret)
{

}

void kmesh::rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, REAL& ret)
{

}

void drawOther()
{

}

void
crigid::checkSdfCollision(crigid*other, std::vector<vec3f>&rets)
{
	const transf& trfA = this->getTrf();
	const transf& trfB = other->getTrf();
	const transf trfA2B = trfB.inverse() * trfA;

	kmesh* mA = this->getMesh();
	kmesh* mB = other->getMesh();

	vec3f *Avtxs = mA->getVtxs();
	aabb<REAL> bx = mB->bound();
	int Anum = mA->getNbVertices();
	for (int i = 0; i < Anum; i++) {
		vec3f& pt = Avtxs[i];
		vec3f ppt = trfA2B.getVertex(pt);

		REAL d;
		vec3f nrm;
		if (bx.inside(ppt)) {
			mB->_sdf->getDistance(ppt, d, nrm);
			if (d < 0) {
				REAL dd;
				vec3f dnrm;

				mB->_sdf->getDistance(ppt, dd, dnrm);

				rets.push_back(ppt);
				rets.push_back(nrm);
			}
		}
	}

}