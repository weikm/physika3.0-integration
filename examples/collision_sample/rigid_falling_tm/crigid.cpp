#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>

//#include "mesh_defs.h"
#include "crigid.h"
#include "box.h"
#include "tmbvh.h"
#include "qbvh.h"

#include <set>
using namespace std;

extern std::vector<int> fset;
extern std::set<int> vset;
extern vec3f projDir;
extern REAL maxDist;

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


#if 0
	if (cyl) {//highlight the fset
		std::vector<tri3f> tris;
		for (int i = 0; i < fset.size(); i++) {
			tris.push_back(_tris[fset[i]]);
		}

		glDrawElements(GL_TRIANGLES, fset.size() * 3, GL_UNSIGNED_INT, tris.data());

	} else
		glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);
#else
	glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);
#endif

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

#if 0
	if (cyl) {//highlight the vset
		glColor3f(1, 1, 1);
		glLineWidth(2.0);

		glBegin(GL_LINES);
		for (auto v : vset) {
			vec3f& p = _vtxs[v];
			vec3f p1 = p + projDir * maxDist;
			glVertex3dv(p.v);
			glVertex3dv(p1.v);
		}
		glEnd();
	}
#endif
}

extern vec3f dPt0, dPt1, dPtw;

void kmesh::displayDynamic(bool cyl, int level)
{
	if (level >= 0)
		_bvh->visualize(level);

#if 0
	if (cyl) {
		glDisable(GL_LIGHTING);
		glColor3f(1, 0, 0);
		glPointSize(5.0);
		glBegin(GL_POINTS);
		glColor3f(1, 0, 0);
		glVertex3dv(dPt0.v);
		glColor3f(0, 0, 1);
		glVertex3dv(dPt1.v);
		glEnd();

		glColor3f(0, 1, 0);
		glLineWidth(2.0);
		glBegin(GL_LINES);
		glVertex3dv(dPt0.v);
		glVertex3dv(dPt1.v);
		glEnd();
		glEnable(GL_LIGHTING);
	}
#endif
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
	if (_bvh != nullptr)
		delete _bvh;
	if (_qbvh != nullptr)
		delete _qbvh;
	if (_sbvh != nullptr)
		delete _sbvh;

	std::vector<kmesh*> ms;
	ms.push_back(this);
	_bvh = new bvh(ms);
	_qbvh = new qbvh(_bvh);
	_sbvh = new sbvh(_bvh, 4096);
}

void kmesh::collide(const kmesh* other, const cbox2boxTrfCache& trf, std::vector<id_pair>& rets)
{
#if 1
	_bvh->collideWithStack(other->_bvh, trf, rets);
#else
	_bvh->collide(other->_bvh, trf, rets);
#endif
}

void kmesh::query2(const BOX& bx, std::vector<int>& rets)
{
	unsigned short minPt[3], maxPt[3];
	_qbvh->quantizePoint(minPt, bx.getMin());
	_qbvh->quantizePoint(maxPt, bx.getMax());

	_sbvh->query(minPt, maxPt, rets);
}

void kmesh::query(const BOX& bx, std::vector<int> &rets)
{
#if 0
	_bvh->query(bx, rets);
#else
	unsigned short minPt[3], maxPt[3];
	_qbvh->quantizePoint(minPt, bx.getMin());
	_qbvh->quantizePoint(maxPt, bx.getMax());

	_qbvh->query(minPt, maxPt, rets);
#endif
}

void kmesh::rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret)
{
	_bvh->rayCasting(pt, dir, this, ret);
}

void kmesh::rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, REAL& ret)
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
#if 0
	glDisable(GL_LIGHTING);
	glPointSize(5.0);
	glBegin(GL_POINTS);
	glColor3f(1, 1, 1);
	/*
	for (auto v : vtxset) {
		glVertex3dv(v.v);
	}
	*/
	glVertex3dv(dPtw.v);
	glEnd();
	glEnable(GL_LIGHTING);
#endif
}

#ifdef GPU
extern void pushMesh2(int, int, int, void*, void*, int, void*, void *, void *, void *, int, int);

void
kmesh::push2G(int i)
{
	pushMesh2(i, _num_vtx, _num_tri, _tris, _vtxs, _bvh->num(), _bvh->root(), _qbvh->root(),
		_sbvh->upperNodes(), _sbvh->lowerNodes(), _sbvh->upperNum(), _sbvh->lowerNum());
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

float getLargestVelocityNorm(crigid *body1, crigid *body2)
{
#if 0
	// dont'need to compute this in the common frame... can simply ask how fast each of the points
	// of each bounding volume are moving in the other object's moving frame

	float largestVelocityNorm = 0;

	//for (int i = 0; i < 2; i++)
	int i = 0;
	{
		crigid *body = (i == 0) ? body1 : body2;
		aabb bx = body->getMesh()->bound();
		std::vector<vec3f> crns;
		bx.getCorners(crns);

		for (int i = 0; i < crns.size(); i++) {
			vec3f& p = crns[i];
			vec3f pp = body->getTrf().getVertex(p);

			vec3f v1 = body1->getSpatialVelocity(pp);
			vec3f v2 = body2->getSpatialVelocity(pp);
			largestVelocityNorm = fmax((v1-v2).length(), largestVelocityNorm);
		}
	}

	return largestVelocityNorm;
#else
	vec3f v1 = body1->getLinearVelocity();
	vec3f v2 = body2->getLinearVelocity();
	return (v1 - v2).length2();

#endif

}
