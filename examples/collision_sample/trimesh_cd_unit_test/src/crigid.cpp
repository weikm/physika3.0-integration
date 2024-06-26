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


#if 0
	if (cyl) {//highlight the fset
		std::vector<tri3f> tris;
		for (int i = 0; i < fset.size(); i++) {
			tris.push_back(_tris[fset[i]]);
		}

		glDrawElements(GL_TRIANGLES, fset.size() * 3, GL_UNSIGNED_INT, tris.data());

	}
	else
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
#endif

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

extern bool triContact(vec3f& P1, vec3f& P2, vec3f& P3, vec3f& Q1, vec3f& Q2, vec3f& Q3);

int kmesh::collide(const kmesh* other,  const transf &trf0, const transf &trf1, std::vector<id_pair>& rets)
{
	cbox2boxTrfCache trf;
	trf.calc_from_homogenic(trf0, trf1);

	std::vector<id_pair> broadRet;

#if 1
	_bvh->collideWithStack(other->_bvh, trf, broadRet);
#else
	_bvh->collide(other->_bvh, trf, rets);
#endif

	int num = 0;
	for (auto t : broadRet) {
		unsigned int t0, t1;
		t.get(t0, t1);
		vec3f v0, v1, v2;
		other->getTriangleVtxs(t1, v0, v1, v2);
		vec3f p0 = trf.transform(v0);
		vec3f p1 = trf.transform(v1);
		vec3f p2 = trf.transform(v2);

		vec3f q0, q1, q2;
		this->getTriangleVtxs(t0, q0, q1, q2);

		if (triContact(p0, p1, p2, q0, q1, q2)) {
			rets.push_back(id_pair(t0, t1, false));
			num++;
		}
	}

#if 0
	const transf trfA2B = trf1.inverse() * trf0;
	int num2 = 0;
	for (auto t : broadRet) {
		unsigned int t0, t1;
		t.get(t0, t1);
		vec3f v0, v1, v2;
		this->getTriangleVtxs(t0, v0, v1, v2);
		vec3f p0 = trfA2B.getVertex(v0);
		vec3f p1 = trfA2B.getVertex(v1);
		vec3f p2 = trfA2B.getVertex(v2);

		vec3f q0, q1, q2;
		other->getTriangleVtxs(t1, q0, q1, q2);

		if (triContact(p0, p1, p2, q0, q1, q2)) {
			num2++;
		}
	}

	if (num != num2)
		printf("ERE!\n");
#endif

	return broadRet.size();
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

int
crigid::checkCollision(crigid *rb, std::vector<id_pair>&pairs)
{
	crigid* ra = this;
	const transf& trfA = ra->getTrf();
	const transf& trfB = rb->getTrf();
	return ra->getMesh()->collide(rb->getMesh(), trfA, trfB, pairs);
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
	BOX bx = mB->bound();
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

REAL
crigid::getMinDistance(crigid* other, std::vector<vec3f>&rets)
{
	const transf& trfA = this->getTrf();
	const transf& trfB = other->getTrf();
	const transf trfA2B = trfB.inverse() * trfA;
	cbox2boxTrfCache trf2to1;
	trf2to1.calc_from_homogenic(trfA, trfB);

	kmesh*mA = this->getMesh();
	kmesh* mB = other->getMesh();

	vec3f p0 = mA->getVtxs()[0];
	vec3f p1 = mB->getVtxs()[0];
	vec3f pp0 = trfA2B.getVertex(p0);

	REAL minDist = (pp0 - p1).squareLength();
	rets.push_back(pp0);
	rets.push_back(p1);

#ifdef PARAL
	mA->_bvh->distance(mB->_bvh, trf2to1, trfA2B, mA, mB, minDist, rets, true);
#else
	mA->_bvh->distance(mB->_bvh, trf2to1, trfA2B, mA, mB, minDist, rets, false);
#endif
	return minDist;
}

extern void checkRigidRigidDistanceGPU0(REAL& iDist, void* iPtA, void* iPtB);
extern void checkRigidRigidDistanceGPU1(REAL& iDist, void* iPtA, void* iPtB);
extern void checkRigidRigidDistanceGPU2(REAL& iDist, void* iPtA, void* iPtB, int, int);

REAL
crigid::getMinDistanceGPU(crigid* other, std::vector<vec3f>& rets)
{
#ifdef GPU
	const transf& trfA = this->getTrf();
	const transf& trfB = other->getTrf();
	const transf trfA2B = trfB.inverse() * trfA;
	cbox2boxTrfCache trf2to1;
	trf2to1.calc_from_homogenic(trfA, trfB);

	kmesh* mA = this->getMesh();
	kmesh* mB = other->getMesh();

	vec3f p0 = mA->getVtxs()[0];
	vec3f p1 = mB->getVtxs()[0];
	vec3f pp0 = trfA2B.getVertex(p0);

	REAL minDist = (pp0 - p1).squareLength();

	if (_levelSt == -1) {
		static std::vector<int> tmpNodes;
		bvh_node* root = mA->_bvh->root();
		root->getChilds(10, root, tmpNodes);
		int st = tmpNodes.front();
		int ed = tmpNodes.back();
		if ((ed - st + 1) != tmpNodes.size()) {
			printf("Something is wrong...\n");
		}
		_levelSt = st;
		_levelNum = tmpNodes.size();
	}

	//checkRigidRigidDistanceGPU0(minDist, pp0.v, p1.v);
	checkRigidRigidDistanceGPU1(minDist, pp0.v, p1.v);
	//checkRigidRigidDistanceGPU2(minDist, pp0.v, p1.v, _levelSt, _levelNum);
	rets.push_back(pp0);
	rets.push_back(p1);

	return minDist;
#else
	return 0;
#endif
}
