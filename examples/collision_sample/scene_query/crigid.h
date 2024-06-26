#pragma once

#include <stdio.h>
#include "vec3f.h"
#include "mat3f.h"

#include "tri.h"
#include "box.h"

#include <set>
#include <vector>
using namespace std;

class bvh;

class kmesh {
public:
	unsigned int _num_vtx;
	unsigned int _num_tri;
	tri3f *_tris;
	vec3f *_vtxs;

	vec3f* _fnrms;
	vec3f *_nrms;
	aabb *_bxs;
	aabb _bx;
	bvh* _bvh;
	
	int _dl;

public:
	kmesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs, bool cyl) {
		_num_vtx = numVtx;
		_num_tri = numTri;
		_tris = tris;
		_vtxs = vtxs;

		_fnrms = nullptr;
		_nrms = nullptr;
		_bxs = nullptr;
		_dl = -1;
		_bvh = nullptr;

		updateNrms();
		updateBxs();
		updateBVH(cyl);
		updateDL(cyl, -1);
	}

	~kmesh() {
		delete[]_tris;
		delete[]_vtxs;

		if (_fnrms != nullptr)
			delete[] _fnrms;
		if (_nrms != nullptr)
			delete[] _nrms;
		if (_bxs != nullptr)
			delete[] _bxs;
		
		destroyDL();
		destroyBVH();
	}

	unsigned int getNbVertices() const { return _num_vtx; }
	unsigned int getNbFaces() const { return _num_tri; }
	vec3f *getVtxs() const { return _vtxs; }
	vec3f* getNrms() const { return _nrms; }
	vec3f* getFNrms() const { return _fnrms; }
	tri3f* getTris() const { return _tris; }

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

	//prepare for bvh
	void updateBVH(bool cyl);
	// finish bvh
	void destroyBVH();

	void updateBxs() {
		if (_bxs == nullptr)
			_bxs = new aabb[_num_tri];

		_bx.init();

		for (int i = 0; i < _num_tri; i++) {
			tri3f &a = _tris[i];
			vec3f p0 = _vtxs[a.id0()];
			vec3f p1 = _vtxs[a.id1()];
			vec3f p2 = _vtxs[a.id2()];

			BOX bx(p0, p1);
			bx += p2;
			_bxs[i] = bx;

			_bx += bx;
		}
	}

	BOX bound() {
		return _bx;
	}

	void rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret);
	void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, REAL& ret);

#ifdef GPU
	void push2G(int i);
#endif
};

class crigid {
private:
	vec3f _linearVelocity;
	vec3f _angularVelocity;
	REAL _invMass;
	vec3f _invInertiaTensor;
	REAL _angularDamping;
	REAL _linearDamping;

	vec3f _totalForce;
	vec3f _totalTorque;

	REAL _mass;
	vec3f _gravity;

	void initPhy() {
		_mass = 4;
		_gravity = vec3f(0, 0, -0.98) * _mass;
		_invMass = 1.0 / _mass;

		_angularDamping = 0;
		_linearDamping = 0;

	}

private:
	kmesh* _mesh;
	vec3f _off;
	matrix3f _trf;
	aabb _bx;

	void updateBx() {
		aabb bx = _mesh->bound();
		std::vector<vec3f> crns;
		bx.getCorners(crns);

		for (int i = 0; i < crns.size(); i++) {
			vec3f& p = crns[i];
			vec3f pp = _trf*p + _off;
			if (i == 0)
				_bx = pp;
			else
				_bx += pp;
		}
	}

public:
	crigid(kmesh*m, vec3f offset = vec3f(), vec3f axis = vec3f(), REAL theta = 0) {
		_mesh = m;
		_off = offset;
		_trf = matrix3f::rotation(axis, (theta / 180.f) * M_PI);

		updateBx();
	}

	~crigid() {
		NULL;
	}

	crigid(kmesh* m, matrix3f &rt, vec3f offset = vec3f()) {
		_mesh = m;
		_off = offset;
		_trf = rt;

		updateBx();
	}

	BOX bound() {
		return _bx;
	}

	kmesh* getMesh() {
		return _mesh;
	}

	matrix3f getTrf() const
	{
		return _trf;
	}
	vec3f getOffset() const
	{
		return _off;
	}

	void updatePos(matrix3f& rt, vec3f offset = vec3f())
	{
		_trf = rt;
		_off = offset;

		updateBx();
	}

#ifdef GPU
	//void push2G(int i);
#endif
};

