#pragma once

#include <stdio.h>
#include "vec3f.h"
#include "mat3f.h"
#include "transf.h"

#include "tri.h"
#include "box.h"
#include "pair.h"

#include <set>
#include <vector>
using namespace std;

class bvh;
class qbvh;
class sbvh;

#define ANGULAR_MOTION_THRESHOLD float(0.5)*M_PI_2

/// Utils related to temporal transforms
class TransformUtil
{
public:

	static void integrateTransform(const transf& curTrans, const vec3f& linvel, const vec3f& angvel, float timeStep, transf& predictedTransform)
	{
		predictedTransform.setOrigin(curTrans.getOrigin() + linvel * timeStep);
		//Exponential map
		//google for "Practical Parameterization of Rotations Using the Exponential Map", F. Sebastian Grassia

		vec3f axis;
		float fAngle = angvel.length();
		//limit the angular motion
		if (fAngle * timeStep > ANGULAR_MOTION_THRESHOLD)
		{
			fAngle = ANGULAR_MOTION_THRESHOLD / timeStep;
		}

		if (fAngle < float(0.001))
		{
			// use Taylor's expansions of sync function
			axis = angvel * (float(0.5) * timeStep - (timeStep * timeStep * timeStep) * (float(0.020833333333)) * fAngle * fAngle);
		}
		else
		{
			// sync(fAngle) = sin(c*fAngle)/t
			axis = angvel * (sinf(float(0.5) * fAngle * timeStep) / fAngle);
		}
		quaternion dorn(axis.x, axis.y, axis.z, cosf(fAngle * timeStep * float(0.5)));
		quaternion orn0 = curTrans.getRotation();

		quaternion predictedOrn = dorn * orn0;
		predictedOrn.normalize();
		predictedTransform.setRotation(predictedOrn);
	}
};

__forceinline vec3f getPointInetia(const vec3f& point, float mass)
{
	float x2 = point[0] * point[0];
	float y2 = point[1] * point[1];
	float z2 = point[2] * point[2];
	return vec3f(mass * (y2 + z2), mass * (x2 + z2), mass * (x2 + y2));
}

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
	qbvh* _qbvh;
	sbvh* _sbvh;
	
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
		_qbvh = nullptr;
		_sbvh = nullptr;

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
	void query(const BOX& bx, std::vector<int>& rets);
	void query2(const BOX& bx, std::vector<int>& rets);

	void collide(const kmesh* other, const cbox2boxTrfCache& trf, std::vector<id_pair>& rets);

#ifdef GPU
	void push2G(int i);
#endif

	void calculateInertia(float mass, vec3f& inertia) {
		inertia = vec3f();

		float pointmass = mass / _num_vtx;

		//for (int i = 0; i < _num_vtx; i++)
		int i = _num_vtx;
		while (i--)
		{
			vec3f& v = _vtxs[i];
			vec3f ptInteria = getPointInetia(v, pointmass);
			inertia += ptInteria;
		}
	}
};

//island management, _activationState1
#define ACTIVE_TAG 1
#define ISLAND_SLEEPING 2
#define WANTS_DEACTIVATION 3
#define DISABLE_DEACTIVATION 4
#define DISABLE_SIMULATION 5

class crigid {
private:
	vec3f _linearVelocity;
	vec3f _angularVelocity;
	REAL _invMass;

	vec3f _invInertiaLocal;
	//vec3f _invInertiaTensor;
	matrix3f _invInertiaTensorWorld;

	REAL _angularDamping;
	REAL _linearDamping;
	float _linearSleepingThreshold;
	float _angularSleepingThreshold;

	vec3f _totalForce;
	vec3f _totalTorque;

	//REAL _mass;
	vec3f _gravity;
	int	_activationState1;
	float _deactivationTime;

	//uninitialized data
	transf	_interpolationWorldTransform;
	vec3f	_interpolationLinearVelocity;
	vec3f	_interpolationAngularVelocity;

	//index for solverBody
	int _companionId;

	//for ptr<->ID
	int _id;

private:
	kmesh* _mesh;
	aabb _bx;
	transf _worldTrf;
	transf _worldTrf0;

	void updateBx(bool ccd) {
#if 0
		aabb bx = _mesh->bound();
		std::vector<vec3f> crns;
		bx.getCorners(crns);

		for (int i = 0; i < crns.size(); i++) {
			vec3f& p = crns[i];
			vec3f pp = _worldTrf._trf * p + _worldTrf._off;
			if (i == 0)
				_bx = pp;
			else
				_bx += pp;
		}

#else
		_bx = _mesh->bound();
		_bx.appyTransform(_worldTrf);

		if (ccd) {
			aabb bx0 = _mesh->bound();
			bx0.appyTransform(_worldTrf0);
			_bx += bx0;
			_bx.enlarge(0.1);
		}
#endif
	}

public:
	crigid(kmesh* km, const transf& trf, float mass) {
		_mesh = km;

		vec3f inertia;
		km->calculateInertia(mass, inertia);
		setupRigidBody(trf, mass, inertia);

		updateBx(false);
	}


	~crigid() {
		NULL;
	}

	void setID(int i) { _id = i; }
	int getID() { return _id; }

	BOX bound() {
		return _bx;
	}

	kmesh* getMesh() {
		return _mesh;
	}

	const transf &getTrf() const {
		return _worldTrf;
	}

	const transf& getTrf0() const {
		return _worldTrf0;
	}

	const matrix3f &getRot() const
	{
		return _worldTrf._trf;
	}

	const vec3f &getOffset() const
	{
		return _worldTrf._off;
	}

#if 0
	void updatePos(matrix3f& rt, vec3f offset = vec3f())
	{
		_worldTrf._trf = rt;
		_worldTrf._off = offset;

		updateBx();
	}
#endif

	int getVertexCount() {
		return _mesh->getNbVertices();
	}

	vec3f getVertex(int i) {
		vec3f& p = _mesh->getVtxs()[i];
		return _worldTrf.getVertex(p);
	}

#ifdef GPU
	//void push2G(int i);
#endif

	////////////////////////////
	void	setupRigidBody(const transf& trf, float mass, const vec3f& inertia)
	{
		_linearVelocity = vec3f();
		_angularVelocity = vec3f();
		_totalForce = vec3f();
		_totalTorque = vec3f();
		_angularDamping = 0;
		_linearDamping = 0;
		_linearSleepingThreshold = float(0.8);
		_angularSleepingThreshold = float(1.f);

		_worldTrf = trf;
		_worldTrf0 = _worldTrf;
		_activationState1 = ACTIVE_TAG;
		_deactivationTime = 0;

		//setMassProps(mass, localInertia);
		_invMass = 1.f / mass;
		_gravity = vec3f(0, -10, 0) * mass;
		_invInertiaLocal = vec3f(
			inertia.x != float(0.0) ? float(1.0) / inertia.x : float(0.0),
			inertia.y != float(0.0) ? float(1.0) / inertia.y : float(0.0),
			inertia.z != float(0.0) ? float(1.0) / inertia.z : float(0.0));

		updateInertiaTensor();
	}

	__forceinline const vec3f& getTotalForce() const
	{
		return _totalForce;
	}

	__forceinline const vec3f& getTotalTorque() const
	{
		return _totalTorque;
	}

	__forceinline float getInvMass() const { return _invMass; }

	__forceinline const matrix3f& getInvInertiaTensorWorld() const {
		return _invInertiaTensorWorld;
	}

	__forceinline transf& getWorldTransform() {
		return _worldTrf;
	}

	__forceinline void	setWorldTransform(const transf& worldTrans) {
		_worldTrf = worldTrans;
	}

	__forceinline const vec3f& getLinearVelocity() const {
		return _linearVelocity;
	}

	__forceinline const vec3f& getAngularVelocity() const {
		return _angularVelocity;
	}

	__forceinline void setLinearVelocity(const vec3f& lin_vel)
	{
		_linearVelocity = lin_vel;
	}

	__forceinline void setAngularVelocity(const vec3f& ang_vel)
	{
		_angularVelocity = ang_vel;
	}

	__forceinline void	setDeactivationTime(float time)
	{
		_deactivationTime = time;
	}

	__forceinline float getDeactivationTime() const
	{
		return _deactivationTime;
	}

	__forceinline int getActivationState() const { return _activationState1; }

	__forceinline void setActivationState(int newState) {
		_activationState1 = newState;
	}

	__forceinline bool isActive() const {
		return ((getActivationState() != ISLAND_SLEEPING) && (getActivationState() != DISABLE_SIMULATION));
	}

	void applyGravity()
	{
		applyCentralForce(_gravity);
	}

	void	applyCentralForce(const vec3f& force)
	{
		_totalForce += force;
	}

	void clearForces()
	{
		_totalForce = vec3f::zero();
		_totalTorque = vec3f::zero();
	}

	__forceinline void	updateDeactivation(float timeStep)
	{
		if ((getActivationState() == ISLAND_SLEEPING) || (getActivationState() == DISABLE_DEACTIVATION))
			return;

		if ((getLinearVelocity().length2() < _linearSleepingThreshold * _linearSleepingThreshold) &&
			(getAngularVelocity().length2() < _angularSleepingThreshold * _angularSleepingThreshold))
		{
			_deactivationTime += timeStep;
		}
		else
		{
			_deactivationTime = float(0.);
			setActivationState(0);
		}

	}

	__forceinline bool	wantsSleeping()
	{
		float gDeactivationTime = float(2.0f);

		if (getActivationState() == DISABLE_DEACTIVATION)
			return false;

		if ((getActivationState() == ISLAND_SLEEPING) || (getActivationState() == WANTS_DEACTIVATION))
			return true;

		if (_deactivationTime > gDeactivationTime)
		{
			return true;
		}
		return false;
	}

	__forceinline int getCompanionId() const
	{
		return	_companionId;
	}

	__forceinline void	setCompanionId(int id)
	{
		_companionId = id;
	}

	///applyDamping damps the velocity, using the given m_linearDamping and m_angularDamping
	void	applyDamping(float timeStep)
	{
		_linearVelocity *= powf(float(1) - _linearDamping, timeStep);
		_angularVelocity *= powf(float(1) - _angularDamping, timeStep);
	}

	vec3f getSpatialVelocity(const vec3f &wpos) const
	{
		vec3f lpos = _worldTrf.getVertexInv(wpos);
		return getVelocityInLocalPoint(lpos);
	}

	vec3f getVelocityInLocalPoint(const vec3f& rel_pos) const
	{
		//we also calculate lin/ang velocity for kinematic objects
		return _linearVelocity + _angularVelocity.cross(rel_pos);

		//for kinematic objects, we could also use use:
		//		return 	(m_worldTransform(rel_pos) - m_interpolationWorldTransform(rel_pos)) / m_kinematicTimeStep;
	}

	transf& getInterpolationWorldTransform()
	{
		return _interpolationWorldTransform;
	}

	void predictIntegratedTransform(float timeStep, transf& predictedTransform)
	{
		TransformUtil::integrateTransform(_worldTrf, _linearVelocity, _angularVelocity, timeStep, predictedTransform);
	}


	void proceedToTransform(const transf& newTrans)
	{
		setCenterOfMassTransform(newTrans);
	}

	void setCenterOfMassTransform(const transf& xform)
	{
		_interpolationWorldTransform = xform;
		_interpolationLinearVelocity = getLinearVelocity();
		_interpolationAngularVelocity = getAngularVelocity();
		_worldTrf0 = _worldTrf;
		_worldTrf = xform;
		updateInertiaTensor();
	}

	void updateInertiaTensor()
	{
		_invInertiaTensorWorld = _worldTrf._trf.scaled(_invInertiaLocal) * _worldTrf._trf.getTranspose();
	}

	void updateContactBx(float threshold, bool ccd)
	{
		updateBx(ccd);

		//need to increase the aabb for contact thresholds
		vec3f contactThreshold(threshold, threshold, threshold);
		_bx._min -= contactThreshold;
		_bx._max += contactThreshold;
	}

	inline void outputM(char* name, const matrix3f& m) {
		const float* data = m.asColMajor();
		printf("%s =\n       %f, %f, %f\n", name, data[0], data[3], data[6]);
		printf("       %f, %f, %f\n", data[1], data[4], data[7]);
		printf("       %f, %f, %f\n", data[2], data[5], data[8]);
	}

	inline void outputV(char *name, const vec3f &v) {
		printf("%s = %f, %f, %f\n", name, v.x, v.y, v.z);
	}
	inline void outputT(char* name, const transf &t) {
		const vec3f& tt = t.getOrigin();
		outputV(name, tt);
		const matrix3f& r = t.getBasis();
		outputM(name, r);
	}

	void output(int i) {
		printf("########################################\n rigid body (%d):\n", i);
		outputV("linearV", _linearVelocity);
		outputV("angularV", _angularVelocity);
		printf("invMass/mass = %f\n", _invMass);
		outputV("invInertiaLocal", _invInertiaLocal);
		outputM("invInertiaTensorWorld", _invInertiaTensorWorld);
		printf("%f, %f, %f, %f\n", _angularDamping, _linearDamping, _linearSleepingThreshold, _angularSleepingThreshold);
		outputV("totalForce", _totalForce);
		outputV("totalTorque", _totalTorque);
		outputV("gravity", _gravity);
		printf("%d, %f\n", _activationState1, _deactivationTime);
		outputT("wroldTrf", _worldTrf);
	}

	int exportObj(FILE *fp, int vtxOffset, int id=0)
	{
		fprintf(fp, "g Rigid%02d\n", id);

		int vtxNum = getVertexCount();
		for (int i = 0; i <vtxNum; i++) {
			vec3f pt = getVertex(i);

			fprintf(fp, "v %f %f %f\n", pt.x, pt.y, pt.z);
		}

		int triNum = _mesh->getNbFaces();
		tri3f* tris = _mesh->_tris;
		for (int i = 0; i < triNum; i++) {
			fprintf(fp, "f %d %d %d\n", tris[i].id0() + 1 + vtxOffset, tris[i].id1() + 1 + vtxOffset, tris[i].id2() + 1 + vtxOffset);
		}
		return vtxNum;
	}
};

