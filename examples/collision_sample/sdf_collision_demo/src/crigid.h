#pragma once
#include "collision/internal/collision_transf.hpp"
#include "collision/internal/collision_tri3f.hpp"
#include "collision/internal/collision_mat3f.hpp"
#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/collision_aabb.hpp"
#include "collision/internal/collision_pair.hpp"
#include "collision/internal/collision_box.hpp"
#include <stdio.h>

//#include "vec3f.h"
//#include "mat3f.h"
//#include "transf.h"
//
//#include "tri.h"
//#include "box.h"
//#include "pair.h"

#include <set>
#include <vector>
using namespace std;
using Physika::vec3f;
using Physika::vec2f;
using Physika::matrix3f;
using Physika::transf;
using Physika::quaternion;
using Physika::tri3f;
using Physika::aabb;
using Physika::id_pair;
using Physika::cbox2boxTrfCache;
class bvh;
class qbvh;
class sbvh;

#define M_PI_2 3.1415926
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

class DistanceField3D;

class kmesh {
public:
	unsigned int _num_vtx;
	unsigned int _num_tri;
	tri3f *_tris;
	vec3f *_vtxs;

	vec3f* _fnrms;
	vec3f *_nrms;
	aabb<REAL> *_bxs;
    aabb<REAL>  _bx;
	bvh* _bvh;
	qbvh* _qbvh;
	sbvh* _sbvh;
	
	int _dl;
	DistanceField3D* _sdf;

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
		_sdf = nullptr;

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
		destroySDF();
	}

	void setSDF(DistanceField3D* sdf) {
		_sdf = sdf;
	}

	unsigned int getNbVertices() const { return _num_vtx; }
	unsigned int getNbFaces() const { return _num_tri; }
	vec3f *getVtxs() const { return _vtxs; }
	vec3f* getNrms() const { return _nrms; }
	vec3f* getFNrms() const { return _fnrms; }
	tri3f* getTris() const { return _tris; }
	void getTriangleVtxs(int fid, vec3f& v0, vec3f& v1, vec3f& v2) const
	{
		tri3f& f = _tris[fid];
		v0 = _vtxs[f.id0()];
		v1 = _vtxs[f.id1()];
		v2 = _vtxs[f.id2()];
	}

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
	//finish sdf
	void destroySDF();

	void updateBxs() {
		if (_bxs == nullptr)
            _bxs = new aabb<REAL>[_num_tri];

		_bx.init();

		for (int i = 0; i < _num_tri; i++) {
			tri3f &a = _tris[i];
			vec3f p0 = _vtxs[a.id0()];
			vec3f p1 = _vtxs[a.id1()];
			vec3f p2 = _vtxs[a.id2()];

			aabb<REAL> bx(p0, p1);
			bx += p2;
			_bxs[i] = bx;

			_bx += bx;
		}
	}

	aabb<REAL> bound() {
		return _bx;
	}

	void rayCasting(const vec3f& pt, const vec3f& dir, REAL& ret);
	void rayCasting2(const vec3f& pt, const vec3f& dir, const vec3f &dirR, REAL& ret);
    void query(const aabb<REAL>& bx, std::vector<int>& rets);
    void query2(const aabb<REAL>& bx, std::vector<int>& rets);

	int collide(const kmesh* other, const transf& trf0, const transf &trf1, std::vector<id_pair>& rets);
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
	aabb<REAL> _bx;
	transf _worldTrf;

	void updateBx() {

		_bx = _mesh->bound();
		_bx.applyTransform(_worldTrf);
	}

	//for bvh
	int _levelSt;
	int _levelNum;

public:
	crigid(kmesh* km, const transf& trf, float mass) {
		_mesh = km;

		vec3f inertia;
		setupRigidBody(trf, mass, inertia);

		updateBx();
		_levelSt = -1;
	}


	~crigid() {
		NULL;
	}

	void setID(int i) { _id = i; }
	int getID() { return _id; }

	aabb<REAL> bound()
    {
		return _bx;
	}

	kmesh* getMesh() {
		return _mesh;
	}

	const transf &getTrf() const {
		return _worldTrf;
	}

	const matrix3f &getRot() const
	{
		return _worldTrf._trf;
	}

	const vec3f &getOffset() const
	{
		return _worldTrf._off;
	}

	void updatePos(matrix3f& rt, vec3f offset = vec3f())
	{
		_worldTrf._trf = rt;
		_worldTrf._off = offset;

		updateBx();
	}

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

	REAL getMinDistance(crigid *, std::vector<vec3f>&);
	REAL getMinDistanceGPU(crigid*, std::vector<vec3f>&);
	int checkCollision(crigid*, std::vector<id_pair>&);
	void checkSdfCollision(crigid*, std::vector<vec3f>&);

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

	__forceinline REAL getInvMass() const { return _invMass; }

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
		_worldTrf = xform;
		updateInertiaTensor();
	}

	void updateInertiaTensor()
	{
		_invInertiaTensorWorld = _worldTrf._trf.scaled(_invInertiaLocal) * _worldTrf._trf.getTranspose();
	}

	void updateContactBx(float threshold)
	{
		updateBx();

		//need to increase the aabb for contact thresholds
		vec3f contactThreshold(threshold, threshold, threshold);
		_bx._min -= contactThreshold;
		_bx._max += contactThreshold;
	}

	inline void outputM(char* name, const matrix3f& m) {
		const REAL* data = m.asColMajor();
		printf("%s =\n       %lf, %lf, %lf\n", name, data[0], data[3], data[6]);
		printf("       %lf, %lf, %lf\n", data[1], data[4], data[7]);
		printf("       %lf, %lf, %lf\n", data[2], data[5], data[8]);
	}

	inline void outputV(char *name, const vec3f &v) {
		printf("%s = %lf, %lf, %lf\n", name, v.x, v.y, v.z);
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

};

