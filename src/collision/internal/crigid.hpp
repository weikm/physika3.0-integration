#pragma once
#include "collision/internal/collision_mat3f.hpp"
#include "collision/internal/collision_vec3.hpp"
#include "collision/internal/trimesh.hpp"
#include "collision/internal/transform_util.hpp"

// island management, _activationState1
#define ACTIVE_TAG 1
#define ISLAND_SLEEPING 2
#define WANTS_DEACTIVATION 3
#define DISABLE_DEACTIVATION 4
#define DISABLE_SIMULATION 5

class crigid
{
private:
    Physika::vec3f _linearVelocity;
    Physika::vec3f _angularVelocity;
    REAL           _invMass;

    Physika::vec3f _invInertiaLocal;
    // Physika::vec3f _invInertiaTensor;
    Physika::matrix3f _invInertiaTensorWorld;

    REAL  _angularDamping;
    REAL  _linearDamping;
    float _linearSleepingThreshold;
    float _angularSleepingThreshold;

    Physika::vec3f _totalForce;
    Physika::vec3f _totalTorque;

    // REAL _mass;
    Physika::vec3f _gravity;
    int            _activationState1;
    float          _deactivationTime;

    // uninitialized data
    Physika::transf _interpolationWorldTransform;
    Physika::vec3f  _interpolationLinearVelocity;
    Physika::vec3f  _interpolationAngularVelocity;

    // index for solverBody
    int _companionId;

private:
    Physika::triMesh*  _mesh;
    Physika::BOX<REAL> _bx;
    Physika::transf    _worldTrf;

    void updateBx()
    {
#if 0
		aabb bx = _mesh->bound();
		std::vector<Physika::vec3f> crns;
		bx.getCorners(crns);

		for (int i = 0; i < crns.size(); i++) {
			Physika::vec3f& p = crns[i];
			Physika::vec3f pp = _worldTrf._trf * p + _worldTrf._off;
			if (i == 0)
				_bx = pp;
			else
				_bx += pp;
		}

#else
        _bx = _mesh->bound();
        _bx.applyTransform(_worldTrf);
#endif
    }

public:
    crigid(Physika::triMesh* km, const Physika::transf& trf, float mass)
    {
        _mesh = km;

        Physika::vec3f inertia;
        km->calculateInertia(mass, inertia);
        setupRigidBody(trf, mass, inertia);

        updateBx();
    }

    ~crigid()
    {
        NULL;
    }

    Physika::BOX<REAL> bound()
    {
        return _bx;
    }

    Physika::triMesh* getMesh()
    {
        return _mesh;
    }

    const Physika::transf& getTrf() const
    {
        return _worldTrf;
    }

    const Physika::matrix3f& getRot() const
    {
        return _worldTrf._trf;
    }

    const Physika::vec3f& getOffset() const
    {
        return _worldTrf._off;
    }

    void updatePos(Physika::matrix3f& rt, Physika::vec3f offset = Physika::vec3f())
    {
        _worldTrf._trf = rt;
        _worldTrf._off = offset;

        updateBx();
    }

    int getVertexCount()
    {
        return _mesh->getNbVertices();
    }

    Physika::vec3f getVertex(int i)
    {
        Physika::vec3f& p = _mesh->getVtxs()[i];
        return _worldTrf.getVertex(p);
    }

#ifdef GPU
    // void push2G(int i);
#endif

    ////////////////////////////
    void setupRigidBody(const Physika::transf& trf, float mass, const Physika::vec3f& inertia)
    {
        _linearVelocity           = Physika::vec3f();
        _angularVelocity          = Physika::vec3f();
        _totalForce               = Physika::vec3f();
        _totalTorque              = Physika::vec3f();
        _angularDamping           = 0;
        _linearDamping            = 0;
        _linearSleepingThreshold  = float(0.8);
        _angularSleepingThreshold = float(1.f);

        _worldTrf         = trf;
        _activationState1 = ACTIVE_TAG;
        _deactivationTime = 0;

        // setMassProps(mass, localInertia);
        _invMass         = 1.f / mass;
        _gravity         = Physika::vec3f(0, -10, 0) * mass;
        _invInertiaLocal = Physika::vec3f(
            inertia.x != float(0.0) ? float(1.0) / inertia.x : float(0.0),
            inertia.y != float(0.0) ? float(1.0) / inertia.y : float(0.0),
            inertia.z != float(0.0) ? float(1.0) / inertia.z : float(0.0));

        updateInertiaTensor();
    }

    __forceinline const Physika::vec3f& getTotalForce() const
    {
        return _totalForce;
    }

    __forceinline const Physika::vec3f& getTotalTorque() const
    {
        return _totalTorque;
    }

    __forceinline float getInvMass() const
    {
        return _invMass;
    }

    __forceinline const Physika::matrix3f& getInvInertiaTensorWorld() const
    {
        return _invInertiaTensorWorld;
    }

    __forceinline Physika::transf& getWorldTransform()
    {
        return _worldTrf;
    }

    __forceinline void setWorldTransform(const Physika::transf& worldTrans)
    {
        _worldTrf = worldTrans;
    }

    __forceinline const Physika::vec3f& getLinearVelocity() const
    {
        return _linearVelocity;
    }

    __forceinline const Physika::vec3f& getAngularVelocity() const
    {
        return _angularVelocity;
    }

    __forceinline void setLinearVelocity(const Physika::vec3f& lin_vel)
    {
        _linearVelocity = lin_vel;
    }

    __forceinline void setAngularVelocity(const Physika::vec3f& ang_vel)
    {
        _angularVelocity = ang_vel;
    }

    __forceinline void setDeactivationTime(float time)
    {
        _deactivationTime = time;
    }

    __forceinline float getDeactivationTime() const
    {
        return _deactivationTime;
    }

    __forceinline int getActivationState() const
    {
        return _activationState1;
    }

    __forceinline void setActivationState(int newState)
    {
        _activationState1 = newState;
    }

    __forceinline bool isActive() const
    {
        return ((getActivationState() != ISLAND_SLEEPING) && (getActivationState() != DISABLE_SIMULATION));
    }

    void applyGravity()
    {
        applyCentralForce(_gravity);
    }

    void applyCentralForce(const Physika::vec3f& force)
    {
        _totalForce += force;
    }

    void clearForces()
    {
        _totalForce  = Physika::vec3f::zero();
        _totalTorque = Physika::vec3f::zero();
    }

    __forceinline void updateDeactivation(float timeStep)
    {
        if ((getActivationState() == ISLAND_SLEEPING) || (getActivationState() == DISABLE_DEACTIVATION))
            return;

        if ((getLinearVelocity().length2() < _linearSleepingThreshold * _linearSleepingThreshold) && (getAngularVelocity().length2() < _angularSleepingThreshold * _angularSleepingThreshold))
        {
            _deactivationTime += timeStep;
        }
        else
        {
            _deactivationTime = float(0.);
            setActivationState(0);
        }
    }

    __forceinline bool wantsSleeping()
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
        return _companionId;
    }

    __forceinline void setCompanionId(int id)
    {
        _companionId = id;
    }

    /// applyDamping damps the velocity, using the given m_linearDamping and m_angularDamping
    void applyDamping(float timeStep)
    {
        _linearVelocity *= powf(float(1) - _linearDamping, timeStep);
        _angularVelocity *= powf(float(1) - _angularDamping, timeStep);
    }

    Physika::vec3f getSpatialVelocity(const Physika::vec3f& wpos) const
    {
        Physika::vec3f lpos = _worldTrf.getVertexInv(wpos);
        return getVelocityInLocalPoint(lpos);
    }

    Physika::vec3f getVelocityInLocalPoint(const Physika::vec3f& rel_pos) const
    {
        // we also calculate lin/ang velocity for kinematic objects
        return _linearVelocity + _angularVelocity.cross(rel_pos);

        // for kinematic objects, we could also use use:
        //		return 	(m_worldTransform(rel_pos) - m_interpolationWorldTransform(rel_pos)) / m_kinematicTimeStep;
    }

    Physika::transf& getInterpolationWorldTransform()
    {
        return _interpolationWorldTransform;
    }

    void predictIntegratedTransform(float timeStep, Physika::transf& predictedTransform)
    {
        Physika::TransformUtil::integrateTransform(_worldTrf, _linearVelocity, _angularVelocity, timeStep, predictedTransform);
    }

    void proceedToTransform(const Physika::transf& newTrans)
    {
        setCenterOfMassTransform(newTrans);
    }

    void setCenterOfMassTransform(const Physika::transf& xform)
    {
        _interpolationWorldTransform  = xform;
        _interpolationLinearVelocity  = getLinearVelocity();
        _interpolationAngularVelocity = getAngularVelocity();
        _worldTrf                     = xform;
        updateInertiaTensor();
    }

    void updateInertiaTensor()
    {
        _invInertiaTensorWorld = _worldTrf._trf.scaled(_invInertiaLocal) * _worldTrf._trf.getTranspose();
    }

    void updateContactBx(float threshold)
    {
        updateBx();

        // need to increase the aabb for contact thresholds
        Physika::vec3f contactThreshold(threshold, threshold, threshold);
        _bx._min -= contactThreshold;
        _bx._max += contactThreshold;
    }

    inline void outputM(char* name, const Physika::matrix3f& m)
    {
        const float* data = m.asColMajor();
        printf("%s =\n       %f, %f, %f\n", name, data[0], data[3], data[6]);
        printf("       %f, %f, %f\n", data[1], data[4], data[7]);
        printf("       %f, %f, %f\n", data[2], data[5], data[8]);
    }

    inline void outputV(char* name, const Physika::vec3f& v)
    {
        printf("%s = %f, %f, %f\n", name, v.x, v.y, v.z);
    }
    inline void outputT(char* name, const Physika::transf& t)
    {
        const Physika::vec3f& tt = t.getOrigin();
        outputV(name, tt);
        const Physika::matrix3f& r = t.getBasis();
        outputM(name, r);
    }

    void output(int i)
    {
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
