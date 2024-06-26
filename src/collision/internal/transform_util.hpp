#pragma once
#include "collision/internal/collision_define.hpp"
#include "collision/internal/collision_transf.hpp"
namespace Physika {
class TransformUtil
{
public:
    static void integrateTransform(const transf& curTrans, const vec3f& linvel, const vec3f& angvel, float timeStep, transf& predictedTransform)
    {
        predictedTransform.setOrigin(curTrans.getOrigin() + linvel * timeStep);
        // Exponential map
        // google for "Practical Parameterization of Rotations Using the Exponential Map", F. Sebastian Grassia

        vec3f axis;
        float fAngle = angvel.length();
        // limit the angular motion
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
}  // namespace Physika
