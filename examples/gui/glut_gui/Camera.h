#pragma once
#include <vector>
//#include "Core/Quaternion/quaternion.h"
#include "height_field_sand/core/Vector/vector2D.hpp"
#include "height_field_sand/core/Vector/vector3D.hpp"
#include "height_field_sand/core/Vector/vector4D.hpp"
#include "height_field_sand/core/quaternion.hpp"

#define M_PI 3.14159265358979323846

namespace Physika {

class Camera
{

public:
    Camera();
    ~Camera(){};

    void zoom(float amount);
    void setGL(float neardist, float fardist, float width, float height);

    Vec3f getViewDir() const;

    float getPixelArea() const;
    int   width() const;
    int   height() const;

    Vec3f getEye() const;
    void     setEyePostion(const Vec3f& pos);

    void  setFocus(float focus);
    float getFocus() const;

    void lookAt(const Vec3f& camPos, const Vec3f& center, const Vec3f& updir);

    void getCoordSystem(Vec3f& view, Vec3f& up, Vec3f& right) const;
    void rotate(Quat1f& rotquat);
    void localTranslate(const Vec3f translation);
    void translate(const Vec3f translation);

    /**
        * @brief Yaw around focus point. Rotation axis is m_worldUp.
        */
    void yawAroundFocus(float radian);

    /**
        * @brief Pitch around focus point. Rotation axis is m_right.
        */
    void pitchAroundFocus(float radian);

    /**
        * @brief Yaw around camera position. Rotation axis is m_worldUp.
        */
    void yaw(float radian);

    /**
        * @brief Pitch around camera position. Rotation axis is m_right.
        */
    void pitch(float radian);

    /**
        * @brief Update camera RIGHT, UP and VIEW direction.
        */
    void updateDir();

    void setWorldUp(const Vec3f& worldup)
    {
        m_worldUp = worldup;
    }
    void setLocalView(const Vec3f& localview)
    {
        m_localView = localview;
    }
    void setLocalRight(const Vec3f& localright)
    {
        m_localRight = localright;
    }
    void setLocalUp(const Vec3f& localup)
    {
        m_localUp = localup;
    }

private:
    void translateLight(const Vec3f translation);

private:
    float m_x;
    float m_y;

    float m_near;
    float m_far;
    float m_fov;

    int m_width;
    int m_height;

    float m_pixelarea;

    Vec3f m_eye;    // Camera position.
    Vec3f m_light;  // Light position.

    // Camera can rotate around focus point.
    // Focus point = m_eye + m_view * focus.
    float m_focus;

    Quat1f m_cameraRot;  // Camera rotation quaternion.
    Vec3f    m_rotation_axis;
    float       m_rotation;

    Vec3f m_right;  // Camera right direction in world frame.
    Vec3f m_up;     // Camera up direction in world frame.
    Vec3f m_view;   // Camera view direction in world frame.

    Vec3f m_worldUp;     // World up direction.
    Vec3f m_localView;   // Camera view direction in camera frame.
    Vec3f m_localRight;  // Camera right direction in camera frame.
    Vec3f m_localUp;     // Camera up direction in camera frame.
};

}  // namespace PhysIKA
