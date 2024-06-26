/******************************************************************************
Copyright (c) 2007 Bart Adams (bart.adams@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. The authors shall be
acknowledged in scientific publications resulting from using the Software
by referencing the ACM SIGGRAPH 2007 paper "Adaptively Sampled Particle
Fluids".

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
******************************************************************************/

#include <GL/gl.h>
#include "Camera.h"
#include <iostream>
#include <math.h>
using namespace std;

namespace Physika {
Camera::Camera()
{
    m_worldUp    = Vec3f(0, 1, 0);
    m_localView  = Vec3f(0, 0, -1);
    m_localRight = Vec3f(1, 0, 0);
    m_localUp    = Vec3f(0, 1, 0);

    m_light = Vec3f(0, 0, 3);

    m_fov   = 0.90f;
    m_focus = 3.0f;

    m_eye = Vec3f(0, 1.5, 3);
    this->lookAt(m_eye, Vec3f(0, 0, 0), Vec3f(0, 1, 0));
}

void Camera::setGL(float neardist, float fardist, float width, float height)
{
    float diag   = sqrt(width * width + height * height);
    float top    = height / diag * 0.5f * m_fov * neardist;
    float bottom = -top;
    float right  = width / diag * 0.5f * m_fov * neardist;
    float left   = -right;

    m_cameraRot.getRotation(m_rotation, m_rotation_axis);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(left, right, bottom, top, neardist, fardist);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRotatef(-180.0f / M_PI * m_rotation, m_rotation_axis[0], m_rotation_axis[1], m_rotation_axis[2]);
    glTranslatef(-m_eye[0], -m_eye[1], -m_eye[2]);

    GLfloat pos[] = { m_light[0], m_light[1], m_light[2], 1 };
    glLightfv(GL_LIGHT0, GL_POSITION, pos);

    m_width     = ( int )width;
    m_height    = ( int )height;
    m_pixelarea = 4 * right * top / (width * height);
    m_near      = neardist;
    m_far       = fardist;
}

int Camera::width() const
{
    return m_width;
}

int Camera::height() const
{
    return m_height;
}

float Camera::getPixelArea() const
{
    return m_pixelarea;
}

Vec3f Camera::getEye() const
{
    return m_eye;
}

void Camera::setEyePostion(const Vec3f& pos)
{
    m_eye = pos;
}

void Camera::setFocus(float focus)
{
    m_focus = focus;
}

float Camera::getFocus() const
{
    return m_focus;
}

void Camera::lookAt(const Vec3f& camPos, const Vec3f& center, const Vec3f& updir_)
{
    Quat1f curQ;
    //Vec3f rightdir(1, 0, 0);
    //Vec3f updir(0, 1, 0);

    Vec3f viewdir = center - camPos;
    viewdir.normalize();
    m_view = viewdir;

    if (abs(viewdir[0]) > 1e-3 || abs(viewdir[2]) > 1e-3)
    {
        Vec3f viewxz = viewdir;
        viewxz[1]       = 0;
        viewxz.normalize();

        Vec3f right0(-viewxz[2], 0, viewxz[0]);
        Vec3f right1(viewxz[2], 0, -viewxz[0]);

        Vec3f targetRight = viewdir.cross(updir_).normalize();
        if (targetRight.dot(right0) > 0)
        {
            m_right = right0;
        }
        else
            m_right = right1;

        m_up = m_right.cross(viewdir).normalize();
    }

    Mat3f mat{
        m_right[0], m_up[0], -m_view[0], m_right[1], m_up[1], -m_view[1], m_right[2], m_up[2], -m_view[2]
    };

    Quat1f q(mat);
    //q.getRotation(m_rotation, m_rotation_axis);
    m_cameraRot = q;

    m_eye   = camPos;
    m_focus = (center - camPos).norm();
}

void Camera::rotate(Quat1f& rotquat)
{
    m_cameraRot = rotquat * m_cameraRot;
}

Vec3f Camera::getViewDir() const
{
    return m_view;
}

void Camera::getCoordSystem(Vec3f& view, Vec3f& up, Vec3f& right) const
{
    view  = m_view;
    up    = m_up;
    right = m_right;
}

void Camera::localTranslate(const Vec3f translation)
{

    m_eye += translation[0] * m_right + translation[1] * m_up - translation[2] * m_view;
}

void Camera::translate(const Vec3f translation)
{
    m_eye += translation;
}

void Camera::yawAroundFocus(float radian)
{
    Vec3f    focus2eye = m_view * (-m_focus);
    Vec3f    focusP    = m_eye - focus2eye;
    Quat1f   rot(radian, m_worldUp);

    m_eye       = focusP + rot.rotate(focus2eye);
    m_cameraRot = m_cameraRot * Quat1f(radian, m_cameraRot.conjugate().rotate(m_worldUp));
    m_cameraRot.normalize();

    this->updateDir();
}

void Camera::pitchAroundFocus(float radian)
{
    Vec3f    focus2eye = m_view * (-m_focus);
    Vec3f    focusP    = m_eye - focus2eye;
    Quat1f   rot(radian, m_right);

    m_eye       = focusP + rot.rotate(focus2eye);
    m_cameraRot = m_cameraRot * Quat1f(radian, m_localRight);
    //m_cameraRot = m_cameraRot * Quaternionf(m_cameraRot.getConjugate().rotate(m_right), radian);
    m_cameraRot.normalize();

    this->updateDir();
}

void Camera::yaw(float radian)
{
    Quat1f rot(radian, m_worldUp);
    m_cameraRot = m_cameraRot * Quat1f(radian, m_cameraRot.conjugate().rotate(m_worldUp));
    m_cameraRot.normalize();

    this->updateDir();
}

void Camera::pitch(float radian)
{
    Quat1f rot(radian, m_right);
    m_cameraRot = m_cameraRot * Quat1f(radian, m_localRight);
    m_cameraRot.normalize();

    this->updateDir();
}

void Camera::updateDir()
{
    m_right = m_cameraRot.rotate(m_localRight);
    m_up    = m_cameraRot.rotate(m_localUp);
    m_view  = m_cameraRot.rotate(m_localView);
}

void Camera::translateLight(const Vec3f translation)
{

    m_light += translation[0] * m_right + translation[1] * m_up + (-translation[2]) * m_view;
}

void Camera::zoom(float amount)
{
    m_fov += amount / 10;
    m_fov = max(m_fov, 0.01f);
}

}  // namespace PhysIKA
