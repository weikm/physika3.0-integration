/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-03
 * @description: vector data structure
 * @version    : 1.0
 */

#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
namespace Physika {
#define REAL float
#define FORCEINLINE __forceinline__
#define COMM_FUNC __device__ __host__
#define MAX_PAIR_NUM 20000000
#define GLH_ZERO REAL(0.0)
#define GLH_EPSILON REAL(10e-6)
#define GLH_EPSILON_2 REAL(10e-12)
#define equivalent(a, b) (((a < b + GLH_EPSILON) && (a > b - GLH_EPSILON)) ? true : false)
#define GLH_LARGE_FLOAT REAL(1e18f)

#define equivalentd(a, b) (((a < b + GLH_EPSILON) && (a > b - GLH_EPSILON)) ? true : false)
inline double fmax(double a, double b)
{
    return (a > b) ? a : b;
}

inline double fmin(double a, double b)
{
    return (a < b) ? a : b;
}

inline bool isEqual(double a, double b, double tol = GLH_EPSILON)
{
    return fabs(a - b) < tol;
}

template <class T>
FORCEINLINE void setMax2(T& a, const T& b)
{
    if (a < b)
    {
        a = b;
    }
}

template <class T>
FORCEINLINE void setMin2(T& a, const T& b)
{
    if (b < a)
    {
        a = b;
    }
}

inline REAL lerp(REAL a, REAL b, REAL t)
{
    return a + t * (b - a);
}

#ifdef USE_DOUBLE
inline REAL fmax(REAL a, REAL b)
{
    return (a > b) ? a : b;
}

inline REAL fmin(REAL a, REAL b)
{
    return (a < b) ? a : b;
}
#endif

inline bool isEqual(REAL a, REAL b, REAL tol = GLH_EPSILON)
{
    return fabs(a - b) < tol;
}

/* This is approximately the smallest number that can be
 * represented by a REAL, given its precision. */
#define ALMOST_ZERO FLT_EPSILON

#ifndef M_PI
#define M_PI 3.14159f
#endif

/**
 * vec2f data structure
 */
class vec2f
{
public:
    union
    {
        struct
        {
            REAL x, y;
        };
        struct
        {
            REAL v[2];
        };
    };

    FORCEINLINE vec2f()
    {
        x = 0;
        y = 0;
    }

    FORCEINLINE vec2f(const vec2f& v)
    {
        x = v.x;
        y = v.y;
    }

    FORCEINLINE vec2f(const REAL* v)
    {
        x = v[0];
        y = v[1];
    }

    FORCEINLINE vec2f(REAL x, REAL y)
    {
        this->x = x;
        this->y = y;
    }

    FORCEINLINE REAL operator[](int i) const
    {
        return v[i];
    }
    FORCEINLINE REAL& operator[](int i)
    {
        return v[i];
    }

    FORCEINLINE vec2f operator-(const vec2f& v) const
    {
        return vec2f(x - v.x, y - v.y);
    }

    // cross product
    FORCEINLINE REAL cross(const vec2f& vec) const
    {
        return x * vec.y - y * vec.x;
    }

    FORCEINLINE REAL dot(const vec2f& vec) const
    {
        return x * vec.x + y * vec.y;
    }
};

/**
 * vec3f data structure
 */
class vec3f
{
public:
    union
    {
        struct
        {
            REAL x, y, z;  //!< x, y, z of 3d-vector
        };
        struct
        {
            REAL v[3];  //!< data of 3d-vector
        };
    };

    /**
     * constructor
     */
    COMM_FUNC vec3f()
    {
        x = 0;
        y = 0;
        z = 0;
    }

    /**
     * copy constructor
     *
     * @param[in] v another vec3f
     */
    COMM_FUNC vec3f(const vec3f& v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
    }

    /**
     * constructor
     *
     * @param[in] v data pointer
     */
    COMM_FUNC vec3f(const float* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    /**
     * constructor
     *
     * @param[in] v data pointer
     */
    COMM_FUNC vec3f(float* v)
    {
        x = v[0];
        y = v[1];
        z = v[2];
    }

    /**
     * constructor
     *
     * @param[in] x x-direction data
     * @param[in] y y-direction data
     * @param[in] z z-direction data
     */
    COMM_FUNC vec3f(float x, float y, float z)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    /**
     * operator[]
     *
     * @param[in] i index
     * @return the i-th data of the vec3f
     */
    COMM_FUNC float operator[](int i) const
    {
        return v[i];
    }

    /**
     * operator[]
     *
     * @param[in] i index
     * @return the i-th data of the vec3f
     */
    COMM_FUNC float& operator[](int i)
    {
        return v[i];
    }

    /**
     * operator += add the vec3f with another one
     *
     * @param[in] v another vec3f
     * @return add result
     */
    COMM_FUNC vec3f& operator+=(const vec3f& v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    /**
     * operator -= substract the vec3f with another one
     *
     * @param[in] v another vec3f
     * @return substract result
     */
    COMM_FUNC vec3f& operator-=(const vec3f& v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    /**
     * operator *= multiply with a scalar
     *
     * @param[in] t scalar
     * @return multiply result
     */
    COMM_FUNC vec3f& operator*=(float t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    /**
     * operator /= divide with a scalar
     *
     * @param[in] t scalar
     * @return divide result
     */
    COMM_FUNC vec3f& operator/=(float t)
    {
        x /= t;
        y /= t;
        z /= t;
        return *this;
    }

    /**
     * negate the current vector
     */
    COMM_FUNC void negate()
    {
        x = -x;
        y = -y;
        z = -z;
    }

    /**
     * get the absolute value of the current vector
     */
    COMM_FUNC vec3f absolute() const
    {
        return vec3f(fabs(x), fabs(y), fabs(z));
    }

    /**
     * negate the current vector
     *
     * @return the negated result
     */
    COMM_FUNC vec3f operator-() const
    {
        return vec3f(-x, -y, -z);
    }

    /**
     * vector add two vector
     *
     * @param[in] v another vec3f
     * @return add result
     */
    COMM_FUNC vec3f operator+(const vec3f& v) const
    {
        return vec3f(x + v.x, y + v.y, z + v.z);
    }

    /**
     * vector substract
     *
     * @param[in] v another vec3f
     * @return the substracted result
     */
    COMM_FUNC vec3f operator-(const vec3f& v) const
    {
        return vec3f(x - v.x, y - v.y, z - v.z);
    }

    /**
     * vector multiply a scalar
     *
     * @param[in] t scaler
     * @return the multiplied result
     */
    COMM_FUNC vec3f operator*(float t) const
    {
        return vec3f(x * t, y * t, z * t);
    }

    /**
     * vector multiply a vector
     *
     * @param[in] t vec3f
     * @return the multiplied result
     */
    COMM_FUNC vec3f operator*(vec3f t) const
    {
        return vec3f(x * t.x, y * t.y, z * t.z);
    }

    /**
     * vector divide a scalar
     *
     * @param[in] t scaler
     * @return the divided result
     */
    COMM_FUNC vec3f operator/(float t) const
    {
        return vec3f(x / t, y / t, z / t);
    }

    /**
     * vector cross product
     *
     * @param[in] v another vec3f
     * @return cross product result
     */
    COMM_FUNC const vec3f cross(const vec3f& vec) const
    {
        return vec3f(y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x);
    }

    /**
     * vector dot product
     *
     * @param[in] v another vec3f
     * @return dot product result
     */
    COMM_FUNC float dot(const vec3f& vec) const
    {
        return x * vec.x + y * vec.y + z * vec.z;
    }

    /**
     * vector normalize
     */
    COMM_FUNC vec3f normalize()
    {
        float sum = x * x + y * y + z * z;
        if (sum > float(10e-12))
        {
            float base = float(1.0 / sqrt(sum));
            x *= base;
            y *= base;
            z *= base;
        }
        return *this;
    }

    /**
     * get vector length
     *
     * @return the length of the vector
     */
    COMM_FUNC float length() const
    {
        return float(sqrt(x * x + y * y + z * z));
    }

    /**
     * get vector square length
     *
     * @return the square length of the vector
     */
    COMM_FUNC float length2() const
    {
        return x * x + y * y + z * z;
    }

    /**
     * get dot3
     *
     * @return the dot3
     */
    COMM_FUNC vec3f dot3(const vec3f& v0, const vec3f& v1, const vec3f& v2) const
    {
        return vec3f(dot(v0), dot(v1), dot(v2));
    }
    /**
     * get unit vector of the current vector
     *
     * @return the unit vector
     */
    COMM_FUNC vec3f getUnit() const
    {
        return (*this) / length();
    }

    /**
     * eplision equal
     *
     * @param[in] a one number
     * @param[in] b the other number
     * @param[in] tol threshold
     * @return whether the two number is equal with respect to the current threshold
     */
    COMM_FUNC inline bool isEqual(float a, float b, float tol = float(10e-6)) const
    {
        return fabs(a - b) < tol;
    }

    /**
     * check unit
     *
     * @return whether the current vector is a unit vector
     */
    COMM_FUNC bool isUnit() const
    {
        return isEqual(squareLength(), 1.f);
    }

    /**
     * get the infinity norm
     *
     * @return the infinity norm
     */
    COMM_FUNC float infinityNorm() const
    {
        return fmax(fmax(fabs(x), fabs(y)), fabs(z));
    }

    /**
     * set the value of the vec3f
     *
     * @param[in] vx x-direction value
     * @param[in] vy y-direction value
     * @param[in] vz z-direction value
     * @return new vec3f
     */
    vec3f& set_value(const float& vx, const float& vy, const float& vz)
    {
        x = vx;
        y = vy;
        z = vz;
        return *this;
    }

    /**
     * check if two vec3f has the same value
     *
     * @param[in] other another vector
     * @return check result
     */
    bool equal_abs(const vec3f& other)
    {
        return x == other.x && y == other.y && z == other.z;
    }

    /**
     * get the square length
     *
     * @return the square length
     */
    float squareLength() const
    {
        return x * x + y * y + z * z;
    }

    /**
     * get a vec3f with all elements 0
     *
     * @return the zero vector
     */
    static vec3f zero()
    {
        return vec3f(0.f, 0.f, 0.f);
    }

    //! Named constructor: retrieve vector for nth axis
    static vec3f axis(int n)
    {
        switch (n)
        {
            case 0: {
                return xAxis();
            }
            case 1: {
                return yAxis();
            }
            case 2: {
                return zAxis();
            }
        }
        return vec3f();
    }

    //! Named constructor: retrieve vector for x axis
    static vec3f xAxis()
    {
        return vec3f(1.f, 0.f, 0.f);
    }
    //! Named constructor: retrieve vector for y axis
    static vec3f yAxis()
    {
        return vec3f(0.f, 1.f, 0.f);
    }
    //! Named constructor: retrieve vector for z axis
    static vec3f zAxis()
    {
        return vec3f(0.f, 0.f, 1.f);
    }
    __forceinline__ void setMax(const vec3f& other)
    {
        setMax2(x, other.x);
        setMax2(y, other.y);
        setMax2(z, other.z);
    }

    /**@brief Set each element to the min of the current values and the values of another btVector3
     * @param other The other btVector3 to compare with
     */
    __forceinline__ void setMin(const vec3f& other)
    {
        setMin2(x, other.x);
        setMin2(y, other.y);
        setMin2(z, other.z);
    }
};

/**
 * vec4f data structure
 */
class vec4f
{
    REAL m_floats[4];

public:
    FORCEINLINE vec4f() {}

    FORCEINLINE vec4f(const REAL& _x, const REAL& _y, const REAL& _z, const REAL& _w)
    {
        m_floats[0] = _x;
        m_floats[1] = _y;
        m_floats[2] = _z;
        m_floats[3] = _w;
    }

    FORCEINLINE vec4f absolute4() const
    {
        return vec4f(
            fabs(m_floats[0]),
            fabs(m_floats[1]),
            fabs(m_floats[2]),
            fabs(m_floats[3]));
    }

    REAL w() const
    {
        return m_floats[3];
    }
    vec3f xyz() const
    {
        return vec3f(m_floats);
    }

    FORCEINLINE int maxAxis4() const
    {
        int  maxIndex = -1;
        REAL maxVal   = REAL(-GLH_LARGE_FLOAT);
        if (m_floats[0] > maxVal)
        {
            maxIndex = 0;
            maxVal   = m_floats[0];
        }
        if (m_floats[1] > maxVal)
        {
            maxIndex = 1;
            maxVal   = m_floats[1];
        }
        if (m_floats[2] > maxVal)
        {
            maxIndex = 2;
            maxVal   = m_floats[2];
        }
        if (m_floats[3] > maxVal)
        {
            maxIndex = 3;
            maxVal   = m_floats[3];
        }

        return maxIndex;
    }

    FORCEINLINE int minAxis4() const
    {
        int  minIndex = -1;
        REAL minVal   = REAL(GLH_LARGE_FLOAT);
        if (m_floats[0] < minVal)
        {
            minIndex = 0;
            minVal   = m_floats[0];
        }
        if (m_floats[1] < minVal)
        {
            minIndex = 1;
            minVal   = m_floats[1];
        }
        if (m_floats[2] < minVal)
        {
            minIndex = 2;
            minVal   = m_floats[2];
        }
        if (m_floats[3] < minVal)
        {
            minIndex = 3;
            minVal   = m_floats[3];
        }

        return minIndex;
    }

    FORCEINLINE int closestAxis4() const
    {
        return absolute4().maxAxis4();
    }

    FORCEINLINE void setValue(const REAL& _x, const REAL& _y, const REAL& _z, const REAL& _w)
    {
        m_floats[0] = _x;
        m_floats[1] = _y;
        m_floats[2] = _z;
        m_floats[3] = _w;
    }

    FORCEINLINE vec4f& operator*=(REAL t)
    {
        m_floats[0] *= t;
        m_floats[1] *= t;
        m_floats[2] *= t;
        m_floats[3] *= t;
        return *this;
    }
};

/**
 * scalar multiply vec3f
 *
 * @param[in] t scalar
 * @param[in] v vec3f
 * @return result
 */
inline vec3f operator*(float t, const vec3f& v)
{
    return vec3f(v.x * t, v.y * t, v.z * t);
}

/**
 * lerp two vec3f with (1 - t) * a + t * b
 * @param[in] a vec3f
 * @param[in] b vec3f
 * @param[in] t scalar
 * @return lerp result
 */
inline vec3f interp(const vec3f& a, const vec3f& b, float t)
{
    return a * (1 - t) + b * t;
}

/**
 * vinterp two vec3f with t * a + (1 - t) * b
 * @param[in] a vec3f
 * @param[in] b vec3f
 * @param[in] t scalar
 * @return vinterp result
 */
inline vec3f vinterp(const vec3f& a, const vec3f& b, float t)
{
    return a * t + b * (1 - t);
}

/**
 * calculate weighted position of three vec3f
 * @param[in] a vec3f
 * @param[in] b vec3f
 * @param[in] c vec3f
 * @param[in] u weight
 * @param[in] v weight
 * @param[in] w weight
 * @return weighted position result
 */
inline vec3f interp(const vec3f& a, const vec3f& b, const vec3f& c, float u, float v, float w)
{
    return a * u + b * v + c * w;
}

/**
 * calculate the distance of to points
 * @param[in] a vec3f
 * @param[in] b vec3f
 * @return distance
 */
inline float vdistance(const vec3f& a, const vec3f& b)
{
    return (a - b).length();
}

/**
 * print vec3f data
 * @param[in] os output stream
 * @param[in] v  vec3f
 * @return output stream
 */
inline std::ostream& operator<<(std::ostream& os, const vec3f& v)
{
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
    return os;
}

/**
 * get the min value of two vec3f in all directions
 * @param[in] a vec3f
 * @param[in] b vec3f
 * @return the min values in vec3f
 */
inline void vmin(vec3f& a, const vec3f& b)
{
    a.set_value(
        fmin(a[0], b[0]),
        fmin(a[1], b[1]),
        fmin(a[2], b[2]));
}

/**
 * get the max value of two vec3f in all directions
 * @param[in] a vec3f
 * @param[in] b vec3f
 * @return the max values in vec3f
 */
inline void vmax(vec3f& a, const vec3f& b)
{
    a.set_value(
        fmax(a[0], b[0]),
        fmax(a[1], b[1]),
        fmax(a[2], b[2]));
}

/**
 * lerp between two vector
 * @param[in] a vec3f
 * @param[in] b vec3f
 * @param[in] t scalar
 * @return lerp result
 */
inline vec3f lerp(const vec3f& a, const vec3f& b, float t)
{
    return a + t * (b - a);
}

inline REAL clamp(REAL f, REAL a, REAL b)
{
    return fmax(a, fmin(f, b));
}

}  // namespace Physika