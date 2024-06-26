#pragma once

#define _USE_MATH_DEFINES
#include <math.h>
#include <ostream>
#include "forceline.h"
#include "real.h"

#define     GLH_ZERO                REAL(0.0)
#define     GLH_EPSILON          REAL(10e-6)
#define		GLH_EPSILON_2		REAL(10e-12)
#define     equivalent(a,b)             (((a < b + GLH_EPSILON) &&\
                                                      (a > b - GLH_EPSILON)) ? true : false)
#define GLH_LARGE_FLOAT REAL(1e18f)

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
	return a + t*(b - a);
}

#ifdef USE_DOUBLE
inline REAL fmax(REAL a, REAL b) {
	return (a > b) ? a : b;
}

inline REAL fmin(REAL a, REAL b) {
	return (a < b) ? a : b;
}
#endif

inline bool isEqual( REAL a, REAL b, REAL tol=GLH_EPSILON )
{
    return fabs( a - b ) < tol;
}

/* This is approximately the smallest number that can be
* represented by a REAL, given its precision. */
#define ALMOST_ZERO		FLT_EPSILON

#ifndef M_PI
#define M_PI 3.14159f
#endif

#include <assert.h>

class vec2f {
public:
	union {
		struct {
		REAL x, y;
		};
		struct {
		REAL v[2];
		};
	};

	FORCEINLINE vec2f ()
	{x=0; y=0;}

	FORCEINLINE vec2f(const vec2f &v)
	{
		x = v.x;
		y = v.y;
	}

	FORCEINLINE vec2f(const REAL *v)
	{
		x = v[0];
		y = v[1];
	}

	FORCEINLINE vec2f(REAL x, REAL y)
	{
		this->x = x;
		this->y = y;
	}

	FORCEINLINE REAL operator [] ( int i ) const {return v[i];}
	FORCEINLINE REAL &operator [] (int i) { return v[i]; }

	FORCEINLINE vec2f operator- (const vec2f &v) const
	{
		return vec2f(x - v.x, y - v.y);
	}

	// cross product
	FORCEINLINE REAL cross(const vec2f &vec) const
	{
		return x*vec.y - y*vec.x;
	}

	FORCEINLINE REAL dot(const vec2f &vec) const {
		return x*vec.x + y*vec.y;
	}
};

class vec3f {
public:
	union {
		struct {
		REAL x, y, z;
		};
		struct {
		REAL v[3];
		};
	};

	FORCEINLINE vec3f ()
	{x=0; y=0; z=0;}

	FORCEINLINE vec3f(const vec3f &v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	FORCEINLINE vec3f(const REAL *v)
	{
		x = v[0];
		y = v[1];
		z = v[2];
	}

	FORCEINLINE vec3f(REAL x, REAL y, REAL z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	FORCEINLINE REAL operator [] ( int i ) const {return v[i];}
	FORCEINLINE REAL &operator [] (int i) { return v[i]; }

	FORCEINLINE vec3f &operator += (const vec3f &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	FORCEINLINE vec3f &operator -= (const vec3f &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	FORCEINLINE vec3f &operator *= (REAL t) {
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	FORCEINLINE vec3f &operator /= (REAL t) {
		x /= t;
		y /= t;
		z /= t;
		return *this;
	}

	FORCEINLINE void negate() {
		x = -x;
		y = -y;
		z = -z;
	}

	FORCEINLINE vec3f absolute() const
	{
		return vec3f(fabs(x), fabs(y), fabs(z));
	}

	FORCEINLINE vec3f operator - () const {
		return vec3f(-x, -y, -z);
	}

	FORCEINLINE vec3f operator+ (const vec3f &v) const
	{
		return vec3f(x+v.x, y+v.y, z+v.z);
	}

	FORCEINLINE vec3f operator- (const vec3f &v) const
	{
		return vec3f(x-v.x, y-v.y, z-v.z);
	}

	FORCEINLINE vec3f operator *(REAL t) const
	{
		return vec3f(x*t, y*t, z*t);
	}

	FORCEINLINE vec3f operator /(REAL t) const
	{
		return vec3f(x/t, y/t, z/t);
	}

     // cross product
     FORCEINLINE const vec3f cross(const vec3f &vec) const
     {
          return vec3f(y*vec.z - z*vec.y, z*vec.x - x*vec.z, x*vec.y - y*vec.x);
     }

	 FORCEINLINE REAL dot(const vec3f &vec) const {
		 return x*vec.x+y*vec.y+z*vec.z;
	 }

	 FORCEINLINE void normalize() 
	 { 
		 REAL sum = x*x+y*y+z*z;
		 if (sum > GLH_EPSILON_2) {
			 REAL base = REAL(1.0/sqrt(sum));
			 x *= base;
			 y *= base;
			 z *= base;
		 }
	 }

	 FORCEINLINE REAL length() const {
		 return REAL(sqrt(x*x + y*y + z*z));
	 }

	 FORCEINLINE vec3f getUnit() const {
		 return (*this)/length();
	 }

	FORCEINLINE bool isUnit() const {
		return isEqual( squareLength(), 1.f );
	}

    //! max(|x|,|y|,|z|)
	FORCEINLINE REAL infinityNorm() const
	{
		return fmax(fmax( fabs(x), fabs(y) ), fabs(z));
	}

	FORCEINLINE vec3f & set_value( const REAL &vx, const REAL &vy, const REAL &vz)
	{ x = vx; y = vy; z = vz; return *this; }

	FORCEINLINE bool equal_abs(const vec3f &other) {
		return x == other.x && y == other.y && z == other.z;
	}

	FORCEINLINE REAL squareLength() const {
		return x*x+y*y+z*z;
	}

	FORCEINLINE REAL length2() const {
		return x * x + y * y + z * z;
	}

	FORCEINLINE vec3f  dot3(const vec3f& v0, const vec3f& v1, const vec3f& v2) const
	{
		return vec3f(dot(v0), dot(v1), dot(v2));
	}

	/**@brief Set each element to the max of the current values and the values of another btVector3
 * @param other The other btVector3 to compare with
 */
	FORCEINLINE void	setMax(const vec3f& other)
	{
		setMax2(x, other.x);
		setMax2(y, other.y);
		setMax2(z, other.z);
	}

	/**@brief Set each element to the min of the current values and the values of another btVector3
 * @param other The other btVector3 to compare with
 */
	FORCEINLINE void	setMin(const vec3f& other)
	{
		setMin2(x, other.x);
		setMin2(y, other.y);
		setMin2(z, other.z);
	}

	static vec3f zero() {
		return vec3f(0.f, 0.f, 0.f);
	}

    //! Named constructor: retrieve vector for nth axis
	static vec3f axis( int n ) {
		assert( n < 3 );
		switch( n ) {
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
	static vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
    //! Named constructor: retrieve vector for y axis
	static vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
    //! Named constructor: retrieve vector for z axis
	static vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }

};

inline vec3f operator * (REAL t, const vec3f &v) {
	return vec3f(v.x*t, v.y*t, v.z*t);
}

inline vec3f interp(const vec3f &a, const vec3f &b, REAL t)
{
	return a*(1-t)+b*t;
}

inline vec3f vinterp(const vec3f &a, const vec3f &b, REAL t)
{
	return a*t+b*(1-t);
}

inline vec3f interp(const vec3f &a, const vec3f &b, const vec3f &c, REAL u, REAL v, REAL w)
{
	return a*u+b*v+c*w;
}

inline REAL clamp(REAL f, REAL a, REAL b)
{
	return fmax(a, fmin(f, b));
}

inline REAL vdistance(const vec3f &a, const vec3f &b)
{
	return (a-b).length();
}


inline std::ostream& operator<<( std::ostream&os, const vec3f &v ) {
	os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
	return os;
}

#define CLAMP(a, b, c)		if((a)<(b)) (a)=(b); else if((a)>(c)) (a)=(c)


FORCEINLINE void
vmin(vec3f &a, const vec3f &b)
{
	a.set_value(
		fmin(a[0], b[0]),
		fmin(a[1], b[1]),
		fmin(a[2], b[2]));
}

FORCEINLINE void
vmax(vec3f &a, const vec3f &b)
{
	a.set_value(
		fmax(a[0], b[0]),
		fmax(a[1], b[1]),
		fmax(a[2], b[2]));
}

FORCEINLINE vec3f lerp(const vec3f &a, const vec3f &b, REAL t)
{
	return a + t*(b - a);
}


/**@brief Return the elementwise product of two vectors */
FORCEINLINE vec3f operator*(const vec3f& v1, const vec3f& v2)
{
	return vec3f(
		v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z);
}

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


	REAL	w() const { return m_floats[3]; }
	vec3f xyz() const { return vec3f(m_floats); }

	FORCEINLINE int maxAxis4() const
	{
		int maxIndex = -1;
		REAL maxVal = REAL(-GLH_LARGE_FLOAT);
		if (m_floats[0] > maxVal)
		{
			maxIndex = 0;
			maxVal = m_floats[0];
		}
		if (m_floats[1] > maxVal)
		{
			maxIndex = 1;
			maxVal = m_floats[1];
		}
		if (m_floats[2] > maxVal)
		{
			maxIndex = 2;
			maxVal = m_floats[2];
		}
		if (m_floats[3] > maxVal)
		{
			maxIndex = 3;
			maxVal = m_floats[3];
		}

		return maxIndex;
	}


	FORCEINLINE int minAxis4() const
	{
		int minIndex = -1;
		REAL minVal = REAL(GLH_LARGE_FLOAT);
		if (m_floats[0] < minVal)
		{
			minIndex = 0;
			minVal = m_floats[0];
		}
		if (m_floats[1] < minVal)
		{
			minIndex = 1;
			minVal = m_floats[1];
		}
		if (m_floats[2] < minVal)
		{
			minIndex = 2;
			minVal = m_floats[2];
		}
		if (m_floats[3] < minVal)
		{
			minIndex = 3;
			minVal = m_floats[3];
		}

		return minIndex;
	}


	FORCEINLINE int closestAxis4() const
	{
		return absolute4().maxAxis4();
	}

	FORCEINLINE void	setValue(const REAL& _x, const REAL& _y, const REAL& _z, const REAL& _w)
	{
		m_floats[0] = _x;
		m_floats[1] = _y;
		m_floats[2] = _z;
		m_floats[3] = _w;
	}

	FORCEINLINE vec4f& operator *= (REAL t) {
		m_floats[0] *= t;
		m_floats[1] *= t;
		m_floats[2] *= t;
		m_floats[3] *= t;
		return *this;
	}


};
