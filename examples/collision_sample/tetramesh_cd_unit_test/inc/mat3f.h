#pragma once

#include <math.h>

#include "forceline.h"
#include "vec3f.h"
#include "quaternion.h"

#include <algorithm>
using namespace std;

#include <assert.h>
#include <string.h>

class alignas(16) matrix3f {
	REAL _data[9]; // stored in column-major format

	// private. Clients should use named constructors colMajor or rowMajor.
	matrix3f(const REAL data[9]) {
		//std::copy(data, data+9, _data);
		memcpy(_data, data, sizeof(REAL)*9);
	}

public:
     // ----- static member functions -----

   //! Named constructor: construct from REAL array, column-major storage
	static matrix3f colMajor( const REAL data[ 9 ] ) {
		return matrix3f(data);
	}

    //! Named constructor: construct from REAL array, row-major storage
	static matrix3f rowMajor( const REAL data[ 9 ] ) {
		return matrix3f(
			data[0], data[3], data[6],
			data[1], data[4], data[7],
			data[2], data[5], data[8]);
	}

    //! Named constructor: construct a scaling matrix, with [sx sy sz] along
    //! the main diagonal
	static matrix3f scaling( REAL sx, REAL sy, REAL sz ) {
		return matrix3f(
			sx, 0, 0,
			0, sy, 0,
			0, 0, sz);
	}

    //! Named constructor: construct a rotation matrix, representing a
    //! rotation of theta about the given axis.
	static matrix3f rotation( const vec3f& axis, REAL theta ) {
		const REAL s = sin( theta );
		const REAL c = cos( theta );
		const REAL t = 1-c;
		const REAL x = axis.x, y = axis.y, z = axis.z;
		return matrix3f(
			t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
			t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
			t*x*z - s*y, t*y*z + s*x, t*z*z + c
		);
	}

    //! Named constructor: create matrix M = a * b^T
	static matrix3f outerProduct( const vec3f& a, const vec3f& b ) {
		return matrix3f(
			a.x * b.x, a.x * b.y, a.x * b.z,
			a.y * b.x, a.y * b.y, a.y * b.z,
			a.z * b.x, a.z * b.y, a.z * b.z
		);
	}

    //! Named constructor: create identity matrix. This is implemented this
    //! way (instead of as a static member) so that the compiler knows the
    //! contents of the matrix and can optimise it out.
	static matrix3f identity() {
		static const REAL entries[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
		return matrix3f(entries);
	}

    //! Named constructor: create zero matrix
	static matrix3f zero() {
		static const REAL entries[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
		return matrix3f(entries);
	}

    // ----- member functions -----
    
	matrix3f() {
		*this = matrix3f::zero();
	}

    matrix3f(REAL entry00, REAL entry01, REAL entry02,
			        REAL entry10, REAL entry11, REAL entry12,
					REAL entry20, REAL entry21, REAL entry22) {
		_data[0] = entry00, _data[3] = entry01, _data[6] = entry02;
		_data[1] = entry10, _data[4] = entry11, _data[7] = entry12;
		_data[2] = entry20, _data[5] = entry21, _data[8] = entry22;
	}

    // Default copy constructor is fine.
    
    // Default assignment operator is fine.
	REAL operator()( size_t row, size_t col ) const {
		assert(row < 3 && col <3);
		return _data[col*3+row];
	}

	REAL& operator()( size_t row, size_t col ) {
		assert(row < 3 && col < 3);
		return _data[col*3+row];
	}

	vec3f operator()(size_t row) const {
		return getRow(row);
	}

	matrix3f operator-() const {
		return matrix3f(
			-_data[0], -_data[1], -_data[2],
			-_data[3], -_data[4], -_data[5],
			-_data[6], -_data[7], -_data[8]);
	}

	matrix3f& operator*=( const matrix3f&rhs) {
		return operator=(operator *(rhs));
	}

	matrix3f& operator*=( REAL rhs) {
		for (int i=0; i<9; i++)
			_data[i] *= rhs;

		return *this;
	}

	matrix3f& operator+=( const matrix3f& rhs ) {
		for (int i=0; i<9; i++)
			_data[i] += rhs._data[i];

		return *this;
	}

	matrix3f& operator-=( const matrix3f& rhs) {
		for (int i=0; i<9; i++)
			_data[i] -= rhs._data[i];

		return *this;
	}

	matrix3f operator*( const matrix3f&rhs) const {
		matrix3f result;
		for ( int r = 0; r < 3; ++r ) {
			for ( int c = 0; c < 3; ++c ) {
				REAL val = 0;
				for ( int i = 0; i < 3; ++i ) {
					val += operator()( r, i ) * rhs( i, c );
				}
				result( r, c ) = val;
			}
		}
		return result;
	}

	vec3f operator*( const vec3f& rhs) const {
		// _data[ r+c*3 ]
		return vec3f(
			_data[ 0+0*3 ]*rhs.x + _data[ 0+1*3 ]*rhs.y + _data[ 0+2*3 ]*rhs.z,
			_data[ 1+0*3 ]*rhs.x + _data[ 1+1*3 ]*rhs.y + _data[ 1+2*3 ]*rhs.z,
			_data[ 2+0*3 ]*rhs.x + _data[ 2+1*3 ]*rhs.y + _data[ 2+2*3 ]*rhs.z);
	}

	matrix3f operator*( REAL rhs) const {
		matrix3f tmp(*this);
		tmp *= rhs;
		return tmp;
	}

	matrix3f operator+( const matrix3f& rhs) const {
		matrix3f tmp(*this);
		tmp += rhs;
		return tmp;
	}

	matrix3f operator-( const matrix3f& rhs) const {
		matrix3f tmp(*this);
		tmp -= rhs;
		return tmp;
	}

	bool operator==( const matrix3f& rhs) const {
		bool ret = true;
		for (int i=0; i<9; i++)
			ret = ret && isEqual(_data[i], rhs._data[i]);
		return ret;
	}

	bool operator!=( const matrix3f& rhs) const {
		return !operator==(rhs);
	}

    //! Sum of diagonal elements.
	REAL getTrace() const {
		return _data[0] + _data[4] + _data[8];
	}

    //! Not the standard definition... max of all elements
	REAL infinityNorm() const {
		return fmax(
			fmax(
				fmax(
					fmax( _data[ 0 ], _data[ 1 ] ),
					fmax( _data[ 2 ], _data[ 3 ] )
				),
				fmax(
					fmax( _data[ 4 ], _data[ 5 ] ),
					fmax( _data[ 6 ], _data[ 7 ] )
				)
			),
			_data[ 8 ]
		);
	}

    //! Retrieve data as a flat array, column-major storage
	const REAL* asColMajor() const {
		return _data;
	}

	matrix3f getTranspose() const {
		//TangMin: error...
		//return matrix3f::rowMajor(_data);

		return matrix3f(
			_data[0], _data[1], _data[2],
			_data[3], _data[4], _data[5],
			_data[6], _data[7], _data[8]);

	}

	matrix3f getInverse() const {
		matrix3f result(
			operator()(1,1) * operator()(2,2) - operator()(1,2) * operator()(2,1),
			operator()(0,2) * operator()(2,1) - operator()(0,1) * operator()(2,2),
			operator()(0,1) * operator()(1,2) - operator()(0,2) * operator()(1,1),

			operator()(1,2) * operator()(2,0) - operator()(1,0) * operator()(2,2),
			operator()(0,0) * operator()(2,2) - operator()(0,2) * operator()(2,0),
			operator()(0,2) * operator()(1,0) - operator()(0,0) * operator()(1,2),

			operator()(1,0) * operator()(2,1) - operator()(1,1) * operator()(2,0),
			operator()(0,1) * operator()(2,0) - operator()(0,0) * operator()(2,1),
			operator()(0,0) * operator()(1,1) - operator()(0,1) * operator()(1,0)
		);

		REAL det =
			operator()(0,0) * result(0,0) +
			operator()(0,1) * result(1,0) +
			operator()(0,2) * result(2,0);

		assert( ! isEqual(det, 0) );

		REAL invDet = 1.0f / det;
		for( int i = 0; i < 9; ++i )
			result._data[ i ] *= invDet;

		return result;
	}

	REAL determinant()
	{
		return 
			operator()(0,0)*(operator()(2,2)*operator()(1,1)-operator()(2,1)*operator()(1,2))-
			operator()(1,0)*(operator()(2,2)*operator()(0,1)-operator()(2,1)*operator()(0,2))+
			operator()(2,0)*(operator()(1,2)*operator()(0,1)-operator()(1,1)*operator()(0,2));
	}

	//////////////////////////////////////////////////////////////////////
	matrix3f scaled(const vec3f& s) const
	{
		return matrix3f(
			_data[0] * s.x, _data[3] * s.y, _data[6] * s.z,
			_data[1] * s.x, _data[4] * s.y, _data[7] * s.z,
			_data[2] * s.x, _data[5] * s.y, _data[8] * s.z);
	}

	/** @brief Set the matrix from a quaternion
*  @param q The Quaternion to match */
	static matrix3f rotation(const quaternion& q)
	{
		float d = q.length2();
		assert(d != float(0.0));
		float s = float(2.0) / d;
		float xs = q.x() * s, ys = q.y() * s, zs = q.z() * s;
		float wx = q.w() * xs, wy = q.w() * ys, wz = q.w() * zs;
		float xx = q.x() * xs, xy = q.x() * ys, xz = q.x() * zs;
		float yy = q.y() * ys, yz = q.y() * zs, zz = q.z() * zs;
		return matrix3f(
			float(1.0) - (yy + zz), xy - wz, xz + wy,
			xy + wz, float(1.0) - (xx + zz), yz - wx,
			xz - wy, yz + wx, float(1.0) - (xx + yy));
	}

	/**@brief Get the matrix represented as a quaternion
	* @param q The quaternion which will be set */
	void getRotation(quaternion& q) const
	{
		float trace = operator()(0, 0) + operator()(1, 1) + operator()(2, 2);

		float temp[4];

		if (trace > float(0.0))
		{
			float s = sqrtf(trace + float(1.0));
			temp[3] = (s * float(0.5));
			s = float(0.5) / s;

			temp[0] = ((operator()(2, 1) - operator()(1, 2)) * s);
			temp[1] = ((operator()(0, 2) - operator()(2, 0)) * s);
			temp[2] = ((operator()(1, 0) - operator()(0, 1)) * s);
		}
		else
		{
			int i = operator()(0, 0) < operator()(1, 1) ?
				(operator()(1, 1) < operator()(2, 2) ? 2 : 1) :
				(operator()(0, 0) < operator()(2, 2) ? 2 : 0);
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;

			float s = sqrtf(operator()(i, i) - operator()(j, j) - operator()(k, k) + float(1.0));
			temp[i] = s * float(0.5);
			s = float(0.5) / s;

			temp[3] = (operator()(k, j) - operator()(j, k)) * s;
			temp[j] = (operator()(j, i) + operator()(i, j)) * s;
			temp[k] = (operator()(k, i) + operator()(i, k)) * s;
		}
		q.setValue(temp[0], temp[1], temp[2], temp[3]);
	}

	/** @brief Get a column of the matrix as a vector
*  @param i Column number 0 indexed */
	__forceinline vec3f getRow(int i) const
	{
		assert(0 <= i && i < 3);
		return vec3f(_data[i], _data[3+i], _data[6+i]);
	}


	/** @brief Get a row of the matrix as a vector
	*  @param i Row number 0 indexed */
	__forceinline const vec3f& getColumn(int i) const
	{
		assert(0 <= i && i < 3);
		return vec3f(_data+i*3);
	}

};

//! Scalar-matrix multiplication
__forceinline matrix3f operator*( REAL lhs, const matrix3f& rhs) {
	return rhs * lhs;
}

//! Multiply row vector by matrix, v^T * M
__forceinline vec3f operator*( const vec3f& lhs, const matrix3f& rhs) {
    return vec3f(
        lhs.x * rhs(0,0) + lhs.y * rhs(1,0) + lhs.z * rhs(2,0),
        lhs.x * rhs(0,1) + lhs.y * rhs(1,1) + lhs.z * rhs(2,1),
        lhs.x * rhs(0,2) + lhs.y * rhs(1,2) + lhs.z * rhs(2,2)
    );
}

#include <ostream>

__forceinline std::ostream& operator<<( std::ostream&out, const matrix3f&m )
{
    out << "M3(" << std::endl;
    out << "  " << m(0,0) << " " << m(0,1) << " " << m(0,2) << std::endl;
    out << "  " << m(1,0) << " " << m(1,1) << " " << m(1,2) << std::endl;
    out << "  " << m(2,0) << " " << m(2,1) << " " << m(2,2) << std::endl;
    out << ")";
    return out;
}