#pragma once


class alignas(16) quaternion {
	float _data[4];

public:
	quaternion() {
		_data[0] = _data[1] = _data[2] = _data[3] = -1;
	}

	quaternion(const float& x, const float& y, const float& z, const float& w)
	{
		setValue(x, y, z, w);
	}

	/**@brief Constructor from Euler angles
 * @param yaw Angle around Y unless BT_EULER_DEFAULT_ZYX defined then Z
 * @param pitch Angle around X unless BT_EULER_DEFAULT_ZYX defined then Y
 * @param roll Angle around Z unless BT_EULER_DEFAULT_ZYX defined then X */
	quaternion(const float& yaw, const float& pitch, const float& roll)
	{
		setEuler(yaw, pitch, roll);
	}

	/**@brief Set the quaternion using Euler angles
 * @param yaw Angle around Y
 * @param pitch Angle around X
 * @param roll Angle around Z */
	__forceinline void setEuler(const float& yaw, const float& pitch, const float& roll)
	{
		float halfYaw = float(yaw) * float(0.5);
		float halfPitch = float(pitch) * float(0.5);
		float halfRoll = float(roll) * float(0.5);
		float cosYaw = cosf(halfYaw);
		float sinYaw = sinf(halfYaw);
		float cosPitch = cosf(halfPitch);
		float sinPitch = sinf(halfPitch);
		float cosRoll = cosf(halfRoll);
		float sinRoll = sinf(halfRoll);
		setValue(cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
			cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
			sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
			cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw);
	}

	/**@brief Set the values
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   * @param w Value of w
   */
	__forceinline void	setValue(const float& _x, const float& _y, const float& _z, const float& _w)
	{
		_data[0] = _x;
		_data[1] = _y;
		_data[2] = _z;
		_data[3] = _w;
	}

	/**@brief Return the x value */
	__forceinline const float& x() const { return _data[0]; }
	/**@brief Return the y value */
	__forceinline const float& y() const { return _data[1]; }
	/**@brief Return the z value */
	__forceinline const float& z() const { return _data[2]; }
	/**@brief Return the w value */
	__forceinline const float& w() const { return _data[3]; }

	/**@brief Return the dot product between this quaternion and another
 * @param q The other quaternion */
	__forceinline float dot(const quaternion& q) const
	{
		return  _data[0] * q._data[0] +
			_data[1] * q._data[1] +
			_data[2] * _data[2] +
			_data[3] * q._data[3];
	}

	/**@brief Return the length squared of the quaternion */
	__forceinline float length2() const
	{
		return dot(*this);
	}

	/**
	 * @brief Return the length of the quaternion.
	 * 
	 * \return 
	 */
	__forceinline float length() const
	{
		return sqrtf(length2());
	}

	/**@brief Normalize the quaternion
	 * Such that x^2 + y^2 + z^2 +w^2 = 1 */
	__forceinline quaternion& normalize()
	{
		return *this /= length();
	}


	/**@brief Inversely scale this quaternion
	 * @param s The scale factor */
	__forceinline quaternion& operator/=(const float& s)
	{
		assert(s != float(0.0));
		return *this *= float(1.0) / s;
	}


	/**@brief Scale this quaternion
	 * @param s The scalar to scale by */
	__forceinline quaternion& operator*=(const float& s)
	{
		_data[0] *= s;
		_data[1] *= s;
		_data[2] *= s;
		_data[3] *= s;

		return *this;
	}

};



/**@brief Return the product of two quaternions */
__forceinline quaternion
operator*(const quaternion& q1, const quaternion& q2)
{
	return quaternion(
		q1.w() * q2.x() + q1.x() * q2.w() + q1.y() * q2.z() - q1.z() * q2.y(),
		q1.w() * q2.y() + q1.y() * q2.w() + q1.z() * q2.x() - q1.x() * q2.z(),
		q1.w() * q2.z() + q1.z() * q2.w() + q1.x() * q2.y() - q1.y() * q2.x(),
		q1.w() * q2.w() - q1.x() * q2.x() - q1.y() * q2.y() - q1.z() * q2.z());
}
