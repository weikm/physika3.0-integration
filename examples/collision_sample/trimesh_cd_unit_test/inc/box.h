#pragma once

#include "aabb.h"

#define BOX aabb

//!  Class for transforming a model1 to the space of model0
class cbox2boxTrfCache
{
public:
	vec3f  m_T1to0;//!< Transforms translation of model1 to model 0
	matrix3f m_R1to0;//!< Transforms Rotation of model1 to model 0, equal  to R0' * R1
	matrix3f m_AR;//!< Absolute value of m_R1to0

	__forceinline void calc_absolute_matrix()
	{
		int i, j;

		for (i = 0; i < 3; i++)
		{
			for (j = 0; j < 3; j++)
			{
				m_AR(i, j) = 1e-6f + fabs(m_R1to0(i, j));
			}
		}
	}

	cbox2boxTrfCache()
	{
	}

	//! Calc the transformation relative  1 to 0. Inverts matrics by transposing
	__forceinline void calc_from_homogenic(const transf& trans0, const transf& trans1)
	{

		transf temp_trans = trans0.inverse();
		temp_trans = temp_trans * trans1;

		m_T1to0 = temp_trans.getOrigin();
		m_R1to0 = temp_trans.getBasis();


		calc_absolute_matrix();
	}

	//! Calcs the full invertion of the matrices. Useful for scaling matrices
	__forceinline void calc_from_full_invert(const transf& trans0, const transf& trans1)
	{
		m_R1to0 = trans0.getBasis().getInverse();
		m_T1to0 = m_R1to0 * (-trans0.getOrigin());

		m_T1to0 += m_R1to0 * trans1.getOrigin();
		m_R1to0 *= trans1.getBasis();

		calc_absolute_matrix();
	}

	__forceinline vec3f transform(const vec3f& point) const
	{
		//return point.dot3(m_R1to0[0], m_R1to0[1], m_R1to0[2]) + m_T1to0;
		return point.dot3(
			vec3f(m_R1to0(0, 0), m_R1to0(0, 1), m_R1to0(0, 2)),
			vec3f(m_R1to0(1, 0), m_R1to0(1, 1), m_R1to0(1, 2)),
			vec3f(m_R1to0(2, 0), m_R1to0(2, 1), m_R1to0(2, 2))) + m_T1to0;

	}
};

#define GREATER(x, y)	fabsf(x) > (y)


//! Returns the dot product between a vec3f and the col of a matrix
__forceinline float mat3_dot_col(
	const matrix3f& mat, const vec3f& vec3, int colindex)
{
	return vec3.x * mat(0)[colindex] + vec3.y * mat(1)[colindex] + vec3.z * mat(2)[colindex];
}

//! transcache is the transformation cache from box to this AABB
__forceinline bool overlapping_trans_cache(const BOX &bx0, const BOX &bx1,
	const cbox2boxTrfCache& transcache, bool fulltest)
{

	//Taken from OPCODE
	vec3f ea, eb;//extends
	vec3f ca, cb;//extends
	bx0.getCenterExtend(ca, ea);
	bx1.getCenterExtend(cb, eb);


	vec3f T;
	float t, t2;
	int i;

	// Class I : A's basis vectors
	for (i = 0; i < 3; i++)
	{
		T[i] = transcache.m_R1to0(i).dot(cb) + transcache.m_T1to0[i] - ca[i];
		t = transcache.m_AR(i).dot(eb) + ea[i];
		if (GREATER(T[i], t))	return false;
	}
	// Class II : B's basis vectors
	for (i = 0; i < 3; i++)
	{
		t = mat3_dot_col(transcache.m_R1to0, T, i);
		t2 = mat3_dot_col(transcache.m_AR, ea, i) + eb[i];
		if (GREATER(t, t2))	return false;
	}
	// Class III : 9 cross products
	if (fulltest)
	{
		int j, m, n, o, p, q, r;
		for (i = 0; i < 3; i++)
		{
			m = (i + 1) % 3;
			n = (i + 2) % 3;
			o = i == 0 ? 1 : 0;
			p = i == 2 ? 1 : 2;
			for (j = 0; j < 3; j++)
			{
				q = j == 2 ? 1 : 2;
				r = j == 0 ? 1 : 0;
				t = T[n] * transcache.m_R1to0(m, j) - T[m] * transcache.m_R1to0(n, j);
				t2 = ea[o] * transcache.m_AR(p, j) + ea[p] * transcache.m_AR(o, j) +
					eb[r] * transcache.m_AR(i, q) + eb[q] * transcache.m_AR(i, r);
				if (GREATER(t, t2))	return false;
			}
		}
	}
	return true;
}