#pragma once

#include "vec3f.h"
#include "plane.h"
#include "transf.h"
#include <float.h>
#include <vector>

class alignas(16) aabb {
public:
	FORCEINLINE void init() {
		_max = vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		_min = vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	}

	vec3f _min;
	vec3f _max;

	FORCEINLINE aabb() {
		init();
	}

	FORCEINLINE aabb(const vec3f &v) {
		_min = _max = v;
	}

	FORCEINLINE aabb(const vec3f &a, const vec3f &b) {
		_min = a;
		_max = a;
		vmin(_min, b);
		vmax(_max, b);
	}

	FORCEINLINE bool overlaps(const aabb& b) const
	{
		if (_min[0] > b._max[0]) return false;
		if (_min[1] > b._max[1]) return false;
		if (_min[2] > b._max[2]) return false;

		if (_max[0] < b._min[0]) return false;
		if (_max[1] < b._min[1]) return false;
		if (_max[2] < b._min[2]) return false;

		return true;
	}

	// https://gdbooks.gitbooks.io/3dcollisions/content/Chapter2/static_aabb_plane.html
	FORCEINLINE bool overlaps(const cplane &pln) const
	{
		vec3f c = (_max + _min) * 0.5f; // Compute AABB center
		vec3f e = _max - c; // Compute positive extents
		const vec3f n = pln.n();

		// Compute the projection interval radius of b onto L(t) = b.c + t * p.n
		float r = e.x * fabs(n.x) + e.y * fabs(n.y) + e.z * fabs(n.z);

		// Compute distance of box center from plane
		float s = n.dot(c) - pln.d();

		// Intersection occurs when distance s falls within [-r,+r] interval
		return fabs(s) <= r;
	}

	FORCEINLINE bool overlaps(const aabb& b, REAL tol) const
	{
		aabb aa = *this;
		aabb bb = b;

		aa.enlarge(tol);
		bb.enlarge(tol);
		return aa.overlaps(bb);
	}

	FORCEINLINE bool overlaps(const aabb &b, aabb &ret) const
	{
		if (!overlaps(b))
			return false;

		ret._min = vec3f(
			fmax(_min[0], b._min[0]),
			fmax(_min[1], b._min[1]),
			fmax(_min[2], b._min[2]));

		ret._max = vec3f(
			fmin(_max[0], b._max[0]),
			fmin(_max[1], b._max[1]),
			fmin(_max[2], b._max[2]));

		return true;
	}

	FORCEINLINE bool inside(const vec3f &p) const
	{
		if (p[0] < _min[0] || p[0] > _max[0]) return false;
		if (p[1] < _min[1] || p[1] > _max[1]) return false;
		if (p[2] < _min[2] || p[2] > _max[2]) return false;

		return true;
	}

	FORCEINLINE aabb &operator += (const vec3f &p)
	{
		vmin(_min, p);
		vmax(_max, p);
		return *this;
	}

	FORCEINLINE aabb &operator += (const aabb &b)
	{
		vmin(_min, b._min);
		vmax(_max, b._max);
		return *this;
	}

	FORCEINLINE aabb operator + (const aabb &v) const
	{
		aabb rt(*this); return rt += v;
	}

	FORCEINLINE REAL width()  const { return _max[0] - _min[0]; }
	FORCEINLINE REAL height() const { return _max[1] - _min[1]; }
	FORCEINLINE REAL depth()  const { return _max[2] - _min[2]; }
	FORCEINLINE vec3f center() const { return (_min + _max)*REAL(0.5); }
	FORCEINLINE REAL volume() const { return width()*height()*depth(); }


	FORCEINLINE bool empty() const {
		return _max[0] < _min[0];
	}

	FORCEINLINE void enlarge(REAL thickness) {
		_max += vec3f(thickness, thickness, thickness);
		_min -= vec3f(thickness, thickness, thickness);
	}

	FORCEINLINE const vec3f &getMax() const { return _max; }
	FORCEINLINE const vec3f &getMin() const { return _min; }
	FORCEINLINE void setMax(vec3f& v) { _max = v; }
	FORCEINLINE void setMin(vec3f& v) { _min = v; }

	void getCorners(std::vector<vec3f> &crns) {
		crns.push_back(_max);
		crns.push_back(vec3f(_max.x, _max.y, _min.z));
		crns.push_back(vec3f(_max.x, _min.y, _min.z));
		crns.push_back(vec3f(_max.x, _min.y, _max.z));
		crns.push_back(_min);
		crns.push_back(vec3f(_min.x, _max.y, _min.z));
		crns.push_back(vec3f(_min.x, _max.y, _max.z));
		crns.push_back(vec3f(_min.x, _min.y, _max.z));
	}

	//! Apply a transform to an AABB
	FORCEINLINE void applyTransform(const transf &trans)
	{
		vec3f c =center();
		vec3f extends = _max - c;
		// Compute new center
		c = trans(c);

		vec3f textends = extends.dot3(trans.getBasis().getRow(0).absolute(),
			trans.getBasis().getRow(1).absolute(),
			trans.getBasis().getRow(2).absolute());

		_min = c - textends;
		_max = c + textends;
	}

	//! Gets the extend and center
	FORCEINLINE void getCenterExtend(vec3f & center, vec3f & extend)  const
	{
		center = (_min + _max) * 0.5f;
		extend = _max - center;
	}

	void print(FILE *fp) {
		//fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	void visualize();
};
