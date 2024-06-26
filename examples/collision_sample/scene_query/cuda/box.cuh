#pragma once

#include "vec3.cuh"
#include "tools.cuh"

#define USE_AABB
#ifdef USE_AABB

typedef struct __align__(16) _box3f {
	REAL3 _min, _max;

	CU_FORCEINLINE __host__ __device__ void set(const REAL3 &a)
	{
		_min = _max = a;
	}

	CU_FORCEINLINE __host__ __device__ void set(const REAL3 &a, const REAL3 &b)
	{
		_min = fminf(a, b);
		_max = fmaxf(a, b);
	}

	CU_FORCEINLINE __host__ __device__  void set(const _box3f &a, const _box3f &b)
	{
		_min = fminf(a._min, b._min);
		_max = fmaxf(a._max, b._max);
	}

	CU_FORCEINLINE __host__ __device__  void add(const REAL3 &a)
	{
		_min = fminf(_min, a);
		_max = fmaxf(_max, a);
	}

	CU_FORCEINLINE __host__ __device__  void add(const _box3f &b)
	{
		_min = fminf(_min, b._min);
		_max = fmaxf(_max, b._max);
	}

	CU_FORCEINLINE __host__ __device__  void enlarge(REAL thickness)
	{
		_min -= make_REAL3(thickness);
		_max += make_REAL3(thickness);
	}

	CU_FORCEINLINE __host__ __device__ bool overlaps(const _box3f& b) const
	{
		if (_min.x > b._max.x) return false;
		if (_min.y > b._max.y) return false;
		if (_min.z > b._max.z) return false;

		if (_max.x < b._min.x) return false;
		if (_max.y < b._min.y) return false;
		if (_max.z < b._min.z) return false;

		return true;
	}
	
	CU_FORCEINLINE __host__ __device__
		REAL3 maxV() const {
		return _max;
	}

	CU_FORCEINLINE __host__ __device__
		REAL3 minV() const {
		return _min;
	}

	CU_FORCEINLINE void print() {
		printf("%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	CU_FORCEINLINE __host__ __device__
	bool inside(const REAL3 &p) const
	{
		if (_min.x > p.x) return false;
		if (_min.y > p.y) return false;
		if (_min.z > p.z) return false;

		if (_max.x < p.x) return false;
		if (_max.y < p.y) return false;
		if (_max.z < p.z) return false;

		return true;
	}

	CU_FORCEINLINE void print(FILE *fp) {
		fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}
} g_box;

#else

CU_FORCEINLINE __host__ __device__  void
__getDistances(const REAL3& p,
	REAL &d3, REAL &d4, REAL &d5, REAL &d6, REAL &d7, REAL &d8)
{
	d3 = p.x + p.y;
	d4 = p.x + p.z;
	d5 = p.y + p.z;
	d6 = p.x - p.y;
	d7 = p.x - p.z;
	d8 = p.y - p.z;
}

typedef struct __align__(16) _box3f{
	REAL _dist[18];

	CU_FORCEINLINE __host__ __device__ void set(const REAL3 &a)
	{
		_dist[0] = _dist[9] = a.x;
		_dist[1] = _dist[10] = a.y;
		_dist[2] = _dist[11] = a.z;

		REAL d3, d4, d5, d6, d7, d8;
		__getDistances(a, d3, d4, d5, d6, d7, d8);

		_dist[3] = _dist[12] = d3;
		_dist[4] = _dist[13] = d4;
		_dist[5] = _dist[14] = d5;
		_dist[6] = _dist[15] = d6;
		_dist[7] = _dist[16] = d7;
		_dist[8] = _dist[17] = d8;
	}

	CU_FORCEINLINE __host__ __device__ void set(const REAL3 &a, const REAL3 &b)
	{
		_dist[0] = fminf(a.x, b.x);
		_dist[9] = fmaxf(a.x, b.x);
		_dist[1] = fminf(a.y, b.y);
		_dist[10] = fmaxf(a.y, b.y);
		_dist[2] = fminf(a.z, b.z);
		_dist[11] = fmaxf(a.z, b.z);

		REAL ad3, ad4, ad5, ad6, ad7, ad8;
		__getDistances(a, ad3, ad4, ad5, ad6, ad7, ad8);
		REAL bd3, bd4, bd5, bd6, bd7, bd8;
		__getDistances(b, bd3, bd4, bd5, bd6, bd7, bd8);

		_dist[3] = fminf(ad3, bd3);
		_dist[12] = fmaxf(ad3, bd3);
		_dist[4] = fminf(ad4, bd4);
		_dist[13] = fmaxf(ad4, bd4);
		_dist[5] = fminf(ad5, bd5);
		_dist[14] = fmaxf(ad5, bd5);
		_dist[6] = fminf(ad6, bd6);
		_dist[15] = fmaxf(ad6, bd6);
		_dist[7] = fminf(ad7, bd7);
		_dist[16] = fmaxf(ad7, bd7);
		_dist[8] = fminf(ad8, bd8);
		_dist[17] = fmaxf(ad8, bd8);
	}

	CU_FORCEINLINE __host__ __device__
	void set(const _box3f &a, const _box3f &b)
	{
		_dist[0] = fminf(b._dist[0], a._dist[0]);
		_dist[9] = fmaxf(b._dist[9], a._dist[9]);
		_dist[1] = fminf(b._dist[1], a._dist[1]);
		_dist[10] = fmaxf(b._dist[10], a._dist[10]);
		_dist[2] = fminf(b._dist[2], a._dist[2]);
		_dist[11] = fmaxf(b._dist[11], a._dist[11]);
		_dist[3] = fminf(b._dist[3], a._dist[3]);
		_dist[12] = fmaxf(b._dist[12], a._dist[12]);
		_dist[4] = fminf(b._dist[4], a._dist[4]);
		_dist[13] = fmaxf(b._dist[13], a._dist[13]);
		_dist[5] = fminf(b._dist[5], a._dist[5]);
		_dist[14] = fmaxf(b._dist[14], a._dist[14]);
		_dist[6] = fminf(b._dist[6], a._dist[6]);
		_dist[15] = fmaxf(b._dist[15], a._dist[15]);
		_dist[7] = fminf(b._dist[7], a._dist[7]);
		_dist[16] = fmaxf(b._dist[16], a._dist[16]);
		_dist[8] = fminf(b._dist[8], a._dist[8]);
		_dist[17] = fmaxf(b._dist[17], a._dist[17]);
	}

	CU_FORCEINLINE __host__ __device__  
	void add(const REAL3 &a)
	{
		_dist[0] = fminf(a.x, _dist[0]);
		_dist[9] = fmaxf(a.x, _dist[9]);
		_dist[1] = fminf(a.y, _dist[1]);
		_dist[10] = fmaxf(a.y, _dist[10]);
		_dist[2] = fminf(a.z, _dist[2]);
		_dist[11] = fmaxf(a.z, _dist[11]);

		REAL d3, d4, d5, d6, d7, d8;
		__getDistances(a, d3, d4, d5, d6, d7, d8);

		_dist[3] = fminf(d3, _dist[3]);
		_dist[12] = fmaxf(d3, _dist[12]);
		_dist[4] = fminf(d4, _dist[4]);
		_dist[13] = fmaxf(d4, _dist[13]);
		_dist[5] = fminf(d5, _dist[5]);
		_dist[14] = fmaxf(d5, _dist[14]);
		_dist[6] = fminf(d6, _dist[6]);
		_dist[15] = fmaxf(d6, _dist[15]);
		_dist[7] = fminf(d7, _dist[7]);
		_dist[16] = fmaxf(d7, _dist[16]);
		_dist[8] = fminf(d8, _dist[8]);
		_dist[17] = fmaxf(d8, _dist[17]);
	}

	CU_FORCEINLINE __host__ __device__
	void add(const _box3f& b)
	{
		_dist[0] = fminf(b._dist[0], _dist[0]);
		_dist[9] = fmaxf(b._dist[9], _dist[9]);
		_dist[1] = fminf(b._dist[1], _dist[1]);
		_dist[10] = fmaxf(b._dist[10], _dist[10]);
		_dist[2] = fminf(b._dist[2], _dist[2]);
		_dist[11] = fmaxf(b._dist[11], _dist[11]);
		_dist[3] = fminf(b._dist[3], _dist[3]);
		_dist[12] = fmaxf(b._dist[12], _dist[12]);
		_dist[4] = fminf(b._dist[4], _dist[4]);
		_dist[13] = fmaxf(b._dist[13], _dist[13]);
		_dist[5] = fminf(b._dist[5], _dist[5]);
		_dist[14] = fmaxf(b._dist[14], _dist[14]);
		_dist[6] = fminf(b._dist[6], _dist[6]);
		_dist[15] = fmaxf(b._dist[15], _dist[15]);
		_dist[7] = fminf(b._dist[7], _dist[7]);
		_dist[16] = fmaxf(b._dist[16], _dist[16]);
		_dist[8] = fminf(b._dist[8], _dist[8]);
		_dist[17] = fmaxf(b._dist[17], _dist[17]);
	}

	CU_FORCEINLINE __host__ __device__
	void enlarge(REAL d)
	{
		for (int i = 0; i < 3; i++) {
			_dist[i] -= d;
			_dist[i + 9] += d;
		}

		for (int i = 0; i < 6; i++) {
			_dist[3 + i] -= REAL(M_SQRT2)*d;
			_dist[3 + i + 9] += REAL(M_SQRT2)*d;
		}
	}

	CU_FORCEINLINE __host__ __device__ 
	bool overlaps(const _box3f& b) const
	{
		for (int i = 0; i<9; i++) {
			if (_dist[i] > b._dist[i + 9]) return false;
			if (_dist[i + 9] < b._dist[i]) return false;
		}

		return true;
	}

	CU_FORCEINLINE __host__ __device__
	bool inside(const REAL3 &p) const
	{
		REAL d[9];
		d[0] = p.x;
		d[1] = p.y;
		d[2] = p.z;
		__getDistances(p, d[3], d[4], d[5], d[6], d[7], d[8]);

		for (int i = 0; i<9; i++) {
			if (d[i] < _dist[i] || d[i] > _dist[i + 9])
				return false;
		}

		return true;
	}

	CU_FORCEINLINE __host__ __device__
	REAL3 maxV() const {
		return make_REAL3(_dist[9], _dist[10], _dist[11]);
	}

	CU_FORCEINLINE __host__ __device__
	REAL3 minV() const {
		return make_REAL3(_dist[0], _dist[1], _dist[2]);
	}

	/*
	CU_FORCEINLINE void print() {
		//printf("%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}

	CU_FORCEINLINE void print(FILE *fp) {
		//fprintf(fp, "%lf, %lf, %lf, %lf, %lf, %lf\n", _min.x, _min.y, _min.z, _max.x, _max.y, _max.z);
	}
	*/
} g_box;

#endif
