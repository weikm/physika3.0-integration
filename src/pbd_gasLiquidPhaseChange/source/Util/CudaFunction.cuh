#pragma once

#include "helper_math.h"

#define EPSILON (1e-6f)
#define PI (3.14159265358979323846f)
#define block_size (256)
#define MAX_A (1000.0f)

static inline __device__ float cubic_spline_kernel(const float r, const float radius)
{
	const auto q = 2.0f * fabs(r) / radius;
	if (q > 2.0f || q < EPSILON) return 0.0f;
	else {
		const auto a = 0.25f / (PI * radius * radius * radius);
		return a * ((q > 1.0f) ? (2.0f - q) * (2.0f - q) * (2.0f - q) : ((3.0f * q - 6.0f) * q * q + 4.0f));
	}
}

static inline __device__ float3 cubic_spline_kernel_gradient(const float3 r, const float radius)
{
	const auto q = 2.0f * length(r) / radius;
	if (q > 2.0f) return make_float3(0.0f);
	else {
		const auto a = r / (PI * (q + EPSILON) * radius * radius * radius * radius * radius);
		return a * ((q > 1.0f) ? ((12.0f - 3.0f * q) * q - 12.0f) : ((9.0f * q - 12.0f) * q));
	}
}

static inline __device__ float cubic_spline_kernel_laplacian(const float r, const float radius) 
{
	const auto q = 2.0f * fabs(r) / radius;
	if (q > 2.0f) return 0.0f;
	else {
		const auto a = 0.25f / (PI * radius * radius * radius);
		return a * ((q > 1.0f) ? (2.0f - q) * (2.0f - q) * (2.0f - q) : ((3.0f * q - 6.0f) * q * q + 4.0f));
	}
}

static inline __device__ float viscosity_kernel_laplacian(const float r, const float radius) {
	return (r <= radius) ? (45.0f * (radius - r) / (PI * powf(radius, 6))) : 0.0f;
}
/*
	Locate particles
*/
static inline __device__ unsigned int particlePos2cellIdx(const int3 pos, const int3 cellSize)
{
	return (pos.x >= 0 && pos.x < cellSize.x && pos.y >= 0 && pos.y < cellSize.y && pos.z >= 0 && pos.z < cellSize.z) ?
		(((pos.x * cellSize.y) + pos.y) * cellSize.z + pos.z)
		: (cellSize.x * cellSize.y * cellSize.z);
}

inline __device__ float3 surface_tension_kernel_gradient(float3 r, const float radius)
{
	const auto x = length(r);
	//return
	//	(x < EPSILON) ? make_float3(0.0f) : (
	//		2.0f * x <= radius ? 2.0f * powf((radius - x), 3) * powf(x, 3) - 0.0156f * powf(radius, 6) :
	//		x <= radius ? powf((radius - x), 3) * powf(x, 3) :
	//		0.0f) * 136.0241f / (PI * powf(radius, 9)) * -r / fmaxf(EPSILON, x);
	if (x > radius || x < EPSILON) return make_float3(0.0f);
	else {
		auto cube = [](float x) {return x * x * x; };
		const float3 a = 136.0241f * -r / (PI * cube(radius) * cube(radius) * cube(radius) * x);
		return a * ((2.0f * x <= radius) ?
			(2.0f * cube(radius - x) * cube(x) - 0.0156f * cube(radius) * cube(radius)) :
			(cube(radius - x) * cube(x)));
	}
}


namespace ThrustHelper {
	template<typename T>
	struct plus {
		T _a;
		plus(const T a) :_a(a) {}
		__host__ __device__
			T operator()(const T& lhs) const {
			return lhs + _a;
		}
	};

	template <typename T>
	struct abs_plus
	{
		__host__ __device__
			T operator()(const T& lhs, const T& rhs) const {
			return abs(lhs) + abs(rhs);
		}
	};
}
