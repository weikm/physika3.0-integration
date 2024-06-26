//#define USE_DOUBLE
#pragma once

#include "real.h"

#ifdef USE_DOUBLE
#define REAL2 double2
#define REAL3 double3
#define REAL4 double4
#else
#define REAL2 float2
#define REAL3 float3
#define REAL4 float4
#endif

#define CU_FORCEINLINE __forceinline__

/////////////////////////////////////////////////////////////////////////////////////////////
// write based on cutil_math.h
// for REAL3
CU_FORCEINLINE __host__ __device__ REAL2 make_REAL2(REAL s1, REAL s2)
{
#ifdef USE_DOUBLE
	return make_double2(s1, s2);
#else
	return make_float2(s1, s2);
#endif
}


CU_FORCEINLINE __host__ __device__ REAL3 make_REAL3(REAL s1, REAL s2, REAL s3)
{
#ifdef USE_DOUBLE
	return make_double3(s1, s2, s3);
#else
	return make_float3(s1, s2, s3);
#endif
}


CU_FORCEINLINE __host__ __device__ REAL4 make_REAL4(REAL s1, REAL s2, REAL s3, REAL s4)
{
#ifdef USE_DOUBLE
	return make_double4(s1, s2, s3, s4);
#else
	return make_float4(s1, s2, s3, s4);
#endif
}

