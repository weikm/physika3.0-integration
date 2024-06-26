#pragma once

// #define USE_DOUBLE

#ifdef USE_DOUBLE
#define REAL double
#else
#define REAL float
#endif

#define MAX_PAIR_NUM 20000000