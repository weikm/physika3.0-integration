#pragma once
#include <cuda_runtime.h>

#include <assert.h>
#include <omp.h>
#include <string>

#define USE_TIMER

struct GPUTimer2 {
private:
	cudaEvent_t start, stop;
	float		accum;

public:
	GPUTimer2() :accum(0) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	~GPUTimer2() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

#ifdef USE_TIMER
	void tick() {
		cudaEventRecord(start, 0);
	}
	void tock() {
		cudaEventRecord(stop, 0);
	}
	float tock(std::string msg) {
		cudaDeviceSynchronize();
		//cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		printTime(msg);
		float tmp;
		cudaEventElapsedTime(&tmp, start, stop);
		return tmp;
	}
	float tock2() {
		cudaDeviceSynchronize();
		//cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float tmp;
		cudaEventElapsedTime(&tmp, start, stop);
		return tmp;
	}
	void printTime(std::string msg = std::string()) {
		float	costtime;
		cudaEventElapsedTime(&costtime, start, stop);
		accum += costtime;
		printf("%s %.6f ms\n", msg.c_str(), costtime);
	}
	void reset() {
		accum = 0;
	}
	void reset(std::string msg) {
		printf("%s %.6f ms\n", msg.c_str(), accum);
		accum = 0;
	}
#else
	void tick() {
	}
	void tock() {
	}
	float tock(std::string msg) {
		return 0;
	}
	void printTime(std::string msg = std::string()) {
	}
	void reset() {
	}
	void reset(std::string msg) {
	}
#endif
};

#ifdef USE_TIMER
# define	TIMING_BEGIN \
		{REAL tmp_timing_start = omp_get_wtime(); GPUTimer2 g; g.tick();

# define	TIMING_END(message) \
		{float gpuT = g.tock2();\
	REAL tmp_timing_finish = omp_get_wtime();\
	REAL  tmp_timing_duration = tmp_timing_finish - tmp_timing_start;\
	printf("%s: %2.5f s (%3.5f ms) \n", (message), tmp_timing_duration, gpuT);}}

#else
# define	TIMING_BEGIN {
# define	TIMING_END(message) }
#endif

#define VLST_BEGIN(lstIdx, lstData, idd) \
	{int vst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int vnum = lstIdx[idd] - vst;\
	for (int vi=0; vi<vnum; vi++) {\
		int vid = lstData[vi+vst];\

#define VLST_END }}

#define FLST_BEGIN(lstIdx, lstData, idd) \
	{int fst = (idd == 0) ? 0 : lstIdx[idd-1];\
	int fnum = lstIdx[idd] - fst;\
	for (int fi=0; fi<fnum; fi++) {\
		int fid = lstData[fi+fst];\

#define FLST_END }}

void  reportMemory(char *);

///////////////////////////////////////////////////////
// show memory usage of GPU

#define BLOCK_DIM 64
#define BLOCK_DIM_1 256


inline int BPG(int N, int TPB)
{
	int blocksPerGrid = (N + TPB - 1) / (TPB);
	//printf("(N=%d, TPB=%d, stride=1, BPG=%d)\n", N, TPB, blocksPerGrid);
	
	if (blocksPerGrid > 65536) {
		printf("TM: blocksPerGrid is larger than 65536, aborting ... (N=%d, TPB=%d, BPG=%d)\n", N, TPB, blocksPerGrid);
		exit(0);
	}

	return blocksPerGrid;
}

inline int BPG(int N, int TPB, int &stride)
{
	int blocksPerGrid = 0;
	
	do {
		blocksPerGrid = (N + TPB*stride - 1) / (TPB*stride);
		//printf("(N=%d, TPB=%d, stride=%d, BPG=%d)\n", N, TPB, stride, blocksPerGrid);
	
		if (blocksPerGrid <= 65536)
			return blocksPerGrid;

#ifdef OUTPUT_TXT
		//printf("blocksPerGrid is larger than 65536, double the stride ... (N=%d, TPB=%d, stride=%d, BPG=%d)\n", N, TPB, stride, blocksPerGrid);
#endif
		stride *= 2;
	} while (1);

	assert(0);
	return 0;
}

#include "cuda_occupancy.h"
extern cudaDeviceProp deviceProp;

inline int evalOptimalBlockSize(cudaFuncAttributes attribs, cudaFuncCache cachePreference, size_t smemBytes) {
	cudaOccDeviceProp prop = deviceProp;
	cudaOccFuncAttributes occAttribs = attribs;
	cudaOccDeviceState occCache;

	switch (cachePreference) {
	case cudaFuncCachePreferNone:
		occCache.cacheConfig = CACHE_PREFER_NONE;
		break;
	case cudaFuncCachePreferShared:
		occCache.cacheConfig = CACHE_PREFER_SHARED;
		break;
	case cudaFuncCachePreferL1:
		occCache.cacheConfig = CACHE_PREFER_L1;
		break;
	case cudaFuncCachePreferEqual:
		occCache.cacheConfig = CACHE_PREFER_EQUAL;
		break;
	default:
		;	///< should throw error
	}

	int minGridSize, blockSize;
	cudaOccMaxPotentialOccupancyBlockSize(
		&minGridSize, &blockSize, &prop, &occAttribs, &occCache, nullptr, smemBytes);
	return blockSize;
}

#define LEN_CHK(l) \
    int idx = blockDim.x * blockIdx.x + threadIdx.x;\
	if (idx >= l) return;

#define LEN_CHK1(l) \
	int bidx = blockIdx.x; \
	if (bidx >= l) return;

#define LEN_CHK2(o, l, n) \
	int tidx = o + threadIdx.x; \
	int idx = threadIdx.x; \
	if (threadIdx.x >= n) return;
	//if (tidx >= l) return;\
	 

#define LEN_CHK3(l) \
	int widx = blockIdx.x * 2 + threadIdx.x / 32; \
	int laneidx = threadIdx.x % 32; \
	if (widx >= l) return;

#define LEN_CHK4(lidx, nodeCount, icount)\
	if (lidx >= nodeCount) return;


#define BLK_PAR(l) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T);

#define BLK_PAR2(l, s) \
   int T = BLOCK_DIM; \
    int B = BPG(l, T, s);

#define BLK_PAR3(l, s, n) \
   int T = n; \
    int B = BPG(l, T, s);

#define BLK_PAR4(l) \
	int T = BLOCK_DIM_1; \
	int B = l;

#define cutilSafeCall checkCudaErrors

#define M_PI       3.14159265358979323846
#define M_SQRT2    1.41421356237309504880

#include <map>
using namespace std;

typedef map<void *, int> FUNC_INT_MAP;
static  FUNC_INT_MAP blkSizeTable;

inline int getBlkSize(void *func)
{
	FUNC_INT_MAP::iterator it;

	it = blkSizeTable.find(func);
	if (it == blkSizeTable.end()) {
		cudaFuncAttributes attr;
		cudaFuncGetAttributes(&attr, func);
		int num = evalOptimalBlockSize(attr, cudaFuncCachePreferL1, 0);
		blkSizeTable[func] = num;
		return num;
	}
	else {
		return it->second;
	}
}


static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#if defined(PROF) || defined(GPU)
#define getLastCudaError(msg) {}
#else
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
#endif

#define DEVICE_RESET cudaDeviceReset();

inline void __getLastCudaError(const char *errorMessage, const char *file,
	const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
			"%s(%i) : getLastCudaError() CUDA error :"
			" %s : (%d) %s.\n",
			file, line, errorMessage, static_cast<int>(err),
			cudaGetErrorString(err));
		DEVICE_RESET
			exit(EXIT_FAILURE);
	}
}

void getCudaError(const char *msg)
{
	getLastCudaError(msg);
}


///////////////////////////////////////////////////////
// show memory usage of GPU
void  reportMemory(const char* tag)
{
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}

	REAL free_db = (REAL)free_byte;
	REAL total_db = (REAL)total_byte;
	REAL used_db = total_db - free_db;
	printf("%s: GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		tag, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);

}


