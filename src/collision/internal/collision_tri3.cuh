/**
 * @author     : Tang Min (tang_m@zju.edu.cn)
 * @date       : 2023-07-06
 * @description: device triangle data structure, should not been used directly
 * @version    : 1.0
 */

 #pragma once
#include "collision/internal/collision_tool.cuh"
 

 #define MAX_PAIR_NUM 40000000
 
 /**
  * device triangle data structure
  */
 typedef struct tri3f
 {
     uint3 _ids;  //!< face indices of the triangle
 
     inline __device__ __host__ uint id0() const
     {
         return _ids.x;
     }
     inline __device__ __host__ uint id1() const
     {
         return _ids.y;
     }
     inline __device__ __host__ uint id2() const
     {
         return _ids.z;
     }
     inline __device__ __host__ uint id(int i) const
     {
         return (i == 0 ? id0() : ((i == 1) ? id1() : id2()));
     }
 } g_tri3f;
 