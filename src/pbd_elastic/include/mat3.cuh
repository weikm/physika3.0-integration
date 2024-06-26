#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include"svd3_cuda.cuh"

namespace Physika {
    class mat3 {
    public:

     float value[9];

    __host__ __device__  __forceinline__ mat3::mat3() {
        for (int i = 0; i < 9; i++) {
            value[i] = 0.0f;
        }
    };

    __host__ __device__ __forceinline__ mat3::mat3(float x1,float x2,float x3,float x4,float x5,float x6,float x7,float x8,float x9){
        value[0] = x1;
        value[1] = x2;
        value[2] = x3;
        value[3] = x4;
        value[4] = x5;
        value[5] = x6;
        value[6] = x7;
        value[7] = x8;
        value[8] = x9;
    }
    __host__ __device__ __forceinline__ mat3::mat3(float3 x1,float3 x2,float3 x3) {
        value[0] = x1.x;
        value[1] = x1.y;
        value[2] = x1.z;
        value[3] = x2.x;
        value[4] = x2.y;
        value[5] = x2.z;
        value[6] = x3.x;
        value[7] = x3.y;
        value[8] = x3.z;
    }
     __host__ __device__ __forceinline__ const mat3 mat3::operator+ (const mat3 &rhs) const {
        return mat3(*this) += rhs;
    }

    __host__ __device__ __forceinline__ mat3& mat3::operator+= (const mat3 &rhs) {
        
        this->value[0] += rhs.value[0];
        this->value[1] += rhs.value[1];
        this->value[2] += rhs.value[2];
        this->value[3] += rhs.value[3];
        this->value[4] += rhs.value[4];
        this->value[5] += rhs.value[5];
        this->value[6] += rhs.value[6];
        this->value[7] += rhs.value[7];
        this->value[8] += rhs.value[8];

        return *this;
    }

    __host__ __device__ __forceinline__ const mat3 mat3::operator- (const mat3 &rhs) const {
        return mat3(*this) -= rhs;
    }

    __host__ __device__ __forceinline__ mat3& mat3::operator-= (const mat3 &rhs) {

        this->value[0] -= rhs.value[0];
        this->value[1] -= rhs.value[1];
        this->value[2] -= rhs.value[2];
        this->value[3] -= rhs.value[3];
        this->value[4] -= rhs.value[4];
        this->value[5] -= rhs.value[5];
        this->value[6] -= rhs.value[6];
        this->value[7] -= rhs.value[7];
        this->value[8] -= rhs.value[8];

        return *this;
    }

    __host__ __device__ __forceinline__ const mat3 mat3::operator* (const mat3 &rhs) const {
        return mat3(*this) *= rhs;
    }

    __host__ __device__ __forceinline__ mat3& mat3::operator*= (const mat3 &rhs) {
        
        mat3 t = transpose();

        value[0] = t.value[0]*rhs.value[0] + t.value[1]*rhs.value[3] + t.value[2]*rhs.value[6];
        value[1] = t.value[0]*rhs.value[1] + t.value[1]*rhs.value[4] + t.value[2]*rhs.value[7];
        value[2] = t.value[0]*rhs.value[2] + t.value[1]*rhs.value[5] + t.value[2]*rhs.value[8];

        value[3] = t.value[3]*rhs.value[0] + t.value[4]*rhs.value[3] + t.value[5]*rhs.value[6];
        value[4] = t.value[3]*rhs.value[1] + t.value[4]*rhs.value[4] + t.value[5]*rhs.value[7];
        value[5] = t.value[3]*rhs.value[2] + t.value[4]*rhs.value[5] + t.value[5]*rhs.value[8];

        value[6] = t.value[6]*rhs.value[0] + t.value[7]*rhs.value[3] + t.value[8]*rhs.value[6];
        value[7] = t.value[6]*rhs.value[1] + t.value[7]*rhs.value[4] + t.value[8]*rhs.value[7];
        value[8] = t.value[6]*rhs.value[2] + t.value[7]*rhs.value[5] + t.value[8]*rhs.value[8];

        return *this;
    }

    __host__ __device__ __forceinline__ const mat3 mat3::operator/ (const mat3 &rhs) const {
        return mat3(*this) *= rhs.inverse();
    }

    __host__ __device__ __forceinline__ mat3& mat3::operator/= (const mat3 &rhs) {
        mat3 result;
        result = mat3(*this) * rhs.inverse();
        return result;
    }

    __host__ __device__ __forceinline__ void mat3::operator= (const mat3& rhs) {
        for (int i = 0; i < 9; ++i) {
            value[i] = rhs.value[i];
        }
    }
    __host__ __device__ __forceinline__ bool mat3::operator== (const mat3& rhs) const{
        for (int i = 0; i < 9; ++i) {
            if (value[i] != rhs.value[i]) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ __forceinline__ bool mat3::operator!= (const mat3& rhs) const{
        return !(*this == rhs);        
    }

    __host__ __device__ __forceinline__ const float3 mat3::operator*(const float3& vec) const {
        float3 res;
        res.x = value[0]*vec.x + value[1]*vec.y + value[2]*vec.z;
        res.y = value[3]*vec.x + value[4]*vec.y + value[5]*vec.z;
        res.z = value[6]*vec.x + value[7]*vec.y + value[8]*vec.z;
        return res; 
    }

    __host__ __device__ __forceinline__  const mat3 mat3::operator*(const float& rhs) const {
        return mat3(*this) *= rhs;
    }

    __host__ __device__ __forceinline__ mat3& mat3::operator*=(const float& rhs) {

        this->value[0] *= rhs;
        this->value[1] *= rhs;
        this->value[2] *= rhs;
        this->value[3] *= rhs;
        this->value[4] *= rhs;
        this->value[5] *= rhs;
        this->value[6] *= rhs;
        this->value[7] *= rhs;
        this->value[8] *= rhs;

        return *this;
    }

    __host__ __device__ __forceinline__ mat3 mat3::identity() const {
        mat3 res;
        res.value[0] = 1.f;
        res.value[1] = 0.f;
        res.value[2] = 0.f;

        res.value[3] = 0.f;
        res.value[4] = 1.f;
        res.value[5] = 0.f;

        res.value[6] = 0.f;
        res.value[7] = 0.f;
        res.value[8] = 1.f;

        return res;
    }

    __host__ __device__ __forceinline__ mat3 mat3::transpose() const {
        mat3 res;
        res.value[0] = value[0];
        res.value[1] = value[3];
        res.value[2] = value[6];

        res.value[3] = value[1];
        res.value[4] = value[4];
        res.value[5] = value[7];

        res.value[6] = value[2];
        res.value[7] = value[5];
        res.value[8] = value[8];
		return res;
    } 

    __host__ __device__ __forceinline__ mat3 mat3::inverse() const {
        float cur_determinant =  1.f / determinant();
        mat3 res;

        res.value[0] = +(value[4] * value[8] - value[7] * value[5]) * cur_determinant;
        res.value[3] = -(value[3] * value[8] - value[6] * value[5]) * cur_determinant;
        res.value[6] = +(value[3] * value[7] - value[6] * value[4]) * cur_determinant;

        res.value[1] = -(value[1] * value[8] - value[7] * value[2]) * cur_determinant;
        res.value[4] = +(value[0] * value[8] - value[6] * value[2]) * cur_determinant;
        res.value[7] = -(value[0] * value[7] - value[6] * value[1]) * cur_determinant;

        res.value[2] = +(value[1] * value[5] - value[4] * value[2]) * cur_determinant;
        res.value[5] = -(value[0] * value[5] - value[3] * value[2]) * cur_determinant;
        res.value[8] = +(value[0] * value[4] - value[3] * value[1]) * cur_determinant;

        return res;
    }

    __host__ __device__ __forceinline__ float mat3::determinant() const {
        return value[0] * (value[4] * value[8] - value[7] * value[5])
                - value[3] * (value[1] * value[8] - value[7] * value[2])
                + value[6] * (value[1] * value[5] - value[4] * value[2]);

    }
    __host__ __device__ __forceinline__ float mat3::trace() const{
        return value[0] + value[4] + value[8]; 
    }
    __host__ __device__ __forceinline__ float mat3::frobeniusNorm() const{
        float result = 0;

        for (unsigned int i = 0; i < 9; ++i)
            result += value[i] * value[i];

        return sqrt(result);
    }

    __device__ __forceinline__ void mat3::polarDecomposition (mat3& S, mat3& Q) {
        mat3 D, U, V;

        svd(this->value[0], this->value[1], this->value[2],
            this->value[3], this->value[4], this->value[5],
            this->value[6], this->value[7], this->value[8],

            U.value[0], U.value[1], U.value[2],
            U.value[3], U.value[4], U.value[5],
            U.value[6], U.value[7], U.value[8],

            D.value[0], D.value[1], D.value[2],
            D.value[3], D.value[4], D.value[5],
            D.value[6], D.value[7], D.value[8],

            V.value[0], V.value[1], V.value[2],
            V.value[3], V.value[4], V.value[5],
            V.value[6], V.value[7], V.value[8]);

        /*mat3 H;
        H.value[0] = 1;
        H.value[4] = 1;
        H.value[8] = (V * U.transpose()).determinant();*/

        S = U * D * U.transpose();
        Q = U * V.transpose();

    }

   };
}

#endif