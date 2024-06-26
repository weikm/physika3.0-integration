#pragma once
#include <cuda_runtime.h>
#include <iostream>

#include "Util/helper_cuda.h"
struct DisableCopy {
    DisableCopy() = default;
    DisableCopy(DisableCopy const &) = delete;
    DisableCopy &operator=(DisableCopy const &) = delete;
};

template <class T>
struct DataArray: DisableCopy{
    /*
    static_assert(
        std::is_same<T, float3>::value || std::is_same<T, float>::value || std::is_same<T, uint1>::value ||
        std::is_same<T, int>::value || std::is_same<T, uint3>::value, "Data must be of int, uint3, float or float3.");
    */
    const unsigned int _length;
    const std::shared_ptr<T> d_array;

    explicit DataArray(const unsigned int length) :
        _length(length),
        d_array([length](){
            T* ptr;
            checkCudaErrors(cudaMalloc((void **)& ptr, sizeof(T) * length)); // Malloc on device
            std::shared_ptr<T> t(new(ptr) T[length], 
            [](T* ptr){ // Deleter called
                checkCudaErrors(cudaFree(ptr));
            }
            );
            return t;
        }()){
            this->clear(); // Initialize to 0
        }
    T* addr(const int offset=0) const{
        return d_array.get() + offset;
    }
    unsigned int length() const { 
        return _length;
    }
    void clear(){
        checkCudaErrors(cudaMemset(this->addr(), 0, sizeof(T) * this->length()));
    }
    ~DataArray() noexcept{}
};