#pragma once
#include <assert.h>
#include <cstring>
#include <cuda_runtime.h>
namespace Physika {

#define INVALID -1
#define COMM_FUNC __device__ __host__

class Array3D
{
public:
    Array3D()
        : m_nx(0)
        , m_ny(0)
        , m_nz(0)
        , m_nxy(0)
        , m_totalNum(0)
        , m_data(NULL){};

    Array3D(int nx, int ny, int nz)
        : m_nx(nx)
        , m_ny(ny)
        , m_nz(nz)
        , m_nxy(nx * ny)
        , m_totalNum(nx * ny * nz)
        , m_data(NULL)
    {
        AllocMemory();
    };

    /*!
     *    \brief    Should not release data here, call Release() explicitly.
     */
    ~Array3D(){};

    void Release()
    {
        if (m_data != NULL)
        {
            cudaFree(m_data);
        }

        m_data     = NULL;
        m_nx       = 0;
        m_ny       = 0;
        m_nz       = 0;
        m_nxy      = 0;
        m_totalNum = 0;
    }

    void Resize(int nx, int ny, int nz)
    {
        if (NULL != m_data)
            Release();
        m_nx       = nx;
        m_ny       = ny;
        m_nz       = nz;
        m_nxy      = m_nx * m_ny;
        m_totalNum = m_nxy * m_nz;
        AllocMemory();
    }

    void Reset()
    {
        cudaMemset(m_data, 0, m_totalNum * sizeof(float));
    }

    inline float* GetDataPtr()
    {
        return m_data;
    }
    void SetDataPtr(float* _data)
    {
        m_data = _data;
    }

    COMM_FUNC inline int Nx()
    {
        return m_nx;
    }
    COMM_FUNC inline int Ny()
    {
        return m_ny;
    }
    COMM_FUNC inline int Nz()
    {
        return m_nz;
    }

    COMM_FUNC inline float operator()(const int i, const int j, const int k) const
    {
        return m_data[i + j * m_nx + k * m_nxy];
    }

    COMM_FUNC inline float& operator()(const int i, const int j, const int k)
    {
        return m_data[i + j * m_nx + k * m_nxy];
    }

    COMM_FUNC inline int Index(const int i, const int j, const int k)
    {
        return i + j * m_nx + k * m_nxy;
    }

    COMM_FUNC inline float operator[](const int id) const
    {
        return m_data[id];
    }

    COMM_FUNC inline float& operator[](const int id)
    {
        return m_data[id];
    }

    COMM_FUNC inline int Size()
    {
        return m_totalNum;
    }

public:
    void AllocMemory()
    {
        cudaMalloc(( void** )&m_data, m_totalNum * sizeof(float));

        Reset();
    }

private:
    int    m_nx;
    int    m_ny;
    int    m_nz;
    int    m_nxy;
    int    m_totalNum;
    float* m_data;
};

}  // namespace Physika