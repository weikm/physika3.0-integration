//
// Created by sl936 on 2023/7/26.
//

#ifndef AQUASIM_IOTOOL_H
#define AQUASIM_IOTOOL_H

#include <fstream>
#include "helper_math.hpp"

/**
 * @brief  : write pos to ply file
 *
 * @param[in]  : filename  file path
 * @param[in]  : points  set of points
 */
static void write_ply(const std::string& filename, const std::vector<float3>& points)
{
    std::ofstream ofs(filename);

    ofs << "ply\n";
    ofs << "format ascii 1.0\n";
    ofs << "element vertex " << points.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";

    for (const auto& point : points)
    {
        ofs << point.x << " " << point.y << " " << point.z << "\n";
    }

    ofs.close();
}

/**
 * @brief  : dump average value of specified param
 *
 * @param[in]  : num  particle num
 * @param[in]  : d_params  device pointer of param
 */
template <typename ParamPType>
static void dump_avg(const unsigned int num, ParamPType d_params)
{
    if (std::is_same<ParamPType, float*>::value)
    {
        auto* c_params = new float[num];
        cudaMemcpy(c_params, d_params, num * sizeof(float), cudaMemcpyDeviceToHost);

        float avg = 0.f;
        for (int i = 0; i < num; ++i)
            avg += c_params[i];

        std::cout << "dump_avg():: The specified param's AVG_VALUE: " << avg / static_cast<float>(num) << "\n";

        delete[] c_params;
    }
    else if (std::is_same<ParamPType, unsigned int*>::value)
    {
        auto* c_params = new unsigned int[num];
        cudaMemcpy(c_params, d_params, num * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        float avg = 0.f;
        for (int i = 0; i < num; ++i)
            avg += c_params[i];

        std::cout << "dump_avg():: The specified param's AVG_VALUE: " << avg / static_cast<float>(num) << "\n";

        delete[] c_params;
    }
    else if (std::is_same<ParamPType, float3*>::value)
    {
        auto* c_params = new float3[num];
        cudaMemcpy(c_params, d_params, num * sizeof(float3), cudaMemcpyDeviceToHost);

        float3 avg{ 0, 0, 0 };
        for (int i = 0; i < num; ++i)
            avg += c_params[i];

        avg /= static_cast<float>(num);
        std::cout << "dump_avg():: The specified param's AVG_VALUE: "
                  << "[ " << avg.x << ", " << avg.y << ", " << avg.z << " ]\n";

        delete[] c_params;
    }
    else
    {
        throw std::runtime_error("dump_avg():: Param type not supported!");
    }
}

/**
 * @brief  : average value output of specified param
 *
 * @param[in]  : num  particle num
 * @param[in]  : d_params  device pointer of param
 * @param[in]  : value  output reference
 */
template <typename ParamPType, typename OutputType>
static void avg_output(const unsigned int num, ParamPType d_params, OutputType& value)
{
    if (std::is_same<ParamPType, float*>::value)
    {
        auto* c_params = new float[num];
        cudaMemcpy(c_params, d_params, num * sizeof(float), cudaMemcpyDeviceToHost);

        float avg = 0.f;
        for (int i = 0; i < num; ++i)
            avg += c_params[i];

        value = avg / static_cast<float>(num);

        delete[] c_params;
    }
    else if (std::is_same<ParamPType, unsigned int*>::value)
    {
        auto* c_params = new unsigned int[num];
        cudaMemcpy(c_params, d_params, num * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        float avg = 0.f;
        for (int i = 0; i < num; ++i)
            avg += c_params[i];

        value = avg / static_cast<float>(num);

        delete[] c_params;
    }
    else
    {
        throw std::runtime_error("dump_avg():: Param type not supported!");
    }
}

#endif  // AQUASIM_IOTOOL_H
