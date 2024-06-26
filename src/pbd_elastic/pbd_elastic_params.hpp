/*
 * @Author: pgpgwhp 1213388412@qq.com
 * @Date: 2023-09-19 15:48:32
 * @LastEditors: pgpgwhp 1213388412@qq.com
 * @LastEditTime: 2023-11-09 11:50:32
 * @FilePath: \physika\src\pbd-elastic\pbd_elastoc_params.hpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef ELASTIC_PARAMS_H
#define ELASTIC_PARAMS_H

#include <vector_types.h>
#define M_PI 3.14159265358979323846f

namespace Physika{


struct ElasticSolverParams {
		
	float3 m_lb_boundary; // lower boundary for world  
	float3 m_rt_boundary; // upper boundary for world
	float3 m_cell_size; // cell size of the grid for the hash search
	float3 m_world_orgin; // world origin for the hash search 
	uint3 m_grid_num; // grid number in three dimension
    unsigned int grid_size; // total grid number

	float m_particle_radius; //particle radius 
	float m_particle_dimcenter; // particle dimcenter
	float m_sph_radius; // SPH neighbour search radius
	float m_volume; // particle volume
	float m_mass;	// particle mass
	float m_invMass; // particle inverse mass
	
	// elastic attribute
	float young_modules; // Young's modules  https://zh.wikipedia.org/zh-tw/%E6%9D%A8%E6%B0%8F%E6%A8%A1%E9%87%8F
	float possion_ratio;  // Possion ratio  https://zh.wikipedia.org/wiki/%E6%B3%8A%E6%9D%BE%E6%AF%94
	float lame_first;  // lame first parameter  https://zh.wikipedia.org/wiki/%E6%8B%89%E6%A2%85%E5%8F%82%E6%95%B0
	float lame_second;  // lame second parameter 
};

}

#endif