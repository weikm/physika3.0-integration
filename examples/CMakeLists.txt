#@author     : Chen Chonghan
#@date       : 2023-11-18
#@description: Top-level CMakeLists.txt for building Physika examples
#@version    : 1.0

cmake_minimum_required(VERSION 3.11.4)

project(physika_examples)

add_subdirectory(framework_sample)


if(BUILD_WITH_CUDA)

#nku
    add_subdirectory(multiphase_fluid_accelerate_sample)

#ustb

    add_subdirectory(elastic_sample)
    add_subdirectory(anisotropy_sample)
    add_subdirectory(granular_sample)
    add_subdirectory(granular_integration_sample)
    add_subdirectory(solidLiquidPhaseChange_sample)
    add_subdirectory(fluid_sample)
    add_subdirectory(fluid_sample2)
    add_subdirectory(fluid_solid_coupling_sample)
    add_subdirectory(snow_sample)
    add_subdirectory(hard_snow_sample)
    add_subdirectory(thinning_nonnewton_sample)
    add_subdirectory(thickening_nonnewton_sample)
    #add_subdirectory(thickening_nonnewtonpinn_sample)
    add_subdirectory(gasLiquidPhaseChange_sample)
    add_subdirectory(model_helper_sample)

    # FOR PARTICLE RENDERER (wqw)
    add_subdirectory(gl_particle_render)
    add_subdirectory(gl_dependencies/glfw-3.3.0)
    add_subdirectory(gl_dependencies/glad-4.6)

# zju
    add_subdirectory(collision_sample)

#hnu
    #add_subdirectory(gnn_damage_sample)
endif()

