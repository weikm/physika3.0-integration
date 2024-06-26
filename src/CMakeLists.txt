# @author     : Zhu Fei
# @date       : 2023-03-23
# @description: Top-level CMakeLists.txt for building Physika library sources
# @version    : 1.0

cmake_minimum_required(VERSION 3.11.4)

project(physika_lib)

# build framework
add_subdirectory(framework)


if(BUILD_WITH_CUDA)

#nku
    add_subdirectory(multiphase_fluids_accelerate)

#usbt
    add_subdirectory(utils)

    add_subdirectory(pbd_elastic)
    add_subdirectory(pbd_granular)
    add_subdirectory(pbd_solidLiquidPhaseChange)
    add_subdirectory(pbd_fluid)
    add_subdirectory(pbd_fluid_solid_coupling)
    add_subdirectory(mpm_snow)
    add_subdirectory(pbf_nonnewton)
    add_subdirectory(pbd_gasLiquidPhaseChange)

#zju
    add_subdirectory(collision)

#hnu
    #add_subdirectory(gnn_damage)

endif()
