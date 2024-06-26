#@author     : Yan Xiao xiao
#@date       : 2023-07-12
#@description: Top-level CMakeLists.txt for building Physika framework sample
#@version    : 1.0

cmake_minimum_required(VERSION 3.11.4)

project(multiphase_fluids_accelerate)

enable_language(CUDA)

FILE(GLOB SRC_FILES 
${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp 
${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu 
${CMAKE_CURRENT_SOURCE_DIR}/source/*.cuh 
)
FILE(GLOB HEADER_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/source/*.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.hpp)
set(TARGET_NAME multiphase_fluids_accelerate)



find_package(CUDA)

add_library(${TARGET_NAME} ${SRC_FILES} ${HEADER_FILES})

if(BUILD_WITH_CUDA)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}"
                                                 CUDA_SEPARABLE_COMPILATION ON
                                                 CUDA_RESOLVE_DEVICE_SYMBOLS ON) ##all-major
    target_include_directories(${TARGET_NAME} SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  # include as system path so that compiler won't complain the warnings
    if(NOT MSVC)
        find_library(CUDADEVRT_LIB cudadevrt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        target_link_libraries(${TARGET_NAME} PUBLIC ${CUDADEVRT_LIB})
        find_library(CUDART_LIB cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        target_link_libraries(${TARGET_NAME} PUBLIC ${CUDART_LIB})
    endif()
endif()

target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT} )

target_link_libraries(${TARGET_NAME} PRIVATE physika_framework ${CUDA_cublas_LIBRARY} ${CUDA_LIBRARIES})

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Engine/sph_multiphase")

install(FILES ${HEADER_FILES}
        DESTINATION ${PHYSIKA_HEADER_INSTALL_PATH}/multiphase_fluids_accelerate)
install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_LIB_INSTALL_PATH})