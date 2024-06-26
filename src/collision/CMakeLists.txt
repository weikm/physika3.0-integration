# @author     : Wang Wei
# @date       : 2023-07-05
# @description: CMakeLists.txt for collision
# @version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

if(BUILD_WITH_CUDA)
    project(physika_collision LANGUAGES C CXX CUDA)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/**/*.hpp ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cuh)
else()
    project(physika_collision)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cpp)
endif()

set(TARGET_NAME physika_collision)
add_library(${TARGET_NAME} ${SRC_FILES})

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Engine/collision")

if(BUILD_WITH_CUDA)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON) # #all-major
    target_include_directories(${TARGET_NAME} SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # include as system path so that compiler won't complain the warnings

    if(NOT MSVC)
        find_library(CUDADEVRT_LIB cudadevrt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        target_link_libraries(${TARGET_NAME} PUBLIC ${CUDADEVRT_LIB})
        find_library(CUDART_LIB cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        target_link_libraries(${TARGET_NAME} PUBLIC ${CUDART_LIB})
    endif()
endif()

find_package(OpenMP REQUIRED)
target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_link_libraries(${TARGET_NAME} PRIVATE physika_framework OpenMP::OpenMP_CXX)

install(TARGETS ${TARGET_NAME}
    DESTINATION ${PHYSIKA_LIB_INSTALL_PATH})
