# @author     : Wangwei
# @date       : 2023-11-18
# @description: CMakeLists.txt for particles_cd_unit_test
# @version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

if(BUILD_WITH_CUDA)
    project(particle_collision_demo LANGUAGES C CXX CUDA)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu)
else()
    project(particle_collision_demo)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
endif()

set(TARGET_NAME particle_collision_demo)
add_executable(${TARGET_NAME} ${SRC_FILES})

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
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${PHYSIKA_SRC_ROOT}
    ${CMAKE_CURRENT_SOURCE_DIR}/inc)

target_link_libraries(${TARGET_NAME} PRIVATE physika_framework
    physika_collision
    opengl32.lib
    ${PHYSIKA_EXAMPLE_ROOT}/collision_sample/rigid_falling_tm/lib/glut32.lib
    ${PHYSIKA_EXAMPLE_ROOT}/collision_sample/rigid_falling_tm/lib/glut64.lib)

# copy dependency dlls to target bin dir
if(MSVC)
    add_custom_command(TARGET ${TARGET_NAME}
        POST_BUILD

        # 'copy' command cannot be used here, because MSVC treats zero-value exit code as success, while unix-style 'copy' command
        # treats 1-value exit code as success. I encountered this issue before and resolved it with MSVC settings, just don't know how to handle it
        # in CMake. I failed with multiple tries, and complained about it to ChatGPT. He saved me with a solution. Amazing!
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PHYSIKA_EXAMPLE_ROOT}/collision_sample/bin/glut32.dll $<TARGET_FILE_DIR:${TARGET_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:physika_collision> $<TARGET_FILE_DIR:${TARGET_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:physika_framework> $<TARGET_FILE_DIR:${TARGET_NAME}>
        COMMAND_EXPAND_LISTS)
    install(FILES $<TARGET_FILE:physika_collision> ${PHYSIKA_EXAMPLE_ROOT}/collision_sample/bin/glut32.dll
        DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
endif()

#install(TARGETS ${TARGET_NAME} DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})