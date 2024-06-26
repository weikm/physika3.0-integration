#@author     : YuanmuXu
#@date       : 2023-06-10
#@description: CMakeLists.txt for building granular particle sample
#@version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

# project(granular_sample)

if(BUILD_WITH_CUDA)
    project(granular_sample LANGUAGES C CXX CUDA)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
else()
    project(granular_sample)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
endif()


# Enable CUDA support in CMake
# enable_language(CUDA)

# Note: Exclude granular_sample.cpp
# FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(TARGET_NAME granular_sample)

# Change this line to create a library instead of an executable
add_executable(${TARGET_NAME} ${SRC_FILES})

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

# set_property(TARGET ${TARGET_NAME} 
#              PROPERTY CUDA_ARCHITECTURES 75)

# or you can let cmake automaticly chose CMAKE_CUDA_ARCHITECTURES
#if(POLICY CMP0104)
#    cmake_policy(SET CMP0104 NEW)
#endif()

if(BUILD_SHARED_LIBS)
    set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH $ORIGIN/../../lib)
endif()
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Examples/bku_sample")
target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_link_libraries(${TARGET_NAME} PRIVATE physika_framework pbd_granular gl_particle_render glfw glad)

# CUDA dependencies
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})

#copy dependency dlls to target bin dir
if(MSVC)
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:physika_framework> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pbd_granular> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:gl_particle_render> $<TARGET_FILE_DIR:${TARGET_NAME}> 
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glfw> $<TARGET_FILE_DIR:${TARGET_NAME}> 
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glad> $<TARGET_FILE_DIR:${TARGET_NAME}> 

                       COMMAND_EXPAND_LISTS)
    install(FILES $<TARGET_FILE:physika_framework>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    install(FILES $<TARGET_FILE:pbd_granular>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    install(FILES $<TARGET_FILE:gl_particle_render>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    install(FILES $<TARGET_FILE:glfw>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    install(FILES $<TARGET_FILE:glad>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})

endif()

install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
