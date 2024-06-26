#@author     : YuegeXiong
#@date       : 2023-11-22
#@description: CMakeLists.txt for building fluid particle sample
#@version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

project(pbd_fluid)

# Enable CUDA support in CMake
enable_language(CUDA)

# Note: Exclude fluid_sample.cpp
FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cuh  ${CMAKE_CURRENT_SOURCE_DIR}/source/*.hpp  ${CMAKE_CURRENT_SOURCE_DIR}/interface/*.hpp)

set(TARGET_NAME pbd_fluid)

# Change this line to create a library instead of an executable
add_library(${TARGET_NAME} SHARED ${SRC_FILES})

# TODO(Yuanmu Xu): Set CUDA_ARCHITECTURES uniformly from the upper cmake
# set_property(TARGET ${TARGET_NAME} 
#            PROPERTY CUDA_ARCHITECTURES ${})

# or you can let cmake automaticly chose CMAKE_CUDA_ARCHITECTURES
# if(POLICY CMP0104)
#     cmake_policy(SET CMP0104 NEW)
# endif()

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
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Engine/bku_algorithm")
target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_link_libraries(${TARGET_NAME} PRIVATE physika_framework)

# CUDA dependencies
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/interface ${CMAKE_CURRENT_SOURCE_DIR}/source)
target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})

set(INSTALL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)

foreach(INCLUDE_DIR ${INSTALL_INCLUDE_DIRS})
    file(GLOB_RECURSE HEADERS ${INCLUDE_DIR}/*.hpp ${INCLUDE_DIR}/*.h ${INCLUDE_DIR}/*.cuh)
    foreach(HEADER ${HEADERS})
        file(RELATIVE_PATH HEADER_REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${HEADER})
        get_filename_component(HEADER_DIR ${HEADER_REL_PATH} DIRECTORY)
        install(FILES ${HEADER} DESTINATION ${PHYSIKA_HEADER_INSTALL_PATH}/pbd_fluid/${HEADER_DIR})
    endforeach()
endforeach()
install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_LIB_INSTALL_PATH})