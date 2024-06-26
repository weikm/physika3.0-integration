#@author     : YuanmuXu
#@date       : 2023-06-10
#@description: CMakeLists.txt for building granular particle sample
#@version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

project(multiphase_fluid_accelerate_sample)

# Enable CUDA support in CMake
enable_language(CUDA)

# Note: Exclude granular_sample.cpp
FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(TARGET_NAME multiphase_fluid_accelerate_sample)

# Change this line to create a library instead of an executable
add_executable(${TARGET_NAME} ${SRC_FILES})

target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_link_libraries(${TARGET_NAME} PRIVATE physika_framework multiphase_fluids_accelerate)

# CUDA dependencies
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_cublas_LIBRARY})

file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Examples/multiphase_fluids_accelerate")

#copy dependency dlls to target bin dir
if(MSVC)
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:physika_framework> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:multiphase_fluids_accelerate> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND_EXPAND_LISTS)
    install(FILES $<TARGET_FILE:physika_framework>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH}/multiphase_fluids_accelerate)
    install(FILES $<TARGET_FILE:multiphase_fluids_accelerate>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
endif()

install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH}/multiphase_fluids_accelerate)
