#@author     : YuanmuXu
#@date       : 2023-06-10
#@description: CMakeLists.txt for building snow solver sample
#@version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

project(hard_snow_sample)

# build with gui
option(BUILD_WITH_GUI "Build with GUI" on)

# Note: Exclude granular_sample.cpp
FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

set(TARGET_NAME hard_snow_sample)

# Change this line to create a library instead of an executable
add_executable(${TARGET_NAME} ${SRC_FILES})

if(BUILD_WITH_CUDA)
    # Enable CUDA support in CMake
    enable_language(CUDA)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}"
                                                 CUDA_SEPARABLE_COMPILATION ON
                                                 CUDA_RESOLVE_DEVICE_SYMBOLS ON) ##all-major
    target_include_directories(${TARGET_NAME} SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  # include as system path so that compiler won't complain the warnings
    if(NOT MSVC)
        find_library(CUDADEVRT_LIB cudadevrt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        target_link_libraries(${TARGET_NAME} PUBLIC ${CUDADEVRT_LIB})
        find_library(CUDART_LIB cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
        target_link_libraries(${TARGET_NAME} PUBLIC ${CUDART_LIB})
    endif()# CUDA dependencies
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${CUDA_LIBRARIES})    
endif()

if(BUILD_SHARED_LIBS)
    set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH $ORIGIN/../../lib)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Examples/bku_sample")
target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
if(BUILD_WITH_GUI)
    target_link_libraries(${TARGET_NAME} PRIVATE physika_framework mpm_snow model_helper gl_particle_render glfw glad)
else()
    target_link_libraries(${TARGET_NAME} PRIVATE physika_framework mpm_snow model_helper)
endif()
#copy dependency dlls to target bin dir
if(MSVC)
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:physika_framework> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:mpm_snow> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:model_helper> $<TARGET_FILE_DIR:${TARGET_NAME}>                       
                       COMMAND_EXPAND_LISTS)
    if(BUILD_WITH_GUI)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:gl_particle_render> $<TARGET_FILE_DIR:${TARGET_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glfw> $<TARGET_FILE_DIR:${TARGET_NAME}>
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glad> $<TARGET_FILE_DIR:${TARGET_NAME}>
        )
    endif()
    install(FILES $<TARGET_FILE:physika_framework>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    install(FILES $<TARGET_FILE:mpm_snow>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    install(FILES $<TARGET_FILE:model_helper>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    if(BUILD_WITH_GUI)
    # koko(wqw)
    install(FILES $<TARGET_FILE:gl_particle_render>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
            install(FILES $<TARGET_FILE:glfw>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
            install(FILES $<TARGET_FILE:glad>
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    endif()
endif()

install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})