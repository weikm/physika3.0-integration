#@author     : RuolanLi
#@date       : 2023-11-17
#@description: CMakeLists.txt for building solidLiquidPhaseChange sample
#@version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

# project(solidLiquidPhaseChange_sample)

if(BUILD_WITH_CUDA)
    project(solidLiquidPhaseChange_sample LANGUAGES C CXX CUDA)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
else()
    project(solidLiquidPhaseChange_sample)
    FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
endif()

# 遍历所有源文件
foreach(SRC_FILE IN LISTS SRC_FILES)
    # 获取源文件的名称（不含路径和扩展名）
    get_filename_component(EXECUTABLE_NAME ${SRC_FILE} NAME_WE)
    # 创建目标可执行文件
    add_executable(${EXECUTABLE_NAME} ${SRC_FILE})

    if(BUILD_WITH_CUDA)
        set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH}"
                                                    CUDA_SEPARABLE_COMPILATION ON
                                                    CUDA_RESOLVE_DEVICE_SYMBOLS ON) ##all-major
        target_include_directories(${EXECUTABLE_NAME} SYSTEM PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  # include as system path so that compiler won't complain the warnings
        if(NOT MSVC)
            find_library(CUDADEVRT_LIB cudadevrt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
            target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${CUDADEVRT_LIB})
            find_library(CUDART_LIB cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
            target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${CUDART_LIB})
        endif()
    endif()
    if(BUILD_SHARED_LIBS)
        set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH $ORIGIN/../../lib)
    endif()
    set_target_properties(${EXECUTABLE_NAME}  PROPERTIES FOLDER "Examples/bku_sample")
    target_include_directories(${EXECUTABLE_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE physika_framework pbd_solidLiquidPhaseChange gl_particle_render glfw glad)

    # CUDA dependencies
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${CUDA_LIBRARIES})

    # 复制依赖的动态链接库到目标可执行文件目录
    if(MSVC)
        add_custom_command(TARGET ${EXECUTABLE_NAME}
                           POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:physika_framework> $<TARGET_FILE_DIR:${EXECUTABLE_NAME}>
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:pbd_solidLiquidPhaseChange> $<TARGET_FILE_DIR:${EXECUTABLE_NAME}>
                            # install particle render module(wqw)
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:gl_particle_render> $<TARGET_FILE_DIR:${EXECUTABLE_NAME}> 
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glfw> $<TARGET_FILE_DIR:${EXECUTABLE_NAME}> 
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:glad> $<TARGET_FILE_DIR:${EXECUTABLE_NAME}> 

                           COMMAND_EXPAND_LISTS)
        install(FILES $<TARGET_FILE:physika_framework>
                DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
        install(FILES $<TARGET_FILE:pbd_solidLiquidPhaseChange>
                DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})        
        install(FILES $<TARGET_FILE:gl_particle_render>
                DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
        install(FILES $<TARGET_FILE:glfw>
                DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
        install(FILES $<TARGET_FILE:glad>
                DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
    endif() 
    install(TARGETS ${EXECUTABLE_NAME}
            DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})    
         
endforeach()