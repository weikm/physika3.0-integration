#@author     : Yuhang Xu (mr.xuyh@qq.com)
#@date       : 2023-10-17
#@description: CMakeLists.txt for building fluid solid coupling sample
#@version    : 1.0
cmake_minimum_required(VERSION 3.11.4)

project(datadriven_sample)

FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(TARGET_NAME datadriven_sample)

# Change this line to create a library instead of an executable
add_executable(${TARGET_NAME} ${SRC_FILES})

find_package(PythonLibs 3.8 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

include_directories(/usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy)

add_definitions(-DSOURCE_PATH="${CMAKE_SOURCE_DIR}")


target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_link_libraries(${TARGET_NAME} PRIVATE datadriven)
target_link_libraries(${TARGET_NAME} PRIVATE ${PYTHON_LIBRARIES})
if(MSVC)
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:physika_framework> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:datadriven> $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND_EXPAND_LISTS)
    install(FILES $<TARGET_FILE:physika_framework>
            DESTINATION ${PHYSIKA_TEST_INSTALL_PATH})
    install(FILES $<TARGET_FILE:datadriven>
            DESTINATION ${PHYSIKA_TEST_INSTALL_PATH})
endif()

install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_EXAMPLE_INSTALL_PATH})
