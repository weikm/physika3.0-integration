cmake_minimum_required(VERSION 3.0)
project(datadriven)

option(BUILD_WITH_GUI "Build with GUI" on)

FILE(GLOB SRC_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
    )

set(TARGET_NAME datadriven)

add_library(${TARGET_NAME} ${SRC_FILES})

find_package(PythonLibs 3.8 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

include_directories(/usr/local/lib/python3.8/dist-packages/numpy/core/include/numpy)

add_definitions(-DSOURCE_PATH="${CMAKE_SOURCE_DIR}")

target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_link_libraries(${TARGET_NAME} PRIVATE ${PYTHON_LIBRARIES})

FILE(GLOB HEADER_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h 
    )
target_sources(datadriven PRIVATE ${HEADER_FILES})

install(FILES ${HEADER_FILES}
        DESTINATION ${PHYSIKA_HEADER_INSTALL_PATH}/datadriven)
install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_LIB_INSTALL_PATH})