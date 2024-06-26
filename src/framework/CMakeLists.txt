#@author     : Zhu Fei
#@date       : 2023-03-23
#@description: CMakeLists.txt for building Physika framework module
#@version    : 1.0

cmake_minimum_required(VERSION 3.11.4)

project(physika_framework)

FILE(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(TARGET_NAME physika_framework)
add_library(${TARGET_NAME} SHARED ${SRC_FILES})

target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_include_directories(${TARGET_NAME} SYSTEM PUBLIC ${PHYSIKA_THIRD_PATH}/entt/)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Engine/framework")
FILE(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
install(FILES ${HEADER_FILES}
        DESTINATION ${PHYSIKA_HEADER_INSTALL_PATH}/framework)
install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_LIB_INSTALL_PATH})
install(DIRECTORY ${PHYSIKA_THIRD_PATH}/entt/
        DESTINATION ${PHYSIKA_THIRD_INSTALL_PATH})