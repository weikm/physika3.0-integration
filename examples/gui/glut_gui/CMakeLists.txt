cmake_minimum_required(VERSION 3.10)

#project(physika_sand LANGUAGES C CXX CUDA)
project(physika_glut_gui)
find_package(CUDAToolkit REQUIRED)
#find_package(THRUST REQUIRED)

FILE(GLOB SRC_FILES
	LIST_DIRECTORIES false
	CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.c*"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h*")

message(${SRC_FILES})
set(TARGET_NAME physika_glut_gui)
#set(LIB_NAME physika_core)
add_library(${TARGET_NAME} ${SRC_FILES})

if(WIN32)
    foreach(SRC IN ITEMS ${SRC_FILES})
        get_filename_component(SRC_PATH "${SRC}" PATH)
        file(RELATIVE_PATH SRC_PATH_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SRC_PATH}")
        string(REPLACE "/" "\\" GROUP_PATH "${SRC_PATH_REL}")
        source_group("${GROUP_PATH}" FILES "${SRC}")
    endforeach()
endif()


file(RELATIVE_PATH PROJECT_PATH_REL "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Engine/GUI")
#set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX) 


target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_EXAMPLE_ROOT})
target_include_directories(${TARGET_NAME} PUBLIC ${PHYSIKA_THIRD_PATH}/glm/)
target_include_directories(${TARGET_NAME} PUBLIC ${PHYSIKA_THIRD_PATH}/freeglut-3.2.1/include)
target_include_directories(${TARGET_NAME} PUBLIC ${PHYSIKA_THIRD_PATH}/glew-1.10.0/include)

target_link_libraries(${TARGET_NAME} PRIVATE physika_framework glew freeglut physika_rendering) 
#link_directories(${CUDA_LIBRARY_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart)

#target_link_libraries(${TARGET_NAME} PUBLIC CUDA::cuda_driver)

FILE(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
install(FILES ${HEADER_FILES}
		DESTINATION ${PHYSIKA_HEADER_INSTALL_PATH}/glut_gui)
install(TARGETS ${TARGET_NAME}
		DESTINATION ${PHYSIKA_LIB_INSTALL_PATH})