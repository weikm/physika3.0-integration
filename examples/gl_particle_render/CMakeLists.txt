file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h src/*.cu include/*.cuh
)
set(TARGET_NAME gl_particle_render)

add_library(${TARGET_NAME} SHARED ${srcs}) 
target_include_directories(${TARGET_NAME} PUBLIC include ../gl_dependencies/glm tools)
target_link_libraries(${TARGET_NAME} PUBLIC glfw glad)
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Examples/bku_sample")
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
target_compile_definitions(${TARGET_NAME} PUBLIC -DSTYLE_HOME=\"${CMAKE_SOURCE_DIR}/examples\")