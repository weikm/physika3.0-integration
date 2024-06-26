# @author     : Zhu Fei
# @date       : 2023-03-23
# @description: Project-level CMakeLists.txt for building Physika
# @version    : 1.0

cmake_minimum_required(VERSION 3.11.4)

option(BUILD_SHARED_LIBS "Build as shared lib" ON)
option(BUILD_WITH_CUDA "Build with CUDA support" ON)
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_WITH_CUDA)
    project(physika LANGUAGES C CXX CUDA)
else()
    project(physika)
endif()

if(BUILD_WITH_CUDA)
    set(CUDA_ARCH "61" CACHE STRING "CUDA Architecture to generate code for (e.g.,60).") # "86;75"
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(BUILD_WITH_CUDA)
endif()

set(PHYSIKA_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PHYSIKA_DATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/data)
set(PHYSIKA_EXAMPLE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/examples)
set(PHYSIKA_TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(PHYSIKA_THIRD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
set(PHYSIKA_INSTALL_PATH ${CMAKE_INSTALL_PREFIX})
set(PHYSIKA_THIRD_INSTALL_PATH ${PHYSIKA_INSTALL_PATH}/thirdparty)
set(PHYSIKA_EXAMPLE_INSTALL_PATH ${PHYSIKA_INSTALL_PATH}/bin/examples)
set(PHYSIKA_TEST_INSTALL_PATH ${PHYSIKA_INSTALL_PATH}/bin/tests)
set(PHYSIKA_HEADER_INSTALL_PATH ${PHYSIKA_INSTALL_PATH}/include)
set(PHYSIKA_LIB_INSTALL_PATH ${PHYSIKA_INSTALL_PATH}/lib)

# Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(HOST_FLAGS "-O3")

    if(CMAKE_BUILD_TYPE STREQUAL Debug)
        set(HOST_FLAGS "-g")
    endif()

    set(HOST_FLAGS "${HOST_FLAGS} -Wall -Wextra -Wfatal-errors -fexceptions -fPIC")
elseif(MSVC)
    # These lines are meaningless because MSVC will override them anyway
    # set(HOST_FLAGS "-O2")
    # if(CMAKE_BUILD_TYPE STREQUAL Debug)
    # set(HOST_FLAGS "-DEBUG")
    # endif()
    set(HOST_FLAGS "${HOST_FLAGS} -W4 -utf-8 -EHa")

    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON) # TODO(ZhuFei): will export APIs only in the future
else()
    message(WARNING "Not tested compiler")
endif()

if(BUILD_WITH_CUDA)
    set(CUDA_FLAGS "${HOST_FLAGS} -forward-unknown-to-host-compiler --extended-lambda --relocatable-device-code=true")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -extended-lambdas")
     set(CMAKE_CUDA_FLAGS    "${CUDA_FLAGS} ${CMAKE_CUDA_FLAGS}")
endif()

set(CMAKE_C_FLAGS "${HOST_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${HOST_FLAGS} ${CMAKE_CXX_FLAGS}")

# build library
add_subdirectory(${PHYSIKA_SRC_ROOT})

# build example
add_subdirectory(${PHYSIKA_EXAMPLE_ROOT})

if(BUILD_TESTS)
    # build test
    enable_testing()
    add_subdirectory(${PHYSIKA_TEST_ROOT})
endif()
