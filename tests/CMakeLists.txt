#@author     : Zhu Fei
#@date       : 2023-04-07
#@description: CMakeLists.txt for building Physika tests
#@version    : 1.0

cmake_minimum_required(VERSION 3.11.4)

project(physika_tests LANGUAGES C CXX)

include(FetchContent)

FetchContent_Declare(
        googletest
        # GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_REPOSITORY https://gitee.com/PhysikaTeam/googletest.git
        GIT_TAG main
        GIT_SHALLOW 1
        EXCLUDE_FROM_ALL
    )
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF) # disable googletest install, replace it with install scripts in our project
# # so that googletest does not get installed with our project
# option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)

# fix gtest unresolved external symbol issue:
# https://github.com/google/googletest/issues/2535
if(MSVC)
    add_compile_definitions(GTEST_LINKED_AS_SHARED_LIBRARY)
endif()

add_subdirectory(${PHYSIKA_TEST_ROOT}/framework)

install(FILES $<TARGET_FILE:GTest::gtest_main>
              $<TARGET_FILE:GTest::gmock_main>
              $<TARGET_FILE:gmock>
              $<TARGET_FILE:gtest>
              DESTINATION ${PHYSIKA_THIRD_INSTALL_PATH}/googletest)
