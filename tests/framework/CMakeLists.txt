#@author     : Zhu Fei
#@date       : 2023-04-07
#@description: CMakeLists.txt for building Physika framework tests
#@version    : 1.0

cmake_minimum_required(VERSION 3.11.4)

project(physika_framework_tests)

set(TARGET_NAME framework_test)
add_executable(${TARGET_NAME} object_test.cpp scene_test.cpp world_test.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "Tests")

target_include_directories(${TARGET_NAME} PRIVATE ${PHYSIKA_SRC_ROOT})
target_link_libraries(${TARGET_NAME} PRIVATE physika_framework)

target_link_libraries(${TARGET_NAME} PRIVATE GTest::gtest_main gmock)

if(BUILD_SHARED_LIBS)
    set_property(TARGET   ${TARGET_NAME}
                 PROPERTY INSTALL_RPATH 
                 $ORIGIN/../../lib
                 $ORIGIN/../../thirdparty/googletest)
endif()

#copy dependency dlls to target bin dir
if(MSVC)
    add_custom_command(TARGET ${TARGET_NAME}
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different    $<TARGET_FILE:physika_framework>
                                                                        $<TARGET_FILE:GTest::gtest_main>
                                                                        $<TARGET_FILE:GTest::gmock_main>
                                                                        $<TARGET_FILE:gmock>
                                                                        $<TARGET_FILE:gtest>
                                                                        $<TARGET_FILE_DIR:${TARGET_NAME}>
                       COMMAND_EXPAND_LISTS)
    install(FILES $<TARGET_FILE:physika_framework>
                  $<TARGET_FILE:GTest::gtest_main>
                  $<TARGET_FILE:GTest::gmock_main>
                  $<TARGET_FILE:gmock>
                  $<TARGET_FILE:gtest>
            DESTINATION ${PHYSIKA_TEST_INSTALL_PATH})
endif()

include(GoogleTest)
gtest_discover_tests(${TARGET_NAME})

install(TARGETS ${TARGET_NAME}
        DESTINATION ${PHYSIKA_TEST_INSTALL_PATH})