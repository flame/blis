# Standalone: cmake -S <installed-or-source testsuite dir> -B out
#   -DCMAKE_PREFIX_PATH=<blis prefix>, then `cmake --build` and run the
#   binary from the (source) testsuite directory where input.general and
#   input.operations live.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.20...4.4)
    project(blis-testsuite C)
    find_package(blis REQUIRED)
    set(_blis_lib blis::blis)
    set(BLIS_TESTSUITE_STANDALONE ON)
else()
    set(_blis_lib ${BLIS_LINK_TARGET})
endif()

file(GLOB _ts_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
add_executable(test_libblis.x ${_ts_srcs})
target_include_directories(test_libblis.x PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(test_libblis.x PRIVATE ${_blis_lib})

if(BLIS_TESTSUITE_STANDALONE)
    if(NOT MSVC AND NOT BLIS_IS_MSVC)
        target_link_libraries(test_libblis.x PRIVATE m)
    endif()
else()
    blis_in_tree_flags(test_libblis.x)
endif()
