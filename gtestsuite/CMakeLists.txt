##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_CXX_COMPILER ${CXX_COMPILER})

project(Blis_GtestSuite)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

enable_testing()

# Set the path to the BLIS installation.
if(NOT(BLIS_PATH))
    message(FATAL_ERROR "Need to provide a BLIS installation path during CMake invocation. Please use
    $ cmake .. -DBLIS_PATH=/home/username/blis_installation")
endif()
# Set the path to BLIS include directory.
set(BLIS_INCLUDE ${BLIS_PATH}/include/blis)

# Set OpenMP as the default option
set(ENABLE_THREADING "openmp" CACHE STRING "Setting OpenMP as the threading library")
# Set the possible values of theading libraries for cmake-gui
set_property(CACHE ENABLE_THREADING PROPERTY STRINGS "openmp" "pthreads" "no")

# Set static BLIS as the default library we build against.
set(BLIS_LINKING_TYPE "static" CACHE STRING "Linking to a static BLIS library")
# Set the possible values of BLIS linking type for cmake-gui
set_property(CACHE BLIS_LINKING_TYPE PROPERTY STRINGS "static" "shared")

if(BLIS_LINKING_TYPE STREQUAL "shared")
    message(FATAL_ERROR "Using shared BLIS library is currently disabled.")
endif()

option(ENABLE_VALGRIND "Run tests using valgrind" OFF)

option(ENABLE_ASAN "Run tests using Address Sanatizer" OFF)

# Set variable if the platform is Linux based.
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Throw an error if the platform is Apple.
if(APPLE)
    message(FATAL_ERROR "Build system does not support Apple platform.")
endif()

# Set the include paths.
set(INC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/inc)

# Set compiler options and BLIS library for Linux.
if(LINUX)
    # Add compiler definition.

    add_compile_options(-g -Wall -Wno-unused-function -Wfatal-errors -fPIC)

    if(ENABLE_ASAN)
       add_definitions(-D__GTEST_VALGRIND_TEST__)
       add_compile_options(-fsanitize=address -static-libsan)
    endif()

    # Set GNU OpenMP library as the default option
    set(OpenMP_LIBRARY "GNU" CACHE STRING "Using GNU OpenMP library")
    # Set the possibe values of OpenMP runtimes
    set_property(CACHE OpenMP_LIBRARY PROPERTY STRINGS "GNU" "Intel")

    if(ENABLE_THREADING STREQUAL "no")
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis.a" CACHE STRING "blis library path")
        else()
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis.so" CACHE STRING "blis library path")
        endif()
    else()
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis-mt.a" CACHE STRING "blis library path")
        else()
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis-mt.so" CACHE STRING "blis library path")
        endif()
    endif()
endif()

# Set BLIS library for Windows.
if(WIN32)
    if(ENABLE_THREADING STREQUAL "no")
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_PATH}/bin/AOCL-LibBlis-Win.a" CACHE STRING "blis library path")
        else()
            set(Blis_LIBRARY "${BLIS_PATH}/bin/AOCL-LibBlis-Win-dll.lib" CACHE STRING "blis library path")
        endif()
    else()
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_PATH}/bin/AOCL-LibBlis-Win-MT.a" CACHE STRING "blis library path")
        else()
            set(Blis_LIBRARY "${BLIS_PATH}/bin/AOCL-LibBlis-Win-MT-dll.lib" CACHE STRING "blis library path")
        endif()
    endif()
endif()

# Since this is an out-of-source build we need to copy the input files in the correct destination.
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/input.general
          ${CMAKE_CURRENT_SOURCE_DIR}/alphabeta.dat
          ${CMAKE_CURRENT_SOURCE_DIR}/input.operations
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(gtest_libblis
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gtest_suite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gtest_pthread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blis_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blis_inpfile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blis_utils_int.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blis_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_process.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_addv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_amaxv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_axpbyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_axpyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_copyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_dotv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_dotxv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_normfv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_scalv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_scal2v.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_subv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_xpbyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_addm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_axpym.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_copym.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_normfm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_scalm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_scal2m.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_subm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_xpbym.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_axpy2v.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_dotaxpyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_axpyf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_dotxf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_dotxaxpyf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_gemv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_ger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_hemv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_her.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_her2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_symv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_syr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_syr2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_trmv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_trsv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_gemmt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_hemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_herk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_her2k.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_symm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_syrk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_syr2k.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_trmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_trmm3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ref_trsm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_randv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_randm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_addv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_amaxv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_axpbyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_axpyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_copyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_dotv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_dotxv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_normfv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_scalv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_scal2v.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_setv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_subv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_xpbyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_addm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_axpym.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_copym.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_normfm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_scalm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_scal2m.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_setm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_subm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_xpbym.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_axpy2v.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_dotaxpyv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_axpyf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_dotxf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_dotxaxpyf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_ger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_hemv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_her.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_her2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_symv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_syr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_syr2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_trmv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_trsv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemmt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_hemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_herk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_her2k.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_symm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_syrk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_syr2k.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_trmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_trmm3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_trsm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lpgemm_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm_u8s8s32os32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm_u8s8s32os8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm_f32f32f32of32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm_u8s8s16os16.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm_u8s8s16os8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm_bf16bf16f32of32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test_gemm_bf16bf16f32obf16.cpp
)

target_include_directories(gtest_libblis PUBLIC ${INC_PATH} ${BLIS_INCLUDE})
target_link_libraries(gtest_libblis gtest gtest_main ${Blis_LIBRARY} pthread)

# Linking appropriate threading library.
if(ENABLE_THREADING STREQUAL "openmp")
    if(LINUX)
        if(OpenMP_LIBRARY STREQUAL "GNU")
            target_link_libraries(gtest_libblis -fopenmp)
        else()
            target_link_libraries(gtest_libblis iomp5)
        endif()
    endif()
endif()

add_test(
    NAME gtest_libblis
    COMMAND gtest_libblis
)

#gtest_discover_tests(gtest_libblis)