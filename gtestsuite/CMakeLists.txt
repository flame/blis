#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

]=]

cmake_minimum_required(VERSION 3.22.0)
set(CMAKE_CXX_COMPILER ${CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)

project(BLIS_GtestSuite)

enable_testing()

# Set variable if the platform is Linux based.
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Throw an error if the platform is Apple.
if(APPLE)
    message(FATAL_ERROR "Build system does not support Apple platform.")
endif()

# Set the path to the BLIS installation.
set(BLIS_PATH $ENV{AOCL_BLAS_PATH} CACHE STRING "Setting the path to a BLIS installation that needs testing.")
if(BLIS_PATH STREQUAL "")
    message(FATAL_ERROR "Need to provide a BLIS installation path during CMake invocation.\
    Set environment variable \$AOCL_BLAS_PATH or set the cmake variable directly using\
    $ cmake .. -DBLIS_PATH=/home/username/blis_installation")
endif()

# Set the path to BLIS include directory.
# Adding both paths so that testing works with installation using configure/Make or CMake.
set(BLIS_INCLUDE ${BLIS_PATH}/include/ ${BLIS_PATH}/include/blis CACHE STRING "Setting the path to the BLIS headers.")
set(BLIS_LIB_PATH ${BLIS_PATH}/lib CACHE STRING "Setting the path to the BLIS library.")

# Use REF_BLAS to set the library that will be used for reference results.
set(REF_CBLAS "Netlib" CACHE STRING "Library used to compute reference results.")
# Use REF_LIB to set the library that will be used for reference results.
set(REF_LIB $ENV{CBLAS_REF_LIB} CACHE STRING "Path to a shared library that will be used as a reference.")

# Set OpenMP as the default option
set(ENABLE_THREADING "openmp" CACHE STRING "the threading flag")
# Set the possible values of theading libraries for cmake-gui
if(WIN32)
    set_property(CACHE ENABLE_THREADING PROPERTY STRINGS "openmp" "no")
    if( NOT ((ENABLE_THREADING STREQUAL "openmp") OR (ENABLE_THREADING STREQUAL "no")) )
        message(FATAL_ERROR "ENABLE_THREADING option '${ENABLE_THREADING}' is not supported. Please use one of the following options \
                during CMake invokation: openmp, no")
    endif()
else()
    set_property(CACHE ENABLE_THREADING PROPERTY STRINGS "openmp" "pthreads" "no")
    if( NOT ((ENABLE_THREADING STREQUAL "openmp") OR (ENABLE_THREADING STREQUAL "pthreads") OR (ENABLE_THREADING STREQUAL "no")) )
        message(FATAL_ERROR "ENABLE_THREADING option '${ENABLE_THREADING}' is not supported. Please use one of the following options \
                during CMake invokation: openmp, pthreads, no")
    endif()
endif()

# Setting path to OpenMP runtime.
if(WIN32)
   set(OpenMP_libomp_LIBRARY "C:/Program Files/LLVM/lib/libomp.lib" CACHE STRING "openmp library path")
endif()

# Set up OpenMP flags correctly if it's required.
if( (ENABLE_THREADING STREQUAL "openmp") OR (MKL_ENABLE_THREADING STREQUAL "openmp") )
    find_package(OpenMP)
    if(NOT OPENMP_FOUND)
        message (FATAL_ERROR "Openmp Not Found, please provide an OpenMP library using -DOpenMP_libomp_LIBRARY=path_to_omp_lib.")
    endif()
endif()

# If MKL is used as a reference set up the threading library options.
if(REF_CBLAS STREQUAL "MKL")
    # MKL threading option is set up as BLIS threading option by default.
    set(MKL_ENABLE_THREADING ${ENABLE_THREADING} CACHE STRING "Setting MKL threading option.")
endif()

# Set static BLIS as the default library we build against.
set(BLIS_LINKING_TYPE "static" CACHE STRING "Type of BLIS library (shared or static) that is being tested.")
# Set the possible values of BLIS linking type for cmake-gui
set_property(CACHE BLIS_LINKING_TYPE PROPERTY STRINGS "static" "shared")
if( NOT ((BLIS_LINKING_TYPE STREQUAL "static") OR (BLIS_LINKING_TYPE STREQUAL "shared")) )
    message(FATAL_ERROR "BLIS_LINKING_TYPE option '${BLIS_LINKING_TYPE}' is not supported. Please use one of the following options \
            during CMake invokation: static, shared")
endif()

# Set common libraries.
if(LINUX)
    set(COMMON_LIBS pthread m dl)
    option(ENABLE_ASAN "Run tests using Address Sanitizer" OFF)
    option(ENABLE_COVERAGE "Run tests for Code Coverage" OFF)
endif()

# Use INT_SIZE to set the int type used for testing.
set(INT_SIZE "32" CACHE STRING "Integer size used in testing suite. Must match the integer size of BLIS.")
# Set the possible values of reference CBLAS for cmake-gui
set_property(CACHE INT_SIZE PROPERTY STRINGS "32" "64")
if( NOT ((INT_SIZE STREQUAL "32") OR (INT_SIZE STREQUAL "64")) )
    message(FATAL_ERROR "INT_SIZE option '${INT_SIZE}' is not supported. Please use one of the following options \
            during CMake invokation: 32, 64")
endif()

# Use TEST_INTERFACE to set which interface, supported by BLIS is meant to be tested.
set(TEST_INTERFACE "BLAS" CACHE STRING "Interface of BLIS that is being tested.")
# Set the possible values of interfaces for cmake-gui
set_property(CACHE TEST_INTERFACE PROPERTY STRINGS "BLAS" "BLAS_BLIS_IMPL" "CBLAS" "BLIS_TYPED")
if( NOT ((TEST_INTERFACE STREQUAL "BLAS") OR (TEST_INTERFACE STREQUAL "BLAS_BLIS_IMPL") OR (TEST_INTERFACE STREQUAL "CBLAS") OR (TEST_INTERFACE STREQUAL "BLIS_TYPED")) )
    message(FATAL_ERROR "TEST_INTERFACE option ${TEST_INTERFACE} is not supported. Please use on of the following options \
            during CMake invokation: BLAS, BLAS_BLIS_IMPL, CBLAS, BLIS_TYPED")
endif()

# Use BLIS_ELEMENT_TYPE to set whether the elements of any matrix/vector tested are integers or floating point values.
set(BLIS_ELEMENT_TYPE "f" CACHE STRING "Type of elements of matrices/vectors")
# Set the possible values of element types for cmake-gui
set_property(CACHE BLIS_ELEMENT_TYPE PROPERTY STRINGS "f" "i")
if( NOT ((BLIS_ELEMENT_TYPE STREQUAL "f") OR (BLIS_ELEMENT_TYPE STREQUAL "i")) )
    message(FATAL_ERROR "BLIS_ELEMENT_TYPE option ${BLIS_ELEMENT_TYPE} is not supported. Please use on of the following options \
            during CMake invokation: f, i")
endif()

# Option to enable testing with upper case character arguments in BLAS and BLIS calls.
option(TEST_UPPERCASE_ARGS "Test upper case character arguments" OFF)

# Option to enable testing with thresholds set to zero.
option(THRESHOLD_ZERO "Set thresholds to zero" OFF)

# Can we test the value of info stored within BLIS and returned by a call to
# bli_info_get_info_value (introduced at AMD BLAS 4.2).
option(CAN_TEST_INFO_VALUE "Can test value of info" ON)

# Use EXT_VAL to get the extreme value (NaN or Inf) used for testing data that shouldn't be read.
set(EXT_VAL "Inf" CACHE STRING "Extreme value (NaN or Inf) used for testing data that shouldn't be read")
# Set the possible values of reference CBLAS for cmake-gui
set_property(CACHE EXT_VAL PROPERTY STRINGS "NaN" "Inf")
if( NOT ((EXT_VAL STREQUAL "NaN") OR (EXT_VAL STREQUAL "Inf")) )
    message(FATAL_ERROR "EXT_VAL option '${EXT_VAL}' is not supported. Please use one of the following options \
            during CMake invokation: NaN, Inf")
endif()

# Option to enable testing of input arguments to BLAS APIs.
# Note: This imposes a significant runtime overhead.
option(TEST_INPUT_ARGS "Test input arguments" OFF)

if(REF_LIB)
    get_filename_component(REFLIB_PATH ${REF_LIB}/.. ABSOLUTE)
    get_filename_component(library ${REF_LIB} NAME)
    find_library(reflib NAMES ${library} PATHS ${REFLIB_PATH} NO_DEFAULT_PATH)
    if(${reflib} STREQUAL reflib-NOTFOUND)
        message(FATAL_ERROR "Reference Library not found : " ${REF_LIB})
    else()
        message(STATUS "Found Reference Library : " ${reflib})
    endif()
else()
    # Set the possible values of theading libraries for cmake-gui
    set_property(CACHE REF_CBLAS PROPERTY STRINGS "OpenBLAS" "Netlib" "MKL")
    if(NOT ((REF_CBLAS STREQUAL "OpenBLAS") OR (REF_CBLAS STREQUAL "Netlib") OR(REF_CBLAS STREQUAL "MKL")))
        message(FATAL_ERROR "REF_CBLAS option '${REF_CBLAS}' is not supported. Please, use one of the following options \
        during CMake invokation: OpenBLAS, Netlib, MKL or modify CMakeLists.txt to include this option.")
    endif()

    if(LINUX)
        set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
        set(LIBOpenBLAS openblas)
        set(LIBCLAS cblas)
        set(LIBCLAS64 cblas64)
        set(LIBMKL mkl_rt)
    else()
        set(CMAKE_FIND_LIBRARY_PREFIXES "")
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")
        set(LIBOpenBLAS libopenblas)
        set(LIBMKL mkl_rt.2)
    endif()

    if(REF_CBLAS STREQUAL "OpenBLAS")
        if(NOT(OPENBLAS_PATH))
            message(FATAL_ERROR "Need to provide an OpenBLAS installation path \
            during CMake invokation when OpenBLAS is used for reference results. Please use \
            $ cmake .. -DOPENBLAS_PATH=/home/username/openblas_installation")
        endif()
        find_library(reflib NAMES ${LIBOpenBLAS} PATHS ${OPENBLAS_PATH} NO_DEFAULT_PATH)
        if(${reflib} STREQUAL reflib-NOTFOUND)
            message(FATAL_ERROR "OpenBLAS Reference Library not found : " ${OPENBLAS_PATH})
        else()
            message(STATUS "Found OpenBLAS Reference Library : " ${reflib})
        endif()
        set(REF_LIB ${reflib})
    elseif(REF_CBLAS STREQUAL "Netlib")
        if(NOT(NETLIB_PATH))
            message(FATAL_ERROR "Need to provide a Netlib installation path \
            during CMake invokation when Netlib is used for reference results. Please use \
            $ cmake .. -DNETLIB_PATH=/home/username/netlib_installation")
        endif()
        if(INT_SIZE STREQUAL "32")
            find_library(netlib NAMES ${LIBCLAS} PATHS ${NETLIB_PATH} NO_DEFAULT_PATH)
        else()
            find_library(netlib NAMES ${LIBCLAS64} PATHS ${NETLIB_PATH} NO_DEFAULT_PATH)
        endif()
        if(${netlib} STREQUAL netlib-NOTFOUND)
            message(FATAL_ERROR "Netlib Reference Library not found : "  ${NETLIB_PATH})
        else()
            message(STATUS "Found Netlib Reference Library : "  ${netlib})
        endif()
        set(REF_LIB ${netlib})
    elseif(REF_CBLAS STREQUAL "MKL")
        set(MKL_PATH $ENV{MKLROOT}/lib/intel64 CACHE STRING "The path to MKL.")
        find_library(mkllib NAMES ${LIBMKL} PATHS ${MKL_PATH} NO_DEFAULT_PATH)
        if(${mkllib} STREQUAL mkllib-NOTFOUND)
            message(FATAL_ERROR "MKL Reference Library not found : " ${MKL_PATH})
        else()
            message(STATUS "Found MKL Reference Library  : " ${mkllib})
        endif()
        set(REF_LIB ${mkllib})
    else()
        message(FATAL_ERROR "Need to set up a reference library. Please use on of the following options \
        during CMake invokation: -DREF_CBLAS=Netlib or -DREF_CBLAS=OpenBLAS or -DREF_CBLAS=MKL")
    endif()
endif()

# Set up the library name.
if(WIN32)
    set(LIBBLIS AOCL-LibBlis-Win)
else()
    set(LIBBLIS blis)
endif()
# Append if threading is required.
if(NOT (ENABLE_THREADING STREQUAL "no"))
    if(WIN32)
        string(APPEND LIBBLIS -MT)
    else()
        string(APPEND LIBBLIS -mt)
    endif()
endif()
# Append for dll if necessary.
if(WIN32 AND (BLIS_LINKING_TYPE STREQUAL "shared"))
    string(APPEND LIBBLIS -dll)
endif()
# Setting the suffix for find_library().
if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib)
else()
    if(BLIS_LINKING_TYPE STREQUAL "shared")
        set(CMAKE_FIND_LIBRARY_SUFFIXES .so)
    else()
        set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif()
endif()

find_library(BLIS_LIBRARY NAMES ${LIBBLIS} PATHS ${BLIS_LIB_PATH} NO_DEFAULT_PATH)
if(${BLIS_LIBRARY} STREQUAL BLIS_LIBRARY-NOTFOUND)
    message(FATAL_ERROR "Blis Library ${LIBBLIS} not found in BLIS_LIB_PATH=${BLIS_LIB_PATH}")
else()
    message(STATUS "Found ${LIBBLIS} BLIS Library : " ${BLIS_LIBRARY})
endif()

# Set compiler options and BLIS library for Linux.
if(LINUX)
    # Add compiler definition.
    add_compile_options(-g -Wall -Wno-unused-function -Wfatal-errors -fPIC )

    if(ENABLE_ASAN)
        set(ASAN_FLAGS "-fsanitize=address")
        list(APPEND CMAKE_C_FLAGS ${ASAN_FLAGS})
    endif()

    if(ENABLE_COVERAGE)
        set(COVERAGE_FLAGS "-O0 --coverage")
        list(APPEND CMAKE_C_FLAGS ${COVERAGE_FLAGS})
    endif()
endif()

#Setting up the correct Windows Runtime Library.
if(WIN32)
    cmake_policy(SET CMP0091 NEW)
    if(BUILD_SHARED_LIBS)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# The following part will be used to set up a list of defines that dictate
# which kernel tests can be build and run on the current architecture.
# Given that the symbols of kernel functions are not exported for shared libraries
# we only set up those defines for static libs.
# This way, kernel tests won't be compiled/run for shared versions of BLIS.
if(BLIS_LINKING_TYPE STREQUAL "static")
    if(ENABLE_THREADING STREQUAL "openmp")
        try_run(RUNRESULT COMPILERESULT "${CMAKE_BINARY_DIR}/temp" SOURCES ${CMAKE_SOURCE_DIR}/cmake/config_ukr_tests.cpp
            COMPILE_DEFINITIONS -I${BLIS_PATH}/include/ -I${BLIS_PATH}/include/blis
            LINK_LIBRARIES ${BLIS_LIBRARY} ${COMMON_LIBS} OpenMP::OpenMP_CXX ${ASAN_FLAGS} ${COVERAGE_FLAGS}
            RUN_OUTPUT_VARIABLE UKR_CONFIG
            COMPILE_OUTPUT_VARIABLE COMP_VAR
        )
    else()
        try_run(RUNRESULT COMPILERESULT "${CMAKE_BINARY_DIR}/temp" SOURCES ${CMAKE_SOURCE_DIR}/cmake/config_ukr_tests.cpp
            COMPILE_DEFINITIONS -I${BLIS_PATH}/include/ -I${BLIS_PATH}/include/blis
            LINK_LIBRARIES ${BLIS_LIBRARY} ${COMMON_LIBS} ${ASAN_FLAGS} ${COVERAGE_FLAGS}
            RUN_OUTPUT_VARIABLE UKR_CONFIG
            COMPILE_OUTPUT_VARIABLE COMP_VAR
        )
    endif()
    # Uncomment this to debug this snippet above, if necessary.
    if(NOT COMPILERESULT)
        message(FATAL_ERROR "Compiling config_ukr_tests.cpp failed with the following error ${COMP_VAR}.")
    endif()
    # Remove all empty items from the list.
    list(REMOVE_ITEM UKR_CONFIG "")
    # We iterate through the list returned from the snippet above.
    # For example, UKR_CONFIG = AVX2FMA3 for zen3
    # or UKR_CONFIG = AVX2FMA3;AVX512;AVX512VNNI;AVX512BF16 for zen4
    # Depending on the values of this list we define corresponding macros
    # -DGTEST_AVX2FMA3 on zen3
    # or -DGTEST_AVX2FMA3;-DGTEST_AVX512;-DGTEST_AVX512VNNI;-DGTEST_AVX512BF16 on zen4
    # Those macros are passed when compiling the tests in testsuite/CMakeLists.txt.
    foreach(ukrconf ${UKR_CONFIG})
        list(APPEND UKR_DEFINES "-DGTEST_${ukrconf}")
    endforeach()
    message(STATUS "Since BLIS GTestSuite is used to check the static version of blis, all kernel tests are enabled.")
else()
    message(WARNING "Since BLIS GTestSuite is used to check the shared version of blis, all kernel tests are disabled.")
endif()

add_subdirectory(testinghelpers)
add_subdirectory(testsuite)

add_custom_target(distclean
    COMMAND ${CMAKE_MAKE_PROGRAM} clean
    COMMAND rm ${CMAKE_BINARY_DIR}/*.txt
    COMMAND rm ${CMAKE_BINARY_DIR}/*.cmake
    COMMAND rm ${CMAKE_BINARY_DIR}/Makefile
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/bin
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Remove cmake_generated files and executables"
)
