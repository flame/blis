#[=[
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

cmake_minimum_required(VERSION 3.14.0)
set(CMAKE_CXX_COMPILER ${CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)

project(BLIS_GtestSuite)

enable_testing()

# Set the path to the BLIS installation.
if(NOT(BLIS_PATH))
    message(FATAL_ERROR "Need to provide a BLIS installation path during CMake invocation. Please use \
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

option(ENABLE_ASAN "Run tests using Address Sanatizer" OFF)

option(ENABLE_COVERAGE "Run tests for Code Coderage" OFF)

# Set variable if the platform is Linux based.
if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Throw an error if the platform is Apple.
if(APPLE)
    message(FATAL_ERROR "Build system does not support Apple platform.")
endif()

# Use INT_SIZE to set the int type used for testing.
set(INT_SIZE "32" CACHE STRING "Library used to compute reference results.")
# Set the possible values of reference CBLAS for cmake-gui
set_property(CACHE INT_SIZE PROPERTY STRINGS "32" "64")
if( NOT ((INT_SIZE STREQUAL "32") OR (INT_SIZE STREQUAL "64")) )
    message(FATAL_ERROR "INT_SIZE option ${INT_SIZE} is not supported. Must be 32 or 64.")
endif()

# Use REF_BLAS to set the library that will be used for reference results.
set(REF_CBLAS CACHE STRING "Library used to compute reference results.")
# Set the possible values of reference CBLAS for cmake-gui
set_property(CACHE REF_CBLAS PROPERTY STRINGS "OpenBLAS" "Netlib" "MKL")


if(REF_LIB)
    set(REFLIB_PATH ${REF_LIB}/..)
    find_library(reflib NAMES openblas cblas mkl_intel_lp64 mkl_intel_ilp64 PATHS ${REFLIB_PATH})
    if(${reflib} STREQUAL reflib-NOTFOUND)
        message(FATAL_ERROR "Reference Library not found : " ${REF_LIB})
    else()
        message(STATUS "Found Reference Library : " ${reflib})
    endif()
    message( "Setting REF_LIB to ${REF_LIB}")
else()
    if(REF_CBLAS STREQUAL "OpenBLAS")
        if(NOT(OPENBLAS_PATH))
            message(FATAL_ERROR "Need to provide an OpenBLAS installation path \
            during CMake invokation when OpenBLAS is used for reference results. Please use \
            $ cmake .. -DOPENBLAS_PATH=/home/username/openblas_installation")
        endif()
        find_library(reflib NAMES openblas PATHS ${OPENBLAS_PATH})
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
            find_library(netlib NAMES cblas PATHS ${NETLIB_PATH})
        else()
            find_library(netlib NAMES cblas64 PATHS ${NETLIB_PATH})
        endif()
        if(${netlib} STREQUAL netlib-NOTFOUND)
            message(FATAL_ERROR "Netlib Reference Library not found : "  ${NETLIB_PATH})
        else()
            message(STATUS "Found Netlib Reference Library : "  ${netlib})
        endif()
        set(REF_LIB ${netlib})
    elseif(REF_CBLAS STREQUAL "MKL")
        set(MKL_PATH $ENV{MKLROOT}/lib/intel64
                CACHE STRING "The path to MKL.")
        if(INT_SIZE STREQUAL "32")
            find_library(mkllib NAMES mkl_intel_lp64 PATHS ${MKL_PATH})
        else()
            find_library(mkllib NAMES mkl_intel_ilp64 PATHS ${MKL_PATH})
        endif()
        if(${mkllib} STREQUAL mkllib-NOTFOUND)
            message(FATAL_ERROR "MKL Reference Library not found : " ${MKL_PATH})
        else()
            message(STATUS "Found MKL Reference Library  : " ${mkllib})
        endif()
        set(REF_LIB ${mkllib})
        find_library(mklcore NAMES mkl_core PATHS ${MKL_PATH})
        if(${mklcore} STREQUAL mklcore-NOTFOUND)
            message(FATAL_ERROR "MKL_CORE Library not found : " ${MKL_PATH})
        else()
            message(STATUS "Found MKL_CORE Library       : " ${mklcore})
        endif()
        set(MKL_CORE_PATH ${mklcore})
        find_library(mklthread NAMES mkl_gnu_thread PATHS ${MKL_PATH})
        if(${mklthread} STREQUAL mklthread-NOTFOUND)
            message(FATAL_ERROR "MKL_GNU_THREAD Library not found : " ${MKL_PATH})
        else()
            message(STATUS "Found MKL_GNU_THREAD Library : " ${mklthread})
        endif()
        set(MKL_GNU_THREAD_PATH ${mklthread})
    else()
        message(FATAL_ERROR "Need to set up a reference library. Please use on of the following options \
                during CMake invokation: -DREF_CBLAS=Netlib or -DREF_CBLAS=OpenBLAS or -DREF_CBLAS=MKL")
    endif()
endif()

# Use TEST_INTERFACE to set which interface, supported by BLIS is meant to be tested.
set(TEST_INTERFACE "BLAS" CACHE STRING "Interface that is being tested.")
# Set the possible values of interfaces for cmake-gui
set_property(CACHE TEST_INTERFACE PROPERTY STRINGS "BLAS" "CBLAS" "BLIS_TYPED")
if( NOT ((TEST_INTERFACE STREQUAL "BLAS") OR (TEST_INTERFACE STREQUAL "CBLAS") OR (TEST_INTERFACE STREQUAL "BLIS_TYPED")) )
    message(FATAL_ERROR "TEST_INTERFACE option ${TEST_INTERFACE} is not supported. Please use on of the following options \
            during CMake invokation: -DTEST_INTERFACE=BLAS or -DTEST_INTERFACE=CBLAS or -DTEST_INTERFACE=BLIS_TYPED")
endif()

# Use ELEMENT_TYPE to set whether the elements of any matrix/vector tested are integers or floating point values.
set(ELEMENT_TYPE "f" CACHE STRING "Type of elements of matrices/vectors")
# Set the possible values of element types for cmake-gui
set_property(CACHE ELEMENT_TYPE PROPERTY STRINGS "f" "i")
if( NOT ((ELEMENT_TYPE STREQUAL "f") OR (ELEMENT_TYPE STREQUAL "i")) )
    message(FATAL_ERROR "ELEMENT_TYPE option ${ELEMENT_TYPE} is not supported. Please use on of the following options \
            during CMake invokation: -DELEMENT_TYPE=f or -DELEMENT_TYPE=i")
endif()

# Set common libraries.
set(COMMON_LIBS pthread m dl)

# Set compiler options and BLIS library for Linux.
if(LINUX)
    # Add compiler definition.
    add_compile_options(-g -Wall -Wno-unused-function -Wfatal-errors -fPIC )

    if(ENABLE_ASAN)
       add_compile_options(-fsanitize=address)
       add_definitions(-DENABLE_ASAN)
    endif()

    if(ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "-O0 --coverage")
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
        find_library(libblis NAMES blis PATHS ${BLIS_PATH}/lib)
    else()
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis-mt.a" CACHE STRING "blis library path")
        else()
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis-mt.so" CACHE STRING "blis library path")
        endif()
        find_library(libblis NAMES blis-mt PATHS ${BLIS_PATH}/lib)
    endif()
    if(${libblis} STREQUAL libblis-NOTFOUND)
        message(FATAL_ERROR "Blis Library not found : " ${BLIS_PATH})
    else()
        message(STATUS "Found BLIS Library : " ${Blis_LIBRARY})
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
