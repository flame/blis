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

cmake_minimum_required(VERSION 3.20.0)
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
set(BLIS_PATH "undefined" CACHE STRING "Setting the path to a BLIS installation that needs testing.")
if(BLIS_PATH STREQUAL "undefined")
    message(FATAL_ERROR "Need to provide a BLIS installation path during CMake invocation. Please use \
    $ cmake .. -DBLIS_PATH=/home/username/blis_installation")
endif()

# Set the path to BLIS include directory.
# Adding both paths so that testing works with installation using configure/Make or CMake.
set(BLIS_INCLUDE ${BLIS_PATH}/include/ ${BLIS_PATH}/include/blis CACHE STRING "Setting the path to the BLIS headers.")
set(BLIS_LIB_PATH ${BLIS_PATH}/lib CACHE STRING "Setting the path to the BLIS library.")

# Use REF_BLAS to set the library that will be used for reference results.
set(REF_CBLAS CACHE STRING "Library used to compute reference results.")
# Set the possible values of reference CBLAS for cmake-gui and throw errors for disabled options.
if(LINUX)
    set_property(CACHE REF_CBLAS PROPERTY STRINGS "OpenBLAS" "Netlib" "MKL")
    if(NOT ((REF_CBLAS STREQUAL "OpenBLAS") OR (REF_CBLAS STREQUAL "Netlib") OR(REF_CBLAS STREQUAL "MKL")))
        message(FATAL_ERROR "REF_CBLAS option '${REF_CBLAS}' is not supported. Please, use one of the following options \
                during CMake invokation: OpenBLAS, Netlib, MKL or modify CMakeLists.txt to include this option.")
    endif()
else()
    set_property(CACHE REF_CBLAS PROPERTY STRINGS "OpenBLAS" "MKL")
    if(NOT ((REF_CBLAS STREQUAL "OpenBLAS") OR (REF_CBLAS STREQUAL "MKL")))
        message(FATAL_ERROR "REF_CBLAS option '${REF_CBLAS}' is not supported. Please, use one of the following options \
                during CMake invokation: OpenBLAS, MKL or modify CMakeLists.txt to include this option.")
    endif()
endif()

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

# If MKL is used as a reference set up the threading library options.
if(REF_CBLAS STREQUAL "MKL")
    # MKL threading option is set up as BLIS threading option by default.
    set(MKL_ENABLE_THREADING ${ENABLE_THREADING} CACHE STRING "Setting MKL threading option.")
endif()

# Set up OpenMP flags correctly if it's required.
if( (ENABLE_THREADING STREQUAL "openmp") OR (MKL_ENABLE_THREADING STREQUAL "openmp") )
    find_package(OpenMP)
    if(NOT OPENMP_FOUND)
        message (FATAL_ERROR "Openmp Not Found, please provide an OpenMP library using -DOpenMP_libomp_LIBRARY=path_to_omp_lib.")
    endif()
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
    option(ENABLE_ASAN "Run tests using Address Sanatizer" OFF)
    option(ENABLE_COVERAGE "Run tests for Code Coderage" OFF)
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
set_property(CACHE TEST_INTERFACE PROPERTY STRINGS "BLAS" "CBLAS" "BLIS_TYPED")
if( NOT ((TEST_INTERFACE STREQUAL "BLAS") OR (TEST_INTERFACE STREQUAL "CBLAS") OR (TEST_INTERFACE STREQUAL "BLIS_TYPED")) )
    message(FATAL_ERROR "TEST_INTERFACE option ${TEST_INTERFACE} is not supported. Please use on of the following options \
            during CMake invokation: BLAS, CBLAS, BLIS_TYPED")
endif()

# Use BLIS_ELEMENT_TYPE to set whether the elements of any matrix/vector tested are integers or floating point values.
set(BLIS_ELEMENT_TYPE "f" CACHE STRING "Type of elements of matrices/vectors")
# Set the possible values of element types for cmake-gui
set_property(CACHE BLIS_ELEMENT_TYPE PROPERTY STRINGS "f" "i")
if( NOT ((BLIS_ELEMENT_TYPE STREQUAL "f") OR (BLIS_ELEMENT_TYPE STREQUAL "i")) )
    message(FATAL_ERROR "BLIS_ELEMENT_TYPE option ${BLIS_ELEMENT_TYPE} is not supported. Please use on of the following options \
            during CMake invokation: f, i")
endif()

if(LINUX)
    if(REF_LIB)
        set(REFLIB_PATH ${REF_LIB}/..)
        find_library(reflib NAMES openblas cblas mkl_rt HINTS ${REFLIB_PATH} PATHS ${REFLIB_PATH})
        if(${reflib} STREQUAL reflib-NOTFOUND)
            message(FATAL_ERROR "Reference Library not found : " ${REF_LIB})
        else()
            message(STATUS "Found Reference Library : " ${reflib})
        endif()
    else()
        if(REF_CBLAS STREQUAL "OpenBLAS")
            if(NOT(OPENBLAS_PATH))
                message(FATAL_ERROR "Need to provide an OpenBLAS installation path \
                during CMake invokation when OpenBLAS is used for reference results. Please use \
                $ cmake .. -DOPENBLAS_PATH=/home/username/openblas_installation")
            endif()
            find_library(reflib NAMES openblas HINTS ${OPENBLAS_PATH} PATHS ${OPENBLAS_PATH})
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
                find_library(netlib NAMES cblas HINTS ${NETLIB_PATH} PATHS ${NETLIB_PATH})
            else()
                find_library(netlib NAMES cblas64 HINTS ${NETLIB_PATH} PATHS ${NETLIB_PATH})
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
            find_library(mkllib NAMES mkl_rt HINTS ${MKL_PATH} PATHS ${MKL_PATH})
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
else() #WIN32
    if( NOT ((REF_CBLAS STREQUAL "OpenBLAS") OR (REF_CBLAS STREQUAL "MKL")) )
        message(FATAL_ERROR "REF_CBLAS option ${REF_CBLAS} is not supported. Please use on of the following options \
                during CMake invokation: -DREF_CBLAS=OpenBLAS or -DREF_CBLAS=MKL")
    endif()
    if(REF_CBLAS STREQUAL "OpenBLAS")
        if(NOT(OPENBLAS_PATH))
            message(FATAL_ERROR "Need to provide an OpenBLAS installation path \
                    during CMake invokation when OpenBLAS is used for reference results. Please use \
                    $ cmake .. -DOPENBLAS_PATH=/home/username/openblas_installation")
        endif()
        set(REF_LIB "${OPENBLAS_PATH}/libopenblas.dll" CACHE STRING "Reference OpenBLAS Library")
        message(STATUS "Found OpenBLAS Reference Library : " ${REF_LIB})
    elseif(REF_CBLAS STREQUAL "MKL")
        if(NOT(MKL_PATH))
            message(FATAL_ERROR "Need to provide an MKL_PATH installation path \
                    during CMake invokation when MKL] is used for reference results. Please use \
                    $ cmake .. -DMKL_PATH=/home/username/path_to_mkl_rt")
        endif()
        set(REF_LIB "${MKL_PATH}/mkl_rt.2.dll" CACHE STRING "Reference MKL Library")
        message(STATUS "Found MKL Reference Library : " ${REF_LIB})      
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

find_library(BLIS_LIBRARY NAMES ${LIBBLIS} PATHS ${BLIS_LIB_PATH})
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
       add_compile_options(-fsanitize=address)
       add_definitions(-DENABLE_ASAN)
    endif()

    if(ENABLE_COVERAGE)
        set(CMAKE_CXX_FLAGS "-O0 --coverage")
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
