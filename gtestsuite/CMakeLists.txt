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
if(LINUX)
    if(NOT(BLIS_PATH))
        message(FATAL_ERROR "Need to provide a BLIS installation path during CMake invocation. Please use \
        $ cmake .. -DBLIS_PATH=/home/username/blis_installation")
    endif()
    # Set the path to BLIS include directory.
    set(BLIS_INCLUDE ${BLIS_PATH}/include/blis)
else()
    if(NOT(BLIS_LIB_PATH))
        message(FATAL_ERROR "Need to provide a path to BLIS library during CMake invocation. Please use \
        $ cmake .. -DBLIS_LIB_PATH=/home/username/blis_installation/path_to_library")
    endif()
    # Set the path to BLIS include directory.
    if(NOT(BLIS_INCLUDE))
        message(FATAL_ERROR "Need to provide a path to BLIS headers during CMake invocation. Please use \
        $ cmake .. -DBLIS_INCLUDE=/home/username/blis_installation/path_to_headers")
    endif()
endif()

# Use REF_BLAS to set the library that will be used for reference results.
set(REF_CBLAS CACHE STRING "Library used to compute reference results.")
# Set the possible values of reference CBLAS for cmake-gui
if(LINUX)
    set_property(CACHE REF_CBLAS PROPERTY STRINGS "OpenBLAS" "Netlib" "MKL")
else()
    set_property(CACHE REF_CBLAS PROPERTY STRINGS "OpenBLAS" "MKL")
endif()

# Set OpenMP as the default option
set(ENABLE_THREADING "openmp" CACHE STRING "Setting OpenMP as the threading library")
# Set the possible values of theading libraries for cmake-gui
if(LINUX)
    set_property(CACHE ENABLE_THREADING PROPERTY STRINGS "openmp" "pthreads" "no")
else()
    set_property(CACHE ENABLE_THREADING PROPERTY STRINGS "openmp" "no")
endif()


# Set the possibe values of OpenMP runtimes
if(WIN32)
    # Set LLVM OpenMP library as the default option
    set(OpenMP_LIBRARY "LLVM" CACHE STRING "Using LLVM OpenMP library")
    set_property(CACHE OpenMP_LIBRARY PROPERTY STRINGS "LLVM" "Intel")
endif()

# If MKL is used as a reference set up the threading library options.
if(REF_CBLAS STREQUAL "MKL")
    # MKL threading option is set up as BLIS threading option by default.
    set(MKL_ENABLE_THREADING  ${ENABLE_THREADING} CACHE STRING "Setting MKL threading the same as BLIS threading")
endif()

# Set up OpenMP flags correctly if it's required.
if( (ENABLE_THREADING STREQUAL "openmp") OR (MKL_ENABLE_THREADING STREQUAL "openmp") )
    if(WIN32)
        set(OpenMP_libomp_LIBRARY "C:/Program Files/LLVM/lib/libomp.lib" CACHE STRING "openmp library path")
    endif()
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
        message (FATAL_ERROR "Openmp Not Found, please provide an OpenMP library using -DOpenMP_libomp_LIBRARY=path_to_omp_lib.")
    endif()
endif()

# Set static BLIS as the default library we build against.
set(BLIS_LINKING_TYPE "static" CACHE STRING "Linking to a static BLIS library")
# Set the possible values of BLIS linking type for cmake-gui
set_property(CACHE BLIS_LINKING_TYPE PROPERTY STRINGS "static" "shared")

# Set common libraries.
if(LINUX)
    set(COMMON_LIBS pthread m dl)
    option(ENABLE_ASAN "Run tests using Address Sanatizer" OFF)
    option(ENABLE_COVERAGE "Run tests for Code Coderage" OFF)
endif()

# Use INT_SIZE to set the int type used for testing.
set(INT_SIZE "32" CACHE STRING "Integer size used in testing suite.")
# Set the possible values of reference CBLAS for cmake-gui
set_property(CACHE INT_SIZE PROPERTY STRINGS "32" "64")
if( NOT ((INT_SIZE STREQUAL "32") OR (INT_SIZE STREQUAL "64")) )
    message(FATAL_ERROR "INT_SIZE option ${INT_SIZE} is not supported. Must be 32 or 64.")
endif()

# Use TEST_INTERFACE to set which interface, supported by BLIS is meant to be tested.
set(TEST_INTERFACE "BLAS" CACHE STRING "Interface that is being tested.")
# Set the possible values of interfaces for cmake-gui
set_property(CACHE TEST_INTERFACE PROPERTY STRINGS "BLAS" "CBLAS" "BLIS_TYPED")
if( NOT ((TEST_INTERFACE STREQUAL "BLAS") OR (TEST_INTERFACE STREQUAL "CBLAS") OR (TEST_INTERFACE STREQUAL "BLIS_TYPED")) )
    message(FATAL_ERROR "TEST_INTERFACE option ${TEST_INTERFACE} is not supported. Please use on of the following options \
            during CMake invokation: -DTEST_INTERFACE=BLAS or -DTEST_INTERFACE=CBLAS or -DTEST_INTERFACE=BLIS_TYPED")
endif()

# Use BLIS_ELEMENT_TYPE to set whether the elements of any matrix/vector tested are integers or floating point values.
set(BLIS_ELEMENT_TYPE "f" CACHE STRING "Type of elements of matrices/vectors")
# Set the possible values of element types for cmake-gui
set_property(CACHE BLIS_ELEMENT_TYPE PROPERTY STRINGS "f" "i")
if( NOT ((BLIS_ELEMENT_TYPE STREQUAL "f") OR (BLIS_ELEMENT_TYPE STREQUAL "i")) )
    message(FATAL_ERROR "BLIS_ELEMENT_TYPE option ${BLIS_ELEMENT_TYPE} is not supported. Please use on of the following options \
            during CMake invokation: -DBLIS_ELEMENT_TYPE=f or -DBLIS_ELEMENT_TYPE=i")
endif()

if(LINUX)
    if(REF_LIB)
        set(REFLIB_PATH ${REF_LIB}/..)
        find_library(reflib NAMES openblas cblas mkl_rt PATHS ${REFLIB_PATH})
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
            find_library(mkllib NAMES mkl_rt PATHS ${MKL_PATH})
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

    if(ENABLE_THREADING STREQUAL "no")
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis.a")
        else()
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis.so")
        endif()
        find_library(libblis NAMES blis PATHS ${BLIS_PATH}/lib)
    else()
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis-mt.a")
        else()
            set(Blis_LIBRARY "${BLIS_PATH}/lib/libblis-mt.so")
        endif()
        find_library(libblis NAMES blis-mt PATHS ${BLIS_PATH}/lib)
    endif()
    if(${libblis} STREQUAL libblis-NOTFOUND)
        message(FATAL_ERROR "Blis Library not found : " ${BLIS_PATH})
    else()
        message(STATUS "Found BLIS Library : " ${Blis_LIBRARY})
    endif()
else()
    add_definitions(-DBOOST_THREAD_USE_LIB)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)  
    cmake_policy(SET CMP0091 NEW)
    if(BLIS_LINKING_TYPE STREQUAL "shared")
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    
    if(ENABLE_THREADING STREQUAL "no")
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_LIB_PATH}/AOCL-LibBlis-Win.lib")
        else()
            set(Blis_LIBRARY "${BLIS_LIB_PATH}/AOCL-LibBlis-Win-dll.lib")
            set(BLIS_DLL "${BLIS_LIB_PATH}/AOCL-LibBlis-Win-dll.dll")
        endif()
    else()
        if(BLIS_LINKING_TYPE STREQUAL "static")
            set(Blis_LIBRARY "${BLIS_LIB_PATH}/AOCL-LibBlis-Win-MT.lib")
        else()
            set(Blis_LIBRARY "${BLIS_LIB_PATH}/AOCL-LibBlis-Win-MT-dll.lib")
            set(BLIS_DLL "${BLIS_LIB_PATH}/AOCL-LibBlis-Win-MT-dll.dll")
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
