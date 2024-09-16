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

# Comments:
# Set the path to the BLIS installation.
set(BLIS_INSTALL_PATH "" CACHE STRING "Setting the path to a BLIS installation that needs testing.")
if(BLIS_INSTALL_PATH)
    message(STATUS "BLIS_INSTALL_PATH :" ${BLIS_INSTALL_PATH})
endif()

# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.

#if(NOT DEFINED BLIS_INSTALL_PATH)
if(BLIS_INSTALL_PATH STREQUAL "")
    set(DIST_PATH ${CMAKE_BINARY_DIR})
    set(LIB_PATH ${DIST_PATH}/lib/${BLIS_CONFIG_FAMILY})
    set(INC_PATH ${DIST_PATH}/include/${BLIS_CONFIG_FAMILY})
    set(CINFLAGS ${INC_PATH})
    set(LIBBLIS  ${libblis_link})
else()
    set(LIB_PATH ${BLIS_INSTALL_PATH}/lib)
    set(INC_PATH ${BLIS_INSTALL_PATH}/include/${BLIS_CONFIG_FAMILY})
    set(CINFLAGS ${INC_PATH})
    # Set up the library name.
    if(WIN32)
        set(LIB_BLIS AOCL-LibBlis-Win)
    else()
        set(LIB_BLIS ${libblis_link})
    endif()
    # Append if threading is required.
    if(NOT (ENABLE_THREADING STREQUAL "no"))
        if(WIN32)
            string(APPEND LIB_BLIS -MT)
        else()
            string(APPEND LIB_BLIS -mt)
        endif()
    endif()
    # Append for dll if necessary.
    if(WIN32 AND BUILD_SHARED_LIBS)
        string(APPEND LIB_BLIS -dll)
    endif()
    # Setting the suffix for find_library().
    if(WIN32)
        set(LIB_BLIS .lib)
    else()
        if(BUILD_SHARED_LIBS)
            string(APPEND LIB_BLIS .so)
        else()
            string(APPEND LIB_BLIS .a)
        endif()
    endif()
    set(LIBBLIS  ${LIB_PATH}/${LIB_BLIS})
    message(STATUS "BLIS_INSTALL_PATH : " ${LIBBLIS})
endif()

if(WIN32)
    set(LIBSUFFIX dll)
else()
    set(LIBSUFFIX so)
endif()

set(NREPEATS "1000" CACHE STRING "Set no. of times loop repeats.")
set(MKL_PATH $ENV{MKLROOT} CACHE STRING "Set MKL_PATH.")
if(THREADING_MODEL STREQUAL "no")
    set(MKL_THREAD "${MKL_PATH}/libmkl_sequential.${LIBSUFFIX}")
else()
    set(MKL_THREAD "${MKL_PATH}/libmkl_gnu_thread.${LIBSUFFIX}")
    set(MKL_OMP iomp5)
endif()
set(INTEL_LP64 "${MKL_PATH}/libmkl_intel_lp64.${LIBSUFFIX}")
set(MKL_CORE "${MKL_PATH}/libmkl_core.${LIBSUFFIX}")
set(COMMON_LIBS pthread m dl ${MKL_OMP})
set(MKL_LIB ${INTEL_LP64} ${MKL_CORE} ${MKL_THREAD} ${COMMON_LIBS})
set(OPENBLAS_PATH "/home/amd/mylibs/openblas" CACHE STRING "Set OPENBLAS_PATH.")
set(OPENBLAS_LIB "${OPENBLAS_PATH}/libopenblas.${LIBSUFFIX}")
set(ATLAS_PATH "/home/amd/mylibs/atlas" CACHE STRING "Set ATLAS_PATH.")
set(F77BLAS_LIB "${ATLAS_PATH}/libf77blas.${LIBSUFFIX}")
set(ATLAS_LIB "${ATLAS_PATH}/libatlas.${LIBSUFFIX}")
set(ATLAS_LIB ${ATLAS_LIB} ${F77BLAS_LIB})


# Include the corresponding make_defs.cmake that holds the required compiler options.
include(${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY}/make_defs.cmake)

# Gather all local source files.
file(GLOB file_list LIST_DIRECTORIES false RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" "*.c")

# Defining the format specifiers to read long double value from input file using fscanf
if (WIN32 AND ((INT_SIZE STREQUAL "auto") OR (INT_SIZE STREQUAL "64")))
    set(BENCH_FLAGS -DN_REPEAT=${NREPEATS} -DINT_FS="%lld" -DUINT_FS="%llu")
elseif ((INT_SIZE STREQUAL "auto") OR (INT_SIZE STREQUAL "64"))
    set(BENCH_FLAGS -DN_REPEAT=${NREPEATS} -DINT_FS="%ld" -DUINT_FS="%lu")
else()
    set(BENCH_FLAGS -DN_REPEAT=${NREPEATS} -DINT_FS="%d" -DUINT_FS="%u")
endif()

# Create an executable using the sources above.
function(benchexe extn)
    set(dblas "aocl")
    if(extn STREQUAL "mkl")
        set(BLAS_LIBS ${MKL_LIB})
        set(dblas ${extn})
    elseif(extn STREQUAL "openblas")
        set(BLAS_LIBS ${OPENBLAS_LIB})
        set(dblas ${extn})
    elseif(extn STREQUAL "atlas")
        set(BLAS_LIBS ${ATLAS_LIB})
        set(dblas ${extn})
    endif()
    set(BENCH_FLAGS "${BENCH_FLAGS}" -DBLAS="${dblas}")
    foreach(src ${file_list})
        string(REGEX REPLACE ".c$" "" exec_name ${src})
        set(exec_name "${exec_name}_${extn}")
        add_executable(${exec_name}.x ${src})
        target_compile_options(${exec_name}.x
                                PRIVATE
                                # load-var-for,COPTFLAGS
                                ${COPTFLAGS}
                            )
        if(WIN32 AND BUILD_SHARED_LIBS)
            target_compile_definitions(${exec_name}.x
                                        PRIVATE
                                        # in get-noopt-cflags-for
                                        ${VERS_DEF}
                                        "-DBLIS_EXPORT=__declspec(dllimport)"
                                        ${BENCH_FLAGS}
                                    )
        else()
            target_compile_definitions(${exec_name}.x
                                        PRIVATE
                                        # in get-noopt-cflags-for
                                        ${VERS_DEF}
                                        ${BENCH_FLAGS}
                                    )
        endif()
        target_include_directories(${exec_name}.x
                                BEFORE
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CINFLAGS}
                            )
        target_link_libraries(${exec_name}.x PRIVATE ${BLAS_LIBS} ${LIBBLIS} ${LDFLAGS})
        if(THREADING_MODEL STREQUAL "openmp")
            if((NOT ${OpenMP_libomp_LIBRARY} STREQUAL "") AND (NOT WIN32))
                target_link_libraries(${exec_name}.x PRIVATE ${OpenMP_libomp_LIBRARY})
            else()
                target_link_libraries(${exec_name}.x PRIVATE OpenMP::OpenMP_C)
            endif()
        endif()
        list(APPEND temp_executables ${exec_name}.x)
    endforeach()
     set(bench_executables ${temp_executables} PARENT_SCOPE)
endfunction()

benchexe("blis")
add_custom_target(bench_blis DEPENDS ${bench_executables})
benchexe("mkl")
add_custom_target(bench_mkl DEPENDS ${bench_executables})
benchexe("openblas")
add_custom_target(bench_openblas DEPENDS ${bench_executables})
benchexe("atlas")
add_custom_target(bench_atlas DEPENDS ${bench_executables})
add_custom_target(benchmark DEPENDS bench_blis bench_mkl bench_openblas)

# Put all those targets under bench-targets folder name so that they appear all together in IDE.
# NOTE : To run bench for atlas, add bench_atlas to the bench-targets
set_target_properties(benchmark bench_blis bench_mkl bench_openblas PROPERTIES FOLDER bench-targets)

# Add bench_aocl_gemm only if aocl_gemm is in the ENABLE_ADDON list.
# This needs to work in cases where both aocl_gemm and gemmd are requested.
# lpgemm_index will be -1 if it's not found in ENABLE_ADDON list.
list(FIND ENABLE_ADDON "aocl_gemm" lpgemm_index)
if(NOT (lpgemm_index STREQUAL -1))
    add_subdirectory(bench_aocl_gemm EXCLUDE_FROM_ALL)
endif()
