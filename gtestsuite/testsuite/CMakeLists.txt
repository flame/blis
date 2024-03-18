#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

# Fetch and Build GTest at configure time
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

# Set corresponding environment variables when we compare against MKL.
if(REF_CBLAS STREQUAL "MKL")
  # Since we test with MKL as reference we need to ensure that the correct interface is being picked up from mkl_rt library.
  if(INT_SIZE STREQUAL "32")
    set(MKL_ENV ${MKL_ENV};MKL_INTERFACE_LAYER=LP64)
  else()
    set(MKL_ENV ${MKL_ENV};MKL_INTERFACE_LAYER=ILP64)
  endif()
  # Chose which threading library to use with MKL depending on the option.
  if(MKL_ENABLE_THREADING STREQUAL "no")
    set(MKL_ENV ${MKL_ENV};MKL_THREADING_LAYER=SEQUENTIAL)
  else()
    if(WIN32)
      set(MKL_ENV ${MKL_ENV};MKL_THREADING_LAYER=INTEL)
    else() # if the system is Linux
      set(MKL_ENV ${MKL_ENV};MKL_THREADING_LAYER=GNU)
    endif()
  endif()
endif()

# Note: Once we integrate with the blis CMake system, we will update and use
# this functionality from the build/cmake directory.
#--------------------------------------------
#  Important sets of header files and paths
#--------------------------------------------
# Get a list of all sub-directories of a given directory
macro(get_dirpaths_with_suffixes result curdir sufflist)
    set(dirlist "")
    # dirlist will have all files which are below this directory.
    file(GLOB_RECURSE children LIST_DIRECTORIES true ${curdir}/*)
    # Adding current directory in the list.
    list(PREPEND children ${curdir})
    # Filter out anything that is not a directory.
    foreach(child ${children})
        if(IS_DIRECTORY ${child})
            set(HAS_SUFF_FILE "false")
            foreach(suff ${sufflist})
                file(GLOB suff_files LIST_DIRECTORIES false ${child}/*\.${suff})
                list(LENGTH suff_files list_size)
                if(NOT (${list_size} STREQUAL 0))
                    set(HAS_SUFF_FILE "true")
                    # If there is at least one file with a specific suffix break from for-loop.
                    break()
                endif()
            endforeach()
            # If there is at least one *.suff file, add directory path in the list.
            if(HAS_SUFF_FILE STREQUAL "true")
                list(APPEND dirlist "${child}")
            endif()
        endif()
    endforeach()
    # Get the name of the current directory, after removing the source directory
    # from the name, so that we can exclude the files that are part of the ignore
    # list even if the blis directory is located in a directory with a name that
    # would be ignored.
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" curdirsimple ${curdir})
    # Filter out anything that is part of the IGNORE_LIST.
    foreach(item ${IGNORE_LIST})
        list(FILTER dirlist EXCLUDE REGEX ${curdirsimple}.*/${item}/)
    endforeach()
    list(APPEND ${result} ${dirlist})
endmacro()

get_dirpaths_with_suffixes(test_files ${CMAKE_CURRENT_SOURCE_DIR} cpp)
set(target_name "testsuite")
foreach(dir ${test_files})
  file(GLOB files ${dir}/*.cpp)
  STRING(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" exec_name ${dir})
  STRING(REPLACE "/" "." exec_name ${exec_name})
  STRING(PREPEND exec_name ${target_name}.)
  if(files)
    add_executable(${exec_name} ${files})
    set_target_properties(${exec_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set_target_properties(${exec_name} PROPERTIES OUTPUT_NAME ${exec_name})
    target_include_directories(${exec_name} PUBLIC ${BLIS_INCLUDE} ${CMAKE_SOURCE_DIR}/testinghelpers/inc ${CMAKE_SOURCE_DIR}/testsuite/)
    target_link_libraries(${exec_name} gtest gtest_main testinghelpers ${BLIS_LIBRARY} ${COMMON_LIBS})
    # if we test serial BLIS, but MKL is used as a reference we still need to set up OpenMP.
    if( (ENABLE_THREADING STREQUAL "openmp") OR (MKL_ENABLE_THREADING STREQUAL "openmp") OR (NOT(OPENMP_NESTED STREQUAL "0")))
      target_link_libraries(${exec_name} OpenMP::OpenMP_CXX)
    endif()
    target_link_libraries(${exec_name} ${ASAN_FLAGS} ${COVERAGE_FLAGS})
    target_compile_definitions(${exec_name} PUBLIC ${TEST_BLIS_VERSION})
    if(TEST_INTERFACE STREQUAL "BLAS")
      target_compile_definitions(${exec_name} PUBLIC TEST_BLAS TEST_BLAS_LIKE API_PRINT="blas")
    elseif(TEST_INTERFACE STREQUAL "BLAS_BLIS_IMPL")
      target_compile_definitions(${exec_name} PUBLIC TEST_BLAS_BLIS_IMPL TEST_BLAS_LIKE API_PRINT="blas_blis_impl")
    elseif(TEST_INTERFACE STREQUAL "CBLAS")
      target_compile_definitions(${exec_name} PUBLIC TEST_CBLAS API_PRINT="cblas")
    else() # BLIS_TYPED option
      target_compile_definitions(${exec_name} PUBLIC TEST_BLIS_TYPED API_PRINT="bli")
    endif()
    target_compile_definitions(${exec_name} PUBLIC ${UKR_DEFINES})
    if(TEST_UPPERCASE_ARGS)
      target_compile_definitions(${exec_name} PUBLIC TEST_UPPERCASE_ARGS)
    endif()
    if(THRESHOLD_ZERO)
      target_compile_definitions(${exec_name} PUBLIC THRESHOLD_ZERO)
    endif()
    if(CAN_TEST_INFO_VALUE)
      target_compile_definitions(${exec_name} PUBLIC CAN_TEST_INFO_VALUE)
    endif()
    if(TEST_INPUT_ARGS)
      target_compile_definitions(${exec_name} PUBLIC TEST_INPUT_ARGS)
    endif()

    if(OPENMP_NESTED STREQUAL "1")
      target_compile_definitions(testinghelpers PUBLIC OPENMP_NESTED_1)
    elseif(OPENMP_NESTED STREQUAL "2")
      target_compile_definitions(testinghelpers PUBLIC OPENMP_NESTED_2)
    elseif(OPENMP_NESTED STREQUAL "1diff")
      target_compile_definitions(testinghelpers PUBLIC OPENMP_NESTED_1diff)
    endif()

    add_test(NAME ${exec_name} COMMAND ${exec_name})
    if(REF_CBLAS STREQUAL "MKL")
      set_property(TEST ${exec_name} PROPERTY ENVIRONMENT ${MKL_ENV})
    endif()
    if(BLIS_LINKING_TYPE STREQUAL "shared")
      set_property(TEST ${exec_name} PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${BLIS_LIB_PATH}")
    endif()
  endif()
  list(APPEND all_execs ${exec_name})
endforeach()

# Return the list of the subdirectories in the directory curdir.
macro(SUBDIRLIST result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    ENDIF()
  endforeach()
  set(${result} ${dirlist})
endmacro()

# Add dependencies to build all level1 or level2, etc., tests with one target.
SUBDIRLIST(subdirs ${CMAKE_CURRENT_SOURCE_DIR})
foreach(dir ${subdirs})
  set(child_execs ${all_execs})
  add_custom_target(${target_name}.${dir})
  list(FILTER child_execs INCLUDE REGEX ${dir})
  foreach(child ${child_execs})
    add_dependencies(${target_name}.${dir} ${child})
  endforeach()
endforeach()
