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

# Return the list of the subdirectories in the directory curdir.
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(DIRS ${CMAKE_CURRENT_SOURCE_DIR})

set(target_name "testsuite")
foreach(dir ${DIRS})
    add_custom_target(${target_name}.${dir})
    SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    foreach(subdir ${SUBDIRS})
      file(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${subdir}/*.cpp)
      if(files)
        add_executable(${target_name}.${dir}.${subdir} ${files})
        set_target_properties(${target_name}.${dir}.${subdir} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        set_target_properties(${target_name}.${dir}.${subdir} PROPERTIES OUTPUT_NAME ${target_name}.${dir}.${subdir})
        target_include_directories(${target_name}.${dir}.${subdir} PUBLIC ${BLIS_INCLUDE} ${CMAKE_SOURCE_DIR}/testinghelpers/inc ${CMAKE_SOURCE_DIR}/testsuite/)
        target_link_libraries(${target_name}.${dir}.${subdir} gtest gtest_main testinghelpers ${Blis_LIBRARY} ${COMMON_LIBS})
        # if we test serial BLIS, but MKL is used as a reference we still need to set up OpenMP.
        if( (ENABLE_THREADING STREQUAL "openmp") OR (REF_CBLAS STREQUAL "MKL"))
            if(LINUX)
                if(OpenMP_LIBRARY STREQUAL "GNU")
                    target_link_libraries(${target_name}.${dir}.${subdir} -fopenmp)
                else()
                    target_link_libraries(${target_name}.${dir}.${subdir} iomp5)
                endif()
            endif()
        endif()
        if(ENABLE_ASAN)
          target_link_libraries(${target_name}.${dir}.${subdir} -fsanitize=address)
        endif()
        if(ENABLE_COVERAGE)
          target_link_libraries(${target_name}.${dir}.${subdir} "--coverage")
        endif()
        if(TEST_INTERFACE STREQUAL "BLAS")
          target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC TEST_BLAS)
        elseif(TEST_INTERFACE STREQUAL "CBLAS")
          target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC TEST_CBLAS)
        else() # BLIS_TYPED option
          target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC TEST_BLIS_TYPED)
        endif()
        target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC ELEMENT_TYPE='${ELEMENT_TYPE}')
        add_test(NAME ${target_name}.${dir}.${subdir} COMMAND ${target_name}.${dir}.${subdir})
        add_dependencies(${target_name}.${dir} ${target_name}.${dir}.${subdir})
      endif()
    endforeach()
endforeach()


