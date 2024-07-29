#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
if(NOT DEFINED BLIS_INSTALL_PATH)
    set(DIST_PATH ${CMAKE_BINARY_DIR})
    set(LIB_PATH ${DIST_PATH}/lib/${BLIS_CONFIG_FAMILY})
    set(INC_PATH ${DIST_PATH}/include/${BLIS_CONFIG_FAMILY})
else()
    set(LIB_PATH ${BLIS_INSTALL_PATH}/lib)
    set(INC_PATH ${BLIS_INSTALL_PATH}/include/${BLIS_CONFIG_FAMILY})
endif()

# Include the corresponding make_defs.cmake that holds the required compiler options.
include(${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY}/make_defs.cmake)

# Gather all local source files.
file(GLOB testcpp_sources LIST_DIRECTORIES false ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
list(TRANSFORM testcpp_sources REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/ "")

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
set(CINFLAGS ${INC_PATH})

# Create one executable for each of the sources.
foreach(source ${testcpp_sources})
    string(REPLACE .cc "" exec_name ${source})
    string(APPEND exec_name "_blis")
    add_executable(${exec_name} ${source})
    target_compile_options(${exec_name}
                            PRIVATE
                            # load-var-for,COPTFLAGS
                            ${COPTFLAGS}
                            # get-noopt-cflags-for
                            ${CDBGFLAGS}
                            ${CWARNFLAGS}
                            ${CPICFLAGS}
                            ${CMISCFLAGS}
                            ${CXXLANGFLAGS}

                        )
    target_include_directories(${exec_name}
                            BEFORE
                            PRIVATE
                            # in get-noopt-cflags-for
                            ${CINFLAGS}
                            # Add local header paths
                            ${CMAKE_CURRENT_SOURCE_DIR}
                            ${CMAKE_SOURCE_DIR}/vendor/cpp
                        )
    target_link_libraries(${exec_name} PRIVATE ${LDFLAGS} ${libblis_link})
    if(THREADING_MODEL STREQUAL "openmp")
        if((NOT ${OpenMP_libomp_LIBRARY} STREQUAL "") AND (NOT WIN32))
            target_link_libraries(${exec_name} PRIVATE ${OpenMP_libomp_LIBRARY})
        else()
            target_link_libraries(${exec_name} PRIVATE OpenMP::OpenMP_C)
        endif()
    endif()
    set_target_properties(${exec_name} PROPERTIES CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # Put all those targets under vendor-testcpp-targets folder name so that they appear all together in IDE.
    set_target_properties(${exec_name} PROPERTIES FOLDER vendor-testcpp-targets)
    add_custom_target(${exec_name}.x
                        COMMAND ${exec_name})
    # Put all those targets under vendor-testcpp-targets folder name so that they appear all together in IDE.
    set_target_properties(${exec_name}.x PROPERTIES FOLDER vendor-testcpp-targets)
    list(APPEND test_executables "${exec_name}.x")
endforeach()

add_custom_target(checkbliscpp DEPENDS ${test_executables})
# Put all those targets under vendor-testcpp-targets folder name so that they appear all together in IDE.
set_target_properties(checkbliscpp PROPERTIES FOLDER vendor-testcpp-targets)
