#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
file(GLOB testsuite_sources LIST_DIRECTORIES false ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
set(CINFLAGS ${INC_PATH})
if((NOT WIN32) AND ENABLE_COVERAGE)
    include(coverage.cmake)
    list(APPEND LDFLAGS ${COVERAGE_FLAGS})
endif()
if((NOT WIN32) AND ENABLE_ASAN)
    list(APPEND LDFLAGS ${ASAN_FLAGS})
endif()

# Create an executable using the sources above.
add_executable(test_libblis.x ${testsuite_sources})
target_compile_options(test_libblis.x
                        PRIVATE
                        # load-var-for,COPTFLAGS
                        ${COPTFLAGS}
                        # get-noopt-cflags-for
                        ${CDBGFLAGS}
                        ${CWARNFLAGS}
                        ${CPICFLAGS}
                        ${CMISCFLAGS}
                        ${CLANGFLAGS}
                    )
if(WIN32 AND BUILD_SHARED_LIBS)
    target_compile_definitions(test_libblis.x
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${VERS_DEF}
                                # Need to import symbols because the testsuite is using
                                # static variables which always need to be imported.
                                "-DBLIS_EXPORT=__declspec(dllimport)"
                            )
else()
    target_compile_definitions(test_libblis.x
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${VERS_DEF}
                            )
endif()
target_include_directories(test_libblis.x
                        BEFORE
                        PRIVATE
                        # in get-noopt-cflags-for
                        ${CINFLAGS}
                        # Add local header paths
                        ${CMAKE_CURRENT_SOURCE_DIR}/src
                    )
target_link_libraries(test_libblis.x PRIVATE ${libblis_link} ${LDFLAGS})
if(THREADING_MODEL STREQUAL "openmp")
    if((NOT ${OpenMP_libomp_LIBRARY} STREQUAL "") AND (NOT WIN32))
        target_link_libraries(test_libblis.x PRIVATE ${OpenMP_libomp_LIBRARY})
    else()
        target_link_libraries(test_libblis.x PRIVATE OpenMP::OpenMP_C)
    endif()
endif()

# -- Test run/check rules --
# Wrap the creation of testing helpers in this function.
function(add_testblis flavour)
    if (NOT(flavour STREQUAL ""))
        set(dotflavour .${flavour})
        set(dashflavour -${flavour})
        set(printflavour "(${flavour})")
    endif()
    # A rule to run the testsuite using the input.*${dotflavour} files.
    add_custom_target(testblis${dashflavour}
                        COMMAND test_libblis.x -g ${CMAKE_CURRENT_SOURCE_DIR}/input.general${dotflavour} -o ${CMAKE_CURRENT_SOURCE_DIR}/input.operations${dotflavour} > ${CMAKE_CURRENT_BINARY_DIR}/output.testsuite${dotflavour}
                        COMMENT "Running test_libblis.x ${printflavour} with output redirected to ${CMAKE_CURRENT_BINARY_DIR}/output.testsuite${dotflavour}"
                        DEPENDS test_libblis.x ${CMAKE_CURRENT_SOURCE_DIR}/input.general${dotflavour} ${CMAKE_CURRENT_SOURCE_DIR}/input.operations${dotflavour}
                        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/output.testsuite${dotflavour}
                        WORKING_DIRECTORY $<TARGET_FILE_DIR:${libblis_link}>
                        VERBATIM
                    )
    # Check the results of the BLIS testsuite.
    add_custom_target(checkblis${dashflavour}
                        COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/cmake/check-blistest.py ${CMAKE_CURRENT_BINARY_DIR}/output.testsuite${dotflavour}
                        DEPENDS testblis${dashflavour}
                  )
endfunction()

# Add testing targets using functions above for all input file options.
add_testblis("")
add_testblis("fast")
add_testblis("mixed")
add_testblis("salt")

add_custom_target(checkblis-md DEPENDS checkblis-mixed)
add_custom_target(testblis-md DEPENDS testblis-mixed)
add_custom_target(testsuite DEPENDS testblis)

# Put all those targets under testsuite-targets folder name so that they appear all together in IDE.
set_target_properties(test_libblis.x testblis checkblis testblis-fast checkblis-fast testblis-md checkblis-md testblis-mixed checkblis-mixed testblis-salt checkblis-salt
                        PROPERTIES FOLDER testsuite-targets)
