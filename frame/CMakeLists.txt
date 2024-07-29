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

# Collect all subdirectory paths that have at least one file with suffix in FRAME_SRC_SUFS list.
get_filepaths_with_suffixes(LOCAL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR} "${FRAME_SRC_SUFS}")

# AMD has optimized some of the framework files, these optimizations
# may not be compatible with other platforms.
#
# In order to keep main framework code independent of AMD changes,e
# AMD has duplicated the files and updated them for example
# frame/compact/bla_gemm.c : generic framework file
# frame/compact/bla_gemm_amd.c : AMD optimized framework file
# Based on the archiecture we choose correct files
if(ENABLE_AOCL_ZEN)
    # Build is being done for AMD platforms, remove the objects which
    # don't have amd suffix (for which exists AMD specific implementation).
    # Create a copy of the source files so that we can do regex matching.
    set(TEMP_SOURCE_LIST ${LOCAL_SOURCE_FILES})
    # Create a list with all the files that have _amd. in the name.
    list(FILTER TEMP_SOURCE_LIST INCLUDE REGEX "_amd\.c$")
    # Remove _amd from all items in the list. This will leave us with all
    # source files, for which an AMD-optimized file exists.
    list(TRANSFORM TEMP_SOURCE_LIST REPLACE "_amd\.c$" "\.c")
    list(REMOVE_ITEM LOCAL_SOURCE_FILES ${TEMP_SOURCE_LIST})
else()
    # Build is done for non AMD platforms, remove the amd specific files.
    list(FILTER LOCAL_SOURCE_FILES EXCLUDE REGEX "_amd\.c$")
endif()

# Remove frame/compat/f2c/bla_xerbla_array.c from the list when building on Windows
# to avoid duplicate symbols issue when we use BLIS with LIBFLAME.
# This is a hack and the symbol should be removed from LIBFLAME instead.
list(FILTER LOCAL_SOURCE_FILES EXCLUDE REGEX "bla_xerbla_array\.c$")

# Add corresponding definitions for API that is being exported.
if(WIN32)
    if(ENABLE_NO_UNDERSCORE_API)
        set(EXPORT_API_DEFS "-DENABLE_NO_UNDERSCORE_API")
    endif()
    if(ENABLE_UPPERCASE_API)
        list(APPEND EXPORT_API_DEFS "-DBLIS_ENABLE_UPPERCASE_API")
    endif()
endif()

# Create an object library using the source file list above.
add_library(FRAME
                OBJECT
                ${LOCAL_SOURCE_FILES}
            )

# Include the corresponding make_defs.cmake that holds the required compiler options.
include(${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY}/make_defs.cmake)
# Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
# mimicing get-frame-cflags-for
target_compile_options(FRAME
                        PRIVATE
                        # load-var-for,COPTFLAGS
                        ${COPTFLAGS}
                        # get-noopt-cflags-for
                        ${CDBGFLAGS}
                        # get-noopt-cflags-for
                        ${CWARNFLAGS}
                        # get-noopt-cflags-for
                        ${CMISCFLAGS}
                        # get-noopt-cflags-for
                        ${CLANGFLAGS}
                        # in get-frame-cflags-for
                        ${BUILD_SYMFLAGS}
                    )
target_compile_definitions(FRAME
                            PRIVATE
                            # get-noopt-cflags-for
                            ${CPPROCFLAGS}
                            # in get-noopt-cflags-for
                            ${VERS_DEF}
                            # in get-frame-cflags-for
                            ${BUILD_CPPFLAGS}
                            # Windows-specific definitions
                            ${EXPORT_API_DEFS}
                        )
target_include_directories(FRAME
                            BEFORE
                            PRIVATE
                            # in get-noopt-cflags-for
                            ${CINFLAGS}
                        )
if(THREADING_MODEL STREQUAL "openmp")
    # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
    target_link_libraries(FRAME PRIVATE OpenMP::OpenMP_C)
elseif(THREADING_MODEL STREQUAL "pthreads")
    # in get-noopt-cflags-for
    target_compile_options(FRAME PRIVATE ${CTHREADFLAGS})
endif()
# Equivalent to CPICFLAGS in get-noopt-cflags-for
set_target_properties(FRAME PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_dependencies(FRAME flat-header)
# Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
set_target_properties(FRAME PROPERTIES FOLDER object-libs-targets)
