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

# Writing a function that will be used to generate the required object
# libraries for the required kernels.
function(generate_kernel_targets kernel_target)
    # Collect all subdirectory paths that have at least one file with suffix in KERNELS_SRC_SUFS list.
    get_filepaths_with_suffixes(LOCAL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_target}" "${KERNELS_SRC_SUFS}")

    # Choose correct sub-configurarion name for the given kernel set.
    get_config_for_kernel_from_kconfig_map(LOCAL_CONFIG ${kernel_target} "${KCONFIG_MAP}")

    # filter the lpgemm source files to a different array
    set(LOCAL_LPEGMM_SOURCE_FILES ${LOCAL_SOURCE_FILES})
    list(FILTER LOCAL_SOURCE_FILES EXCLUDE REGEX ".*/lpgemm/.*")
    list(FILTER LOCAL_LPEGMM_SOURCE_FILES INCLUDE REGEX ".*/lpgemm/.*")

    # Only generate the object library if there is at least one source file.
    list(LENGTH LOCAL_SOURCE_FILES size)
    if(size GREATER 0)
        # Create an object library using the source file list above.
        add_library(${kernel_target}_KERNELS
                        OBJECT
                        ${LOCAL_SOURCE_FILES}
                    )
        # Include the corresponding make_defs.cmake that holds the required compiler options.
        include(${CMAKE_SOURCE_DIR}/config/${LOCAL_CONFIG}/make_defs.cmake)
        # Use PRIVATE keyword for option setting since we do not want the
        # properties to propagate in other targets.
        # mimicing get-kernel-cflags-for
        target_compile_options(${kernel_target}_KERNELS
                                PRIVATE
                                # load-var-for,CKOPTFLAGS
                                ${CKOPTFLAGS}
                                # load-var-for,CKVECFLAGS
                                ${CKVECFLAGS}
                                # get-noopt-cflags-for
                                ${CDBGFLAGS}
                                # get-noopt-cflags-for
                                ${CWARNFLAGS}
                                # get-noopt-cflags-for
                                ${CMISCFLAGS}
                                # get-noopt-cflags-for
                                ${CLANGFLAGS}
                                # in get-kernel-cflags-for
                                ${COMPSIMDFLAGS}
                                # in get-kernel-cflags-for
                                ${BUILD_SYMFLAGS}
                            )
        target_compile_definitions(${kernel_target}_KERNELS
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CPPROCFLAGS}
                                    # in get-noopt-cflags-for
                                    ${VERS_DEF}
                                    # in get-kernel-cflags-for
                                    ${BUILD_CPPFLAGS}
                                )
        target_include_directories(${kernel_target}_KERNELS
                                    BEFORE
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CINFLAGS}
                                )
        if(THREADING_MODEL STREQUAL "openmp")
            # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
            target_link_libraries(${kernel_target}_KERNELS PRIVATE OpenMP::OpenMP_C)
        elseif(THREADING_MODEL STREQUAL "pthreads")
            # in get-noopt-cflags-for
            target_compile_options(${kernel_target}_KERNELS PRIVATE ${CTHREADFLAGS})
        endif()
        # Equivalent to CPICFLAGS in get-noopt-cflags-for
        set_target_properties(${kernel_target}_KERNELS PROPERTIES POSITION_INDEPENDENT_CODE ON)
        add_dependencies(${kernel_target}_KERNELS flat-header)
        # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
        set_target_properties(${kernel_target}_KERNELS PROPERTIES FOLDER object-libs-targets)
    endif()

    # Only generate the object library if there is at least one source file.
    list(LENGTH LOCAL_LPEGMM_SOURCE_FILES size_lpgemm)
    if (size_lpgemm GREATER 0)
        # Create an object library using the source file list above.
        add_library(${kernel_target}_LPGEMM_KERNELS
                OBJECT
                ${LOCAL_LPEGMM_SOURCE_FILES}
            )
        # Include the corresponding make_defs.cmake that holds the required compiler options.
        include(${CMAKE_SOURCE_DIR}/config/${LOCAL_CONFIG}/make_defs.cmake)
        # Use PRIVATE keyword for option setting since we do not want the
        # properties to propagate in other targets.
        # mimicing get-kernel-cflags-for
        target_compile_options(${kernel_target}_LPGEMM_KERNELS
                                PRIVATE
                                # load-var-for,CKOPTFLAGS
                                ${CKOPTFLAGS}
                                # load-var-for,CKLPOPTFLAGS
                                ${CKLPOPTFLAGS}
                                # load-var-for,CKVECFLAGS
                                ${CKVECFLAGS}
                                # get-noopt-cflags-for
                                ${CDBGFLAGS}
                                # get-noopt-cflags-for
                                ${CWARNFLAGS}
                                # get-noopt-cflags-for
                                ${CMISCFLAGS}
                                # get-noopt-cflags-for
                                ${CLANGFLAGS}
                                # in get-kernel-cflags-for
                                ${COMPSIMDFLAGS}
                                # in get-kernel-cflags-for
                                ${BUILD_SYMFLAGS}
                            )
        target_compile_definitions(${kernel_target}_LPGEMM_KERNELS
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CPPROCFLAGS}
                                    # in get-noopt-cflags-for
                                    ${VERS_DEF}
                                    # in get-kernel-cflags-for
                                    ${BUILD_CPPFLAGS}
                                )
        target_include_directories(${kernel_target}_LPGEMM_KERNELS
                                    BEFORE
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CINFLAGS}
                                )
        if(THREADING_MODEL STREQUAL "openmp")
        # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
        target_link_libraries(${kernel_target}_LPGEMM_KERNELS PRIVATE OpenMP::OpenMP_C)
        elseif(THREADING_MODEL STREQUAL "pthreads")
        # in get-noopt-cflags-for
        target_compile_options(${kernel_target}_LPGEMM_KERNELS PRIVATE ${CTHREADFLAGS})
        endif()
        # Equivalent to CPICFLAGS in get-noopt-cflags-for
        set_target_properties(${kernel_target}_LPGEMM_KERNELS PROPERTIES POSITION_INDEPENDENT_CODE ON)
        add_dependencies(${kernel_target}_LPGEMM_KERNELS flat-header)
        # Put all those targets under object-libs-targets folder name so that they appear
        # all together in IDE.
        set_target_properties(${kernel_target}_LPGEMM_KERNELS PROPERTIES FOLDER object-libs-targets)
        endif()
    endfunction()

# Generate targets for each of the kernels present
# in the kernel list.
foreach(KERN ${KERNEL_LIST})
    generate_kernel_targets(${KERN})
endforeach()
