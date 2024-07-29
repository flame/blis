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

# Writing a function that will be used to generate the required object
# libraries for the required configs.
function(generate_config_targets config_target)
    # Collect all subdirectory paths that have at least one file with suffix in CONFIG_SRC_SUFS list.
    get_filepaths_with_suffixes(LOCAL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${config_target}" "${CONFIG_SRC_SUFS}")

    # Create an object library using the source file list above.
    add_library(${config_target}_CONFIG
                    OBJECT
                    ${LOCAL_SOURCE_FILES}
                )
    # Include the corresponding make_defs.cmake that holds the required compiler options.
    include(${CMAKE_SOURCE_DIR}/config/${config_target}/make_defs.cmake)
    # Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
    # mimicing get-config-cflags-for
    target_compile_options(${config_target}_CONFIG
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
                            # in get-config-cflags-for
                            ${BUILD_SYMFLAGS}
                        )
    target_compile_definitions(${config_target}_CONFIG
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CPPROCFLAGS}
                                # in get-noopt-cflags-for
                                ${VERS_DEF}
                                # in get-config-cflags-for
                                ${BUILD_CPPFLAGS}
                            )
    target_include_directories(${config_target}_CONFIG
                                BEFORE
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CINFLAGS}
                            )
    if(THREADING_MODEL STREQUAL "openmp")
        # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
        target_link_libraries(${config_target}_CONFIG PRIVATE OpenMP::OpenMP_C)
    elseif(THREADING_MODEL STREQUAL "pthreads")
        # in get-noopt-cflags-for
        target_compile_options(${config_target}_CONFIG PRIVATE ${CTHREADFLAGS})
    endif()
    # Equivalent to CPICFLAGS in get-noopt-cflags-for
    set_target_properties(${config_target}_CONFIG PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_dependencies(${config_target}_CONFIG flat-header)
    # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
    set_target_properties(${config_target}_CONFIG PROPERTIES FOLDER object-libs-targets)

    # Create on object library using the corresponding reference kernel initialization file.
    add_library(${config_target}_REFINIT
                OBJECT
                ${CMAKE_SOURCE_DIR}/ref_kernels/bli_cntx_ref.c
            )
    # Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
    # mimicing get-refinit-cflags-for
    target_compile_options(${config_target}_REFINIT
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
                            # in get-refinit-cflags-for
                            ${BUILD_SYMFLAGS}
                        )
    target_compile_definitions(${config_target}_REFINIT
                                PRIVATE
                                # get-noopt-cflags-for
                                ${CPPROCFLAGS}
                                # in get-noopt-cflags-for
                                ${VERS_DEF}
                                # in get-refinit-cflags-for
                                ${BUILD_CPPFLAGS}
                                # get-noopt-cflags-for
                                ${CPPROCFLAGS}
                                # in get-refinit-cflags-for
                                -DBLIS_CNAME=${config_target}
                            )
    target_include_directories(${config_target}_REFINIT
                                BEFORE
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CINFLAGS}
                            )
    if(THREADING_MODEL STREQUAL "openmp")
        # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
        target_link_libraries(${config_target}_REFINIT PRIVATE OpenMP::OpenMP_C)
    elseif(THREADING_MODEL STREQUAL "pthreads")
        # in get-noopt-cflags-for
        target_compile_options(${config_target}_REFINIT PRIVATE ${CTHREADFLAGS})
    endif()
    # Equivalent to CPICFLAGS in get-noopt-cflags-for
    set_target_properties(${config_target}_REFINIT PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_dependencies(${config_target}_REFINIT flat-header)
    # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
    set_target_properties(${config_target}_REFINIT PROPERTIES FOLDER object-libs-targets)

    # Collect all subdirectory paths that have at least one file with suffix in KERNELS_SRC_SUFS list.
    set(REFKERN_PATH ${CMAKE_SOURCE_DIR}/ref_kernels)
    get_filepaths_with_suffixes(LOCAL_REFKERN_FILES ${REFKERN_PATH} ${KERNELS_SRC_SUFS})
    # Remove bli_cntx_ref.c from source list.
    list(FILTER LOCAL_REFKERN_FILES EXCLUDE REGEX bli_cntx_ref.c)

    # Create on object library using the corresponding reference implementations being targeted.
    add_library(${config_target}_REFKERN
                OBJECT
                ${LOCAL_REFKERN_FILES}
            )
    # Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
    # mimicing get-refkern-cflags-for
    target_compile_options(${config_target}_REFKERN
                            PRIVATE
                            # load-var-for,CROPTFLAGS
                            ${CROPTFLAGS}
                            # load-var-for,CRVECFLAGS
                            ${CRVECFLAGS}
                            # get-noopt-cflags-for
                            ${CDBGFLAGS}
                            # get-noopt-cflags-for
                            ${CWARNFLAGS}
                            # get-noopt-cflags-for
                            ${CMISCFLAGS}
                            # get-noopt-cflags-for
                            ${CLANGFLAGS}
                            # in get-refkernel-cflags-for
                            ${COMPSIMDFLAGS}
                            # in get-refkern-cflags-for
                            ${BUILD_SYMFLAGS}
                        )
    target_compile_definitions(${config_target}_REFKERN
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CPPROCFLAGS}
                                # in get-noopt-cflags-for
                                ${VERS_DEF}
                                # in get-refkern-cflags-for
                                -DBLIS_CNAME=${config_target}
                                # in get-refkern-cflags-for
                                ${BUILD_CPPFLAGS}
                            )
    target_include_directories(${config_target}_REFKERN
                                BEFORE
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CINFLAGS}
                            )
    if(THREADING_MODEL STREQUAL "openmp")
        # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
        target_link_libraries(${config_target}_REFKERN PRIVATE OpenMP::OpenMP_C)
    elseif(THREADING_MODEL STREQUAL "pthreads")
        # in get-noopt-cflags-for
        target_compile_options(${config_target}_REFKERN PRIVATE ${CTHREADFLAGS})
    endif()
    # Equivalent to CPICFLAGS in get-noopt-cflags-for
    set_target_properties(${config_target}_REFKERN PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_dependencies(${config_target}_REFKERN flat-header)
    # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
    set_target_properties(${config_target}_REFKERN PROPERTIES FOLDER object-libs-targets)
endfunction()

# Generate targets for each of the configs.
foreach(CONF ${CONFIG_LIST})
    generate_config_targets(${CONF})
endforeach()
