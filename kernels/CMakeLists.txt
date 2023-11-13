##Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.##

# Writing a function that will be used to generate the required object
# libraries for the required kernels.
function(generate_kernel_targets kernel_target)
    # Collect all subdirectory paths that have at least one file with suffix in KERNELS_SRC_SUFS list.
    get_filepaths_with_suffixes(LOCAL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${kernel_target}" "${KERNELS_SRC_SUFS}")

    # Choose correct sub-configurarion name for the given kernel set.
    get_config_for_kernel_from_kconfig_map(LOCAL_CONFIG ${kernel_target} "${KCONFIG_MAP}")

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
        # Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
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
        if(BUILD_SHARED_LIBS)
            # Equivalent to CPICFLAGS in get-noopt-cflags-for
            set_target_properties(${kernel_target}_KERNELS PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        add_dependencies(${kernel_target}_KERNELS flat-header)
        # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
        set_target_properties(${kernel_target}_KERNELS PROPERTIES FOLDER object-libs-targets)
    endif()
endfunction()

# Generate targets for each of the kernels present
# in the kernel list.
foreach(KERN ${KERNEL_LIST})
    generate_kernel_targets(${KERN})
endforeach()
