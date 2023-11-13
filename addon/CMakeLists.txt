##Copyright (C) 2023, Advanced Micro Devices, Inc ##

# Writing a function that will be used to generate the required object
# libraries for the required addons.
function(generate_addon_targets addon_target)
    # Collect all subdirectory paths that have at least one file with suffix in ADDON_C99_SUFS list.
    get_filepaths_with_suffixes(LOCAL_SOURCE_C99_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${addon_target}" "${ADDON_C99_SUFS}")
    # We want to break the files above in 2 categories, files in kernel directory and the rest.
    # Only list files in kernel directory.
    set(LOCAL_KERNEL_FILES_C99 ${LOCAL_SOURCE_FILES})
    list(FILTER LOCAL_KERNEL_FILES_C99 INCLUDE REGEX ${addon_target}/kernels/)
    # All C99 files, except of the ones in kernels directory.
    list(REMOVE_ITEM LOCAL_SOURCE_C99_FILES ${LOCAL_KERNEL_FILES_C99})

    # Collect all subdirectory paths that have at least one file with suffix in ADDON_H99_SUFS list.
    get_dirpaths_with_suffixes(CADDONINCFLAGS "${CMAKE_CURRENT_SOURCE_DIR}/${addon_target}" "${ADDON_H99_SUFS}")

    # Only generate the object library if there is at least one source file.
    list(LENGTH LOCAL_SOURCE_C99_FILES size)
    if(size GREATER 0)
        # Create an object library using the source file list above.
        add_library(${addon_target}_C99_ADDON
                        OBJECT
                        ${LOCAL_SOURCE_C99_FILES}
                    )
        # Include the corresponding make_defs.cmake that holds the required compiler options.
        include(${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY}/make_defs.cmake)
        # Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
        # mimicing get-addon-c99flags-for
        target_compile_options(${addon_target}_C99_ADDON
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
                                # in get-addon-c99flags-for
                                ${BUILD_SYMFLAGS}
                            )
        target_compile_definitions(${addon_target}_C99_ADDON
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CPPROCFLAGS}
                                    # in get-noopt-cflags-for
                                    ${VERS_DEF}
                                    # in get-addon-c99flags-for
                                    ${BUILD_CPPFLAGS}
                                )
        target_include_directories(${addon_target}_C99_ADDON
                                    BEFORE
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CINFLAGS}
                                    # in get-addon-c99flags-for
                                    ${CADDONINCFLAGS}
                                )
        if(THREADING_MODEL STREQUAL "openmp")
            # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
            target_link_libraries(${addon_target}_C99_ADDON PRIVATE OpenMP::OpenMP_C)
        elseif(THREADING_MODEL STREQUAL "pthreads")
            # in get-noopt-cflags-for
            target_compile_options(${addon_target}_C99_ADDON PRIVATE ${CTHREADFLAGS})
        endif()
        if(BUILD_SHARED_LIBS)
            # Equivalent to CPICFLAGS in get-noopt-cflags-for
            set_target_properties(${addon_target}_C99_ADDON PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        add_dependencies(${addon_target}_C99_ADDON flat-header)
        # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
        set_target_properties(${addon_target}_C99_ADDON PROPERTIES FOLDER object-libs-targets)
    endif()

    # Only generate the object library if there is at least one source file.
    list(LENGTH LOCAL_KERNEL_FILES_C99 size)
    if(size GREATER 0)
        # Create an object library using the kernel source file list above.
        add_library(${addon_target}_C99_KERNEL_ADDON
                        OBJECT
                        ${LOCAL_KERNEL_FILES_C99}
                    )
        # Include the corresponding make_defs.cmake that holds the required compiler options.
        include(${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY}/make_defs.cmake)
        # Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
        # mimicing get-addon-c99flags-for
        target_compile_options(${addon_target}_C99_KERNEL_ADDON
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
                                # in get-addon-kernel-c99flags-for
                                ${BUILD_SYMFLAGS}
                            )
        target_compile_definitions(${addon_target}_C99_KERNEL_ADDON
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CPPROCFLAGS}
                                    # in get-noopt-cflags-for
                                    ${VERS_DEF}
                                    # in get-addon-kernel-c99flags-for
                                    ${BUILD_CPPFLAGS}
                                )
        target_include_directories(${addon_target}_C99_KERNEL_ADDON
                                    BEFORE
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CINFLAGS}
                                    # in get-addon-kernel-c99flags-for
                                    ${CADDONINCFLAGS}
                                )
        if(THREADING_MODEL STREQUAL "openmp")
            # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
            target_link_libraries(${addon_target}_C99_KERNEL_ADDON PRIVATE OpenMP::OpenMP_C)
        elseif(THREADING_MODEL STREQUAL "pthreads")
            # in get-noopt-cflags-for
            target_compile_options(${addon_target}_C99_KERNEL_ADDON PRIVATE ${CTHREADFLAGS})
        endif()
        if(BUILD_SHARED_LIBS)
            # Equivalent to CPICFLAGS in get-noopt-cflags-for
            set_target_properties(${addon_target}_C99_KERNEL_ADDON PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        add_dependencies(${addon_target}_C99_KERNEL_ADDON flat-header)
        # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
        set_target_properties(${addon_target}_C99_KERNEL_ADDON PROPERTIES FOLDER object-libs-targets)
    endif()

    # Collect all subdirectory paths that have at least one file with suffix in ADDON_CXX_SUFS list.
    get_filepaths_with_suffixes(LOCAL_SOURCE_CXX_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${addon_target}" "${ADDON_CXX_SUFS}")

    # Only generate the object library if there is at least one source file.
    list(LENGTH LOCAL_SOURCE_CXX_FILES size)
    if(size GREATER 0)
        # Create an object library using the source file list above.
        add_library(${addon_target}_CXX_ADDON
                        OBJECT
                        ${LOCAL_SOURCE_CXX_FILES}
                    )

        # Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
        # mimicing get-addon-cxxflags-for
        target_compile_options(${addon_target}_CXX_ADDON
                                PRIVATE
                                # load-var-for,COPTFLAGS
                                ${COPTFLAGS}
                                # get-noopt-cxxflags-for
                                ${CDBGFLAGS}
                                # get-noopt-cxxflags-for
                                ${CWARNFLAGS}
                                # get-noopt-cxxflags-for
                                ${CMISCFLAGS}
                                # get-noopt-cxxflags-for
                                ${CXXLANGFLAGS}
                                # in get-addon-cxxflags-for
                                ${BUILD_SYMFLAGS}
                            )
        target_compile_definitions(${addon_target}_CXX_ADDON
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CPPROCFLAGS}
                                    # in get-noopt-cflags-for
                                    ${VERS_DEF}
                                    # in get-addon-cxxflags-for
                                    ${BUILD_CPPFLAGS}
                                )
        target_include_directories(${addon_target}_CXX_ADDON
                                    BEFORE
                                    PRIVATE
                                    # in get-noopt-cflags-for
                                    ${CINFLAGS}
                                    # in get-addon-cxxflags-for
                                    ${CADDONINCFLAGS}
                                )
        if(THREADING_MODEL STREQUAL "openmp")
            # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
            target_link_libraries(${addon_target}_CXX_ADDON PRIVATE OpenMP::OpenMP_C)
        elseif(THREADING_MODEL STREQUAL "pthreads")
            # in get-noopt-cflags-for
            target_compile_options(${addon_target}_CXX_ADDON PRIVATE ${CTHREADFLAGS})
        endif()
        if(BUILD_SHARED_LIBS)
            # Equivalent to CPICFLAGS in get-noopt-cflags-for
            set_target_properties(${addon_target}_CXX_ADDON PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
        add_dependencies(${addon_target}_CXX_ADDON flat-header)
        # Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
        set_target_properties(${addon_target}_CXX_ADDON PROPERTIES FOLDER object-libs-targets)
    endif()
endfunction()

# Generate targets for each of the addons.
foreach(ADDON ${ENABLE_ADDON})
    generate_addon_targets(${ADDON})
endforeach()
