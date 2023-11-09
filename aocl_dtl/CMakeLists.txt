##Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved. ##

# Collect all subdirectory paths that have at least one file with suffix in AOCLDTL_SRC_SUFS list.
get_filepaths_with_suffixes(LOCAL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR} "${AOCLDTL_SRC_SUFS}")

# Create an object library using the source file list above.
add_library(AOCL_DTL
                OBJECT
                ${LOCAL_SOURCE_FILES}
            )

# Include the corresponding make_defs.cmake that holds the required compiler options.
include(${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY}/make_defs.cmake)
# Use PRIVATE keyword for option setting since we do not want the properties to propagate in other targets.
# mimicing get-aocldtl-cflags-for
target_compile_options(AOCL_DTL
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
                        # in get-aocldtl-cflags-for
                        ${BUILD_SYMFLAGS}
                    )
target_compile_definitions(AOCL_DTL
                            PRIVATE
                            # in get-noopt-cflags-for
                            ${VERS_DEF}
                            # in get-aocldtl-cflags-for
                            ${BUILD_CPPFLAGS}
                            # in get-aocldtl-cflags-for
                            ${CPPROCFLAGS}
                        )
target_include_directories(AOCL_DTL
                            BEFORE
                            PRIVATE
                            # in get-noopt-cflags-for
                            ${CINFLAGS}
                        )
if(THREADING_MODEL STREQUAL "openmp")
    # Equivalent to CTHREADFLAGS in get-noopt-cflags-for
    target_link_libraries(AOCL_DTL PRIVATE OpenMP::OpenMP_C)
elseif(THREADING_MODEL STREQUAL "pthreads")
    # in get-noopt-cflags-for
    target_compile_options(AOCL_DTL PRIVATE ${CTHREADFLAGS})
endif()
if(BUILD_SHARED_LIBS)
    # Equivalent to CPICFLAGS in get-noopt-cflags-for
    set_target_properties(AOCL_DTL PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_dependencies(AOCL_DTL flat-header)
# Put all those targets under object-libs-targets folder name so that they appear all together in IDE.
set_target_properties(AOCL_DTL PROPERTIES FOLDER object-libs-targets)
