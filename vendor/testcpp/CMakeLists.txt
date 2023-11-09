##Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.##

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
    set(INC_PATH ${BLIS_INSTALL_PATH}/include/blis)
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
    target_link_libraries(${exec_name} PRIVATE ${LDFLAGS} libblis)
    if(THREADING_MODEL STREQUAL "openmp")
        target_link_libraries(${exec_name} PRIVATE OpenMP::OpenMP_C)
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
