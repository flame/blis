##Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.##

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

# Create a static library using the sources in f2c directory.
file(GLOB f2c_sources LIST_DIRECTORIES false ${CMAKE_CURRENT_SOURCE_DIR}/f2c/*.c)
add_library(f2c STATIC ${f2c_sources})
target_compile_options(f2c
                        PRIVATE
                        # load-var-for,COPTFLAGS
                        ${COPTFLAGS}
                        # get-noopt-cflags-for
                        ${CDBGFLAGS}
                        ${CWARNFLAGS}
                        ${CPICFLAGS}
                        ${CMISCFLAGS}
                        ${CLANGFLAGS}
                        # Suppress warnings about uninitialized functions
                        -Wno-maybe-uninitialized -Wno-parentheses -Wfatal-errors
                    )
target_compile_definitions(f2c
                            PRIVATE
                            # in get-noopt-cflags-for
                            ${VERS_DEF}
                            ${CPPROCFLAGS}
                            -DHAVE_BLIS_H
                        )
target_include_directories(f2c
                        BEFORE
                        PRIVATE
                        # Add local header paths
                        ${CMAKE_CURRENT_SOURCE_DIR}/f2c
                        # and the path to blis.h
                        ${INC_PATH}
                    )
target_link_libraries(f2c PRIVATE ${LDFLAGS})
if(THREADING_MODEL STREQUAL "openmp")
    target_link_libraries(f2c PRIVATE OpenMP::OpenMP_C)
endif()
# Put all those targets under blastest-targets-targets folder name so that they appear all together in IDE.
set_target_properties(f2c PROPERTIES FOLDER blastest-targets)
add_dependencies(f2c flat-header)

# Gather all local source files.
file(GLOB blastest_sources LIST_DIRECTORIES false ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
list(TRANSFORM blastest_sources REPLACE ${CMAKE_CURRENT_SOURCE_DIR}/src/ "")

# Create one executable for each of the sources.
foreach(source ${blastest_sources})
    string(REPLACE .c "" exec_name ${source})
    add_executable(${exec_name}.x src/${source})
    target_compile_options(${exec_name}.x
                            PRIVATE
                            # load-var-for,COPTFLAGS
                            ${COPTFLAGS}
                            # get-noopt-cflags-for
                            ${CDBGFLAGS}
                            ${CWARNFLAGS}
                            ${CPICFLAGS}
                            ${CMISCFLAGS}
                            ${CLANGFLAGS}
                            # Suppress warnings about uninitialized functions
                            -Wno-parentheses -Wno-maybe-uninitialized 
                        )
    target_compile_definitions(${exec_name}.x
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${VERS_DEF}
                                ${CPPROCFLAGS}
                                -DHAVE_BLIS_H
                            )
    target_include_directories(${exec_name}.x
                            BEFORE
                            PRIVATE
                            # Add local header paths
                            ${CMAKE_CURRENT_SOURCE_DIR}/f2c
                            # and the path to blis.h
                            ${INC_PATH}
                        )
    target_link_libraries(${exec_name}.x PRIVATE f2c libblis ${LDFLAGS})
    if(THREADING_MODEL STREQUAL "openmp")
        target_link_libraries(${exec_name}.x PRIVATE OpenMP::OpenMP_C)
    endif()
    set_target_properties(${exec_name}.x PROPERTIES CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # Put all those targets under blastest-targets-targets folder name so that they appear all together in IDE.
    set_target_properties(${exec_name}.x PROPERTIES FOLDER blastest-targets)
    # Add a target for running the tests. Rules are different for level-1 APIs, compared to levels 2 and 3.
    if(${exec_name} MATCHES 1)
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/out.${exec_name}
                            COMMAND ${exec_name}.x > ${CMAKE_BINARY_DIR}/out.${exec_name}
                            COMMENT "Running ${exec_name}.x with output redirected to ${CMAKE_BINARY_DIR}/out.${exec_name}"
                            DEPENDS ${exec_name}.x
                            WORKING_DIRECTORY $<TARGET_FILE_DIR:libblis>
                            VERBATIM
                            )
    else()# name has 2 or 3
        add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/out.${exec_name}
                            COMMAND ${exec_name}.x < ${CMAKE_CURRENT_SOURCE_DIR}/input/${exec_name}.in
                            COMMENT "Running ${exec_name}.x with output saved to ${CMAKE_BINARY_DIR}/out.${exec_name}"
                            DEPENDS ${exec_name}.x
                            WORKING_DIRECTORY $<TARGET_FILE_DIR:libblis>
                            VERBATIM
                        )
    endif()
    add_custom_target(run-${exec_name} DEPENDS ${CMAKE_BINARY_DIR}/out.${exec_name})
    # Put all those targets under blastest-targets-targets folder name so that they appear all together in IDE.
    set_target_properties(run-${exec_name} PROPERTIES FOLDER blastest-targets)
    list(APPEND test_executables "run-${exec_name}")
endforeach()

add_custom_target(testblas DEPENDS ${test_executables})
add_custom_target(checkblas
                    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/cmake/check-blastest.py ${CMAKE_BINARY_DIR}
                    DEPENDS testblas
                )
# Put all those targets under blastest-targets-targets folder name so that they appear all together in IDE.
set_target_properties(testblas checkblas PROPERTIES FOLDER blastest-targets)
