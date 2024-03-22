##Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.##
# Comments:
# Set the path to the BLIS installation.
set(BLIS_INSTALL_PATH "" CACHE STRING "Setting the path to a BLIS installation that needs testing.")
if(BLIS_INSTALL_PATH)
    message(STATUS "BLIS_INSTALL_PATH :" ${BLIS_INSTALL_PATH})
endif()

# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.

#if(NOT DEFINED BLIS_INSTALL_PATH)
if(BLIS_INSTALL_PATH STREQUAL "")
    set(DIST_PATH ${CMAKE_BINARY_DIR})
    set(LIB_PATH ${DIST_PATH}/lib/${BLIS_CONFIG_FAMILY})
    set(INC_PATH ${DIST_PATH}/include/${BLIS_CONFIG_FAMILY})
    set(CINFLAGS ${INC_PATH})
    set(LIBBLIS  ${libblis_link})
else()
    set(LIB_PATH ${BLIS_INSTALL_PATH}/lib)
    set(INC_PATH ${BLIS_INSTALL_PATH}/include)
    set(CINFLAGS ${INC_PATH})
    # Set up the library name.
    if(WIN32)
        set(LIB_BLIS AOCL-LibBlis-Win)
    else()
        set(LIB_BLIS ${libblis_link})
    endif()
    # Append if threading is required.
    if(NOT (ENABLE_THREADING STREQUAL "no"))
        if(WIN32)
            string(APPEND LIB_BLIS -MT)
        else()
            string(APPEND LIB_BLIS -mt)
        endif()
    endif()
    # Append for dll if necessary.
    if(WIN32 AND BUILD_SHARED_LIBS)
        string(APPEND LIB_BLIS -dll)
    endif()
    # Setting the suffix for find_library().
    if(WIN32)
        string(APPEND LIB_BLIS .lib)
    else()
        if(BUILD_SHARED_LIBS)
            string(APPEND LIB_BLIS .so)
        else()
            string(APPEND LIB_BLIS .a)
        endif()
    endif()
    set(LIBBLIS  ${LIB_PATH}/${LIB_BLIS})
    message(STATUS "BLIS_INSTALL_PATH : " ${LIBBLIS})
endif()

if(WIN32)
    set(LIBSUFFIX lib)
else()
    set(LIBSUFFIX so)
endif()
set(CMAKE_EXECUTABLE_SUFFIX ".x")
set(MKL_PATH $ENV{MKLROOT} CACHE STRING "Set MKL_PATH.")
if(WIN32)
    set(mkllib "${MKL_PATH}\\mkl_rt.lib" CACHE STRING "Set MKL_PATH.")
else()
    set(mkllib "${MKL_PATH}/libmkl_rt.so" CACHE STRING "Set MKL_PATH.")
endif()
set(MKL_LIB ${mkllib})
set(OPENBLAS_PATH "/home/amd/mylibs/openblas" CACHE STRING "Set OPENBLAS_PATH.")
set(OPENBLAS_LIB "${OPENBLAS_PATH}/libopenblas.${LIBSUFFIX}")


# Include the corresponding make_defs.cmake that holds the required compiler options.
include(${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY}/make_defs.cmake)

# Gather all local source files.
file(GLOB file_list LIST_DIRECTORIES false RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" "*.c")

# Create an executable using the sources above.
function(testexe extn)
    set(dblas "aocl")
    if(extn STREQUAL "mkl")
        set(BLAS_LIBS ${MKL_LIB})
        set(dblas ${extn})
    elseif(extn STREQUAL "openblas")
        set(BLAS_LIBS ${OPENBLAS_LIB})
        set(dblas ${extn})
    endif()
    set(TEST_FLAGS -DBLAS="${dblas}")
    foreach(src ${file_list})
        string(REGEX REPLACE ".c$" "" exec_name ${src})
        set(exec_name "${exec_name}_${extn}")
        add_executable(${exec_name} ${src})
        target_compile_options(${exec_name}
                                PRIVATE
                                # load-var-for,COPTFLAGS
                                ${COPTFLAGS}
                            )
        if(WIN32 AND BUILD_SHARED_LIBS)
            target_compile_definitions(${exec_name}
                                        PRIVATE
                                        # in get-noopt-cflags-for
                                        ${VERS_DEF}
                                        "-DBLIS_EXPORT=__declspec(dllimport)"
                                        ${TEST_FLAGS}
                                    )
        else()
            target_compile_definitions(${exec_name}
                                        PRIVATE
                                        # in get-noopt-cflags-for
                                        ${VERS_DEF}
                                        ${TEST_FLAGS}
                                    )
        endif()
        target_include_directories(${exec_name}
                                BEFORE
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CINFLAGS}
                            )
        target_link_libraries(${exec_name} PRIVATE ${BLAS_LIBS} ${LIBBLIS} ${LDFLAGS})
        if(THREADING_MODEL STREQUAL "openmp")
            target_link_libraries(${exec_name} PRIVATE OpenMP::OpenMP_C)
        endif()
        list(APPEND temp_executables ${exec_name})
    endforeach()
     set(test_executables ${temp_executables} PARENT_SCOPE)
endfunction()

testexe("blas")
add_custom_target(test_blis DEPENDS ${test_executables})
testexe("mkl")
add_custom_target(test_mkl DEPENDS ${test_executables})
testexe("openblas")
add_custom_target(test_openblas DEPENDS ${test_executables})
add_custom_target(testall DEPENDS test_blis test_mkl test_openblas)

# Put all those targets under test-targets folder name so that they appear all together in IDE.
set_target_properties(testall test_blis test_mkl test_openblas PROPERTIES FOLDER test-targets)