##Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.##

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
file(GLOB testsuite_sources LIST_DIRECTORIES false ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
set(CINFLAGS ${INC_PATH})

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
target_link_libraries(test_libblis.x PRIVATE ${LDFLAGS} libblis)
if(THREADING_MODEL STREQUAL "openmp")
    target_link_libraries(test_libblis.x PRIVATE OpenMP::OpenMP_C)
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
                        WORKING_DIRECTORY $<TARGET_FILE_DIR:libblis>
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
