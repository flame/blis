# Semantics mirror testsuite/check-blistest.sh and blastest/check-blastest.sh.

# Same meaning as make's TESTSUITE_WRAPPER: a command that prefixes every
# testsuite and blastest run (for example `qemu-aarch64 -L /usr/aarch64-linux-gnu`).
set(TESTSUITE_WRAPPER "$ENV{TESTSUITE_WRAPPER}" CACHE STRING "Command that prefixes every testsuite and blastest run")

# One launcher per test command: TESTSUITE_WRAPPER replaces the cross-compiling
# emulator, as make's TESTSUITE_WRAPPER does, so a command never gets two prefixes.
if(TESTSUITE_WRAPPER)
    set(_blis_launcher "${TESTSUITE_WRAPPER}")
elseif(CMAKE_CROSSCOMPILING AND CMAKE_CROSSCOMPILING_EMULATOR)
    list(JOIN CMAKE_CROSSCOMPILING_EMULATOR " " _blis_launcher)
else()
    set(_blis_launcher "")
endif()

function(blis_register_testsuite_tests)
    set(_variants
        "full:testsuite/input.general:testsuite/input.operations"
        "fast:testsuite/input.general.fast:testsuite/input.operations.fast"
        "mixed:testsuite/input.general.mixed:testsuite/input.operations.mixed"
        "salt:testsuite/input.general.salt:testsuite/input.operations.salt"
    )
    foreach(v ${_variants})
        string(REPLACE ":" ";" parts "${v}")
        list(GET parts 0 name)
        list(GET parts 1 gen)
        list(GET parts 2 ops)
        set(_exe "$<TARGET_FILE:test_libblis.x>")
        add_test(
            NAME blis-test-${name}
            COMMAND
                ${CMAKE_COMMAND} "-DEXE=${_exe}" "-DLAUNCHER=${_blis_launcher}" -Darg1=-g
                -Darg2=${BLIS_SOURCE_DIR}/${gen} -Darg3=-o -Darg4=${BLIS_SOURCE_DIR}/${ops}
                -DWORK_DIR=${PROJECT_BINARY_DIR} -P ${BLIS_SOURCE_DIR}/cmake/BlisRunTest.cmake
        )
        set_tests_properties(blis-test-${name} PROPERTIES LABELS "${name}")
        if(WIN32)
            # The testsuite links the shared library on Windows; add its dir to PATH.
            set_tests_properties(blis-test-${name} PROPERTIES ENVIRONMENT "PATH=${PROJECT_BINARY_DIR}/lib\;$ENV{PATH}")
        endif()
    endforeach()
endfunction()

function(blis_register_blastest_tests)
    foreach(p s d c z)
        foreach(l 1 2 3)
            set(drv "${p}blat${l}")
            set(_scratch "${PROJECT_BINARY_DIR}/blastest_run/${drv}")
            file(MAKE_DIRECTORY "${_scratch}")
            set(_exe "$<TARGET_FILE:${drv}.x>")
            if(_blis_launcher)
                # The runner executes a single file, so hand it a launcher script.
                set(_exe "${PROJECT_BINARY_DIR}/blastest_run/${drv}.sh")
                file(
                    GENERATE OUTPUT "${_exe}"
                    CONTENT "#!/bin/sh\nexec ${_blis_launcher} $<TARGET_FILE:${drv}.x> \"$@\"\n"
                    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                )
            endif()
            add_test(
                NAME blastest-${drv}
                COMMAND
                    ${CMAKE_COMMAND} -DEXE=${_exe} -DINPUT=${BLIS_SOURCE_DIR}/blastest/input/${drv}.in -DLEVEL=${l}
                    -DSCRATCH=${_scratch} -P ${BLIS_SOURCE_DIR}/cmake/BlisRunBlastest.cmake
            )
            set_tests_properties(blastest-${drv} PROPERTIES LABELS "blastest")
        endforeach()
    endforeach()
endfunction()
