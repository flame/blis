# Runs one blastest driver in SCRATCH and mirrors blastest/check-blastest.sh.

# Always start from a clean scratch: a stale out.* from a previously crashed
# run must not false-fail this one.
file(REMOVE_RECURSE "${SCRATCH}")
file(MAKE_DIRECTORY "${SCRATCH}")
set(out "${SCRATCH}/out.driver")

if(LEVEL STREQUAL "1")
    execute_process(COMMAND "${EXE}" RESULT_VARIABLE rv OUTPUT_FILE "${out}" ERROR_QUIET)
else()
    execute_process(
        COMMAND "${EXE}"
        RESULT_VARIABLE rv
        INPUT_FILE "${INPUT}"
        OUTPUT_FILE "${out}"
        ERROR_QUIET
        WORKING_DIRECTORY "${SCRATCH}"
    )
endif()
if(NOT rv EQUAL 0)
    message(FATAL_ERROR "${EXE} exited with status ${rv}")
endif()

# Drivers write their summaries into files named by the first record of the
# .in file (out.<drvname> in SCRATCH); level-1 goes to stdout (out.driver).
file(GLOB summaries "${SCRATCH}/out.*")
set(failed FALSE)
foreach(s ${summaries})
    file(STRINGS "${s}" bad REGEX "\\*\\*\\*\\*")
    if(bad)
        set(failed TRUE)
        message(STATUS "failures found in ${s}:\n${bad}")
    endif()
endforeach()
if(failed)
    message(FATAL_ERROR "BLAS test driver reported failures (see ${SCRATCH}/out.*)")
endif()
