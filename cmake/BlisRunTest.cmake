# Test runner for the BLIS testsuite.

set(_args)
foreach(_n RANGE 1 20)
    if(DEFINED arg${_n})
        list(APPEND _args "${arg${_n}}")
    else()
        break()
    endif()
endforeach()

if(NOT EXE)
    message(FATAL_ERROR "EXE must be specified")
endif()

if(NOT WORK_DIR)
    set(WORK_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

set(_cmd "${EXE}" ${_args})
if(LAUNCHER)
    separate_arguments(_launch_list NATIVE_COMMAND "${LAUNCHER}")
    list(LENGTH _launch_list _launch_len)
    if(_launch_len GREATER 0)
        list(INSERT _cmd 0 ${_launch_list})
    endif()
endif()

execute_process(
    COMMAND ${_cmd}
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE _exit_code
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(NOT (_exit_code EQUAL 0 AND "${_output}" MATCHES "Exiting normally" AND NOT "${_output}" MATCHES "FAILURE"))
    if(NOT _exit_code EQUAL 0)
        message(STATUS "Test failed: exit code ${_exit_code}")
    elseif(NOT "${_output}" MATCHES "Exiting normally")
        message(STATUS "Test failed: missing 'Exiting normally' in output")
    else()
        message(STATUS "Test failed: 'FAILURE' found in output")
    endif()
    message(FATAL_ERROR "Test failed")
endif()
