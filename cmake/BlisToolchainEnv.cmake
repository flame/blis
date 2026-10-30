# configure honors AR, RANLIB and PYTHON from the environment, splits the value
# on whitespace (program first, then arguments), records the raw string in
# config.mk and exits 1 when the program is missing. Do the same: seed the cache
# before project() so CMake's own find_program keeps the program, keep the raw
# string in the cache entry BLIS_<T>_CMD (CMake's automatic reconfigure runs
# without the environment), and put the arguments into the archive rules and the
# python commands below. An explicit -D value wins and takes no arguments.
# shortcut: arguments are split on whitespace like configure's unquoted ${tool}; no shell quoting.
set(_blis_envs AR RANLIB PYTHON)
set(_blis_vars CMAKE_AR CMAKE_RANLIB Python3_EXECUTABLE)
set(_blis_descs "library archiver" "archive indexer" "python interpreter")
foreach(_t _v _d IN ZIP_LISTS _blis_envs _blis_vars _blis_descs)
    get_property(_blis_help CACHE ${_v} PROPERTY HELPSTRING)
    # Seed when unset, or again when an earlier run seeded it (reconfigure keeps the arguments).
    if(NOT "$ENV{${_t}}" STREQUAL "" AND (NOT DEFINED ${_v} OR _blis_help STREQUAL "${_t} from the environment"))
        separate_arguments(_blis_words UNIX_COMMAND "$ENV{${_t}}")
        list(POP_FRONT _blis_words _blis_prog)
        # Unset after the copy: each configure must re-search, and the cache floor
        # (3.20) has no find_program NO_CACHE (3.21).
        find_program(_blis_env_${_t} NAMES "${_blis_prog}")
        set(_blis_prog_path "${_blis_env_${_t}}")
        unset(_blis_env_${_t} CACHE)
        if(NOT _blis_prog_path)
            message(
                FATAL_ERROR
                "BLIS: *** Could not find the ${_d} specified via ${_t} ('$ENV{${_t}}').\n"
                "BLIS: *** A working ${_d} is required. Please set ${_t}\n"
                "BLIS: *** to a ${_d} that exists (or unset ${_t})."
            )
        endif()
        set(${_v} "${_blis_prog_path}" CACHE FILEPATH "${_t} from the environment" FORCE)
        set(BLIS_${_t}_CMD "$ENV{${_t}}" CACHE INTERNAL "raw ${_t} command from the environment")
    elseif(DEFINED BLIS_${_t}_CMD AND NOT _blis_help STREQUAL "${_t} from the environment")
        # An explicit -D value now selects the program; its seeded arguments do not apply.
        unset(BLIS_${_t}_CMD CACHE)
    endif()
endforeach()
# Capture the raw -D CMAKE_C_COMPILER so config.mk records the value the user
# gave; project() below resolves CMAKE_C_COMPILER to an absolute path.
if(DEFINED CMAKE_C_COMPILER AND NOT DEFINED BLIS_CC_CMD)
    set(BLIS_CC_CMD "${CMAKE_C_COMPILER}" CACHE INTERNAL "raw CC command from -D")
endif()
# configure honors LIBPTHREAD from the environment (default -lpthread); keep it
# in the cache so an automatic reconfigure without the environment keeps the
# value, and let an explicit -D win.
get_property(_blis_pth_help CACHE LIBPTHREAD PROPERTY HELPSTRING)
if(DEFINED ENV{LIBPTHREAD} AND (NOT DEFINED LIBPTHREAD OR _blis_pth_help STREQUAL "LIBPTHREAD from the environment"))
    set(LIBPTHREAD "$ENV{LIBPTHREAD}" CACHE STRING "LIBPTHREAD from the environment")
endif()
# Arguments of AR and RANLIB go into every archive rule (CMake's default rules
# take the program only); the rules must exist before project() loads the languages.
foreach(_t _k IN ZIP_LISTS _blis_envs _blis_vars)
    if(_t STREQUAL "PYTHON")
        continue()
    endif()
    separate_arguments(_blis_words UNIX_COMMAND "${BLIS_${_t}_CMD}")
    list(POP_FRONT _blis_words)
    string(JOIN " " _blis_args ${_blis_words})
    if(NOT "${_blis_args}" STREQUAL "")
        foreach(_l C CXX)
            if(_t STREQUAL "AR")
                set(CMAKE_${_l}_ARCHIVE_CREATE "<CMAKE_AR> ${_blis_args} qc <TARGET> <LINK_FLAGS> <OBJECTS>")
                set(CMAKE_${_l}_ARCHIVE_APPEND "<CMAKE_AR> ${_blis_args} q <TARGET> <LINK_FLAGS> <OBJECTS>")
            else()
                set(CMAKE_${_l}_ARCHIVE_FINISH "<CMAKE_RANLIB> ${_blis_args} <TARGET>")
            endif()
        endforeach()
    endif()
endforeach()
separate_arguments(BLIS_PYTHON_ARGS UNIX_COMMAND "${BLIS_PYTHON_CMD}")
list(POP_FRONT BLIS_PYTHON_ARGS)

# config.mk records the raw AR, RANLIB and PYTHON strings, as configure does;
# BlisMakeShare reads the CMake variables, so shadow them for its include.
macro(blis_raw_tools_begin)
    foreach(_t _v IN ZIP_LISTS _blis_envs _blis_vars)
        if(DEFINED BLIS_${_t}_CMD)
            set(${_v} "${BLIS_${_t}_CMD}")
        endif()
    endforeach()
endmacro()
macro(blis_raw_tools_end)
    foreach(_t _v IN ZIP_LISTS _blis_envs _blis_vars)
        if(DEFINED BLIS_${_t}_CMD)
            unset(${_v})
        endif()
    endforeach()
endmacro()
