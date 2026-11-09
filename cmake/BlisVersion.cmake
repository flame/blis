# Computes the BLIS version string the same way configure does:
# an exact-match git tag wins; otherwise build/version is used; everything
# from the first '-' onwards is dropped. BLIS_FORCE_VERSION overrides both.
# Sets: BLIS_VERSION, BLIS_VERSION_MAJOR/MINOR/REVISION, BLIS_SO_MAJOR,
#       BLIS_SO_MINORB (so_version minor.build).

find_package(Git QUIET)
if(BLIS_FORCE_VERSION)
    set(_version_raw "${BLIS_FORCE_VERSION}")
else()
    set(_version_raw "")
    if(GIT_FOUND AND EXISTS "${BLIS_SOURCE_DIR}/.git")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${BLIS_SOURCE_DIR}" describe --tags --exact-match
            OUTPUT_VARIABLE _version_raw
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()
    if(NOT _version_raw)
        file(STRINGS "${BLIS_SOURCE_DIR}/build/version" _version_raw LIMIT_COUNT 1)
    endif()
endif()

string(REGEX REPLACE "-.*$" "" BLIS_VERSION "${_version_raw}")
string(REPLACE "." ";" _vparts "${BLIS_VERSION}")
list(GET _vparts 0 BLIS_VERSION_MAJOR)
list(GET _vparts 1 BLIS_VERSION_MINOR)
list(LENGTH _vparts _vlen)
if(_vlen GREATER 2)
    list(GET _vparts 2 BLIS_VERSION_REVISION)
else()
    set(BLIS_VERSION_REVISION 0)
endif()

file(STRINGS "${BLIS_SOURCE_DIR}/build/so_version" _so_lines)
list(GET _so_lines 0 BLIS_SO_MAJOR)
list(GET _so_lines 1 BLIS_SO_MINORB)
set(BLIS_SO_MMB "${BLIS_SO_MAJOR}.${BLIS_SO_MINORB}")
