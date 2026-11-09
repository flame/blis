# The script expects a make-configured tree (config.mk + include/<config>/blis.h),
# so we create a minimal shim layout.
set(shim "${CMAKE_CURRENT_BINARY_DIR}/symbols-shim")
file(REMOVE_RECURSE "${shim}")
file(MAKE_DIRECTORY "${shim}/include/${CFG}")
file(WRITE "${shim}/config.mk" "CC := ${CC_BIN}\nCONFIG_NAME := ${CFG}\n")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E create_symlink "${FLAT}" "${shim}/include/${CFG}/blis.h"
)
execute_process(COMMAND sh "${SCRIPT}" WORKING_DIRECTORY "${shim}" OUTPUT_FILE "${OUT}" RESULT_VARIABLE rv)
file(REMOVE_RECURSE "${shim}")
if(NOT rv EQUAL 0)
    file(REMOVE "${OUT}")
    message(FATAL_ERROR "symbols generation failed (${rv})")
endif()
