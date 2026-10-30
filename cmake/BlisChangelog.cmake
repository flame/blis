# Regenerates CHANGELOG from git history (make's `make changelog`).
execute_process(
    COMMAND "${GIT}" -C "${SRC}" log --decorate
    OUTPUT_FILE "${OUT}"
    RESULT_VARIABLE rv
)
if(NOT rv EQUAL 0)
    file(REMOVE "${OUT}")
    message(FATAL_ERROR "changelog generation failed (${rv})")
endif()
