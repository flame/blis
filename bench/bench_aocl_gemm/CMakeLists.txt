##Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.##

# Comments:
# Gather all local source files.
file(GLOB file_list LIST_DIRECTORIES false RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" "*.c")

set(LPGEMM_FLAGS -DBLAS="aocl" -DN_REPEAT=${NREPEATS} -DINT_FS="%ld" -DUINT_FS="%lu")
# Create an executable using the sources above.
function(lpgemmbenchexe extn)
    foreach(src ${file_list})
        string(REGEX REPLACE ".c$" "" exec_name ${src})
        set(exec_name "${exec_name}_${extn}")
        add_executable(${exec_name}.x ${src})
        target_compile_options(${exec_name}.x
                                PRIVATE
                                # load-var-for,COPTFLAGS
                                ${COPTFLAGS}
                            )
        if(WIN32 AND BUILD_SHARED_LIBS)
            target_compile_definitions(${exec_name}.x
                                        PRIVATE
                                        # in get-noopt-cflags-for
                                        ${VERS_DEF}
                                        "-DBLIS_EXPORT=__declspec(dllimport)"
                                        ${LPGEMM_FLAGS}
                                    )
        else()
            target_compile_definitions(${exec_name}.x
                                        PRIVATE
                                        # in get-noopt-cflags-for
                                        ${VERS_DEF}
                                        ${LPGEMM_FLAGS}
                                    )
        endif()
        target_include_directories(${exec_name}.x
                                BEFORE
                                PRIVATE
                                # in get-noopt-cflags-for
                                ${CINFLAGS}
                            )
        target_link_libraries(${exec_name}.x PRIVATE ${LIBBLIS} ${LDFLAGS})
        if(THREADING_MODEL STREQUAL "openmp")
            target_link_libraries(${exec_name}.x PRIVATE OpenMP::OpenMP_C)
        endif()
        list(APPEND temp_executables ${exec_name}.x)
    endforeach()
     set(bench_executables ${temp_executables} PARENT_SCOPE)
endfunction()

lpgemmbenchexe("blis")
add_custom_target(lpgemm_blis DEPENDS ${bench_executables})
add_custom_target(benchmark_lpgemm DEPENDS lpgemm_blis)

# Put all those targets under bench_aocl_gemm-targets folder name so that they appear all together in IDE.
set_target_properties(benchmark_lpgemm lpgemm_blis PROPERTIES FOLDER bench_aocl_gemm-targets)
