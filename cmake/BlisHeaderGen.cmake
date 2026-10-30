macro(blis_config_headers)
    string(TOUPPER "${BLIS_CONFIG_NAME}" _cn_upper)
    set(config_name_define "#define BLIS_FAMILY_${_cn_upper}\n")
    set(config_list_defines "")
    foreach(c ${BLIS_CONFIG_LIST})
        string(TOUPPER "${c}" CU)
        string(APPEND config_list_defines "#define BLIS_CONFIG_${CU}\n")
    endforeach()
    set(kernel_list_defines "")
    foreach(k ${BLIS_KERNEL_LIST})
        string(TOUPPER "${k}" KU)
        string(APPEND kernel_list_defines "#define BLIS_KERNELS_${KU}\n")
    endforeach()

    # Template anchors bound by BlisVersion/BlisOptions:
    set(version "${BLIS_VERSION}")
    set(version_major "${BLIS_VERSION_MAJOR}")
    set(version_minor "${BLIS_VERSION_MINOR}")
    set(version_revision "${BLIS_VERSION_REVISION}")

    _blis_probe_omp_simd(_omp_simd_ok)
    if(_omp_simd_ok)
        set(enable_pragma_omp_simd 1)
        set(BLIS_PRAGMA_OMP_SIMD yes)
    else()
        set(enable_pragma_omp_simd 0)
        set(BLIS_PRAGMA_OMP_SIMD no)
    endif()
    if(BLIS_PRAGMA_OMP_SIMD AND NOT BLIS_CC_VENDOR MATCHES "gcc|clang|icc")
        set(enable_pragma_omp_simd 0)
    endif()

    configure_file("${BLIS_SOURCE_DIR}/build/bli_config.h.in" "${PROJECT_BINARY_DIR}/bli_config.h" @ONLY)
    configure_file("${BLIS_SOURCE_DIR}/build/bli_addon.h.in" "${PROJECT_BINARY_DIR}/bli_addon.h" @ONLY)
endmacro()

macro(blis_flat_headers_gen)
    blis_header_dirs(FLAT_HEADER_DIRS "${BLIS_SOURCE_DIR}/frame" "${BLIS_SOURCE_DIR}/ref_kernels")
    blis_collect(ALL_HEADERS "${BLIS_SOURCE_DIR}/frame" h)
    blis_collect(_h1 "${BLIS_SOURCE_DIR}/ref_kernels" h)
    list(APPEND ALL_HEADERS ${_h1})
    foreach(c ${BLIS_CONFIG_LIST_FAM})
        blis_header_dirs(_hdirs "${BLIS_SOURCE_DIR}/config/${c}")
        list(APPEND FLAT_HEADER_DIRS ${_hdirs})
        blis_collect(_h1 "${BLIS_SOURCE_DIR}/config/${c}" h)
        list(APPEND ALL_HEADERS ${_h1})
    endforeach()
    foreach(k ${BLIS_KERNEL_LIST})
        blis_header_dirs(_hdirs "${BLIS_SOURCE_DIR}/kernels/${k}")
        list(APPEND FLAT_HEADER_DIRS ${_hdirs})
        blis_collect(_h1 "${BLIS_SOURCE_DIR}/kernels/${k}" h)
        list(APPEND ALL_HEADERS ${_h1})
    endforeach()
    foreach(a ${BLIS_ENABLE_ADDONS})
        blis_header_dirs(_hdirs "${BLIS_SOURCE_DIR}/addon/${a}")
        list(APPEND FLAT_HEADER_DIRS ${_hdirs})
        blis_collect(_h1 "${BLIS_SOURCE_DIR}/addon/${a}" h hh hpp hxx)
        list(APPEND ALL_HEADERS ${_h1})
    endforeach()
    if(BLIS_SANDBOX)
        blis_header_dirs(_hdirs "${BLIS_SOURCE_DIR}/sandbox/${BLIS_SANDBOX}")
        list(APPEND FLAT_HEADER_DIRS ${_hdirs})
        blis_collect(_h1 "${BLIS_SOURCE_DIR}/sandbox/${BLIS_SANDBOX}" h hh hpp hxx)
        list(APPEND ALL_HEADERS ${_h1})
    endif()
    # The build directory comes first (it holds the generated bli_config.h and
    # bli_addon.h, like '.' in the make-time in-tree build).
    list(PREPEND FLAT_HEADER_DIRS "${PROJECT_BINARY_DIR}")
    list(REMOVE_DUPLICATES FLAT_HEADER_DIRS)

    if(NOT Python3_Interpreter_FOUND)
        message(FATAL_ERROR "BLIS: python3 is required (build/flatten-headers.py)")
    endif()

    # Keep the paths recorded in #line directives relative to the build dir
    # (like the make-based build), instead of leaking absolute build paths.
    set(_flat_in "${BLIS_SOURCE_DIR}/frame/include/blis.h")
    file(RELATIVE_PATH _flat_in_rel "${PROJECT_BINARY_DIR}" "${_flat_in}")
    set(_flat_dirs_rel "")
    foreach(d ${FLAT_HEADER_DIRS})
        file(RELATIVE_PATH _r "${PROJECT_BINARY_DIR}" "${d}")
        if(_r STREQUAL "")
            set(_r ".")
        endif()
        list(APPEND _flat_dirs_rel "${_r}")
    endforeach()
    string(REPLACE ";" " " FLAT_HEADER_DIRS_STR "${_flat_dirs_rel}")

    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include")
    add_custom_command(
        OUTPUT "${PROJECT_BINARY_DIR}/include/blis.h"
        COMMAND
            ${Python3_EXECUTABLE} ${BLIS_PYTHON_ARGS} "${BLIS_SOURCE_DIR}/build/flatten-headers.py" -l -v1
            "${_flat_in_rel}" "include/blis.h" "include" "${FLAT_HEADER_DIRS_STR}"
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
        DEPENDS
            ${ALL_HEADERS}
            "${BLIS_SOURCE_DIR}/build/flatten-headers.py"
            "${PROJECT_BINARY_DIR}/bli_config.h"
            "${PROJECT_BINARY_DIR}/bli_addon.h"
        COMMENT "Flattening monolithic header blis.h"
    )

    set(_flat_outputs "${PROJECT_BINARY_DIR}/include/blis.h")
    if(BLIS_ENABLE_CBLAS)
        file(RELATIVE_PATH _cblas_in_rel "${PROJECT_BINARY_DIR}" "${BLIS_SOURCE_DIR}/frame/compat/cblas/src/cblas.h")
        add_custom_command(
            OUTPUT "${PROJECT_BINARY_DIR}/include/cblas.h"
            COMMAND
                ${Python3_EXECUTABLE} ${BLIS_PYTHON_ARGS} "${BLIS_SOURCE_DIR}/build/flatten-headers.py" -l -v1
                "${_cblas_in_rel}" "include/cblas.h" "include" "${FLAT_HEADER_DIRS_STR}"
            WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
            DEPENDS ${ALL_HEADERS} "${PROJECT_BINARY_DIR}/include/blis.h" "${BLIS_SOURCE_DIR}/build/flatten-headers.py"
            COMMENT "Flattening monolithic header cblas.h"
        )
        list(APPEND _flat_outputs "${PROJECT_BINARY_DIR}/include/cblas.h")
    endif()
    add_custom_target(blis_flat_headers DEPENDS ${_flat_outputs})

    file(STRINGS "${BLIS_SOURCE_DIR}/ref_kernels/bli_cntx_ref.c" _ref_inc REGEX "^[ \t]*#[ \t]*include")
    set(BLIS_CINCDIRS "")
    foreach(l ${_ref_inc})
        string(REGEX MATCH "[\"<]([a-zA-Z0-9_./-]+)[\">]" _m "${l}")
        if(CMAKE_MATCH_1 AND NOT CMAKE_MATCH_1 STREQUAL "blis.h")
            foreach(hf ${ALL_HEADERS})
                if(hf MATCHES "/${CMAKE_MATCH_1}$")
                    get_filename_component(_d "${hf}" DIRECTORY)
                    list(APPEND BLIS_CINCDIRS "${_d}")
                endif()
            endforeach()
        endif()
    endforeach()
    list(APPEND BLIS_CINCDIRS "${BLIS_SOURCE_DIR}/frame/include")
    if(BLIS_ENABLE_CBLAS)
        list(APPEND BLIS_CINCDIRS "${BLIS_SOURCE_DIR}/frame/compat/cblas/src")
    endif()
    if("hpx" IN_LIST BLIS_THREADING_MODELS)
        list(APPEND BLIS_CINCDIRS ${HPX_INCLUDE_DIRS})
    endif()
    list(REMOVE_DUPLICATES BLIS_CINCDIRS)
endmacro()
