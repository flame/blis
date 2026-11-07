if(BLIS_ENABLE_STATIC)
    set(_export_name_static "blis-static")
    if(NOT BLIS_ENABLE_SHARED)
        set(_export_name_static "blis")
    endif()
    set_target_properties(blis_static PROPERTIES EXPORT_NAME ${_export_name_static})
    install(TARGETS blis_static EXPORT blisTargets ARCHIVE DESTINATION ${BLIS_INSTALL_LIBDIR} COMPONENT libs)
endif()
if(BLIS_ENABLE_SHARED)
    set_target_properties(blis_shared PROPERTIES EXPORT_NAME blis)
    # The double install is deliberate: the components feed make's partial install targets (install-libs gets the real file, install-lib-symlinks the symlinks).
    install(
        TARGETS blis_shared
        EXPORT blisTargets
        LIBRARY DESTINATION ${BLIS_INSTALL_LIBDIR} COMPONENT lib-symlinks
        RUNTIME DESTINATION ${BLIS_INSTALL_LIBDIR} COMPONENT lib-symlinks
        ARCHIVE DESTINATION ${BLIS_INSTALL_LIBDIR} COMPONENT lib-symlinks
    )
    if(NOT WIN32)
        install(
            FILES $<TARGET_FILE:blis_shared>
            DESTINATION ${BLIS_INSTALL_LIBDIR}
            COMPONENT libs
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
    endif()
    if(APPLE AND NOT BLIS_DARWIN_RPATH)
        # The link stamps the configure-time libdir, as make does. This rewrite
        # follows the effective prefix with the blis.pc mechanism; DESTDIR stays out.
        # A plain install matches the stamped id, so it skips the rewrite.
        if(IS_ABSOLUTE "${BLIS_INSTALL_LIBDIR}")
            set(_blis_iddir "${BLIS_INSTALL_LIBDIR}")
            set(_blis_idcfgdir "${BLIS_INSTALL_LIBDIR}")
        else()
            set(_blis_iddir "@BLIS_INSTALL_PREFIX@/${BLIS_INSTALL_LIBDIR}")
            set(_blis_idcfgdir "${CMAKE_INSTALL_PREFIX}/${BLIS_INSTALL_LIBDIR}")
        endif()
        set(_blis_idtool "${CMAKE_INSTALL_NAME_TOOL}")
        if(_blis_idtool MATCHES "NOTFOUND")
            set(_blis_idtool "")
        elseif(NOT IS_ABSOLUTE "${_blis_idtool}")
            # find_program keeps its result when the result variable has a value.
            # The minimum is 3.20 and NO_CACHE is a 3.21 option. Clear the normal and cache entry.
            unset(_blis_idtool_search)
            unset(_blis_idtool_search CACHE)
            find_program(_blis_idtool_search "${_blis_idtool}")
            if(_blis_idtool_search MATCHES "NOTFOUND")
                set(_blis_idtool "")
            else()
                set(_blis_idtool "${_blis_idtool_search}")
            endif()
        endif()
        foreach(_blis_idc libs lib-symlinks)
            install(
                CODE
                    "set(_blis_iddir \"${_blis_iddir}\")\nset(_blis_idcfgdir \"${_blis_idcfgdir}\")\nset(_blis_idtool \"${_blis_idtool}\")\nset(_blis_dylib \"$<TARGET_FILE_NAME:blis_shared>\")\nset(_blis_idname \"$<TARGET_SONAME_FILE_NAME:blis_shared>\")"
                COMPONENT ${_blis_idc}
            )
            install(
                CODE
                    [[
    string(REPLACE "@BLIS_INSTALL_PREFIX@" "${CMAKE_INSTALL_PREFIX}" _blis_iddir "${_blis_iddir}")
    set(_blis_idfile "$ENV{DESTDIR}${_blis_iddir}/${_blis_dylib}")
    # IS_EXECUTABLE is a 3.29 condition; the minimum is 3.20. EXISTS catches an
    # empty or missing tool. A tool that exists but cannot start fails below.
    if(NOT EXISTS "${_blis_idtool}")
        message(FATAL_ERROR "BLIS: install-name tool '${_blis_idtool}' is empty or missing; the id of ${_blis_dylib} keeps the stale value ${_blis_idcfgdir}/${_blis_idname}")
    endif()
    if(EXISTS "${_blis_idfile}" AND NOT IS_SYMLINK "${_blis_idfile}")
        execute_process(
            COMMAND "${_blis_idtool}" -id "${_blis_iddir}/${_blis_idname}" "${_blis_idfile}"
            RESULT_VARIABLE _blis_idres
        )
        if(NOT _blis_idres EQUAL 0)
            message(FATAL_ERROR "BLIS: install-name tool '${_blis_idtool}' -id failed for ${_blis_idfile}; the id keeps the stale value ${_blis_idcfgdir}/${_blis_idname}")
        endif()
    endif()
  ]]
                COMPONENT ${_blis_idc}
            )
        endforeach()
    endif()
endif()

# Generated monolithic headers + one-level-up helper headers.
install(FILES "${PROJECT_BINARY_DIR}/include/blis.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/blis COMPONENT headers)
install(FILES "${BLIS_SOURCE_DIR}/build/blis.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT helper-headers)
if(BLIS_ENABLE_CBLAS)
    install(
        FILES "${PROJECT_BINARY_DIR}/include/cblas.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/blis
        COMPONENT headers
    )
    install(FILES "${BLIS_SOURCE_DIR}/build/cblas.h" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT helper-headers)
endif()

install(EXPORT blisTargets NAMESPACE blis:: DESTINATION ${BLIS_INSTALL_LIBDIR}/cmake/blis COMPONENT cmake-package)
if(IS_ABSOLUTE "${BLIS_INSTALL_LIBDIR}")
    install(CODE "set(_blis_tgtdir \"${BLIS_INSTALL_LIBDIR}/cmake/blis\")" COMPONENT cmake-package)
    install(
        CODE
            [[
    set(_f "$ENV{DESTDIR}${_blis_tgtdir}/blisTargets.cmake")
    file(READ "${_f}" _c)
    string(REGEX REPLACE "set\\(_IMPORT_PREFIX \"[^\"]*\"\\)"
           "set(_IMPORT_PREFIX \"${CMAKE_INSTALL_PREFIX}\")" _c "${_c}")
    file(WRITE "${_f}" "${_c}")
  ]]
        COMPONENT cmake-package
    )
endif()

# Per-subconfiguration flag metadata exported for plugins (consumed by BlisPluginHelpers.cmake).
set(BLISPLUGIN_EXPORTS "")
blis_get_cflags(${BLIS_CONFIG_NAME} addon _blisplugin_frame)
string(REPLACE ";" " " _blisplugin_frame_s "${_blisplugin_frame}")
string(APPEND BLISPLUGIN_EXPORTS "set(blis_FRAME_CFLAGS \"${_blisplugin_frame_s}\")\n")
foreach(c ${BLIS_CONFIG_LIST})
    blis_get_cflags(${c} kernel _kf)
    string(REPLACE ";" " " _kfs "${_kf}")
    string(APPEND BLISPLUGIN_EXPORTS "set(blis_KERNEL_CFLAGS_${c} \"${_kfs}\")\n")
    blis_get_cflags(${c} refkern _rf)
    list(FIND _rf "-include" _inc_idx)
    if(_inc_idx GREATER -1)
        math(EXPR _next_idx "${_inc_idx} + 1")
        list(REMOVE_AT _rf ${_inc_idx} ${_next_idx})
    endif()
    list(FILTER _rf EXCLUDE REGEX "^/FI")
    list(REMOVE_ITEM _rf "-DBLIS_IN_REF_KERNEL=1")
    string(REPLACE ";" " " _rfs "${_rf}")
    string(APPEND BLISPLUGIN_EXPORTS "set(blis_REFKERN_CFLAGS_${c} \"${_rfs}\")\n")
endforeach()

include(CMakePackageConfigHelpers)
set(BLIS_NEEDS_THREADS OFF)
if(BLIS_ENABLE_SYSTEM)
    set(BLIS_NEEDS_THREADS ON)
endif()
set(BLIS_NEEDS_OPENMP OFF)
if("openmp" IN_LIST BLIS_THREADING_MODELS)
    set(BLIS_NEEDS_OPENMP ON)
endif()
set(BLIS_NEEDS_MEMKIND OFF)
if(enable_memkind)
    set(BLIS_NEEDS_MEMKIND ON)
endif()
configure_package_config_file(
    "${BLIS_SOURCE_DIR}/cmake/blisConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/blisConfig.cmake"
    INSTALL_DESTINATION ${BLIS_INSTALL_LIBDIR}/cmake/blis
)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/blisConfigVersion.cmake"
    VERSION "${BLIS_VERSION}"
    COMPATIBILITY SameMajorVersion
)
configure_file(
    "${BLIS_SOURCE_DIR}/cmake/BlisPluginHelpers.cmake.in"
    "${PROJECT_BINARY_DIR}/BlisPluginHelpers.cmake"
    @ONLY
)
install(
    FILES
        "${PROJECT_BINARY_DIR}/blisConfig.cmake"
        "${PROJECT_BINARY_DIR}/blisConfigVersion.cmake"
        "${PROJECT_BINARY_DIR}/BlisPluginHelpers.cmake"
    DESTINATION ${BLIS_INSTALL_LIBDIR}/cmake/blis
    COMPONENT cmake-package
)

# @BLIS_INSTALL_PREFIX@ is replaced at install time, so `cmake --install --prefix
# <elsewhere>` yields a .pc that matches the actual install.
set(prefix "@BLIS_INSTALL_PREFIX@")
if(NOT "${BLIS_INSTALL_EXEC_PREFIX}" STREQUAL "")
    set(exec_prefix "${BLIS_INSTALL_EXEC_PREFIX}")
else()
    set(exec_prefix "@BLIS_INSTALL_PREFIX@")
endif()
if(IS_ABSOLUTE "${BLIS_INSTALL_LIBDIR}")
    set(libdir "${BLIS_INSTALL_LIBDIR}")
else()
    set(libdir "@BLIS_INSTALL_PREFIX@/${BLIS_INSTALL_LIBDIR}")
endif()
if(IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
    set(includedir "${CMAKE_INSTALL_INCLUDEDIR}")
else()
    set(includedir "@BLIS_INSTALL_PREFIX@/${CMAKE_INSTALL_INCLUDEDIR}")
endif()
set(PACKAGE_VERSION "${BLIS_VERSION}")
foreach(l ${BLIS_PC_LIBS})
    if(TARGET "${l}" OR l MATCHES "^[^-/][^/]*::")
        message(FATAL_ERROR "BLIS: target name '${l}' cannot go into blis.pc")
    endif()
endforeach()
string(REPLACE ";" " " LDFLAGS "${BLIS_PC_LIBS}")
configure_file("${BLIS_SOURCE_DIR}/blis.pc.in" "${PROJECT_BINARY_DIR}/blis.pc" @ONLY)

# Install a staged file after substituting the real install prefix, so the
# result lands in the install manifest and honors DESTDIR.
function(_blis_install_staged _staged_file _filename _target_dir)
    install(
        CODE "set(_blis_sf \"${_staged_file}\")\nset(_blis_fn \"${_filename}\")\nset(_blis_td \"${_target_dir}\")"
        COMPONENT share
    )
    install(
        CODE
            [[
    file(READ "${_blis_sf}" _c)
    string(REPLACE "@BLIS_INSTALL_PREFIX@" "${CMAKE_INSTALL_PREFIX}" _c "${_c}")
    file(WRITE "${_blis_sf}.d/${_blis_fn}" "${_c}")
    cmake_path(ABSOLUTE_PATH _blis_td BASE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
    file(INSTALL "${_blis_sf}.d/${_blis_fn}" DESTINATION "${_blis_td}"
         PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
  ]]
        COMPONENT share
    )
endfunction()
_blis_install_staged("${PROJECT_BINARY_DIR}/blis.pc" "blis.pc" "${BLIS_SHAREDIR}/pkgconfig")

# Vendor C++ headers (equivalent of `make install INSTALL_HH=yes`).
option(BLIS_INSTALL_VENDOR_CPP_HEADERS "Additionally install AMD's C++ template headers (vendor/cpp/*.hh)" OFF)
if(BLIS_INSTALL_VENDOR_CPP_HEADERS)
    file(GLOB _vendor_hh "${BLIS_SOURCE_DIR}/vendor/cpp/*.hh")
    install(FILES ${_vendor_hh} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/blis COMPONENT headers)
endif()
