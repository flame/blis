set(_blis_record "${PROJECT_BINARY_DIR}/blis_install_record.cmake")
foreach(
    _c
    headers
    helper-headers
    libs
    lib-symlinks
    share
    cmake-package
)
    install(CODE "set(_blis_record \"${_blis_record}\")" COMPONENT ${_c})
    install(
        CODE
            [[
    set(_p "${CMAKE_INSTALL_PREFIX}")
    set(_d "$ENV{DESTDIR}")
    cmake_path(ABSOLUTE_PATH _p NORMALIZE)
    if(NOT "${_d}" STREQUAL "")
      cmake_path(ABSOLUTE_PATH _d NORMALIZE)
    endif()
    file(WRITE "${_blis_record}"
      "set(blis_rec_prefix [==[${_p}]==])\nset(blis_rec_destdir [==[${_d}]==])\n")
  ]]
        COMPONENT ${_c}
    )
endforeach()
set(_blis_unargs "-DRECORD=${_blis_record}" "-DPREFIX=${CMAKE_INSTALL_PREFIX}")

add_custom_target(uninstall-old)
foreach(w libs symlinks headers)
    # Keep the target name on the open-paren line: audits grep for it.
    # gersemi: off
    add_custom_target(uninstall-old-${w}
        COMMAND "${CMAKE_COMMAND}" -DWHAT=${w}
            -DLIBDIR=${BLIS_INSTALL_LIBDIR} -DINCDIR=${CMAKE_INSTALL_INCLUDEDIR}
            -DSO_MAJOR=${BLIS_SO_MAJOR} -DSO_MMB=${BLIS_SO_MMB} -DDARWIN=${APPLE}
            ${_blis_unargs}
            -P "${BLIS_SOURCE_DIR}/cmake/cmake_uninstall.cmake"
        VERBATIM)
    # gersemi: on
    add_dependencies(uninstall-old uninstall-old-${w})
endforeach()

set(_ic_headers headers helper-headers)
foreach(_n helper-headers libs lib-symlinks share)
    set(_ic_${_n} ${_n})
endforeach()
set(_blis_libtargets "")
foreach(_t blis_static blis_shared)
    if(TARGET ${_t})
        list(APPEND _blis_libtargets ${_t})
    endif()
endforeach()
foreach(
    _n
    headers
    helper-headers
    libs
    lib-symlinks
    share
)
    set(_cmds "")
    foreach(_c ${_ic_${_n}})
        list(
            APPEND _cmds
            COMMAND
            "${CMAKE_COMMAND}"
            -DCMAKE_INSTALL_COMPONENT=${_c}
            -P
            "${PROJECT_BINARY_DIR}/cmake_install.cmake"
        )
    endforeach()
    add_custom_target(install-${_n} ${_cmds} VERBATIM)
    if(_n MATCHES "headers$")
        add_dependencies(install-${_n} blis_flat_headers)
    elseif(_n MATCHES "^lib")
        add_dependencies(install-${_n} ${_blis_libtargets})
    endif()
endforeach()

set(_ul_libs "")
set(_ul_syml "")
if(TARGET blis_static)
    list(APPEND _ul_libs "${BLIS_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:blis_static>")
endif()
if(TARGET blis_shared)
    if(WIN32)
        list(
            APPEND _ul_syml
            "${BLIS_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:blis_shared>"
            "${BLIS_INSTALL_LIBDIR}/$<TARGET_LINKER_FILE_NAME:blis_shared>"
        )
    else()
        list(APPEND _ul_libs "${BLIS_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:blis_shared>")
        list(
            APPEND _ul_syml
            "${BLIS_INSTALL_LIBDIR}/$<TARGET_SONAME_FILE_NAME:blis_shared>"
            "${BLIS_INSTALL_LIBDIR}/$<TARGET_LINKER_FILE_NAME:blis_shared>"
        )
    endif()
endif()
set(_ul_hdrs "${CMAKE_INSTALL_INCLUDEDIR}/blis" "${CMAKE_INSTALL_INCLUDEDIR}/blis.h")
if(BLIS_ENABLE_CBLAS)
    list(APPEND _ul_hdrs "${CMAKE_INSTALL_INCLUDEDIR}/cblas.h")
endif()
foreach(_n libs lib-symlinks headers share)
    set(_f "")
    set(_d "")
    if(_n STREQUAL "libs")
        list(JOIN _ul_libs "|" _f)
    elseif(_n STREQUAL "lib-symlinks")
        list(JOIN _ul_syml "|" _f)
    elseif(_n STREQUAL "headers")
        list(JOIN _ul_hdrs "|" _d)
    else()
        set(_d "${BLIS_SHAREDIR}/blis")
    endif()
    # gersemi: off
    add_custom_target(uninstall-${_n}
        COMMAND "${CMAKE_COMMAND}" "-DFILES=${_f}" "-DDIRS=${_d}" ${_blis_unargs}
            -P "${BLIS_SOURCE_DIR}/cmake/cmake_uninstall.cmake"
        VERBATIM)
    # gersemi: on
endforeach()
# gersemi: off
add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" "-DDIRS=${BLIS_INSTALL_LIBDIR}/cmake/blis" ${_blis_unargs}
        -P "${BLIS_SOURCE_DIR}/cmake/cmake_uninstall.cmake"
    VERBATIM)
# gersemi: on
add_dependencies(uninstall uninstall-libs uninstall-lib-symlinks uninstall-headers uninstall-share)
