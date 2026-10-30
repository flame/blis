# One SHELL: option group per target call: CMake de-duplicates option items
# and would split repeated two-word flags ('--diag_suppress A --diag_suppress
# B' becomes '--diag_suppress A B'). The group is re-parsed UNIX_COMMAND-style,
# so each flag is quoted here to survive embedded spaces.
function(blis_flag_group out)
    set(_parts "")
    foreach(_f IN LISTS ARGN)
        string(REPLACE "\\" "\\\\" _f "${_f}")
        string(REPLACE "\"" "\\\"" _f "${_f}")
        list(APPEND _parts "\"${_f}\"")
    endforeach()
    if(_parts)
        list(JOIN _parts " " _g)
        set(${out} "SHELL:${_g}" PARENT_SCOPE)
    else()
        set(${out} "" PARENT_SCOPE)
    endif()
endfunction()

function(blis_add_objects name)
    cmake_parse_arguments(ARG "" "CONFIG;CLASS" "SOURCES;EXTRA_INCLUDES" ${ARGN})
    if(NOT ARG_SOURCES)
        return()
    endif()
    blis_get_cflags("${ARG_CONFIG}" "${ARG_CLASS}" _cflags)
    add_library(${name} OBJECT ${ARG_SOURCES})
    blis_flag_group(_cflags_group ${_cflags})
    target_compile_options(${name} PRIVATE "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:ASM>>:${_cflags_group}>")
    set(_has_cxx FALSE)
    foreach(src ${ARG_SOURCES})
        if(src MATCHES "\\.(cc|cpp|cxx)$")
            set(_has_cxx TRUE)
        endif()
    endforeach()
    if(_has_cxx)
        blis_get_cxxflags("${ARG_CONFIG}" "${ARG_CLASS}" _cxxflags)
        blis_flag_group(_cxxflags_group ${_cxxflags})
        target_compile_options(${name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${_cxxflags_group}>")
    endif()
    target_include_directories(${name} PRIVATE "${PROJECT_BINARY_DIR}/include" ${BLIS_CINCDIRS} ${ARG_EXTRA_INCLUDES})
    add_dependencies(${name} blis_flat_headers)
    # .s/.S compile in the ASM dialect, like make's kernels rule: gcc reads the
    # assembler mode from the extension (.S preprocessed, .s not), no -x needed.
    foreach(src ${ARG_SOURCES})
        if(src MATCHES "\\.(s|S)$")
            set_source_files_properties(${src} PROPERTIES LANGUAGE ASM)
        endif()
    endforeach()
    set_property(GLOBAL APPEND PROPERTY BLIS_OBJLIBS ${name})
endfunction()

# NO_CINCDIRS and NO_LINK cover support libraries that use only the flat header.
function(blis_in_tree_flags target)
    cmake_parse_arguments(ARG "NO_CINCDIRS;NO_LINK" "" "EXTRA_INCLUDES" ${ARGN})
    blis_get_cflags("${BLIS_CONFIG_NAME}" user _uflags)
    blis_flag_group(_uflags_group ${_uflags})
    target_compile_options(${target} PRIVATE ${_uflags_group})
    set(_incdirs "${PROJECT_BINARY_DIR}/include")
    if(NOT ARG_NO_CINCDIRS)
        list(APPEND _incdirs ${BLIS_CINCDIRS})
    endif()
    target_include_directories(${target} PRIVATE ${_incdirs} ${ARG_EXTRA_INCLUDES})
    if(NOT ARG_NO_LINK AND NOT BLIS_LINK_TARGET STREQUAL "blis_static")
        # blis_static's PUBLIC BLIS_LINK_LIBS reach the consumer transitively
        # already; make's link lines carry the presets once.
        set(_consumer_libs ${BLIS_LINK_LIBS})
        if(BLIS_LDFLAGS_PRESET)
            # CMAKE_EXE_LINKER_FLAGS already carries the preset on this consumer.
            list(REMOVE_ITEM _consumer_libs ${BLIS_LDFLAGS_PRESET})
        endif()
        target_link_libraries(${target} PRIVATE ${_consumer_libs})
    endif()
    add_dependencies(${target} blis_flat_headers)
endfunction()

blis_add_objects(blis_obj_frame CONFIG ${BLIS_CONFIG_NAME} CLASS frame SOURCES ${FRAME_SRC})

foreach(c ${BLIS_CONFIG_LIST_FAM})
    blis_add_objects(blis_obj_config_${c} CONFIG ${c} CLASS config SOURCES ${CONFIG_SRC_${c}})
endforeach()
foreach(c ${BLIS_CONFIG_LIST})
    blis_add_objects(blis_obj_refinit_${c} CONFIG ${c} CLASS refinit SOURCES ${REFINIT_SRC})
    blis_add_objects(blis_obj_refkern_${c} CONFIG ${c} CLASS refkern SOURCES ${REFKERN_SRC})
endforeach()

foreach(k ${BLIS_KERNEL_LIST})
    blis_kconfig_for(${k} _kc)
    blis_add_objects(blis_obj_kern_${k} CONFIG ${_kc} CLASS kernel SOURCES ${KERNEL_SRC_${k}})
endforeach()

set(ADDON_INCDIRS "")
foreach(a ${BLIS_ENABLE_ADDONS})
    blis_header_dirs(_ai "${BLIS_SOURCE_DIR}/addon/${a}")
    list(APPEND ADDON_INCDIRS ${_ai})
    blis_add_objects(
        blis_obj_addon_${a}
        CONFIG ${BLIS_CONFIG_NAME}
        CLASS addon
        SOURCES ${ADDON_SRC_${a}}
        EXTRA_INCLUDES ${ADDON_INCDIRS}
    )
    blis_add_objects(
        blis_obj_addonkern_${a}
        CONFIG ${BLIS_CONFIG_NAME}
        CLASS addon-kernel
        SOURCES ${ADDON_KERNEL_SRC_${a}}
        EXTRA_INCLUDES ${ADDON_INCDIRS}
    )
endforeach()

if(BLIS_SANDBOX)
    blis_header_dirs(SANDBOX_INCDIRS "${BLIS_SOURCE_DIR}/sandbox/${BLIS_SANDBOX}")
    blis_add_objects(
        blis_obj_sandbox
        CONFIG ${BLIS_CONFIG_NAME}
        CLASS sandbox
        SOURCES ${SANDBOX_SRC}
        EXTRA_INCLUDES ${SANDBOX_INCDIRS}
    )
endif()

# Libraries
get_property(_objlibs GLOBAL PROPERTY BLIS_OBJLIBS)
set(_obj_exprs "")
foreach(o ${_objlibs})
    list(APPEND _obj_exprs $<TARGET_OBJECTS:${o}>)
endforeach()

set(_blis_iface_incdirs
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/blis>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(BLIS_ENABLE_STATIC)
    add_library(blis_static STATIC ${_obj_exprs})
    set(_blis_static_name blis)
    if(BLIS_IS_MSVC STREQUAL "yes" AND BLIS_ENABLE_SHARED)
        # The MSVC import library takes libblis.lib; move the static name aside.
        set(_blis_static_name blis_static)
    endif()
    set_target_properties(
        blis_static
        PROPERTIES OUTPUT_NAME ${_blis_static_name} PREFIX "lib" ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    )
    # CMAKE_EXE_LINKER_FLAGS already carries the LDFLAGS preset on every in-tree
    # executable consumer; blis.pc keeps its own copy (BLIS_PC_LIBS).
    set(_blis_static_libs ${BLIS_LINK_LIBS})
    set(_blis_static_opts ${BLIS_LINK_OPTS})
    if(BLIS_LDFLAGS_PRESET)
        list(REMOVE_ITEM _blis_static_libs ${BLIS_LDFLAGS_PRESET})
        list(REMOVE_ITEM _blis_static_opts ${BLIS_LDFLAGS_PRESET})
    endif()
    target_link_libraries(blis_static PUBLIC ${_blis_static_libs})
    blis_flag_group(_blis_static_optgroup ${_blis_static_opts})
    target_link_options(blis_static PUBLIC ${_blis_static_optgroup})
    if("openmp" IN_LIST BLIS_THREADING_MODELS)
        target_link_libraries(blis_static INTERFACE "$<INSTALL_INTERFACE:OpenMP::OpenMP_C>")
    endif()
    target_include_directories(blis_static INTERFACE ${_blis_iface_incdirs})
endif()

if(BLIS_ENABLE_SHARED)
    add_library(blis_shared SHARED ${_obj_exprs})
    set_target_properties(
        blis_shared
        PROPERTIES
            OUTPUT_NAME blis
            PREFIX "lib"
            LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
            VERSION "${BLIS_SO_MMB}"
            SOVERSION "${BLIS_SO_MAJOR}"
    )
    if(APPLE AND NOT BLIS_DARWIN_RPATH)
        set_target_properties(blis_shared PROPERTIES INSTALL_NAME_DIR "${BLIS_INSTALL_FULL_LIBDIR}")
    endif()
    if(WIN32)
        set_target_properties(blis_shared PROPERTIES RUNTIME_OUTPUT_NAME "libblis.${BLIS_SO_MAJOR}" PREFIX "")
        if(BLIS_IS_MSVC STREQUAL "yes")
            set_target_properties(blis_shared PROPERTIES ARCHIVE_OUTPUT_NAME libblis)
        endif()
    endif()
    set(_blis_shared_libs ${BLIS_LINK_LIBS})
    set(_blis_shared_opts ${BLIS_LINK_OPTS})
    if(BLIS_LDFLAGS_PRESET)
        list(REMOVE_ITEM _blis_shared_libs ${BLIS_LDFLAGS_PRESET})
        list(REMOVE_ITEM _blis_shared_opts ${BLIS_LDFLAGS_PRESET})
    endif()
    target_link_libraries(blis_shared PRIVATE ${_blis_shared_libs})
    blis_flag_group(_blis_shared_optgroup ${_blis_shared_opts})
    target_link_options(blis_shared PRIVATE ${_blis_shared_optgroup})
    target_link_options(blis_shared INTERFACE "$<BUILD_INTERFACE:${_blis_shared_optgroup}>")
    if(WIN32 AND BLIS_IS_MSVC STREQUAL "no" AND BLIS_CC_VENDOR MATCHES "^(gcc|clang|aocc)$")
        if(BLIS_EXPORT_SHARED STREQUAL "all")
            target_link_options(blis_shared PRIVATE "SHELL:-Wl,--export-all-symbols -Wl,--enable-auto-import")
        else()
            target_link_options(blis_shared PRIVATE "SHELL:-Wl,--exclude-all-symbols")
        endif()
    endif()
    if(NOT BLIS_IS_MSVC STREQUAL "yes")
        target_link_libraries(blis_shared INTERFACE m)
    endif()
    target_include_directories(blis_shared INTERFACE ${_blis_iface_incdirs})
endif()

# In-tree consumers (testsuite, blastest, examples): link static when
# available; on Windows the make build uses the dll regardless.
if(BLIS_ENABLE_STATIC AND NOT (WIN32 AND BLIS_ENABLE_SHARED))
    set(BLIS_LINK_TARGET blis_static)
else()
    set(BLIS_LINK_TARGET blis_shared)
endif()
