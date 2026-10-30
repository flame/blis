cmake_minimum_required(VERSION 3.20...4.4)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BlisToolchainEnv.cmake)

project(blis LANGUAGES C DESCRIPTION "BLAS-like Library Instantiation Software")
if(PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(FATAL_ERROR "BLIS: in-source builds are not supported; use -B <builddir> (out-of-source)")
endif()
if(PROJECT_BINARY_DIR STREQUAL "${PROJECT_SOURCE_DIR}/build")
    message(
        FATAL_ERROR
        "BLIS: <source>/build is BLIS's own helper directory and cannot be a "
        "CMake binary dir: the monolithic-header generator would resolve "
        "\"cblas.h\" against the helper stub build/cblas.h there. "
        "Use a different build directory name (e.g. -B _build)."
    )
endif()
set(BLIS_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)
include(BlisVersion)
include(BlisDetectCompiler)
include(BlisAutoDetect)
include(BlisRegistry)
include(BlisOptions)
include(BlisSources)

find_package(Python3 COMPONENTS Interpreter)

include(BlisConfigSelect)
include(BlisHeaderGen)

blis_config_headers()

include(BlisFlags)

include(BlisLinkItems)
# Frame sources (C; compat subtrees filtered per BLAS/CBLAS; *_amd twins per
# BLIS_ENABLE_AMD_FRAME_TWEAKS).
set(_frame_suffixes c)
if("hpx" IN_LIST BLIS_THREADING_MODELS)
    list(APPEND _frame_suffixes cpp)
endif()
blis_collect(FRAME_SRC "${BLIS_SOURCE_DIR}/frame" ${_frame_suffixes})
if(NOT BLIS_EFFECTIVE_BLAS)
    list(FILTER FRAME_SRC EXCLUDE REGEX "/compat/")
elseif(NOT BLIS_ENABLE_CBLAS)
    list(FILTER FRAME_SRC EXCLUDE REGEX "/compat/cblas/")
endif()

list(FILTER FRAME_SRC EXCLUDE REGEX "/frame/include/")
file(GLOB_RECURSE _amd_twins_all CONFIGURE_DEPENDS "${BLIS_SOURCE_DIR}/frame/*_amd.c")
foreach(f ${_amd_twins_all})
    if(BLIS_ENABLE_AMD_FRAME_TWEAKS)
        string(REGEX REPLACE "_amd\\.c$" ".c" target "${f}")
    else()
        set(target "${f}")
    endif()
    list(FIND FRAME_SRC "${target}" _i)
    if(_i GREATER -1)
        list(REMOVE_AT FRAME_SRC ${_i})
    endif()
endforeach()

# Reference kernels: bli_cntx_ref.c (refinit class) vs everything else.
blis_collect(REFKERN_ALL "${BLIS_SOURCE_DIR}/ref_kernels" c)
set(REFINIT_SRC "")
set(REFKERN_SRC "")
foreach(f ${REFKERN_ALL})
    if(f MATCHES "/bli_cntx_ref\\.c$")
        list(APPEND REFINIT_SRC "${f}")
    else()
        list(APPEND REFKERN_SRC "${f}")
    endif()
endforeach()

# Config / kernel sources per config among CONFIG_LIST_FAM (the family dir
# itself usually holds only headers, but make fragments it too).
foreach(c ${BLIS_CONFIG_LIST_FAM})
    blis_collect(CONFIG_SRC_${c} "${BLIS_SOURCE_DIR}/config/${c}" c)
endforeach()
foreach(k ${BLIS_KERNEL_LIST})
    blis_collect(KERNEL_SRC_${k} "${BLIS_SOURCE_DIR}/kernels/${k}" c s S)
endforeach()

foreach(a ${BLIS_ENABLE_ADDONS})
    blis_collect(ADDON_ALL_${a} "${BLIS_SOURCE_DIR}/addon/${a}" c cc cpp cxx)
    set(ADDON_SRC_${a} "")
    set(ADDON_KERNEL_SRC_${a} "")
    foreach(f ${ADDON_ALL_${a}})
        if(f MATCHES "/kernels/")
            list(APPEND ADDON_KERNEL_SRC_${a} "${f}")
        else()
            list(APPEND ADDON_SRC_${a} "${f}")
        endif()
    endforeach()
endforeach()
if(BLIS_SANDBOX)
    blis_collect(SANDBOX_SRC "${BLIS_SOURCE_DIR}/sandbox/${BLIS_SANDBOX}" c cc cpp cxx)
endif()

set(_need_cxx FALSE)
if("hpx" IN_LIST BLIS_THREADING_MODELS)
    set(_need_cxx TRUE)
endif()
set(_cxx_candidates "")
foreach(a ${BLIS_ENABLE_ADDONS})
    list(APPEND _cxx_candidates ${ADDON_ALL_${a}})
endforeach()
if(BLIS_SANDBOX)
    list(APPEND _cxx_candidates ${SANDBOX_SRC})
endif()
list(FILTER _cxx_candidates INCLUDE REGEX "\\.(cc|cpp|cxx)$")
if(_cxx_candidates)
    set(_need_cxx TRUE)
endif()
if(_need_cxx)
    enable_language(CXX)
endif()

set(_need_asm FALSE)
foreach(k ${BLIS_KERNEL_LIST})
    foreach(f ${KERNEL_SRC_${k}})
        if(f MATCHES "\\.(s|S)$")
            set(_need_asm TRUE)
        endif()
    endforeach()
endforeach()
if(_need_asm)
    enable_language(ASM)
endif()

blis_flat_headers_gen()

include(BlisLibraries)

include(BlisInstall)
# Make-build parity: share/blis fragments, showconfig, uninstall-old, partial
# install and uninstall targets.
blis_raw_tools_begin()
include(make/BlisMakeShare)
include(make/BlisUninstall)
blis_raw_tools_end()

include(CTest)
if(BUILD_TESTING)
    include(BlisTesting)
    add_subdirectory(testsuite)
    blis_register_testsuite_tests()
    # blastest drivers call lsame_/xerbla_, which may be omitted from the
    # library (lapack-compat mode): in that case those symbols must come from
    # a real LAPACK, so the bundled drivers cannot link.
    set(_blat_blocked FALSE)
    if(_omit_list MATCHES "lsame" OR _omit_list MATCHES "xerbla")
        set(_blat_blocked TRUE)
    endif()
    if(BLIS_EFFECTIVE_BLAS AND NOT _blat_blocked)
        add_subdirectory(blastest)
        blis_register_blastest_tests()
    elseif(_blat_blocked)
        message(STATUS "BLIS: blastest disabled (lsame/xerbla omitted; LAPACK-compat mode)")
    endif()
endif()

if(BLIS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Make-build parity: the check*, test*, clean* and other make target names.
include(make/BlisMakeTargets)

# Maintainer helpers: 'symbols' regenerates the MSVC export manifest
# build/libblis-symbols.def from the flattened header; 'changelog' refreshes
# the CHANGELOG file from git. (Both write into the source tree, as upstream.)
# Keep the target name on the open-paren line: audits grep for it.
# gersemi: off
add_custom_target(symbols
  COMMAND "${CMAKE_COMMAND}"
          -DCC_BIN=${CMAKE_C_COMPILER}
          -DFLAT=${PROJECT_BINARY_DIR}/include/blis.h
          -DCFG=${BLIS_CONFIG_NAME}
          -DSCRIPT=${BLIS_SOURCE_DIR}/build/gen-libblis-symbols.sh
          -DOUT=${BLIS_SOURCE_DIR}/build/libblis-symbols.def
          -P ${BLIS_SOURCE_DIR}/cmake/BlisGenSymbols.cmake
  DEPENDS blis_flat_headers)
# gersemi: on
if(GIT_FOUND)
    # gersemi: off
    add_custom_target(changelog
      COMMAND "${CMAKE_COMMAND}"
              -DGIT=${GIT_EXECUTABLE} -DSRC=${BLIS_SOURCE_DIR}
              -DOUT=${BLIS_SOURCE_DIR}/CHANGELOG
              -P ${BLIS_SOURCE_DIR}/cmake/BlisChangelog.cmake)
    # gersemi: on
endif()

blis_config_summary()

# make evaluates a make_defs.mk $(error) before every target, so a deferred
# error must stop every target here, ahead of any compile. Keep this last:
# it must see all targets.
if(TARGET blis_config_error)
    set(_dirs "${CMAKE_CURRENT_SOURCE_DIR}")
    set(_all_targets "")
    while(_dirs)
        list(POP_FRONT _dirs _d)
        get_property(_t DIRECTORY "${_d}" PROPERTY BUILDSYSTEM_TARGETS)
        list(APPEND _all_targets ${_t})
        get_property(_s DIRECTORY "${_d}" PROPERTY SUBDIRECTORIES)
        list(APPEND _dirs ${_s})
    endwhile()
    list(REMOVE_ITEM _all_targets blis_config_error)
    foreach(_t IN LISTS _all_targets)
        add_dependencies(${_t} blis_config_error)
    endforeach()
endif()
