# Evaluates config/<c>/make_defs.mk (via BlisMakeDefs) and assembles
# the per-class compiler flags exactly as common.mk's get-*-cflags-for do.
#
# Inputs expected in scope: BLIS_CC_VENDOR, BLIS_DEBUG_TYPE, the CC
# compiler id/version booleans (BLIS_GCC_OT_* etc.), host OS thread
# flags BLIS_CTHREADFLAGS.
#
# Provides:
#   blis_load_config_defs(<cfg>)        -> BLIS_DEFS_<VAR>_<cfg> parent-scope vars
#   blis_finalize_config_defs(<cfg>)    -> global flag appends (warn/lang/proc/pic)
#   blis_get_cflags(<cfg> <class> <out>) -> flag list; class in
#       frame config kernel refinit refkern addon addon-kernel sandbox user

# Bridge variables for the def files (make naming, "yes"/"no" strings).
set(CC "${CMAKE_C_COMPILER}")
set(CC_VENDOR "${BLIS_CC_VENDOR}")
set(OS_NAME "${BLIS_HOST_OS}")
set(IS_WIN "${BLIS_IS_WIN}")
set(DEBUG_TYPE "${BLIS_DEBUG_TYPE}")
foreach(v 4_9_0 6_1_0 9_1_0 10_3_0)
    set(GCC_OT_${v} "${BLIS_GCC_OT_${v}}")
endforeach()
set(CLANG_OT_9_0_0 "${BLIS_CLANG_OT_9_0_0}")
set(CLANG_OT_12_0_0 "${BLIS_CLANG_OT_12_0_0}")
set(AOCC_OT_2_0_0 "${BLIS_AOCC_OT_2_0_0}")
set(AOCC_OT_3_0_0 "${BLIS_AOCC_OT_3_0_0}")

include(BlisMakeDefs)

# LDFLAGS is captured because some configs append to the link line
# (rv* => -latomic; bgq => SPI libs). The make-defs' per-config
# MK_ENABLE_MEMKIND override is intentionally NOT honored at link time:
# the make system never reads it there either (common.mk computes LDFLAGS
# before including the defs); knl's en-masse '-DBLIS_DISABLE_MEMKIND'
# CPPROCFLAGS define still applies (config/knl def file).
set(_BLIS_DEF_VARS
    CPPROCFLAGS
    CMISCFLAGS
    CPICFLAGS
    CWARNFLAGS
    CDBGFLAGS
    COPTFLAGS
    CKOPTFLAGS
    CKVECFLAGS
    CROPTFLAGS
    CRVECFLAGS
    LDFLAGS
)

function(blis_load_config_defs cfg)
    if(NOT EXISTS "${BLIS_SOURCE_DIR}/config/${cfg}/make_defs.mk")
        message(FATAL_ERROR "BLIS: no flag definitions for config '${cfg}'")
    endif()
    # Init LDFLAGS capture to empty so stale parent values never leak across configs.
    set(LDFLAGS "")
    blis_parse_make_defs(${cfg})
    foreach(v ${_BLIS_DEF_VARS})
        separate_arguments(_l UNIX_COMMAND "${${v}}")
        set(BLIS_DEFS_${v}_${cfg} "${_l}" PARENT_SCOPE)
    endforeach()
endfunction()

# Global appends (common.mk "Configuration-agnostic flags" section). make
# appends the global value onto each config's stored value; MSVC empties PIC.
function(blis_finalize_config_defs cfg)
    set(cwarn ${BLIS_DEFS_CWARNFLAGS_${cfg}} ${BLIS_GLOBAL_CWARNFLAGS})
    set(clang -std=c99)
    set(cproc ${BLIS_DEFS_CPPROCFLAGS_${cfg}} ${BLIS_GLOBAL_CPPFLAGS})
    set(cpic ${BLIS_DEFS_CPICFLAGS_${cfg}})
    if(MSVC OR CMAKE_C_SIMULATE_ID STREQUAL "MSVC")
        set(cpic "")
    endif()
    set(BLIS_DEFS_CWARNFLAGS_${cfg} "${cwarn}" PARENT_SCOPE)
    set(BLIS_DEFS_CLANGFLAGS_${cfg} "${clang}" PARENT_SCOPE)
    set(BLIS_DEFS_CPPROCFLAGS_${cfg} "${cproc}" PARENT_SCOPE)
    set(BLIS_DEFS_CPICFLAGS_${cfg} "${cpic}" PARENT_SCOPE)
endfunction()

# Unified flag assembly for C and C++. The class dispatch table is shared;
# only the base flags differ: C uses BLIS_DEFS_CLANGFLAGS_<cfg> (-std=c99),
# C++ uses BLIS_CXXLANGFLAGS (-std=c++11 or c++17).
function(blis_get_flags__impl cfg class lang out)
    set(f
        ${BLIS_DEFS_CDBGFLAGS_${cfg}}
        ${BLIS_DEFS_CWARNFLAGS_${cfg}}
        ${BLIS_DEFS_CPICFLAGS_${cfg}}
        ${BLIS_DEFS_CMISCFLAGS_${cfg}}
        ${BLIS_DEFS_CPPROCFLAGS_${cfg}}
        ${BLIS_CTHREADFLAGS}
    )

    if(lang STREQUAL "C")
        list(APPEND f ${BLIS_DEFS_CLANGFLAGS_${cfg}})
    elseif(lang STREQUAL "CXX")
        list(APPEND f ${BLIS_CXXLANGFLAGS})
    else()
        message(FATAL_ERROR "blis_get_flags__impl: unknown language '${lang}'")
    endif()

    string(TOUPPER "${cfg}" CFGU)
    if(class STREQUAL "kernel" OR class STREQUAL "addon-kernel")
        list(APPEND f ${BLIS_DEFS_CKOPTFLAGS_${cfg}} ${BLIS_DEFS_CKVECFLAGS_${cfg}})
    elseif(class STREQUAL "refkern")
        list(APPEND f ${BLIS_DEFS_CROPTFLAGS_${cfg}} ${BLIS_DEFS_CRVECFLAGS_${cfg}} ${BLIS_COMPSIMDFLAGS})
    elseif(NOT class MATCHES "^(frame|config|refinit|refkern|addon|sandbox|user)$")
        message(FATAL_ERROR "blis_get_flags__impl: unknown class '${class}'")
    else()
        list(APPEND f ${BLIS_DEFS_COPTFLAGS_${cfg}})
    endif()
    if(class MATCHES "^(config|kernel|addon-kernel|refinit|refkern)$")
        list(APPEND f -DBLIS_CNAME=${cfg} -DBLIS_CNAME_UPPER=${CFGU})
    endif()
    if(class MATCHES "^(frame|config|refinit|refkern)$")
        list(APPEND f ${BLIS_ASANFLAGS})
    endif()
    if(NOT class STREQUAL "user")
        list(APPEND f -DBLIS_IS_BUILDING_LIBRARY ${BLIS_SYMFLAGS})
    endif()
    if(class STREQUAL "refinit" OR class STREQUAL "refkern")
        list(APPEND f -DBLIS_IN_REF_KERNEL=1)
        # clang-cl ignores -include with a warning and reads the path as a source.
        if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            list(APPEND f "/FI${BLIS_SOURCE_DIR}/config/${cfg}/bli_kernel_defs_${cfg}.h")
        else()
            list(APPEND f -include "${BLIS_SOURCE_DIR}/config/${cfg}/bli_kernel_defs_${cfg}.h")
        endif()
    endif()
    set(${out} "${f}" PARENT_SCOPE)
endfunction()

function(blis_get_cxxflags cfg class out)
    blis_get_flags__impl("${cfg}" "${class}" "CXX" _blis_flags_result)
    set(${out} "${_blis_flags_result}" PARENT_SCOPE)
endfunction()

# Mirrors get-noopt-cflags-for + per-class appends. Include dirs and defines
# are the caller's, apart from the class-specific -D flags included here.
function(blis_get_cflags cfg class out)
    blis_get_flags__impl("${cfg}" "${class}" "C" _blis_flags_result)
    set(${out} "${_blis_flags_result}" PARENT_SCOPE)
endfunction()

# Flag scaffolding (mirrors common.mk)
# NOTE: make's CFLAGS/CXXFLAGS "preset" (env at configure time) maps to
# CMAKE_C_FLAGS/CMAKE_CXX_FLAGS: CMake seeds them from the same env vars and
# puts them ahead of target options, the preset's place in the flag chain.
separate_arguments(BLIS_LDFLAGS_PRESET UNIX_COMMAND "$ENV{LDFLAGS}")

set(BLIS_GLOBAL_CPPFLAGS "-D_POSIX_C_SOURCE=200112L")
if(APPLE)
    list(APPEND BLIS_GLOBAL_CPPFLAGS "-D_DARWIN_C_SOURCE")
endif()

set(BLIS_GLOBAL_CWARNFLAGS "")
if(NOT BLIS_CC_VENDOR STREQUAL "ibm")
    list(APPEND BLIS_GLOBAL_CWARNFLAGS -Werror=implicit-function-declaration)
    if(NOT BLIS_CC_VENDOR STREQUAL "NVIDIA")
        list(APPEND BLIS_GLOBAL_CWARNFLAGS -Wall -Wno-unused-function -Wfatal-errors -Werror=pointer-arith)
    else()
        list(APPEND BLIS_GLOBAL_CWARNFLAGS -Wall -Wno-unused-function)
    endif()
endif()
if(BLIS_CC_VENDOR MATCHES "clang|aocc")
    list(APPEND BLIS_GLOBAL_CWARNFLAGS -Wno-tautological-compare -Wno-pass-failed)
endif()
if(BLIS_CC_VENDOR STREQUAL "NVIDIA")
    list(
        APPEND BLIS_GLOBAL_CWARNFLAGS
        --diag_suppress
        unsigned_compare_with_zero
        --diag_suppress
        cast_to_qualified_type
        --diag_suppress
        bad_initializer_type
        --diag_suppress
        used_before_set
        --diag_suppress
        mixed_enum_type
        --diag_suppress
        incompatible_assignment_operands
    )
endif()
if(BLIS_CC_VENDOR STREQUAL "gcc")
    list(APPEND BLIS_GLOBAL_CWARNFLAGS -Wno-comment)
endif()

set(BLIS_CTHREADFLAGS "")
if("openmp" IN_LIST BLIS_THREADING_MODELS)
    list(APPEND BLIS_CTHREADFLAGS -fopenmp)
    find_package(OpenMP REQUIRED COMPONENTS C)
endif()
if("pthreads" IN_LIST BLIS_THREADING_MODELS)
    list(APPEND BLIS_CTHREADFLAGS -pthread)
endif()
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

set(BLIS_ASANFLAGS "")
if(BLIS_ENABLE_ASAN)
    set(BLIS_ASANFLAGS -fsanitize=address)
endif()

set(BLIS_COMPSIMDFLAGS "")
if(enable_pragma_omp_simd)
    if(BLIS_CC_VENDOR STREQUAL "icc")
        set(BLIS_COMPSIMDFLAGS -qopenmp-simd)
    else()
        set(BLIS_COMPSIMDFLAGS -fopenmp-simd)
    endif()
endif()

# Symbol-visibility flags (common.mk BUILD_SYMFLAGS table).
set(BLIS_SYMFLAGS "")
if(BLIS_ENABLE_SHARED AND NOT WIN32)
    if(BLIS_CC_VENDOR STREQUAL "gcc" OR BLIS_CC_VENDOR STREQUAL "icc" OR BLIS_CC_VENDOR MATCHES "clang|aocc")
        if(BLIS_EXPORT_SHARED STREQUAL "all")
            set(BLIS_SYMFLAGS -fvisibility=default)
        else()
            set(BLIS_SYMFLAGS -fvisibility=hidden)
        endif()
    endif()
endif()

# Load per-config defs (family members in sorted order, like CONFIG_LIST_FAM),
# then apply the global appends.
foreach(c ${BLIS_CONFIG_LIST_FAM})
    blis_load_config_defs(${c})
    blis_finalize_config_defs(${c})
endforeach()
