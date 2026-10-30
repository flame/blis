# All user-facing cache options + validation, mirroring ./configure options.
# After inclusion, plain (non-cache) variables with the template anchor names
# of build/bli_config.h.in are set for configure_file(... @ONLY).

set(BLIS_CONFIG "auto" CACHE STRING "BLIS configuration family (see config_registry) or 'auto' for hardware detection")

set(BLIS_THREADING
    "single"
    CACHE STRING
    "Threading models to build (first = runtime default): ';'-separated list of single,openmp,pthreads,hpx,auto"
)
set(BLIS_THREAD_JRIR_PARTITION "slab" CACHE STRING "jr/ir loop partitioning method: slab, rr, or tlb")
set_property(CACHE BLIS_THREAD_JRIR_PARTITION PROPERTY STRINGS slab rr tlb)

set(BLIS_INT_SIZE "auto" CACHE STRING "BLIS integer type size: 32, 64, or auto")
set_property(CACHE BLIS_INT_SIZE PROPERTY STRINGS auto 32 64)
set(BLIS_BLAS_INT_SIZE "32" CACHE STRING "BLAS compatibility integer size: 32 or 64")
set_property(CACHE BLIS_BLAS_INT_SIZE PROPERTY STRINGS 32 64)

option(BLIS_ENABLE_BLAS "Enable the BLAS compatibility layer" ON)
option(BLIS_ENABLE_CBLAS "Enable the CBLAS compatibility layer" OFF)
option(BLIS_ENABLE_SUP_HANDLING "Enable small/unpacked (sup) handling" ON)
option(BLIS_ENABLE_TRSM_PREINVERSION "Enable trsm pre-inversion" ON)
option(BLIS_ENABLE_MIXED_PRECISION_NORM "Accumulate snrm2/scnrm2 in double precision" ON)
option(BLIS_ENABLE_LAPACK_COMPAT "Omit symbols that clash with reference LAPACK" OFF)
option(BLIS_ENABLE_SCALAPACK_COMPAT "Omit symbols that clash with ScaLAPACK" OFF)
set(BLIS_OMIT_SYMBOLS "" CACHE STRING "Comma-separated list of additional symbols to omit")

option(BLIS_ENABLE_STATIC "Build the static library (libblis.a)" ON)
option(BLIS_ENABLE_SHARED "Build the shared library (libblis.so)" ON)
set(BLIS_EXPORT_SHARED "public" CACHE STRING "Shared library exports: public (annotated) or all")
set_property(CACHE BLIS_EXPORT_SHARED PROPERTY STRINGS public all)
# configure defaults to --disable-rpath (absolute install_name on Darwin);
# keep the CMake default off to match.
option(BLIS_DARWIN_RPATH "Use @rpath-based install_name on macOS" OFF)

set(BLIS_DEBUG_TYPE "off" CACHE STRING "Debug info: off, noopt (unoptimized), opt, or sde (Intel SDE)")
set_property(
    CACHE BLIS_DEBUG_TYPE
    PROPERTY STRINGS off noopt opt sde
)
option(BLIS_ENABLE_ASAN "Compile and link with AddressSanitizer" OFF)
option(BLIS_ENABLE_MEM_TRACING "Enable memory-tracing output" OFF)

option(BLIS_ENABLE_PBA_POOLS "Enable packing-block allocator pools" ON)
option(BLIS_ENABLE_SBA_POOLS "Enable small-block allocator pools" ON)
set(BLIS_ENABLE_MEMKIND "auto" CACHE STRING "Use libmemkind for certain allocations: auto, on, or off")
set_property(CACHE BLIS_ENABLE_MEMKIND PROPERTY STRINGS auto on off)
set(BLIS_COMPLEX_RETURN "auto" CACHE STRING "Hidden complex-return-value convention: auto, gnu, or intel")
set_property(CACHE BLIS_COMPLEX_RETURN PROPERTY STRINGS auto gnu intel)

option(BLIS_ENABLE_SYSTEM "Enable system interfaces (os threads, timers)" ON)
option(BLIS_ENABLE_TLS "Use thread-local storage qualifiers" ON)

set(BLIS_ENABLE_ADDONS "" CACHE STRING "';'-separated list of addons under addon/ to build")
set(BLIS_SANDBOX "" CACHE STRING "Name of a sandbox under sandbox/ to enable")

# --- install layout (mirrors configure's --exec-prefix / --sharedir) ---
# Library dir layout. 'make' installs under ${exec_prefix}/lib like the make
# build; 'gnu' keeps the GNUInstallDirs dir (lib/<multiarch> on a /usr prefix).
set(BLIS_INSTALL_LAYOUT "make" CACHE STRING "Library install dir: make (lib) or gnu (GNUInstallDirs)")
set_property(CACHE BLIS_INSTALL_LAYOUT PROPERTY STRINGS make gnu)
if(NOT BLIS_INSTALL_LAYOUT MATCHES "^(make|gnu)$")
    message(FATAL_ERROR "BLIS: BLIS_INSTALL_LAYOUT must be make or gnu")
endif()
set(BLIS_INSTALL_EXEC_PREFIX
    ""
    CACHE STRING
    "Separate prefix for libraries (configure --exec-prefix). Empty = CMAKE_INSTALL_PREFIX"
)
set(BLIS_INSTALL_SHAREDIR
    ""
    CACHE STRING
    "Directory for shared data such as pkg-config files (configure --sharedir). Empty = <prefix>/share"
)
# BLIS_INSTALL_LIBDIR is the one library install dir. Every consumer (install
# DESTINATION, blis.pc, config.mk, uninstall-old, Darwin install_name) derives
# from it. It stays relative without an exec prefix, so --prefix and DESTDIR move it.
# GNUInstallDirs already set CMAKE_INSTALL_LIBDIR, so a caller-provided value is
# told apart from that default by the helpstring the cache entry carries; an
# absolute dir is caller-provided by construction.
get_property(_blis_libdir_hs CACHE CMAKE_INSTALL_LIBDIR PROPERTY HELPSTRING)
if(
    BLIS_INSTALL_LAYOUT STREQUAL "make"
    AND NOT _blis_libdir_hs MATCHES "command line"
    AND NOT IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}"
)
    set(_blis_libdir "lib")
else()
    set(_blis_libdir "${CMAKE_INSTALL_LIBDIR}")
endif()
if(NOT "${BLIS_INSTALL_EXEC_PREFIX}" STREQUAL "" AND NOT IS_ABSOLUTE "${_blis_libdir}")
    set(BLIS_INSTALL_LIBDIR "${BLIS_INSTALL_EXEC_PREFIX}/${_blis_libdir}")
else()
    set(BLIS_INSTALL_LIBDIR "${_blis_libdir}")
endif()
set(BLIS_INSTALL_FULL_LIBDIR "${BLIS_INSTALL_LIBDIR}")
cmake_path(ABSOLUTE_PATH BLIS_INSTALL_FULL_LIBDIR BASE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
if(NOT "${BLIS_INSTALL_SHAREDIR}" STREQUAL "")
    set(BLIS_SHAREDIR "${BLIS_INSTALL_SHAREDIR}")
else()
    set(BLIS_SHAREDIR "${CMAKE_INSTALL_DATAROOTDIR}")
endif()

option(BLIS_ENABLE_AMD_FRAME_TWEAKS "Use AMD-customized framework files (zen configs only)" OFF)
set(BLIS_FORCE_VERSION "" CACHE STRING "Version string override (skips git/version-file detection)")
option(BLIS_BUILD_EXAMPLES "Build the example programs" OFF)

option(BLIS_ENABLE_VERBOSE_MAKE "Enable verbose compilation output during make" OFF)
if(BLIS_ENABLE_VERBOSE_MAKE)
    set(CMAKE_VERBOSE_MAKEFILE ON)
    message(STATUS "BLIS: enabling verbose make output. (disable with 'make V=0'.)")
else()
    message(STATUS "BLIS: disabling verbose make output. (enable with 'make V=1'.)")
endif()

option(
    BLIS_ENABLE_ARG_MAX_HACK
    "Allow archiving and linking when the command line exceeds the system limit (ARG_MAX). This knob only writes ARG_MAX_HACK into config.mk for make-side consumers; the CMake build is unaffected"
    OFF
)
if(BLIS_ENABLE_ARG_MAX_HACK)
    message(STATUS "BLIS: enabling ARG_MAX hack.")
else()
    message(STATUS "BLIS: disabling ARG_MAX hack.")
endif()

option(BLIS_SHOW_CONFIG_LIST "Print the config and kernel lists and the kernel-to-config map at configure time" OFF)

if(NOT BLIS_ENABLE_STATIC AND NOT BLIS_ENABLE_SHARED)
    message(FATAL_ERROR "BLIS: cannot disable both static and shared libraries")
endif()
if(NOT BLIS_BLAS_INT_SIZE MATCHES "^(32|64)$")
    message(FATAL_ERROR "BLIS: BLIS_BLAS_INT_SIZE must be 32 or 64")
endif()
if(BLIS_INT_SIZE STREQUAL "auto")
    set(BLIS_INT_SIZE "0")
endif()
if(NOT BLIS_INT_SIZE MATCHES "^(0|32|64)$")
    message(FATAL_ERROR "BLIS: BLIS_INT_SIZE must be 32, 64, or auto")
endif()
if(BLIS_BLAS_INT_SIZE STREQUAL "64" AND BLIS_INT_SIZE STREQUAL "32")
    message(FATAL_ERROR "BLIS: BLIS_BLAS_INT_SIZE=64 requires BLIS_INT_SIZE=64")
endif()

# AMD frame tweaks validation happens in the top-level CMakeLists, after
# BLIS_CONFIG ('auto') has been resolved.

foreach(a IN LISTS BLIS_ENABLE_ADDONS)
    if(NOT IS_DIRECTORY "${BLIS_SOURCE_DIR}/addon/${a}")
        message(FATAL_ERROR "BLIS: addon '${a}' does not exist under addon/")
    endif()
endforeach()
if(BLIS_SANDBOX AND NOT IS_DIRECTORY "${BLIS_SOURCE_DIR}/sandbox/${BLIS_SANDBOX}")
    message(FATAL_ERROR "BLIS: sandbox '${BLIS_SANDBOX}' does not exist under sandbox/")
endif()

# --- threading model resolution (configure's --enable-threading) ---
# Accept ';' or ',' separators; map synonyms; 'single' is always built and
# the first listed model becomes the runtime default.
string(REPLACE "," ";" _tm "${BLIS_THREADING}")
string(REPLACE "+" ";" _tm "${_tm}")
set(_models "")
foreach(m ${_tm})
    string(STRIP "${m}" m)
    string(TOLOWER "${m}" m)
    if(m MATCHES "^(single|none|off|no)$")
        list(APPEND _models single)
    elseif(m MATCHES "^(openmp|omp)$")
        list(APPEND _models openmp)
    elseif(m MATCHES "^(pthreads|pthread|posix)$")
        list(APPEND _models pthreads)
    elseif(m STREQUAL "hpx")
        list(APPEND _models hpx)
    elseif(m STREQUAL "auto")
        # configure: auto -> openmp for gcc/icc, pthreads for clang;
        # 'auto' is dropped for other vendors (leaving 'single').
        if(BLIS_CC_VENDOR MATCHES "gcc|icc")
            list(APPEND _models openmp)
        elseif(BLIS_CC_VENDOR MATCHES "clang")
            list(APPEND _models pthreads)
        endif()
    else()
        message(FATAL_ERROR "BLIS: unknown threading model '${m}'")
    endif()
endforeach()
list(REMOVE_DUPLICATES _models) # keeps FIRST occurrences; first = default
list(FIND _models single _has_single)
if(_has_single EQUAL -1)
    list(APPEND _models single)
endif()
if(NOT BLIS_ENABLE_SYSTEM)
    set(_models single)
endif()
set(BLIS_THREADING_MODELS "${_models}")
list(GET _models 0 _tm_default)

foreach(t openmp pthreads hpx)
    set(enable_${t} 0)
    set(enable_${t}_as_def 0)
endforeach()
foreach(t ${BLIS_THREADING_MODELS})
    if(NOT t STREQUAL "single")
        set(enable_${t} 1)
        if(t STREQUAL _tm_default)
            set(enable_${t}_as_def 1)
        endif()
    endif()
endforeach()
set(BLIS_THREAD_DEFAULT "${_tm_default}")

# HPX requires C++17 (mirrors common.mk); enable the language lazily from the
# top-level CMakeLists.

set(enable_jrir_slab 0)
set(enable_jrir_rr 0)
set(enable_jrir_tlb 0)
set(enable_jrir_${BLIS_THREAD_JRIR_PARTITION} 1)
if(NOT BLIS_THREAD_JRIR_PARTITION MATCHES "^(slab|rr|tlb)$")
    message(FATAL_ERROR "BLIS: BLIS_THREAD_JRIR_PARTITION must be slab, rr, or tlb")
endif()

macro(_blis_01 anchor opt)
    if(${opt})
        set(${anchor} 1)
    else()
        set(${anchor} 0)
    endif()
endmacro()

_blis_01(enable_system BLIS_ENABLE_SYSTEM)
_blis_01(enable_tls BLIS_ENABLE_TLS)
_blis_01(enable_pba_pools BLIS_ENABLE_PBA_POOLS)
_blis_01(enable_sba_pools BLIS_ENABLE_SBA_POOLS)
_blis_01(enable_mem_tracing BLIS_ENABLE_MEM_TRACING)
_blis_01(enable_mixed_precision_norm BLIS_ENABLE_MIXED_PRECISION_NORM)
_blis_01(enable_sup_handling BLIS_ENABLE_SUP_HANDLING)
_blis_01(enable_trsm_preinversion BLIS_ENABLE_TRSM_PREINVERSION)
_blis_01(enable_scalapack_compat BLIS_ENABLE_SCALAPACK_COMPAT)
_blis_01(enable_sandbox BLIS_SANDBOX)
_blis_01(enable_shared BLIS_ENABLE_SHARED)
# NOTE: like configure, the bli_config.h BLAS macro reflects the *requested*
# value, while the build-level BLAS objects are force-enabled by CBLAS.
_blis_01(enable_blas BLIS_ENABLE_BLAS)
_blis_01(enable_cblas BLIS_ENABLE_CBLAS)
if(BLIS_ENABLE_CBLAS)
    set(BLIS_EFFECTIVE_BLAS ON) # make-level: CBLAS requires BLAS
else()
    set(BLIS_EFFECTIVE_BLAS "${BLIS_ENABLE_BLAS}")
endif()

set(int_type_size "${BLIS_INT_SIZE}")
set(blas_int_type_size "${BLIS_BLAS_INT_SIZE}")

set(_omit "${BLIS_OMIT_SYMBOLS}")
if(BLIS_ENABLE_LAPACK_COMPAT)
    string(APPEND _omit ",crot,zrot,csymv,zsymv,csyr,zsyr,csyr2,zsyr2,lsame,xerbla,xerbla_array")
endif()
if(BLIS_ENABLE_SCALAPACK_COMPAT)
    string(APPEND _omit ",csymv,zsymv,csyr,zsyr,csyr2,zsyr2")
endif()
string(REPLACE "," ";" _omit_list "${_omit}")
set(omit_symbol_list_defines "")
foreach(s ${_omit_list})
    string(STRIP "${s}" s)
    if(NOT s MATCHES "^[a-z0-9_]+$")
        message(FATAL_ERROR "BLIS: invalid symbol name in omit list: '${s}'")
    endif()
    string(TOUPPER "${s}" S)
    string(APPEND omit_symbol_list_defines "#define BLIS_DISABLE_${S}\n")
endforeach()

if(BLIS_COMPLEX_RETURN STREQUAL "auto")
    set(complex_return_intel 0)
    find_program(
        _blis_fc
        NAMES $ENV{FC} gfortran ifort ifx nvfortran
    )
    if(_blis_fc)
        execute_process(
            COMMAND "${_blis_fc}" --version
            OUTPUT_VARIABLE _fc_v
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            TIMEOUT 15
        )
        if(_fc_v MATCHES "IFORT|IFX")
            set(complex_return_intel 1)
        elseif(_fc_v MATCHES "NVIDIA|PGI")
            set(complex_return_intel 1)
            if(BLIS_HOST_MACHINE MATCHES "ppc64le")
                set(complex_return_intel 0)
            endif()
            if(BLIS_HOST_MACHINE MATCHES "aarch64" AND _fc_v MATCHES "([0-9]+)\\.([0-9]+)")
                if(CMAKE_MATCH_1 GREATER 23 OR (CMAKE_MATCH_1 EQUAL 23 AND CMAKE_MATCH_2 GREATER_EQUAL 9))
                    # nvfortran >= 23.9 on aarch64 uses the gnu convention.
                    set(complex_return_intel 0)
                endif()
            endif()
        endif()
    endif()
    unset(_blis_fc CACHE)
elseif(BLIS_COMPLEX_RETURN STREQUAL "intel")
    set(complex_return_intel 1)
else()
    set(complex_return_intel 0)
endif()

if(BLIS_ENABLE_MEMKIND STREQUAL "auto")
    _blis_probe_memkind(_mk_found)
    if(_mk_found)
        set(enable_memkind 1)
    else()
        set(enable_memkind 0)
    endif()
elseif(BLIS_ENABLE_MEMKIND STREQUAL "on")
    _blis_probe_memkind(_mk_found)
    if(NOT _mk_found)
        message(FATAL_ERROR "BLIS: BLIS_ENABLE_MEMKIND=on but libmemkind was not found")
    endif()
    set(enable_memkind 1)
else()
    set(enable_memkind 0)
endif()

set(addon_list_includes "")
_blis_01(enable_addons BLIS_ENABLE_ADDONS)
foreach(a IN LISTS BLIS_ENABLE_ADDONS)
    string(APPEND addon_list_includes "#include \"${a}.h\"\n")
endforeach()
