# Two lists grow in step. BLIS_PC_LIBS holds the flags exactly as make writes
# them (blis.pc, exported package). BLIS_LINK_LIBS holds the CMake link items.
set(BLIS_LINK_LIBS "")
set(BLIS_PC_LIBS "")
macro(blis_add_link _cmake _pc)
    list(APPEND BLIS_LINK_LIBS ${_cmake})
    list(APPEND BLIS_PC_LIBS ${_pc})
endmacro()
blis_add_link("${BLIS_LDFLAGS_PRESET}" "${BLIS_LDFLAGS_PRESET}")
if(NOT BLIS_IS_MSVC STREQUAL "yes")
    blis_add_link(m -lm)
endif()
# pthread library: -lpthread by default (LIBPTHREAD overrides it), needed for
# pthread_once unless --disable-system. common.mk links the raw $(LIBPTHREAD)
# on every OS, so Darwin keeps -lpthread where Threads::Threads drops it.
# The Bionic probe runs here because the pthread items below need its result.
if(BLIS_HOST_OS STREQUAL "Linux")
    include(CheckCSourceCompiles)
    check_c_source_compiles("#ifndef __BIONIC__\n#error not bionic\n#endif\nint main(void){return 0;}" BLIS_ON_BIONIC)
endif()
set(_blis_libpthread FALSE)
if(BLIS_ENABLE_SYSTEM)
    if(BLIS_CC_VENDOR STREQUAL "icc")
        list(REMOVE_ITEM BLIS_LINK_LIBS m -lm)
        list(REMOVE_ITEM BLIS_PC_LIBS -lm)
    endif()
    if(NOT (WIN32 AND BLIS_CC_VENDOR MATCHES "clang|aocc") AND NOT BLIS_IS_MSVC STREQUAL "yes")
        set(_blis_libpthread TRUE)
        if(NOT DEFINED LIBPTHREAD)
            set(_blis_libpthread_lib "-lpthread")
        else()
            set(_blis_libpthread_lib "${LIBPTHREAD}")
        endif()
        # Split the override into one link item for each word, as make's
        # unquoted $(LIBPTHREAD) is shell-split. blis.pc keeps the raw string.
        separate_arguments(_blis_libpthread_items UNIX_COMMAND "${_blis_libpthread_lib}")
        # Bionic keeps pthread in libc and the NDK ships no libpthread, so the
        # default -lpthread becomes Threads::Threads there. An explicit
        # LIBPTHREAD override still goes in raw, split as above.
        # shortcut: Bionic link items use Threads::Threads while make keeps raw
        # -lpthread in LDFLAGS, drop when common.mk drops -lpthread on Bionic.
        if(NOT DEFINED LIBPTHREAD AND (BLIS_ON_BIONIC OR BLIS_HOST_OS STREQUAL "Android"))
            find_package(Threads)
            set(_blis_libpthread_items Threads::Threads)
        endif()
        blis_add_link("${_blis_libpthread_items}" "${_blis_libpthread_lib}")
    endif()
endif()
if(enable_memkind AND NOT BLIS_DEBUG_TYPE STREQUAL "sde")
    blis_add_link(-lmemkind -lmemkind)
endif()
if(BLIS_ENABLE_ASAN)
    blis_add_link(-fsanitize=address -fsanitize=address)
endif()
# shortcut: aocc gets -fopenmp although common.mk omits it; drop when upstream common.mk adds aocc.
if((BLIS_CC_VENDOR STREQUAL "gcc" OR BLIS_CC_VENDOR STREQUAL "icc" OR BLIS_CC_VENDOR STREQUAL "clang"
    OR BLIS_CC_VENDOR STREQUAL "aocc") AND "openmp" IN_LIST BLIS_THREADING_MODELS)
    blis_add_link(-fopenmp -fopenmp)
endif()
# common.mk adds the threading link flags only in its gcc, icc and clang blocks.
# shortcut: aocc gets -pthread although common.mk omits it; drop when upstream common.mk adds aocc.
if(BLIS_CC_VENDOR STREQUAL "gcc" OR BLIS_CC_VENDOR STREQUAL "icc" OR BLIS_CC_VENDOR STREQUAL "clang" OR BLIS_CC_VENDOR STREQUAL "aocc")
    # configure:3821 and BlisOptions.cmake:192 force single threading when system support is off, so this branch never sees pthreads with BLIS_ENABLE_SYSTEM OFF.
    if("pthreads" IN_LIST BLIS_THREADING_MODELS)
        # The pthreads blocks add LIBPTHREAD to LDFLAGS again, so blis.pc
        # carries it twice (the unconditional copy sits above).
        if(_blis_libpthread)
            # shortcut: make links LIBPTHREAD twice for pthreads; CMake links it once so a whole-archive override does not duplicate symbols.
            list(APPEND BLIS_PC_LIBS "${_blis_libpthread_lib}")
        endif()
    endif()
endif()
foreach(c ${BLIS_CONFIG_LIST_FAM})
    if(BLIS_DEFS_LDFLAGS_${c})
        separate_arguments(_cfg_ld UNIX_COMMAND "${BLIS_DEFS_LDFLAGS_${c}}")
        blis_add_link("${_cfg_ld}" "${_cfg_ld}")
    endif()
endforeach()

# HPX threading: pkg-config based, requires C++.
if("hpx" IN_LIST BLIS_THREADING_MODELS)
    enable_language(CXX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(HPX REQUIRED hpx_component)
    set(BLIS_CXXLANGFLAGS -std=c++17)
    # common.mk: LDFLAGS += pkg-config --libs minus -shared (raw flags, no target
    # name, so blis.pc and the exported target carry real linker options).
    set(_hpx_ld ${HPX_LDFLAGS})
    list(FILTER _hpx_ld EXCLUDE REGEX "^-shared$")
    blis_add_link("${_hpx_ld}" "${_hpx_ld}")
    if(APPLE)
        foreach(_f ${_hpx_ld})
            if(_f MATCHES "^-L(.+)")
                blis_add_link("-Wl,-rpath,${CMAKE_MATCH_1}" "-Wl,-rpath,${CMAKE_MATCH_1}")
            endif()
        endforeach()
    endif()
    # common.mk: CTHREADFLAGS += HPX cflags minus -std=*, used for C and C++.
    set(_hpx_cf ${HPX_CFLAGS_OTHER})
    list(FILTER _hpx_cf EXCLUDE REGEX "^-std=")
    list(APPEND BLIS_CTHREADFLAGS ${_hpx_cf})
else()
    set(BLIS_CXXLANGFLAGS -std=c++11)
endif()

if(BLIS_HOST_OS STREQUAL "Linux")
    # common.mk adds -lrt for OS_NAME Linux only, last (after the threading and
    # HPX flags), except on Android/Bionic.
    if(NOT BLIS_ON_BIONIC)
        blis_add_link(rt -lrt)
    endif()
endif()

# CMake links an item as given: "-framework X" is one item, not two, or the
# second word becomes -lX. blis.pc keeps the two words (BLIS_PC_LIBS).
set(_blis_joined "")
set(_blis_pend "")
foreach(_l ${BLIS_LINK_LIBS})
    if(_blis_pend)
        list(APPEND _blis_joined "${_blis_pend} ${_l}")
        set(_blis_pend "")
    elseif(_l MATCHES "^-(weak_|needed_|reexport_)?framework$")
        set(_blis_pend "${_l}")
    else()
        list(APPEND _blis_joined "${_l}")
    endif()
endforeach()
if(_blis_pend)
    list(APPEND _blis_joined "${_blis_pend}")
endif()
set(BLIS_LINK_LIBS "")
set(BLIS_LINK_OPTS "")
foreach(_l ${_blis_joined})
    # target_link_libraries rejects any item with "::" that names no target
    # (CMP0028), so a flag such as -Wl,-rpath,a::b goes through target_link_options.
    if(_l MATCHES "^[-/].*::")
        list(APPEND BLIS_LINK_OPTS "${_l}")
    else()
        list(APPEND BLIS_LINK_LIBS "${_l}")
    endif()
endforeach()
