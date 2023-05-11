##Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.##

cmake_minimum_required(VERSION 3.0.0)

project(AOCL-LibBlis-Win C CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")


SET(AOCL_BLIS_FAMILY "zen" CACHE STRING "AOCL BLIS family name")
SET(OpenMP_libomp_LIBRARY "C:/Program Files/LLVM/lib/libomp.lib" CACHE STRING "openmp library
path")
set(TARGET_ARCH ${AOCL_BLIS_FAMILY})
set(AOCL_BLIS_ZEN TRUE)
set (PYTHON_EXE "python")

if ("${AOCL_BLIS_FAMILY}" STREQUAL "")
  message(FATAL_ERROR "Machine configuration missing! Select one of zen, zen2, zen3, zen4 or amdzen")
endif ()

if (${AOCL_BLIS_FAMILY} STREQUAL "auto")
  set(AUTO_CONFIG_PY "${CMAKE_SOURCE_DIR}/build/auto_config.py")
  # Run python script to find the architecture family name
  execute_process(
        COMMAND ${PYTHON_EXE} ${AUTO_CONFIG_PY}
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
  message( STATUS "Auto configuring the family :" ${CMD_OUTPUT})
  set(AOCL_BLIS_FAMILY ${CMD_OUTPUT})
endif ()

if(${AOCL_BLIS_FAMILY} STREQUAL "zen")
  add_definitions(-DBLIS_FAMILY_ZEN)
  add_definitions(-DBLIS_CONFIG_ZEN)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen2")
  add_definitions(-DBLIS_FAMILY_ZEN2)
  add_definitions(-DBLIS_CONFIG_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen3")
  add_definitions(-DBLIS_FAMILY_ZEN3)
  add_definitions(-DBLIS_CONFIG_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen4")
  add_definitions(-DBLIS_FAMILY_ZEN4)
  add_definitions(-DBLIS_CONFIG_ZEN4)
  add_definitions(-DBLIS_KERNELS_SKX)
  add_definitions(-DBLIS_KERNELS_ZEN4)
  add_definitions(-DBLIS_KERNELS_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_HASWELL)
elseif (${AOCL_BLIS_FAMILY} STREQUAL "amdzen")
  set(AOCL_BLIS_ZEN FALSE)
  add_definitions(-DBLIS_FAMILY_AMDZEN)
  add_definitions(-DBLIS_CONFIG_ZEN4)
  add_definitions(-DBLIS_CONFIG_ZEN3)
  add_definitions(-DBLIS_CONFIG_ZEN2)
  add_definitions(-DBLIS_CONFIG_ZEN)
  add_definitions(-DBLIS_CONFIG_GENERIC)
  add_definitions(-DBLIS_KERNELS_SKX)
  add_definitions(-DBLIS_KERNELS_ZEN4)
  add_definitions(-DBLIS_KERNELS_ZEN3)
  add_definitions(-DBLIS_KERNELS_ZEN2)
  add_definitions(-DBLIS_KERNELS_HASWELL)
  add_definitions(-DBLIS_KERNELS_ZEN)
  add_definitions(-DBLIS_KERNELS_GENERIC)
else ()
  message(FATAL_ERROR "Wrong machine configuration. Select one of zen, zen2, zen3, zen4 or amdzen")
endif ()

set(TARGET_ARCH ${AOCL_BLIS_FAMILY})
message("AOCL_BLIS_FAMILY selected:${AOCL_BLIS_FAMILY}")

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(ENABLE_VERBOSE "Enable VERBOSE mode for build" OFF)
option(ENABLE_MULTITHREADING "Enable Multi threading" OFF)
option(ENABLE_OPENMP "Enable Openmp mode" OFF)
option(ENABLE_JRIR_SLAB "Request slab thread in jr and ir loops" ON)
option(ENABLE_JRIR_RR "Request round robin thread in jr and ir loops" OFF)
option(ENABLE_PBA_POOLS "Internal memory pools for packing blocks" ON)
option(ENABLE_SBA_POOLS "Internal memory pools for small blocks" ON)
option(ENABLE_MEM_TRACING "Memory tracing output" OFF)
option(ENABLE_BLAS "BLAS compatiblity layer" ON)
option(ENABLE_CBLAS "CBLAS compatiblity layer" ON)
option(ENABLE_MIXED_DT "Mixed datatype" ON)
option(ENABLE_MIXED_DT_EXTRA_MEM "Mixed datatype optimization requiring extra memory" ON)
option(ENABLE_SUP_HANDLING "Small matrix handling" ON)
option(ENABLE_MEMKIND "libmemkind for manage memory pools" OFF)
option(ENABLE_PRAGMA_OMP_SIMD "pragma openmp simd" ON)
option(ENABLE_SANDBOX "Sandbox implementation for gemm" OFF)
option(BLIS_ENABLE_ILP64 "ENABLE BLIS ILP64" OFF)
option(ENABLE_INT_TYPE_SIZE " Internal BLIS integers ,used in native BLIS interfaces based on architecture dependent " ON)
option(ENABLE_BLASTEST "Enable the blastest" OFF)
option(ENABLE_TESTCPP_TESTING "Enabling testcpp" OFF)
option (ENABLE_NO_UNDERSCORE_API "export APIs without underscore" OFF)
option (ENABLE_UPPERCASE_API "export APIs with uppercase" OFF)
option (ENABLE_COMPLEX_RETURN_INTEL "Enable complex_return_intel" OFF)
option (ENABLE_TRSM_PREINVERSION "Enable TRSM preinversion" ON)
option (ENABLE_AOCL_DYNAMIC "Enable Dynamic Multi-threading" OFF)
option(DISABLE_BLIS_ARCH_TYPE "Disable BLIS_ARCH_TYPE and BLIS_MODEL_TYPE functionality" OFF)
option(RENAME_BLIS_ARCH_TYPE "Rename BLIS_ARCH_TYPE env var renamed to supplied value" BLIS_ARCH_TYPE)
option(RENAME_BLIS_MODEL_TYPE "Rename BLIS_MODEL_TYPE env var renamed to supplied value" BLIS_MODEL_TYPE)

if (${AOCL_BLIS_FAMILY} STREQUAL "amdzen")
  set(REF_KERNEL_MIRRORING_PY "${CMAKE_SOURCE_DIR}/build/blis_ref_kernel_mirror.py")
  message("ref_kernel mirroring for fat binary")
  # Run python script to find the architecture family name
  execute_process(
        COMMAND ${PYTHON_EXE} ${REF_KERNEL_MIRRORING_PY} ${CMAKE_BINARY_DIR}
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
  message( STATUS "Ref Kernel Mirroring :" ${CMD_OUTPUT})
endif()
if(ENABLE_NO_UNDERSCORE_API)
    add_definitions(-DBLIS_ENABLE_NO_UNDERSCORE_API)
endif()

if(ENABLE_COMPLEX_RETURN_INTEL)
    set(BLIS_ENABLE_COMPLEX_RETURN_INTEL TRUE)
else()
    set(BLIS_DISABLE_COMPLEX_RETURN_INTEL TRUE)
endif()

if(ENABLE_UPPERCASE_API)
     add_definitions(-DBLIS_ENABLE_UPPERCASE_API)
endif()

if(ENABLE_AOCL_DYNAMIC)
     set(AOCL_DYNAMIC TRUE)
endif()

if (BUILD_SHARED_LIBS)
    set(BLIS_ENABLE_SHARED TRUE)
    if(ENABLE_BLASTEST)
        add_definitions(-DAOCL_SUPPORT_BLASTEST_FOR_SHARED)
    endif()
endif ()

# Enable LP64/ILP64
if (BLIS_ENABLE_ILP64)
  set(BLIS_BLAS_INT_TYPE_SIZE TRUE)
  set (BLAS_INT_TYPE_SIZE "64")
  add_definitions(-DF2C_ENABLE_ILP64)
else ()
  set(BLIS_BLAS_INT_TYPE_SIZE TRUE)
  set (BLAS_INT_TYPE_SIZE "32")
endif ()

if (ENABLE_TRSM_PREINVERSION)
  set(BLIS_ENABLE_TRSM_PREINVERSION TRUE)
else()
  add_definitions(-DBLIS_DISABLE_TRSM_PREINVERSION)
endif()

if (ENABLE_INT_TYPE_SIZE)
  set(BLIS_INT_TYPE_SIZE TRUE)
  set (INT_TYPE_SIZE "64")
else ()
  set(BLIS_INT_TYPE_SIZE TRUE)
  set (INT_TYPE_SIZE "32")
endif ()

if (BLIS_ENABLE_ILP64 AND NOT ENABLE_INT_TYPE_SIZE)
    message(FATAL_ERROR "for ILP64 we must enable ENABLE_INT_TYPE_SIZE with BLIS_INT_TYPE_SIZE = 64 ")
endif ()

if (ENABLE_VERBOSE)
    set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)
endif ()

if (ENABLE_JRIR_RR)
    message("Round robin thread method enabled")
    set(BLIS_ENABLE_JRIR_RR TRUE)
    set(BLIS_ENABLE_JRIR_SLAB FALSE)
elseif (ENABLE_JRIR_SLAB)
    message("SLAB thread method enabled")
    set(BLIS_ENABLE_JRIR_SLAB TRUE)
    set(BLIS_ENABLE_JRIR_RR FALSE)
else ()
    message("Unsupported method of thread partitioning in jr and ir loops")
endif ()

if (ENABLE_PBA_POOLS)
    set(BLIS_ENABLE_PBA_POOLS TRUE)
endif ()

if (ENABLE_SBA_POOLS)
    set(BLIS_ENABLE_SBA_POOLS TRUE)
endif ()

if (ENABLE_MEM_TRACING)
    set(BLIS_ENABLE_MEM_TRACING FALSE)
endif ()

if (ENABLE_BLAS)
    add_definitions(-DBLIS_ENABLE_BLAS)
    set(BLIS_ENABLE_BLAS TRUE)
else ()
    add_definitions(-DBLIS_DISABLE_BLAS)
    set(BLIS_ENABLE_BLAS FALSE)
endif ()

if (ENABLE_CBLAS)
    add_definitions(-DBLIS_ENABLE_CBLAS)
    set(BLIS_ENABLE_CBLAS TRUE)
    if (NOT ENABLE_BLAS)
        # Force BLAS layer when CBLAS is enabled
        add_definitions(-DBLIS_ENABLE_BLAS)
        set(BLIS_ENABLE_BLAS TRUE)
    endif ()
else ()
    add_definitions(-DBLIS_DISABLE_CBLAS)
    set(BLIS_ENABLE_CBLAS FALSE)
endif ()

if (ENABLE_BLASTEST)
    add_definitions(-DBLIS_ENABLE_BLAS)
    add_definitions(-DBLIS_ENABLE_CBLAS)
endif()

if (ENABLE_TESTCPP_TESTING)
  add_definitions(-DBLIS_ENABLE_BLAS)
  add_definitions(-DBLIS_ENABLE_CBLAS)
endif ()

if (ENABLE_MIXED_DT)
    set(BLIS_ENABLE_MIXED_DT TRUE)
endif ()

if (ENABLE_MIXED_DT_EXTRA_MEM)
    set(BLIS_ENABLE_MIXED_DT_EXTRA_MEM TRUE)
endif ()

if (ENABLE_SUP_HANDLING)
    set(BLIS_ENABLE_SUP_HANDLING TRUE)
endif ()

if (ENABLE_MEMKIND)
    set(BLIS_ENABLE_MEMKIND FALSE)
endif ()

if (ENABLE_PRAGMA_OMP_SIMD)
    set(BLIS_ENABLE_PRAGMA_OMP_SIMD TRUE)
endif ()

if (ENABLE_SANDBOX)
    set(BLIS_ENABLE_SANDBOX FALSE)
endif ()

include_directories(${PROJECT_SOURCE_DIR}/external/msvc)
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MD ")
#add_definitions(-DBLIS_IS_BUILDING_LIBRARY)
if(NOT BUILD_SHARED_LIBS)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MT ")
add_definitions(-DBLIS_IS_BUILDING_LIBRARY)
endif()

if(ENABLE_MULTITHREADING)
  if(BUILD_SHARED_LIBS)
    set(LIB_NAME "${PROJECT_NAME}-MT-dll")
  elseif(NOT BUILD_SHARED_LIBS)
    set(LIB_NAME "${PROJECT_NAME}-MT")
  endif()
  if(ENABLE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
      set(BLIS_ENABLE_OPENMP TRUE)
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else()
      message (FATAL_ERROR "Openmp Not Found")
    endif()
  endif()
else()
  if(BUILD_SHARED_LIBS)
    set(LIB_NAME "${PROJECT_NAME}-dll")
  elseif(NOT BUILD_SHARED_LIBS)
    set(LIB_NAME "${PROJECT_NAME}")
  endif()
endif()

if(DISABLE_BLIS_ARCH_TYPE)
    set(BLIS_DISABLE_BLIS_ARCH_TYPE TRUE)
    set(BLIS_DISABLE_BLIS_MODEL_TYPE TRUE)
else()
    set(BLIS_DISABLE_BLIS_ARCH_TYPE FALSE)
    set(BLIS_DISABLE_BLIS_MODEL_TYPE FALSE)
endif()

if(RENAME_BLIS_ARCH_TYPE)
    set(__blis_arch_type_name TRUE)
    set(rename_blis_arch_type "${RENAME_BLIS_ARCH_TYPE}")
else()
    set(__blis_arch_type_name TRUE)
    set(rename_blis_arch_type "BLIS_ARCH_TYPE")
endif()

if(RENAME_BLIS_MODEL_TYPE)
    set(__blis_model_type_name TRUE)
    set(rename_blis_model_type "${RENAME_BLIS_MODEL_TYPE}")
else()
    set(__blis_model_type_name TRUE)
    set(rename_blis_model_type "BLIS_MODEL_TYPE")
endif()

set(CMAKE_BUILD_TYPE ${CMAKE_CONFIGURATION_TYPES})

#print configurations
message("---cmake configurations---")
message(CMAKE_C_COMPILER_ID : ${CMAKE_C_COMPILER_ID})
message(CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE})
message(BLIS_ENABLE_OPENMP : ${BLIS_ENABLE_OPENMP})
message(BLIS_ENABLE_JRIR_SLAB : ${BLIS_ENABLE_JRIR_SLAB})
message(BLIS_ENABLE_JRIR_RR : ${BLIS_ENABLE_JRIR_RR})
message(BLIS_ENABLE_PBA_POOLS : ${BLIS_ENABLE_PBA_POOLS})
message(BLIS_ENABLE_SBA_POOLS : ${BLIS_ENABLE_SBA_POOLS})
message(BLIS_ENABLE_MEM_TRACING : ${BLIS_ENABLE_MEM_TRACING})
message(BLIS_INT_TYPE_SIZE : ${BLIS_INT_TYPE_SIZE})
message(BLIS_BLAS_INT_TYPE_SIZE : ${BLIS_BLAS_INT_TYPE_SIZE})
message(BLIS_ENABLE_BLAS : ${BLIS_ENABLE_BLAS})
message(BLIS_ENABLE_CBLAS : ${BLIS_ENABLE_CBLAS})
message(BLIS_ENABLE_MIXED_DT : ${BLIS_ENABLE_MIXED_DT})
message(BLIS_ENABLE_MIXED_DT_EXTRA_MEM : ${BLIS_ENABLE_MIXED_DT_EXTRA_MEM})
message(BLIS_ENABLE_SUP_HANDLING : ${BLIS_ENABLE_SUP_HANDLING})
message(BLIS_ENABLE_MEMKIND : ${BLIS_ENABLE_MEMKIND})
message(BLIS_ENABLE_PRAGMA_OMP_SIMD : ${BLIS_ENABLE_PRAGMA_OMP_SIMD})
message(BLIS_ENABLE_SANDBOX : ${BLIS_ENABLE_SANDBOX})
message(BLIS_ENABLE_SHARED : ${BLIS_ENABLE_SHARED})
message(DISABLE_BLIS_ARCH_TYPE : ${DISABLE_BLIS_ARCH_TYPE})
message(RENAME_BLIS_ARCH_TYPE : ${RENAME_BLIS_ARCH_TYPE})
message(RENAME_BLIS_MODEL_TYPE : ${RENAME_BLIS_MODEL_TYPE})

SET(ENABLE_SIMD_FLAGS "none" CACHE STRING "Set compiler SIMD flags")
SET_PROPERTY(CACHE ENABLE_SIMD_FLAGS PROPERTY STRINGS none SSE2 AVX AVX2)

if(${ENABLE_SIMD_FLAGS} MATCHES "AVX2")
  add_definitions(/arch:AVX2)
elseif(${ENABLE_SIMD_FLAGS} MATCHES "AVX")
  add_definitions(/arch:AVX)
elseif(${ENABLE_SIMD_FLAGS} MATCHES "SSE2")
  add_definitions(/arch:SSE2)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W0 ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Oi")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP")
set(INTR_GENERAL_LINK_FLAGS "${INTR_GENERAL_LINK_FLAGS} /RELEGE")

add_definitions(-D_CRT_SECURE_NO_DEPRECATE)

#add_definitions(-DBLIS_OS_WINDOWS)
add_definitions(-D_MSC_VER)
if (${AOCL_BLIS_FAMILY} STREQUAL "amdzen")
else()
add_definitions(-DBLIS_CNAME=${TARGET_ARCH})
endif()
# Generate the bli_config.h header file
configure_file (build/bli_win_config.h.in ${CMAKE_SOURCE_DIR}/bli_config.h @ONLY)

include_directories(${CMAKE_SOURCE_DIR}/aocl_dtl)
include_directories(${CMAKE_SOURCE_DIR}/.)
include_directories(${CMAKE_SOURCE_DIR}/include/${TARGET_ARCH})
include_directories(${CMAKE_SOURCE_DIR}/frame/include)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/1e)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/1m)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/1r)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/bb)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/io)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/ri)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/ri3)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/rih)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/ro)
include_directories(${CMAKE_SOURCE_DIR}/frame/include/level0/rpi)
include_directories(${CMAKE_SOURCE_DIR}/frame/thread)
include_directories(${CMAKE_SOURCE_DIR}/frame/base)
include_directories(${CMAKE_SOURCE_DIR}/frame/base/cast)
include_directories(${CMAKE_SOURCE_DIR}/frame/base/check)
include_directories(${CMAKE_SOURCE_DIR}/frame/base/noopt)
include_directories(${CMAKE_SOURCE_DIR}/frame/base/proj)
include_directories(${CMAKE_SOURCE_DIR}/frame/0)
include_directories(${CMAKE_SOURCE_DIR}/frame/0/copysc)
include_directories(${CMAKE_SOURCE_DIR}/frame/1)
include_directories(${CMAKE_SOURCE_DIR}/frame/1d)
include_directories(${CMAKE_SOURCE_DIR}/frame/1f)
include_directories(${CMAKE_SOURCE_DIR}/frame/1m)
include_directories(${CMAKE_SOURCE_DIR}/frame/1m/packm)
include_directories(${CMAKE_SOURCE_DIR}/frame/1m/unpackm)
include_directories(${CMAKE_SOURCE_DIR}/frame/2)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/gemv)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/ger)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/hemv)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/her)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/her2)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/symv)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/syr)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/syr2)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/trmv)
include_directories(${CMAKE_SOURCE_DIR}/frame/2/trsv)
include_directories(${CMAKE_SOURCE_DIR}/frame/3)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/gemm)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/gemm/ind)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/gemmt)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/hemm)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/her2k)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/herk)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/symm)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/syr2k)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/syrk)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/trmm)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/trmm3)
include_directories(${CMAKE_SOURCE_DIR}/frame/3/trsm)
include_directories(${CMAKE_SOURCE_DIR}/frame/compat)
include_directories(${CMAKE_SOURCE_DIR}/frame/compat/cblas)
include_directories(${CMAKE_SOURCE_DIR}/frame/compat/cblas/f77_sub)
include_directories(${CMAKE_SOURCE_DIR}/frame/compat/cblas/src)
include_directories(${CMAKE_SOURCE_DIR}/frame/compat/check)
include_directories(${CMAKE_SOURCE_DIR}/frame/compat/f2c)
include_directories(${CMAKE_SOURCE_DIR}/frame/compat/f2c/util)
include_directories(${CMAKE_SOURCE_DIR}/frame/ind)
include_directories(${CMAKE_SOURCE_DIR}/frame/ind/cntx)
include_directories(${CMAKE_SOURCE_DIR}/frame/ind/oapi)
include_directories(${CMAKE_SOURCE_DIR}/frame/ind/tapi)
include_directories(${CMAKE_SOURCE_DIR}/frame/ind/ukernels)
include_directories(${CMAKE_SOURCE_DIR}/frame/util)
include_directories(${CMAKE_SOURCE_DIR}/config/generic)
include_directories(${CMAKE_SOURCE_DIR}/config/zen)
include_directories(${CMAKE_SOURCE_DIR}/config/zen2)
include_directories(${CMAKE_SOURCE_DIR}/config/zen3)
include_directories(${CMAKE_SOURCE_DIR}/config/zen4)
if(${AOCL_BLIS_FAMILY} STREQUAL "amdzen")
    include_directories(${CMAKE_BINARY_DIR}/ref_kernels/generic)
    include_directories(${CMAKE_BINARY_DIR}/ref_kernels/zen)
    include_directories(${CMAKE_BINARY_DIR}/ref_kernels/zen2)
    include_directories(${CMAKE_BINARY_DIR}/ref_kernels/zen3)
    include_directories(${CMAKE_BINARY_DIR}/ref_kernels/zen4)
endif()
include_directories(${CMAKE_SOURCE_DIR}/ref_kernels)
include_directories(${CMAKE_SOURCE_DIR}/kernels)
include_directories(${CMAKE_SOURCE_DIR}/kernels/haswell)
include_directories(${CMAKE_SOURCE_DIR}/kernels/haswell/3)
include_directories(${CMAKE_SOURCE_DIR}/kernels/haswell/3/sup)
include_directories(${CMAKE_SOURCE_DIR}/kernels/haswell/3/sup/d6x8)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen/1)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen/1f)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen/1m)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen/2)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen/3)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen/3/sup)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen2)
include_directories(${CMAKE_SOURCE_DIR}/kernels/zen4)
include_directories(${CMAKE_SOURCE_DIR}/kernels/skx)
include_directories(${CMAKE_SOURCE_DIR}/kernels/skx/3)
file(GLOB headers ${CMAKE_SOURCE_DIR}/*.h)

# Monolithic Header generation
find_package(PythonLibs 3 REQUIRED)

string(APPEND HEADER_PATH
if(${AOCL_BLIS_FAMILY} STREQUAL "zen")
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/haswell/"
elseif (${AOCL_BLIS_FAMILY} STREQUAL "zen2")
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/haswell/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/amdzen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen3/"
     " ${CMAKE_CURRENT_SOURCE_DIR}/config/zen4/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/config/generic/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/zen/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/kernels/haswell/"
endif ()
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/0/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/0/copysc/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1d/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1f/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1m/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1m/packm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/1m/unpackm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/gemv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/ger/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/hemv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/her/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/her2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/symv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/syr/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/syr2/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/trmv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/2/trsv/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/gemm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/gemm/ind/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/gemmt/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/hemm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/her2k/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/herk/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/symm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/syr2k/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/syrk/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/trmm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/trmm3/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/3/trsm/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/cast/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/check/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/noopt/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/base/proj/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/cblas/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/cblas/f77_sub/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/cblas/src/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/check/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/f2c/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/compat/f2c/util/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/1e/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/1m/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/1r/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/bb/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/io/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/ri/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/ri3/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/rih/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/ro/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/include/level0/rpi/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/cntx/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/oapi/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/tapi/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/ind/ukernels/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/thread/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/frame/util/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/aocl_dtl/"
    " ${CMAKE_CURRENT_SOURCE_DIR}/"
)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/include/${TARGET_ARCH})

# Flatten header python script file which expand header contents in blis.h
set(FLATTEN_PY "${CMAKE_SOURCE_DIR}/build/flatten-headers.py")
set(BLIS_H "blis.h")

# Arguements for python script
set(C_COMMENT "-c")
set(VERBOSE "-v1")
set(INPUT "${CMAKE_SOURCE_DIR}/frame/include/${BLIS_H}")
set(OUTPUT "${CMAKE_SOURCE_DIR}/include/${TARGET_ARCH}/${BLIS_H}")
set(TEMP_DIR "${INCLUDE}")
set(DIR_H_PATH "${HEADER_PATH}")

# Run python script to generate monolithic header at configuration time
execute_process(
        COMMAND ${PYTHON_EXE} ${FLATTEN_PY} "${C_COMMENT}" "${VERBOSE}" "${INPUT}" "${OUTPUT}" "${TEMP_DIR}" "${DIR_H_PATH}"
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT)
message( STATUS "Generating monolithic header file :" ${CMD_OUTPUT})

# Logic to generate the cblas.h in include folder.
set(CBLAS_H "cblas.h")
# Arguements for python script
set(C_COMMENT "-c")
set(VERBOSE "-v1")
set(INPUT "${CMAKE_SOURCE_DIR}/frame/compat/cblas/src/${CBLAS_H}")
set(OUTPUT "${CMAKE_SOURCE_DIR}/include/${TARGET_ARCH}/${CBLAS_H}")
set(TEMP_DIR "${INCLUDE}")
set(DIR_H_PATH "${HEADER_PATH}")

# Run python script to generate monolithic header at configuration time
execute_process(
        COMMAND ${PYTHON_EXE} ${FLATTEN_PY} "${C_COMMENT}" "${VERBOSE}" "${INPUT}" "${OUTPUT}" "${TEMP_DIR}" "${DIR_H_PATH}"
        RESULT_VARIABLE CMD_RESULT
        OUTPUT_VARIABLE CMD_OUTPUT)
message( STATUS "Generating monolithic cblas header file :" ${CMD_OUTPUT})

# setting the blis version string
file (STRINGS "version" BLIS_VERSION)
set(BLIS_VERSION_STRING ${BLIS_VERSION})
string(TIMESTAMP BUILD_DATE "%Y%m%d")
add_definitions(-DBLIS_VERSION_STRING="AOCL-BLIS ${BLIS_VERSION_STRING} Build ${BUILD_DATE}")

if(BUILD_SHARED_LIBS)
    add_library("${PROJECT_NAME}" SHARED ${CMAKE_SOURCE_DIR}/bli_config.h
                         ${CMAKE_SOURCE_DIR}/include/${TARGET_ARCH}/blis.h
                         ${headers} 
                         $<TARGET_OBJECTS:zen_1>
                         $<TARGET_OBJECTS:zen_1f>
                         $<TARGET_OBJECTS:zen_2>
                         $<TARGET_OBJECTS:zen_3>
                         $<TARGET_OBJECTS:zen_3_sup>
                         $<TARGET_OBJECTS:zen4_1>
                         $<TARGET_OBJECTS:zen4_1m>
                         $<TARGET_OBJECTS:zen4_3>
                         $<TARGET_OBJECTS:zen4_3sup>
                         $<TARGET_OBJECTS:zen4_3supd24x8>
                         $<TARGET_OBJECTS:skx_3>
                         $<TARGET_OBJECTS:haswell_1m>
                         $<TARGET_OBJECTS:haswell_3>
                         $<TARGET_OBJECTS:haswell_3sup>
                         $<TARGET_OBJECTS:haswell_3supd6x8>
                         )
    if(ENABLE_OPENMP)
        target_link_libraries("${PROJECT_NAME}" PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_definitions("${PROJECT_NAME}" PUBLIC -DBLIS_IS_BUILDING_LIBRARY)
    set_target_properties("${PROJECT_NAME}" PROPERTIES LINKER_LANGUAGE C OUTPUT_NAME "${LIB_NAME}")
endif()
if(NOT BUILD_SHARED_LIBS)
    add_library("${PROJECT_NAME}" STATIC ${CMAKE_SOURCE_DIR}/bli_config.h
                         ${CMAKE_SOURCE_DIR}/include/${TARGET_ARCH}/blis.h
                         ${headers} 
                         $<TARGET_OBJECTS:zen_1>
                         $<TARGET_OBJECTS:zen_1f>
                         $<TARGET_OBJECTS:zen_2>
                         $<TARGET_OBJECTS:zen_3>
                         $<TARGET_OBJECTS:zen_3_sup>
                         $<TARGET_OBJECTS:zen4_1>
                         $<TARGET_OBJECTS:zen4_1m>
                         $<TARGET_OBJECTS:zen4_3>
                         $<TARGET_OBJECTS:zen4_3sup>
                         $<TARGET_OBJECTS:zen4_3supd24x8>
                         $<TARGET_OBJECTS:skx_3>
                         $<TARGET_OBJECTS:haswell_1m>
                         $<TARGET_OBJECTS:haswell_3>
                         $<TARGET_OBJECTS:haswell_3sup>
                         $<TARGET_OBJECTS:haswell_3supd6x8>
                         )
    if(ENABLE_OPENMP)
        set_target_properties("${PROJECT_NAME}" PROPERTIES LINKER_LANGUAGE C OUTPUT_NAME "${LIB_NAME}" STATIC_LIBRARY_OPTIONS "${OpenMP_libomp_LIBRARY}")
    else()
    	set_target_properties("${PROJECT_NAME}" PROPERTIES LINKER_LANGUAGE C OUTPUT_NAME "${LIB_NAME}")
    endif()
endif()

link_directories(${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
add_definitions(-DEXPMODULE)

add_subdirectory(config)
add_subdirectory(ref_kernels)
add_subdirectory(kernels)
add_subdirectory(frame)
add_subdirectory(aocl_dtl)
add_subdirectory(test)
add_subdirectory(testsuite)
add_subdirectory(bench)
if(ENABLE_TESTCPP_TESTING)
    add_subdirectory(vendor/testcpp)
endif()
if (ENABLE_BLASTEST)
    add_subdirectory(blastest)
endif()
