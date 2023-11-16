##Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.##

cmake_minimum_required(VERSION 3.15.0)
if(WIN32)
    project(AOCL-LibBlis LANGUAGES C CXX)
else()
    project(AOCL-LibBlis LANGUAGES C CXX Fortran)
endif()

# Set the C standard to C99.
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
# Set the C++ standard to C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Enable IDE folders for targets.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Find a python interpreter.
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT Python_FOUND)
    message(SEND_ERROR "Could not find working python interperter! Cannot continue.")
endif()
# Functionality that prints configuration usage.
option(PRINT_CONFIGURE_HELP "Print CMake Configuration Usage" OFF)
if(PRINT_CONFIGURE_HELP)
    execute_process(COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/build/cmake/config_print.py)
    return()
endif ()

if(WIN32)
    set(BLIS_CONFIG_FAMILY "auto" CACHE STRING "Set the configuration family for which the BLIS library will be built.")
else()
    set(BLIS_CONFIG_FAMILY "" CACHE STRING "Set the configuration family for which the BLIS library will be built.")
endif()
set_property(CACHE BLIS_CONFIG_FAMILY PROPERTY STRINGS "auto" "generic" "zen" "zen2" "zen3" "zen4" "amdzen")
# Throw an error if CMake was configured with a configuration which is not enabled yet.
if(NOT ((BLIS_CONFIG_FAMILY STREQUAL auto) OR
        (BLIS_CONFIG_FAMILY STREQUAL generic) OR
        (BLIS_CONFIG_FAMILY STREQUAL zen) OR
        (BLIS_CONFIG_FAMILY STREQUAL zen2) OR
        (BLIS_CONFIG_FAMILY STREQUAL zen3) OR
        (BLIS_CONFIG_FAMILY STREQUAL zen4) OR
        (BLIS_CONFIG_FAMILY STREQUAL amdzen)))
    message(FATAL_ERROR "Configuration for ${BLIS_CONFIG_FAMILY} is not supported. \
        Please re-run cmake and specify one of the following configurations for BLIS_CONFIG_FAMILY: \
        auto, zen, zen2, zen3, zen4, amdzen, generic.")
endif()

# automatic hardware detection
if(BLIS_CONFIG_FAMILY STREQUAL "auto")
    message(STATUS "automatic configuration requested")
    set(auto_detect_source_files
        "${CMAKE_SOURCE_DIR}/build/detect/config/config_detect.c"
        "${CMAKE_SOURCE_DIR}/frame/base/bli_arch.c"
        "${CMAKE_SOURCE_DIR}/frame/base/bli_cpuid.c"
        "${CMAKE_SOURCE_DIR}/frame/base/bli_env.c"
       )
    set(frame_include " ${CMAKE_SOURCE_DIR}/frame/include")
    set(base_include " ${CMAKE_SOURCE_DIR}/frame/base")
    set(thread_include " ${CMAKE_SOURCE_DIR}/frame/thread")
    # Try building an executable from one or more source files.
    # Build success returns TRUE and build failure returns FALSE in COMPILERESULT.
    # If the build succeeds, this runs the executable and stores the exit code in RUNRESULT.
    # If the executable was built, but failed to run, then RUNRESULT will be set to FAILED_TO_RUN
    # RUN_OUTPUT_VARIABLE <var> Report the output from running the executable in a given variable
    try_run(RUNRESULT COMPILERESULT "${CMAKE_BINARY_DIR}/temp" SOURCES ${auto_detect_source_files}
            COMPILE_DEFINITIONS -I${frame_include} -I${base_include} -I${thread_include}
                               -DBLIS_CONFIGURETIME_CPUID -DBLIS_CONFIG_SKX -DBLIS_CONFIG_KNL
                               -DBLIS_CONFIG_HASWELL -DBLIS_CONFIG_SANDYBRIDGE -DBLIS_CONFIG_PENRYN
                               -DBLIS_CONFIG_ZEN4 -DBLIS_CONFIG_ZEN3 -DBLIS_CONFIG_ZEN2 -DBLIS_CONFIG_ZEN
                               -DBLIS_CONFIG_EXCAVATOR -DBLIS_CONFIG_STEAMROLLER -DBLIS_CONFIG_PILEDRIVER
                               -DBLIS_CONFIG_BULLDOZER -DBLIS_CONFIG_THUNDERX2 -DBLIS_CONFIG_CORTEXA57
                               -DBLIS_CONFIG_CORTEXA15 -DBLIS_CONFIG_CORTEXA9
                               -D__blis_arch_type_name="BLIS_ARCH_TYPE" -D__blis_model_type_name="BLIS_MODEL_TYPE"
            RUN_OUTPUT_VARIABLE HARDWARE_ARCH
    )
    string(STRIP "${HARDWARE_ARCH}" HARDWARE_ARCH)
    message(STATUS "automatic hardware detection: " ${HARDWARE_ARCH})
    if( NOT(${HARDWARE_ARCH} STREQUAL zen OR
            ${HARDWARE_ARCH} STREQUAL zen2 OR
            ${HARDWARE_ARCH} STREQUAL zen3 OR
            ${HARDWARE_ARCH} STREQUAL zen4) )
           set(BLIS_CONFIG_FAMILY "generic")
           message(WARNING "Only AMD zen architectures are supported. \
           Detected ${HARDWARE_ARCH} hardware. Defaulting to generic configuration.")
    else()
           set(BLIS_CONFIG_FAMILY ${HARDWARE_ARCH})
    endif()
    message(STATUS "automatic configuration registered: " ${BLIS_CONFIG_FAMILY})
endif()

# Read the registered configuration names and lists into associative arrays.
execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/cmake/read_registry.py "${BLIS_CONFIG_FAMILY}" "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE CMD_RESULT
    OUTPUT_VARIABLE CONFIGURATION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE )
# Returns the list of elements specified by indices from the list.
message(STATUS "configuration '${BLIS_CONFIG_FAMILY}' is registered.")
list(GET CONFIGURATION_STRING 0 CONFIG_LIST)
list(GET CONFIGURATION_STRING 1 KERNEL_LIST)
list(GET CONFIGURATION_STRING 2 KCONFIG_MAP)
# Removing leading and trailing spaces in the string.
string(STRIP "${CONFIG_LIST}" CONFIG_LIST)
string(STRIP "${KERNEL_LIST}" KERNEL_LIST)
string(STRIP "${KCONFIG_MAP}" KCONFIG_MAP)
# Convert from string to list(list is a ";"-separated string)
message(STATUS "${BLIS_CONFIG_FAMILY} is defined as having the following sub-configurations:")
message("    ${CONFIG_LIST}    ")
string(REPLACE " " ";" CONFIG_LIST ${CONFIG_LIST})
message(STATUS "which collectively require the following kernels:")
message("    ${KERNEL_LIST}    ")
string(REPLACE " " ";" KERNEL_LIST ${KERNEL_LIST})
message(STATUS "that has kernel:config pairs:")
message("    ${KCONFIG_MAP}    ")
string(REPLACE " " ";" KCONFIG_MAP ${KCONFIG_MAP})
# Create a #define for the configuration family (config_name).
string(TOUPPER ${BLIS_CONFIG_FAMILY} UCONF)
set(CONFIG_NAME_DEFINE "#define BLIS_FAMILY_${UCONF}\n")
#create a AOCL specific #define
#This macro is enabled only for zen family configurations.
#This enables us to use different cache block sizes for TRSM instead of common level-3 block sizes.
if(BLIS_CONFIG_FAMILY MATCHES "zen|amd64|x86_64")
    set(ENABLE_AOCL_ZEN ON)
    set(ENABLE_AOCL_ZEN_01 1)
else()
    set(ENABLE_AOCL_ZEN OFF)
    set(ENABLE_AOCL_ZEN_01 0)
endif()
# Create a list of #defines, one for each configuration in config_list.
set(CONFIG_LIST_DEFINES "")
foreach(CONF ${CONFIG_LIST})
    string(TOUPPER ${CONF} UCONF)
    set(CONFIG_LIST_DEFINES "${CONFIG_LIST_DEFINES}#define BLIS_CONFIG_${UCONF}\n")
endforeach()
# Create a list of #defines, one for each kernel set in kernel_list.
set(KERNEL_LIST_DEFINES "")
foreach(KERN ${KERNEL_LIST})
    string(TOUPPER ${KERN} UCONF)
    set(KERNEL_LIST_DEFINES "${KERNEL_LIST_DEFINES}#define BLIS_KERNELS_${UCONF}\n")
endforeach()

#------------------------------------
#           Option Setting
#------------------------------------
# Options that are specific to Windows.
if(WIN32)
    option(ENABLE_NO_UNDERSCORE_API "Export APIs without underscore." OFF)
    option(ENABLE_UPPERCASE_API "Export APIs with uppercase." OFF)
    # Setting path to OpenMP runtime.
    set(OpenMP_libomp_LIBRARY "C:/Program Files/LLVM/lib/libomp.lib" CACHE STRING "openmp library path")
endif()
# Debug & Release flags option setting is only available for Linux. On Windows the default flags are used.
if(NOT WIN32)
    set(ENABLE_DEBUG "off" CACHE STRING "Enable debugging symbols in the library.")
    set_property(CACHE ENABLE_DEBUG PROPERTY STRINGS "off" "noopt" "opt")
    if( NOT ((ENABLE_DEBUG STREQUAL "off") OR (ENABLE_DEBUG STREQUAL "noopt") OR (ENABLE_DEBUG STREQUAL "opt")) )
        message(FATAL_ERROR "ENABLE_DEBUG option '${ENABLE_DEBUG}' is not supported. Please use one of the following options \
                during CMake invokation: off, noopt, opt")
    endif()
    # Check if user provided CMAKE_BUILD_TYPE. If that's the case, map it to the internal ENABLE_DEBUG type
    # and clean cache from CMAKE_BUILD_TYPE. We do this because CMake will add some flags depending on the
    # the build type and on Linux we want to have more control over what flags are being used.
    if(CMAKE_BUILD_TYPE)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(ENABLE_DEBUG "noopt")
        elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
            set(ENABLE_DEBUG "off")
        elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            set(ENABLE_DEBUG "opt")
        else()
            message(FATAL_ERROR "Configured CMake with incompatible CMAKE_BUILD_TYPE. Only Debug, RelWithDebInfo and Release are supported. \
                    This is due to matching this flag to BLIS internal options corresponding to ENABLE_DEBUG: off, noopt, opt.")
        endif()
        message(WARNING "When CMAKE_BUILD_TYPE is used, BLIS-specific variable ENABLE_DEBUG gets overwritten accordingly.")
        set(CMAKE_BUILD_TYPE "")
    endif()
endif()
# Build shared libraries by default
option(BUILD_SHARED_LIBS "Build shared libraries (.dll/.so) instead of static ones (.lib/.a)" ON)
option(ENABLE_SYSTEM "Check if we are building with or without operating system support" ON)
set(ENABLE_THREADING "no" CACHE STRING "the threading flag")
if(WIN32)
    set_property(CACHE ENABLE_THREADING PROPERTY STRINGS "openmp" "no")
    if( NOT ((ENABLE_THREADING STREQUAL "openmp") OR (ENABLE_THREADING STREQUAL "no")) )
        message(FATAL_ERROR "ENABLE_THREADING option '${ENABLE_THREADING}' is not supported. Please use one of the following options \
                during CMake invokation: openmp, no")
    endif()
else()
    set_property(CACHE ENABLE_THREADING PROPERTY STRINGS "openmp" "pthreads" "no")
    if( NOT ((ENABLE_THREADING STREQUAL "openmp") OR (ENABLE_THREADING STREQUAL "pthreads") OR (ENABLE_THREADING STREQUAL "no")) )
        message(FATAL_ERROR "ENABLE_THREADING option '${ENABLE_THREADING}' is not supported. Please use one of the following options \
                during CMake invokation: openmp, pthreads, no")
    endif()
endif()
set(THREAD_PART_JRIR "slab" CACHE STRING "The method of assigning micropanels to threads in the JR and JR loops.")
set_property(CACHE THREAD_PART_JRIR PROPERTY STRINGS "slab" "rr")
if( NOT ((THREAD_PART_JRIR STREQUAL "slab") OR (THREAD_PART_JRIR STREQUAL "rr")) )
    message(FATAL_ERROR "THREAD_PART_JRIR option '${THREAD_PART_JRIR}' is not supported. Please use one of the following options \
            during CMake invokation: slab, rr")
endif()
# Export symbols only for Linux.
if(NOT WIN32)
    set(EXPORT_SHARED "public" CACHE STRING "Specify the subset of library symbols that are exported within a shared library.")
    set_property(CACHE EXPORT_SHARED PROPERTY STRINGS "public" "all")
    if( NOT ((EXPORT_SHARED STREQUAL "public") OR (EXPORT_SHARED STREQUAL "all")) )
        message(FATAL_ERROR "EXPORT_SHARED option '${EXPORT_SHARED}' is not supported. Please use one of the following options \
        during CMake invokation: public, all")
    endif()
endif()
option(ENABLE_PBA_POOLS "Internal memory pools for packing blocks" ON)
option(ENABLE_SBA_POOLS "Internal memory pools for small blocks" ON)
option(ENABLE_MEM_TRACING "Memory tracing output" OFF)
set(INT_SIZE "auto" CACHE STRING "BLIS API integer size")
set_property(CACHE INT_SIZE PROPERTY STRINGS "auto" "32" "64")
if( NOT ((INT_SIZE STREQUAL "auto") OR (INT_SIZE STREQUAL "32") OR (INT_SIZE STREQUAL "64")) )
    message(FATAL_ERROR "INT_SIZE option '${INT_SIZE}' is not supported. Please use one of the following options \
            during CMake invokation: auto, 32, 64")
endif()
set(BLAS_INT_SIZE "32" CACHE STRING "BLAS/CBLAS API integer size")
set_property(CACHE BLAS_INT_SIZE PROPERTY STRINGS "auto" "32" "64")
if( NOT ((BLAS_INT_SIZE STREQUAL "auto") OR (BLAS_INT_SIZE STREQUAL "32") OR (BLAS_INT_SIZE STREQUAL "64")) )
    message(FATAL_ERROR "BLAS_INT_SIZE option '${BLAS_INT_SIZE}' is not supported. Please use one of the following options \
            during CMake invokation: auto, 32, 64")
endif()
option(ENABLE_BLAS "BLAS compatiblity layer" ON)
option(ENABLE_CBLAS "CBLAS compatiblity layer" OFF)
option(ENABLE_MIXED_DT "Mixed datatype support" ON)
option(ENABLE_MIXED_DT_EXTRA_MEM "Mixed datatype optimization requiring extra memory" ON)
option(ENABLE_SUP_HANDLING "Small matrix handling" ON)
if(WIN32)
    set(ENABLE_MEMKIND "no" CACHE STRING "libmemkind for manage memory pools")
    set_property(CACHE ENABLE_MEMKIND PROPERTY STRINGS "no")
    if( NOT (ENABLE_MEMKIND STREQUAL "no"))
        message(FATAL_ERROR "ENABLE_MEMKIND option is not supported on Windows platforms.")
    endif()
else()
    set(ENABLE_MEMKIND "auto" CACHE STRING "libmemkind for manage memory pools")
    set_property(CACHE ENABLE_MEMKIND PROPERTY STRINGS "auto" "yes" "no")
    if( NOT ((ENABLE_MEMKIND STREQUAL "auto") OR (ENABLE_MEMKIND STREQUAL "yes") OR (ENABLE_MEMKIND STREQUAL "no")) )
        message(FATAL_ERROR "ENABLE_MEMKIND option '${ENABLE_MEMKIND}' is not supported. Please use one of the following options \
                during CMake invokation: auto, yes, no")
    endif()
endif()
option(ENABLE_TRSM_PREINVERSION "Enable TRSM preinversion" ON)
option(ENABLE_AOCL_DYNAMIC "Dynamic selection of number of threads" ON)
set(FORCE_VERSION "no" CACHE STRING "Force configure to use an arbitrary version string")
if(WIN32)
    set(COMPLEX_RETURN "gnu" CACHE STRING "The method used for returning complex numbers")
    set_property(CACHE COMPLEX_RETURN PROPERTY STRINGS "gnu" "intel")
    if( NOT ((COMPLEX_RETURN STREQUAL "gnu") OR (COMPLEX_RETURN STREQUAL "intel")) )
        message(FATAL_ERROR "COMPLEX_RETURN option '${COMPLEX_RETURN}' is not supported. Please use one of the following options \
                during CMake invokation: gnu, intel")
    endif()
else()
    set(COMPLEX_RETURN "default" CACHE STRING "The method used for returning complex numbers")
    set_property(CACHE COMPLEX_RETURN PROPERTY STRINGS "default" "gnu" "intel")
    if( NOT ((COMPLEX_RETURN STREQUAL "default") OR (COMPLEX_RETURN STREQUAL "gnu") OR (COMPLEX_RETURN STREQUAL "intel")) )
        message(FATAL_ERROR "COMPLEX_RETURN option '${COMPLEX_RETURN}' is not supported. Please use one of the following options \
                during CMake invokation: default, gnu, intel")
    endif()
endif()
# If the CONFIG_LIST does not already contain the CONFIG_NAME (i.e.,
# if CONFIG_NAME is an umbrella family), default is to enable BLIS_ARCH_TYPE functionality,
# otherwise default is to disable BLIS_ARCH_TYPE functionality.
list(FIND CONFIG_LIST ${BLIS_CONFIG_FAMILY} IS_UMBRELLA)
if(${IS_UMBRELLA} STREQUAL "-1")
    option(DISABLE_BLIS_ARCH_TYPE "Disable AOCL_ENABLE_INSTRUCTIONS, BLIS_ARCH_TYPE and BLIS_MODEL_TYPE functionality" OFF)
else()
    option(DISABLE_BLIS_ARCH_TYPE "Disable AOCL_ENABLE_INSTRUCTIONS, BLIS_ARCH_TYPE and BLIS_MODEL_TYPE functionality" ON)
endif()
set(RENAME_BLIS_ARCH_TYPE "BLIS_ARCH_TYPE" CACHE STRING "BLIS_ARCH_TYPE env var renamed to supplied value")
set(RENAME_BLIS_MODEL_TYPE "BLIS_MODEL_TYPE" CACHE STRING "BLIS_MODEL_TYPE env var renamed to supplied value")
if(NOT WIN32)
    set(ENABLE_ADDON "" CACHE STRING "Configure with specific addons using a ';'-separated list")
endif()
set(ENABLE_SANDBOX "" CACHE STRING "Enable a separate sandbox implementation of gemm.")
# Do not let ENABLE_SANDBOX appear on cmake-gui since the functionality is not yet implemented.
mark_as_advanced(ENABLE_SANDBOX)

#------------------------------------
#           Check memkind
#------------------------------------
# Using libmemkind is not a valid option on Windows. Check only on Linux platforms.
if(NOT WIN32)
    # In order to determine the default behavior of the --with[out]-memkind
    # option, we try to detect whether libmemkind is available. If it is,
    # the default implied option will be --with-memkind; otherwise, will be
    # --without-memkind.
    try_compile(HAS_MEMKIND "${CMAKE_BINARY_DIR}/temp" SOURCES "${CMAKE_SOURCE_DIR}/build/detect/memkind/libmemkind_detect.c"
                LINK_OPTIONS
                    "-lmemkind"
                )
endif()

#------------------------------------
#       Check #pragma omp simd
#------------------------------------
if(ENABLE_THREADING STREQUAL "openmp")
    # Try to determine whether the chosen compiler supports #pragma omp simd.
    try_compile(PRAGMA_OMP_SIMD "${CMAKE_BINARY_DIR}/temp" SOURCES "${CMAKE_SOURCE_DIR}/build/detect/omp_simd/omp_simd_detect.c"
                CMAKE_FLAGS
                    "-O3 -march=native -fopenmp-simd"
                C_STANDARD 99
                )
endif()
#------------------------------------
#       Acquire the BLIS version
#------------------------------------
# Set the VERSION variable to the default value in the 'version' file.
file(STRINGS ${CMAKE_SOURCE_DIR}/version VERSION)
# Get timestamp.
string(TIMESTAMP BUILD_DATE "%Y%m%d")
# Update using the timestamp.
set(VERSION_STRING "AOCL-BLIS ${VERSION} Build ${BUILD_DATE}")
# Initial message.
message(STATUS "Starting configuration of BLIS ${VERSION_STRING}.")
# Check if the user requested a custom version string.
if(FORCE_VERSION STREQUAL "no")
    message("   Configuring with official version string.")
else()
    set(VERSION_STRING "${FORCE_VERSION}")
    message("   Configuring with custom version string: ${VERSION_STRING}")
endif()
# Set the shared library (.so) version file.
file(STRINGS ${CMAKE_SOURCE_DIR}/so_version SO_VERSION)
# The first line of the 'so_version' file contains the .so major version.
list(GET SO_VERSION 0 SO_VERSION_MAJOR)
# The second line contains the minor and build .so version numbers
# (separated by a '.').
list(GET SO_VERSION 1 SO_VERSION_MINOR)

#------------------------------------
#           Printing Options
#------------------------------------
include(CMakePrintHelpers)
message(STATUS "Printing CMake Configuration Options...")
cmake_print_variables(ENABLE_DEBUG)
# Initialize debug type, using the corresponding cache variable.
set(DEBUG_TYPE ${ENABLE_DEBUG})
if(ENABLE_DEBUG STREQUAL "off")
    message("   Debug symbols disabled.")
elseif(ENABLE_DEBUG STREQUAL "opt")
    message("   Enabling debug symbols with optimizations.")
else() #ENABLE_DEBUG=noopt
    message("   Enabling debug symbols; optimizations disabled.")
endif()
cmake_print_variables(BUILD_SHARED_LIBS)
if(BUILD_SHARED_LIBS)
    message("   Building BLIS as a shared library.")
    set(ENABLE_SHARED_01 1)
else()
    message("   Building BLIS as a static library.")
    set(ENABLE_SHARED_01 0)
endif()
if(NOT WIN32)
cmake_print_variables(EXPORT_SHARED)
    if(EXPORT_SHARED STREQUAL "all")
        if(BUILD_SHARED_LIBS)
            message("   Exporting all symbols within shared library.")
        else()
            message("   Ignoring request to export all symbols within shared library.")
        endif()
    else()
        if(BUILD_SHARED_LIBS)
            message("   Exporting only public symbols within shared library.")
        endif()
    endif()
endif()
cmake_print_variables(ENABLE_SYSTEM)
if(ENABLE_SYSTEM)
    message("   Enabling operating system support.")
    set(ENABLE_SYSTEM_01 1)
    if(NOT WIN32)
        set(LIBPTHREAD "-lpthread")
    endif()
else()
    message("   Disabling operating system support.")
    message("   WARNING: all threading will be disabled!")
    set(ENABLE_THREADING "off")
    set(ENABLE_SYSTEM_01 0)
endif()
# Check the threading model flag and standardize its value, if needed.
cmake_print_variables(ENABLE_THREADING)
set(ENABLE_OPENMP "no")
set(ENABLE_OPENMP_01 0)
set(ENABLE_PTHREADS "no")
set(ENABLE_PTHREADS_01 0)
if(ENABLE_THREADING STREQUAL "openmp")
    message("   Using OpenMP for threading.")
    set(ENABLE_OPENMP "yes")
    set(ENABLE_OPENMP_01 1)
    find_package(OpenMP)
    if(NOT OPENMP_FOUND)
      message(FATAL_ERROR "Openmp Not Found")
    endif()
elseif(ENABLE_THREADING STREQUAL "pthreads")
    message("   Using POSIX threads for threading.")
    set(ENABLE_PTHREADS "yes")
    set(ENABLE_PTHREADS_01 1)
else()
    message("   Threading is disabled.")
endif()
# Check the method of assigning micropanels to threads in the JR and IR
# loops.
cmake_print_variables(THREAD_PART_JRIR)
if(THREAD_PART_JRIR STREQUAL "slab")
    message("   Requesting slab threading in jr and ir loops.")
    set(ENABLE_JRIR_SLAB_01 1)
    set(ENABLE_JRIR_RR_01 0)
else()
    message("   Requesting round-robin threading in jr and ir loops.")
    set(ENABLE_JRIR_SLAB_01 0)
    set(ENABLE_JRIR_RR_01 1)
endif()
# Convert 'yes' and 'no' flags to booleans.
cmake_print_variables(ENABLE_PBA_POOLS)
if(ENABLE_PBA_POOLS)
    message("   Internal memory pools for packing blocks are enabled.")
    set(ENABLE_PBA_POOLS_01 1)
else()
    message("   Internal memory pools for packing blocks are disabled.")
    set(ENABLE_PBA_POOLS_01 0)
endif()
cmake_print_variables(ENABLE_SBA_POOLS)
if(ENABLE_SBA_POOLS)
    message("   Internal memory pools for small blocks are enabled.")
    set(ENABLE_SBA_POOLS_01 1)
else()
    message("   Internal memory pools for small blocks are disabled.")
    set(ENABLE_SBA_POOLS_01 0)
endif()
cmake_print_variables(ENABLE_MEM_TRACING)
if(ENABLE_MEM_TRACING)
    message("   Memory tracing output is enabled.")
    set(ENABLE_MEM_TRACING_01 1)
else()
    message("   Memory tracing output is disabled.")
    set(ENABLE_MEM_TRACING_01 0)
endif()
cmake_print_variables(ENABLE_MEMKIND)
if(HAS_MEMKIND)
    if(ENABLE_MEMKIND STREQUAL "auto")
        # If no explicit option was given for libmemkind one way or the other,
        # we use the value returned previously by has_libmemkind(), in this
        # case "yes", to determine the default.
        message("   libmemkind found; default is to enable use.")
        set(ENABLE_MEMKIND "yes")
        set(ENABLE_MEMKIND_01 1)
    else()
        if(ENABLE_MEMKIND STREQUAL "yes")
            message("   Received explicit request to enable libmemkind.")
            set(ENABLE_MEMKIND_01 1)
        else()
            message("   Received explicit request to disable libmemkind.")
            set(ENABLE_MEMKIND "no")
            set(ENABLE_MEMKIND_01 0)
        endif()
    endif()
else()
    if(WIN32)
        message("   libmemkind option is not supported on Windows.")
    else()
        message("   libmemkind not found; disabling.")
        if(ENABLE_MEMKIND STREQUAL "yes")
            message(WARNING "   Cannot honor explicit request to enable libmemkind.")
        endif()
    endif()
    set(ENABLE_MEMKIND "no")
    set(ENABLE_MEMKIND_01 0)
endif()
cmake_print_variables(PRAGMA_OMP_SIMD)
if(PRAGMA_OMP_SIMD)
    message("   Compiler appears to support #pragma omp simd.")
    set(ENABLE_PRAGMA_OMP_SIMD_01 1)
else()
    message("   Compiler appears to not support #pragma omp simd.")
    set(ENABLE_PRAGMA_OMP_SIMD_01 0)
endif()
cmake_print_variables(ENABLE_CBLAS)
if(ENABLE_CBLAS)
    message("   The CBLAS compatibility layer is enabled.")
    set(ENABLE_CBLAS_01 1)
    # Force BLAS layer when CBLAS is enabled
    set(ENABLE_BLAS ON)
else()
    message("   The CBLAS compatibility layer is disabled.")
    set(ENABLE_CBLAS_01 0)
endif()
cmake_print_variables(ENABLE_BLAS)
if(ENABLE_BLAS)
    message("   The BLAS compatibility layer is enabled.")
    set(ENABLE_BLAS_01 1)
else()
    message("   The BLAS compatibility layer is disabled.")
    set(ENABLE_BLAS_01 0)
endif()
cmake_print_variables(ENABLE_MIXED_DT)
if(ENABLE_MIXED_DT)
    message("   Mixed datatype support is enabled.")
    cmake_print_variables(ENABLE_MIXED_DT_EXTRA_MEM)
    if(ENABLE_MIXED_DT_EXTRA_MEM)
        message("   Mixed datatype optimizations requiring extra memory are enabled.")
        set(ENABLE_MIXED_DT_EXTRA_MEM_01 1)
    else()
        message("   Mixed datatype optimizations requiring extra memory are disabled.")
        set(ENABLE_MIXED_DT_EXTRA_MEM_01 0)
    endif()    
    set(ENABLE_MIXED_DT_01 1)
else()
    message("   Mixed datatype support is disabled.")
    set(ENABLE_MIXED_DT_EXTRA_MEM_01 0)
    set(ENABLE_MIXED_DT_01 0)
endif()
cmake_print_variables(ENABLE_SUP_HANDLING)
if(ENABLE_SUP_HANDLING)
    message("   Small matrix handling is enabled.")
    set(ENABLE_SUP_HANDLING_01 1)
else()
    message("   Small matrix handling is disabled.")
    set(ENABLE_SUP_HANDLING_01 0)
endif()
cmake_print_variables(ENABLE_TRSM_PREINVERSION)
if(ENABLE_TRSM_PREINVERSION)
    message("   trsm diagonal element pre-inversion is enabled.")
    set(ENABLE_TRSM_PREINVERSION_01 1)
else()
    message("   trsm diagonal element pre-inversion is disabled.")
    set(ENABLE_TRSM_PREINVERSION_01 0)
endif()
# Check aocl dynamic threading configuration and enable it only if
# multi-threading is enabled
cmake_print_variables(ENABLE_AOCL_DYNAMIC)
if(ENABLE_AOCL_DYNAMIC)
    if( NOT(ENABLE_THREADING STREQUAL "no"))
        message("   Dynamic selection of number of threads is enabled.")
        set(ENABLE_AOCL_DYNAMIC_01 1)
    else()
        message("   Dynamic threading is disabled as multithreading is disabled.")
        set(ENABLE_AOCL_DYNAMIC OFF)
        set(ENABLE_AOCL_DYNAMIC_01 0)
    endif()
else()
    message("   Dynamic selection of number of threads is disabled.")
    set(ENABLE_AOCL_DYNAMIC_01 0)
endif()
# Report integer sizes.
cmake_print_variables(INT_SIZE)
set(INT_TYPE_SIZE ${INT_SIZE})
if(INT_TYPE_SIZE STREQUAL "32")
    message("   The BLIS API integer size is 32-bit.")
elseif(INT_TYPE_SIZE STREQUAL "64")
    message("   The BLIS API integer size is 64-bit.")
else()
    set(INT_TYPE_SIZE "0")
    message("   The BLIS API integer size is automatically determined.")
endif()
cmake_print_variables(BLAS_INT_SIZE)
set(BLAS_INT_TYPE_SIZE ${BLAS_INT_SIZE})
if(BLAS_INT_TYPE_SIZE STREQUAL "32")
    message("   The BLAS/CBLAS API integer size is 32-bit.")
elseif(BLAS_INT_TYPE_SIZE STREQUAL "64")
    message("   The BLAS/CBLAS API integer size is 64-bit.")
else()
    set(BLAS_INT_TYPE_SIZE "0")
    message("   The BLAS/CBLAS API integer size is automatically determined.")
endif()
# Disallow the simultaneous use of 64-bit integers in the BLAS and
# 32-bit integers in BLIS.
if((INT_TYPE_SIZE STREQUAL "32") AND (BLAS_INT_TYPE_SIZE STREQUAL "64"))
    message(FATAL_ERROR "INT_TYPE_SIZE=${INT_TYPE_SIZE} and BLAS_INT_TYPE_SIZE=${BLAS_INT_TYPE_SIZE}. \
        To avoid the possibility of truncation, we do not allow use of 64-bit integers in the BLAS API with 32-bit integers in BLIS. \
        Please use a different configuration of integers.")
endif()
if(NOT WIN32)
    cmake_print_variables(ENABLE_ADDON)
    if(ENABLE_ADDON STREQUAL "")
        message("   Configuring with no addons.")
        set(ENABLE_ADDONS_01 0)
    else()
        # Remove duplicates in the addon list, if they exist.
        list(REMOVE_DUPLICATES ENABLE_ADDON)
        message("   Configuring with addons:")
        foreach(ADDON ${ENABLE_ADDON})
            message("   ${ADDON}")
            if(NOT (EXISTS ${CMAKE_SOURCE_DIR}/addon/${ADDON}))
                message(FATAL_ERROR "Requested addon sub-directory does not exist! Cannot continue. \
                                        *** Please verify addon existence and name.")
            endif()
        endforeach()
        set(ENABLE_ADDONS_01 1)
    endif()
endif()
cmake_print_variables(ENABLE_SANDBOX)
if(ENABLE_SANDBOX STREQUAL "")
    message("   Configuring for conventional gemm implementation.")
    set(ENABLE_SANDBOX_01 0)
else()
    message("   Configuring with alternate gemm implementation: ${ENABLE_SANDBOX}.")
    message(FATAL_ERROR "Sandbox functionality is not yet integrated in CMake build system.")
    set(ENABLE_SANDBOX_01 1)
endif()
# Check the method used for returning complex numbers. Only for Linux.
if(NOT WIN32)
    if(COMPLEX_RETURN STREQUAL "default")
        if("${CMAKE_Fortran_COMPILER_ID}" MATCHES "Intel")
            set(COMPLEX_RETURN "intel")
        else()
            set(COMPLEX_RETURN "gnu")
        endif()
    endif()
endif()
cmake_print_variables(COMPLEX_RETURN)
if(COMPLEX_RETURN STREQUAL "gnu")
    message("   Configuring with gnu complex return type.")
    set(COMPLEX_RETURN_INTEL_01 0)
else()
    message("   Configuring with intel complex return type.")
    set(COMPLEX_RETURN_INTEL_01 1)
endif()
cmake_print_variables(DISABLE_BLIS_ARCH_TYPE)
if(DISABLE_BLIS_ARCH_TYPE)
    message("   User selection of code path using AOCL_ENABLE_INSTRUCTIONS, BLIS_ARCH_TYPE and")
    message("   BLIS_MODEL_TYPE env vars is disabled.")
    set(DISABLE_BLIS_ARCH_TYPE_01 1)
else()
    set(DISABLE_BLIS_ARCH_TYPE_01 0)
endif()
cmake_print_variables(RENAME_BLIS_ARCH_TYPE)
if(NOT(RENAME_BLIS_ARCH_TYPE STREQUAL "BLIS_ARCH_TYPE"))
    message("   configuring with BLIS_ARCH_TYPE env var renamed to ${RENAME_BLIS_ARCH_TYPE}")
endif()
cmake_print_variables(RENAME_BLIS_MODEL_TYPE)
if(NOT(RENAME_BLIS_MODEL_TYPE STREQUAL "BLIS_MODEL_TYPE"))
    message("   configuring with BLIS_MODEL_TYPE env var renamed to ${RENAME_BLIS_MODEL_TYPE}")
endif()
if(WIN32)
    cmake_print_variables(ENABLE_NO_UNDERSCORE_API)
    if(ENABLE_NO_UNDERSCORE_API)
        message("   Export APIs without underscore.")
    else()
        message("   Export APIs with underscore.")
    endif()
    cmake_print_variables(ENABLE_UPPERCASE_API)
    if(ENABLE_UPPERCASE_API)
        message("   Export APIs with uppercase.")
    else()
        message("   Export APIs with lowercase.")
    endif()
endif()

# Initialize threading model, using the corresponding cache variable.
set(THREADING_MODEL ${ENABLE_THREADING})


#--------------------------------------------
# Instantiate bli_config.h file from template
#--------------------------------------------
# Begin substituting information into the build/cmake/bli_config.h.in file, outputting
# to bli_config.h and store it in build directory of the current project.
configure_file(build/cmake/bli_config.h.in ${PROJECT_BINARY_DIR}/bli_config.h)

#--------------------------------------------
# Instantiate bli_addon.h file from template
#--------------------------------------------
# Create a list of #includes, one for each addon in addon_list.
set(ADDON_LIST_INCLUDES "")
foreach(ADDON ${ENABLE_ADDON})
    if(ADDON STREQUAL "aocl_gemm")
        if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0.0))
            message(FATAL_ERROR "aocl_gemm addon requires a gcc version 11.0.0 or higher.")
        endif()
    endif()
    set(ADDON_HEADER "\"${ADDON}.h\"")
    set(ADDON_LIST_INCLUDES "${ADDON_LIST_INCLUDES}#include ${ADDON_HEADER}\n")
endforeach()
# Begin substituting information into the bli_addon.h.in file, outputting
# to bli_addon.h and store it in build directory of the current project.
configure_file(build/cmake/bli_addon.h.in ${PROJECT_BINARY_DIR}/bli_addon.h)

#--------------------------------------------
#     Collect directory paths for blis.h
#--------------------------------------------
# Variable ALL_HEADER_PATHS_LIST is equivalent to ALL_H99_DIRPATHS in Make system.
# Practically, we collect the required directory paths into a list, which we
# append as we add the corresponding subdirectories. This variable will be
# transformed into a string and will be used to generate the flatten blis.h header.
set(ALL_HEADER_PATHS_LIST "")
# Track files to set dependencies for blis.h.
set(ALL_HEADER_FILES_LIST "")

# Include functionality that returns header paths.
include(${CMAKE_SOURCE_DIR}/build/cmake/subdir_helper_functions.cmake)

# If the CONFIG_LIST does not already contain the CONFIG_NAME (i.e.,
# if CONFIG_NAME is an umbrella family), add in the corresponding
# directory. (In the next step, we will loop over the actual sub-
# configurations and add them as well.)
list(FIND CONFIG_LIST ${BLIS_CONFIG_FAMILY} IS_UMBRELLA)
if(${IS_UMBRELLA} STREQUAL "-1")
    # Collect all subdirectory paths that have at least one file with suffix in ALL_H99_SUFS list.
    get_dirpaths_with_suffixes(${BLIS_CONFIG_FAMILY}_HEADER_PATHS ${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY} "${ALL_H99_SUFS}")
    # Collect all files in the subdirectories.
    get_filepaths_with_suffixes(${BLIS_CONFIG_FAMILY}_HEADER_FILES ${CMAKE_SOURCE_DIR}/config/${BLIS_CONFIG_FAMILY} "${ALL_H99_SUFS}")
endif()
list(APPEND ALL_HEADER_PATHS_LIST "${${BLIS_CONFIG_FAMILY}_HEADER_PATHS}")
list(APPEND ALL_HEADER_FILES_LIST "${${BLIS_CONFIG_FAMILY}_HEADER_FILES}")

# Get header directory paths for each of the sub-configurations present
# in the configuration list.
foreach(CONF ${CONFIG_LIST})
    get_dirpaths_with_suffixes(config_${CONF}_HEADER_PATHS ${CMAKE_SOURCE_DIR}/config/${CONF} "${ALL_H99_SUFS}")
    list(APPEND ALL_HEADER_PATHS_LIST "${config_${CONF}_HEADER_PATHS}")
    get_filepaths_with_suffixes(config_${CONF}_FILES_PATHS ${CMAKE_SOURCE_DIR}/config/${CONF} "${ALL_H99_SUFS}")
    list(APPEND ALL_HEADER_FILES_LIST "${config_${CONF}_HEADER_FILES}")
endforeach()

# Get header directory paths for each of the kernels present
# in the kernel list.
foreach(KERN ${KERNEL_LIST})
    # Collect all subdirectory paths that have at least one file with suffix in ALL_H99_SUFS list.
    get_dirpaths_with_suffixes(kernels_${KERN}_HEADER_PATHS ${CMAKE_SOURCE_DIR}/kernels/${KERN} "${ALL_H99_SUFS}")
    list(APPEND ALL_HEADER_PATHS_LIST "${kernels_${KERN}_HEADER_PATHS}")
    get_filepaths_with_suffixes(kernels_${KERN}_HEADER_FILES ${CMAKE_SOURCE_DIR}/kernels/${KERN} "${ALL_H99_SUFS}")
    list(APPEND ALL_HEADER_PATHS_FILES "${kernels_${KERN}_HEADER_FILES}")
endforeach()

# Get header directory paths for framework directory.
get_dirpaths_with_suffixes(frame_HEADER_PATHS ${CMAKE_SOURCE_DIR}/frame "${ALL_H99_SUFS}")
list(APPEND ALL_HEADER_PATHS_LIST "${frame_HEADER_PATHS}")
get_filepaths_with_suffixes(frame_HEADER_FILES ${CMAKE_SOURCE_DIR}/frame "${ALL_H99_SUFS}")
list(APPEND ALL_HEADER_FILES_LIST "${frame_HEADER_FILES}")

# Get header directory paths for AOCL DTL logs directory.
get_dirpaths_with_suffixes(aocl_dtl_HEADER_PATHS ${CMAKE_SOURCE_DIR}/aocl_dtl "${ALL_H99_SUFS}")
list(APPEND ALL_HEADER_PATHS_LIST "${aocl_dtl_HEADER_PATHS}")
get_filepaths_with_suffixes(aocl_dtl_HEADER_FILES ${CMAKE_SOURCE_DIR}/aocl_dtl "${ALL_H99_SUFS}")
list(APPEND ALL_HEADER_FILES_LIST "${aocl_dtl_FILES_PATHS}")

# Get a copy of the header paths without including the addons and the sandbox.
set(FRAME_HEADER_DIRPATHS_LIST ${ALL_HEADER_PATHS_LIST})

# Get header directory paths for each of the addons.
foreach(ADDON ${ENABLE_ADDON})
    get_dirpaths_with_suffixes(addon_${ADDON}_HEADER_PATHS ${CMAKE_SOURCE_DIR}/addon/${ADDON} "${ALL_H99_SUFS}")
    list(APPEND ALL_HEADER_PATHS_LIST "${addon_${ADDON}_HEADER_PATHS}")
    get_filepaths_with_suffixes(addon_${ADDON}_HEADER_FILES ${CMAKE_SOURCE_DIR}/addon/${ADDON} "${ALL_H99_SUFS}")
    list(APPEND ALL_HEADER_FILES_LIST "${addon_${ADDON}_HEADER_FILES}")
endforeach()

# Pick up generated bli_config.h and bli_addon.h that get generated in
# current build directory.
list(PREPEND ALL_HEADER_PATHS_LIST ${PROJECT_BINARY_DIR}/)
list(PREPEND ALL_HEADER_FILES_LIST ${PROJECT_BINARY_DIR}/bli_config.h)
if(NOT (ENABLE_ADDON STREQUAL ""))
    list(PREPEND ALL_HEADER_FILES_LIST ${PROJECT_BINARY_DIR}/bli_addon.h)
endif()

# Create a string out of this list so that it can be processed by flatten-headers.py.
list(JOIN ALL_HEADER_PATHS_LIST " " ALL_HEADER_PATHS_STRING)

#--------------------------------------------
#     Consolidated blis.h header creation
#--------------------------------------------
# Creating a directory for the generated flatten headers.
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY})
# Flatten header python script file which expand header contents in blis.h.
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/blis.h
                COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/flatten-headers.py -c -v1
                "${CMAKE_SOURCE_DIR}/frame/include/blis.h"
                "${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/blis.h"
                "${PROJECT_BINARY_DIR}/include"
                "${ALL_HEADER_PATHS_STRING}"
                COMMENT "Generating monolithic blis header file: ${CMAKE_SOURCE_DIR}/include/${BLIS_CONFIG_FAMILY}/blis.h"
                DEPENDS ${ALL_HEADER_FILES_LIST}
                )
add_custom_target(flat-header DEPENDS ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/blis.h)
#--------------------------------------------
#     Consolidated cblas.h header creation
#--------------------------------------------
# Flatten header python script file which expand header contents in cblas.h.
if(ENABLE_CBLAS)
    add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/cblas.h
                    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/build/flatten-headers.py -c -v1
                    "${CMAKE_SOURCE_DIR}/frame/compat/cblas/src/cblas.h"
                    "${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/cblas.h"
                    "${PROJECT_BINARY_DIR}/${include}"
                    "${ALL_HEADER_PATHS_STRING}"
                    COMMENT "Generating monolithic cblas header file: ${CMAKE_SOURCE_DIR}/include/${BLIS_CONFIG_FAMILY}/cblas.h"
                    DEPENDS ${ALL_HEADER_FILES_LIST}
                    )
    add_custom_target(flat-cblas-header DEPENDS ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/cblas.h)
endif()

#--------------------------------------------
#         Default linker definitions
#--------------------------------------------
# NOTE: This section needs to reside before the inclusion of make_defs.mk
# files (just below), as most configurations' make_defs.mk don't tinker
# with things like LDFLAGS, but some do (or may), in which case they can
# manually override whatever they need.

# Define the external libraries we may potentially need at link-time.
# Add libm only on Linux and only if Intel compiler is not used.
if((NOT WIN32) AND (NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")))
    set(LIBM -lm)
endif()
set(LIBMEMKIND -lmemkind)

# Default linker flags.
# NOTE: -lpthread is needed unconditionally because BLIS uses pthread_once()
# to initialize itself in a thread-safe manner. The one exception to this
# rule: if --disable-system is given at configure-time, LIBPTHREAD is empty.
if(NOT WIN32)
    set(LDFLAGS ${LIBM} ${LIBPTHREAD})
endif()
# Add libmemkind to the link-time flags, if it was enabled at configure-time.
if(ENABLE_MEMKIND STREQUAL "yes")
    list(APPEND LDFLAGS ${LIBMEMKIND})
endif()

#--------------------------------------------
#         Configuration-agnostic flags
#--------------------------------------------
# --- Warning flags ---

# Disable unused function warnings and stop compiling on first error for
# all compilers that accept such options: gcc, clang, and icc.
set(CWARNFLAGS -Wno-unused-function -Wfatal-errors)
if(NOT WIN32)
    list(PREPEND CWARNFLAGS -Wall)
endif()

# Disable tautological comparision warnings in clang.
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    list(APPEND CWARNFLAGS -Wno-tautological-compare -Wno-pass-failed)
endif()

# Add extra warning flags for Windows builds.
if(WIN32)
    list(APPEND CWARNFLAGS -Wno-unused-variable -Wno-deprecated-declarations)
endif()

#Setting up the correct Windows Runtime Library.
if(WIN32)
    cmake_policy(SET CMP0091 NEW)
    if(BUILD_SHARED_LIBS)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# --- Symbol exporting flags (shared libraries only) --

# NOTE: These flags are only applied when building BLIS and not used by
# applications.

# Determine default export behavior / visibility of symbols for gcc, icc and clang.
if(NOT WIN32)
    if(EXPORT_SHARED STREQUAL "all")
        # Export all symbols by default.
        set(BUILD_SYMFLAGS -fvisibility=default)
    else() # ifeq ($(EXPORT_SHARED),public)
        # Hide all symbols by default and export only those that have been annotated
        # as needing to be exported.
        set(BUILD_SYMFLAGS -fvisibility=hidden)
    endif()
endif()

# --- C Preprocessor flags ---
# Enable clock_gettime() in time.h.
set(CPPROCFLAGS -D_POSIX_C_SOURCE=200112L)

# --- Threading flags ---
# NOTE: We don't have to explicitly omit -pthread when --disable-system is given
# since that option forces --enable-threading=none, and thus -pthread never gets
# added to begin with.
if(NOT WIN32)
    if(THREADING_MODEL STREQUAL "pthreads")
        set(CTHREADFLAGS "-pthread")
    endif()
endif()

# --- #pragma omp simd flags (used for reference kernels only) ---
if(PRAGMA_OMP_SIMD)
    if(WIN32)
        set(COMPSIMDFLAGS /openmp:experimental)
    else()
        set(COMPSIMDFLAGS -fopenmp-simd)
    endif()
endif()

#--------------------------------------------
#     Compiler include path definitions
#--------------------------------------------
# Obtain a list of header files #included inside of the bli_cntx_ref.c file.
# Due to the way that bli_cntx_ref.c uses headers and macros, paths to these
# files will be needed when compiling bli_cntx_ref.c with the monolithic header.

# Read content of bli_cntx_ref.c and put it in REF_KER_HEADERS_TEMP.
file(STRINGS ${CMAKE_SOURCE_DIR}/ref_kernels/bli_cntx_ref.c REF_KER_HEADERS_TEMP)
# Only keep the lines where there are includes.
list(FILTER REF_KER_HEADERS_TEMP INCLUDE REGEX "\#include")
# REF_KER_HEADERS has a list of all files that are included in bli_cntx_ref.c.
set(REF_KER_HEADERS "")
foreach(header ${REF_KER_HEADERS_TEMP})
    string(REGEX MATCH "\#include [\"<]\([a-zA-Z0-9\_\.\/\-]*\)[\">].*" helper ${header})
    list(APPEND REF_KER_HEADERS ${CMAKE_MATCH_1})
endforeach()
# Remove blis.h from the list.
list(FILTER REF_KER_HEADERS EXCLUDE REGEX "blis.h")
set(REF_KER_H_PATHS "")
foreach(header_name ${REF_KER_HEADERS})
    foreach(header_dir ${FRAME_HEADER_DIRPATHS_LIST})
        if(EXISTS ${header_dir}/${header_name})
            list(APPEND REF_KER_H_PATHS ${header_dir})
            break()
        endif()
    endforeach()
endforeach()
# Remove duplicates, if they exist.
list(REMOVE_DUPLICATES REF_KER_H_PATHS)

# Create list of include directories, to be used while creating the library.
# NOTE: We no longer need every header path in the source tree since we
# now #include the monolithic/flattened blis.h instead.
set(CINFLAGS ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY})
list(APPEND CINFLAGS ${REF_KER_H_PATHS})
# Then add frame/include since it's needed for bli_oapi_w[o]_cntx.h.
list(APPEND CINFLAGS ${CMAKE_SOURCE_DIR}/frame/include)
# If CBLAS is enabled, we also include the path to the cblas.h directory so
# that the compiler will be able to find cblas.h as the CBLAS source code is
# being compiled.
if(ENABLE_CBLAS)
    set(CBLAS_H_DIRPATH "")
    foreach(header_dir ${FRAME_HEADER_DIRPATHS_LIST})
        if(EXISTS ${header_dir}/cblas.h)
            list(APPEND CBLAS_H_DIRPATH ${header_dir})
            break()
        endif()
    endforeach()
    list(APPEND CINFLAGS ${CBLAS_H_DIRPATH})
endif()

#--------------------------------------------
#   Special preprocessor macro definitions
#--------------------------------------------
# Define a C preprocessor macro to communicate the current version so that it
# can be embedded into the library and queried later.
set(VERS_DEF -DBLIS_VERSION_STRING="${VERSION_STRING}")

# Define a C preprocessor flag that is *only* defined when BLIS is being
# compiled. (In other words, an application that #includes blis.h will not
# get this cpp macro.)
set(BUILD_CPPFLAGS -DBLIS_IS_BUILDING_LIBRARY)

#--------------------------------------------
#     Add CMakeLists.txt from directories
#--------------------------------------------
# Add config subdirectory.
add_subdirectory(config)
# Add kernel subdirectory.
add_subdirectory(kernels)
# Add framework directory.
add_subdirectory(frame)
# Add AOCL DTL logs directory.
add_subdirectory(aocl_dtl)
# Add subdirectory for each of the addons.
list(LENGTH ENABLE_ADDON addon_list_size)
if(addon_list_size GREATER 0)
    add_subdirectory(addon)
endif()

# Collect all object libraries that are required to build the blis library.
set(OBJECT_LIBRARIES "")
# Add objects from config.
foreach(conf ${CONFIG_LIST})
    list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:${conf}_CONFIG>)
endforeach()
# Add objects from kernels.
foreach(ker ${KERNEL_LIST})
    if(TARGET ${ker}_KERNELS)
        list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:${ker}_KERNELS>)
    endif()
endforeach()
# Add objects for reference kernels.
foreach(conf ${CONFIG_LIST})
    list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:${conf}_REFINIT>)
    list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:${conf}_REFKERN>)
endforeach()
# Add objects for frame.
list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:FRAME>)
# Add objects for aocl-dtl.
list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:AOCL_DTL>)
# Add objects for addons.
foreach(addon ${ENABLE_ADDON})
    if(TARGET ${addon}_C99_ADDON)
        list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:${addon}_C99_ADDON>)
    endif()
    if(TARGET ${addon}_C99_KERNEL_ADDON)
        list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:${addon}_C99_KERNEL_ADDON>)
    endif()
    if(TARGET ${addon}_CXX_ADDON)
        list(APPEND OBJECT_LIBRARIES $<TARGET_OBJECTS:${addon}_CXX_ADDON>)
    endif()
endforeach()

#--------------------------------------------
#         Building BLIS Library
#--------------------------------------------
# Public blis headers.
set(BLIS_PUBLIC_HEADERS
    ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/blis.h
    # Include AMD's C++ template header files in the list of headers
    # to install.
    ${CMAKE_SOURCE_DIR}/vendor/cpp/blis.hh
    ${CMAKE_SOURCE_DIR}/vendor/cpp/cblas.hh
)
if(ENABLE_CBLAS)
    list(APPEND BLIS_PUBLIC_HEADERS ${PROJECT_BINARY_DIR}/include/${BLIS_CONFIG_FAMILY}/cblas.h)
endif()

# --- Library name and local paths ---
# From old CMake
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)    
    add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Oi")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /MP${CMake_MSVC_PARALLEL}")
    set(INTR_GENERAL_LINK_FLAGS "${INTR_GENERAL_LINK_FLAGS} /RELEGE")
    add_definitions(-DEXPMODULE)
endif()

# Set up the library name.
if(WIN32)
    set(LIBBLIS AOCL-LibBlis-Win)
else()
    set(LIBBLIS blis)
endif()

# Append if threading is required.
if(NOT (THREADING_MODEL STREQUAL "no"))
    if(WIN32)
        string(APPEND LIBBLIS -MT)
    else()
        string(APPEND LIBBLIS -mt)
    endif()
endif()

if(BUILD_SHARED_LIBS)
    if(WIN32)
        string(APPEND LIBBLIS -dll)
    endif()
    # Build shared library.
    add_library(libblis SHARED ${OBJECT_LIBRARIES})
    target_link_libraries(libblis PRIVATE ${LDFLAGS})
    set_target_properties(libblis PROPERTIES LINKER_LANGUAGE C VERSION ${VERSION} SOVERSION ${SO_VERSION_MAJOR})
    set_target_properties(libblis PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(THREADING_MODEL STREQUAL "openmp")
        target_link_libraries(libblis PRIVATE OpenMP::OpenMP_C)
    endif()
else()
    # Build static library.
    add_library(libblis STATIC ${OBJECT_LIBRARIES})
    set_target_properties(libblis PROPERTIES LINKER_LANGUAGE C)
endif()
add_dependencies(libblis flat-header)
if(ENABLE_CBLAS)
    add_dependencies(libblis flat-cblas-header)
endif()
# Add headers as a property to the library.
set_target_properties(libblis PROPERTIES PUBLIC_HEADER "${BLIS_PUBLIC_HEADERS}")
set_target_properties(libblis PROPERTIES OUTPUT_NAME ${LIBBLIS})

# Install targets.
install(TARGETS libblis LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

# --- Primary targets ---
add_custom_target(libs DEPENDS libblis)

# Multiple BLIS API testing targets. Result files are generated in ${CMAKE_BINARY_DIR}/testsuite.
add_subdirectory(testsuite EXCLUDE_FROM_ALL)

# Check results of BLIS CPP Template tests
add_subdirectory(vendor/testcpp EXCLUDE_FROM_ALL)

# Add BLAS tests if BLAS interface is enabled.
if(ENABLE_BLAS)
    add_subdirectory(blastest EXCLUDE_FROM_ALL)
endif()

# Add generic testing target `test`.
set(available_testsuites checkblis)
if(ENABLE_BLAS)
    list(APPEND available_testsuites checkblas)
endif()
add_custom_target(test DEPENDS ${available_testsuites})

# Add generic testing target `check`.
set(available_testsuites checkblis-fast)
if(ENABLE_BLAS)
    list(APPEND available_testsuites checkblas)
endif()
add_custom_target(check DEPENDS ${available_testsuites})