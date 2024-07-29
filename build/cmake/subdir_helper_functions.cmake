#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

]=]

# Create a list of keywords for files that need to be ignored by the system.
file(READ ${CMAKE_SOURCE_DIR}/build/gen-make-frags/ignore_list IGNORE_LIST)
string(REPLACE "\n" ";" IGNORE_LIST ${IGNORE_LIST})

# Create a list of suffixes for files that need to be compiled to create the library.
file(READ ${CMAKE_SOURCE_DIR}/build/gen-make-frags/suffix_list SUFFIX_LIST)
string(REPLACE "\n" ";" SUFFIX_LIST ${SUFFIX_LIST})

#--------------------------------------------
#               SUFFIX LISTS
#--------------------------------------------
# Source suffixes.
set(CONFIG_SRC_SUFS "c")
set(KERNELS_SRC_SUFS "c;s;S")
set(FRAME_SRC_SUFS "c")

set(AOCLDTL_SRC_SUFS "c")
set(ADDON_C99_SUFS "c")
set(ADDON_CXX_SUFS "cc;cpp;cxx")
set(ADDON_SRC_SUFS "${ADDON_C99_SUFS};${ADDON_CXX_SUFS}")

set(SANDBOX_C99_SUFS "c")
set(SANDBOX_CXX_SUFS "cc;cpp;cxx")
set(SANDBOX_SRC_SUFS "${SANDBOX_C99_SUFS};${SANDBOX_CXX_SUFS}")

# Header suffixes.
set(FRAME_HDR_SUFS "h")

set(AOCLDTL_HDR_SUFS "h")
set(ADDON_H99_SUFS "h")
set(ADDON_HXX_SUFS "hh;hpp;hxx")
set(ADDON_HDR_SUFS "${ADDON_H99_SUFS};${ADDON_HXX_SUFS}")

set(SANDBOX_H99_SUFS "h")
set(SANDBOX_HXX_SUFS "hh;hpp;hxx")
set(SANDBOX_HDR_SUFS "$(SANDBOX_H99_SUFS);$(SANDBOX_HXX_SUFS)")

# Combine all header suffixes and remove duplicates.
set(ALL_HDR_SUFS "${FRAME_HDR_SUFS};${ADDON_HDR_SUFS};${SANDBOX_HDR_SUFS};${AOCLDTL_HDR_SUFS}")
list(REMOVE_DUPLICATES ALL_HDR_SUFS)

set(ALL_H99_SUFS "${FRAME_HDR_SUFS};${ADDON_HDR_SUFS};${SANDBOX_H99_SUFS};${AOCLDTL_HDR_SUFS}")
list(REMOVE_DUPLICATES ALL_H99_SUFS)

#--------------------------------------------
#  Important sets of header files and paths
#--------------------------------------------
# Get a list of all sub-directories of a given directory
macro(get_dirpaths_with_suffixes result curdir sufflist)
    set(dirlist "")
    # dirlist will have all files which are below this directory.
    file(GLOB_RECURSE children LIST_DIRECTORIES true ${curdir}/*)
    # Adding current directory in the list.
    list(PREPEND children ${curdir})
    # Filter out anything that is not a directory.
    foreach(child ${children})
        if(IS_DIRECTORY ${child})
            set(HAS_SUFF_FILE "false")
            foreach(suff ${sufflist})
                file(GLOB suff_files LIST_DIRECTORIES false ${child}/*\.${suff})
                list(LENGTH suff_files list_size)
                if(NOT (${list_size} STREQUAL 0))
                    set(HAS_SUFF_FILE "true")
                    # If there is at least one file with a specific suffix break from for-loop.
                    break()
                endif()
            endforeach()
            # If there is at least one *.suff file, add directory path in the list.
            if(HAS_SUFF_FILE STREQUAL "true")                
                list(APPEND dirlist "${child}/")
            endif()
        endif()
    endforeach()
    # Get the name of the current directory, after removing the source directory
    # from the name, so that we can exclude the files that are part of the ignore
    # list even if the blis directory is located in a directory with a name that
    # would be ignored.
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" curdirsimple ${curdir})
    # Filter out anything that is part of the IGNORE_LIST.
    foreach(item ${IGNORE_LIST})
        list(FILTER dirlist EXCLUDE REGEX ${curdirsimple}.*/${item}/)
    endforeach()
    list(APPEND ${result} ${dirlist})
endmacro()

# Get a list of all source files of a given directory based on the suffix list.
# Returns a list which can be transfored to a string when needed
# from high level CMake.
macro(get_filepaths_with_suffixes result curdir sufflist)
    set(sourcelist "")
    # Get the name of the current directory, after removing the source directory
    # from the name, so that we can exclude the files that are part of the ignore
    # list even if the blis directory is located in a directory with a name that
    # would be ignored.
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" curdirsimple ${curdir})
    foreach(suff ${sufflist})
        # dirlist will have all files which are below this directory.
        file(GLOB_RECURSE suff_files LIST_DIRECTORIES false ${curdir}/*\.${suff})
        # Filter out anything that is part of the IGNORE_LIST.
        foreach(item ${IGNORE_LIST})
            list(FILTER suff_files EXCLUDE REGEX ${curdirsimple}.*/${item}/)
        endforeach()
        list(APPEND sourcelist "${suff_files}")
    endforeach()
    list(APPEND ${result} ${sourcelist})
endmacro()

# Choose correct sub-configurarion name for the given kernel set. 
# Behaves similary to get-config-for-kset.
macro(get_config_for_kernel_from_kconfig_map config kernel kconfig_map)
    set(conf ${kconfig_map})
    # Since kconfig_map has as elements pairs of the form kernel:config,
    # to find the element with the corresponding config we need to filter
    # with respect to the kernel first.
    list(FILTER conf INCLUDE REGEX ${kernel}:)
    # Now that the list has only one element, we can remove the part
    # of kernel: and then we will be left with config.
    list(TRANSFORM conf REPLACE ${kernel}: "")
    list(APPEND ${config} ${conf})
endmacro()
