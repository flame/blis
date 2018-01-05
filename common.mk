#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas at Austin nor the names
#     of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

# Only include this block of code once
ifndef COMMON_MK_INCLUDED
COMMON_MK_INCLUDED := yes



#
# --- CFLAGS storage functions -------------------------------------------------
#

# Define a function that stores the value of a variable to a different
# variable containing a specified suffix (corresponding to a configuration).
define store-var-for
$(strip $(1)).$(strip $(2)) := $($(strip $(1)))
endef

# Define a function similar to store-var-for, except that appends instead
# of overwriting.
define append-var-for
$(strip $(1)).$(strip $(2)) += $($(strip $(1)))
endef

# Define a function that stores the value of all of the variables in a
# make_defs.mk file to other variables with the configuration (the
# argument $(1)) added as a suffix. This function is called once from
# each make_defs.mk. Also, add the configuration to CONFIGS_INCL.
define store-make-defs
$(eval $(call store-var-for,CC,         $(1)))
$(eval $(call store-var-for,CC_VENDOR,  $(1)))
$(eval $(call store-var-for,CPPROCFLAGS,$(1)))
$(eval $(call store-var-for,CMISCFLAGS, $(1)))
$(eval $(call store-var-for,CPICFLAGS,  $(1)))
$(eval $(call store-var-for,CWARNFLAGS, $(1)))
$(eval $(call store-var-for,CDBGFLAGS,  $(1)))
$(eval $(call store-var-for,COPTFLAGS,  $(1)))
$(eval $(call store-var-for,CKOPTFLAGS, $(1)))
$(eval $(call store-var-for,CVECFLAGS,  $(1)))
CONFIGS_INCL += $(1)
endef

# Define a function that retreives the value of a variable for a
# given configuration.
define load-var-for
$($(strip $(1)).$(strip $(2)))
endef

# Define some functions that return the appropriate CFLAGS for a given
# configuration. This assumes that the make_defs.mk files have already been
# included, which results in those values having been stored to
# configuration-qualified variables.



#
# --- CFLAGS query functions ---------------------------------------------------
#

get-noopt-cflags-for   = $(strip $(call load-var-for,CDBGFLAGS,$(1)) \
                                 $(call load-var-for,CWARNFLAGS,$(1)) \
                                 $(call load-var-for,CPICFLAGS,$(1)) \
                                 $(call load-var-for,CMISCFLAGS,$(1)) \
                                 $(call load-var-for,CPPROCFLAGS,$(1)) \
                                 $(CTHREADFLAGS) \
                                 $(INCLUDE_PATHS) $(VERS_DEF) \
                          )

get-kernel-cflags-for  = $(call load-var-for,CKOPTFLAGS,$(1)) \
                         $(call load-var-for,CVECFLAGS,$(1)) \
                         $(call get-noopt-cflags-for,$(1))

get-refkern-cflags-for = $(call get-kernel-cflags-for,$(1)) \
                         -DBLIS_CNAME=$(1)

get-frame-cflags-for   = $(call load-var-for,COPTFLAGS,$(1)) \
                         $(call load-var-for,CVECFLAGS,$(1)) \
                         $(call get-noopt-cflags-for,$(1))

get-config-cflags-for  = $(call get-kernel-cflags-for,$(1))

get-noopt-text       = "(CFLAGS for no optimization)"
get-kernel-text-for  = "('$(1)' CFLAGS for kernels)"
get-refkern-text-for = "('$(1)' CFLAGS for ref. kernels)"
get-frame-text-for   = "('$(1)' CFLAGS for framework code)"
get-config-text-for  = "('$(1)' CFLAGS for config code)"



#
# --- Miscellaneous helper functions -------------------------------------------
#

# Define functions that filters a list of filepaths $(1) that contain (or
# omit) an arbitrary substring $(2).
files-that-contain      = $(strip $(foreach f, $(1), $(if $(findstring $(2),$(f)),$(f),)))
files-that-dont-contain = $(strip $(foreach f, $(1), $(if $(findstring $(2),$(f)),,$(f))))



#
# --- Include makefile configuration file --------------------------------------
#

# The path to the directory in which BLIS was built.
ifeq ($(strip $(BUILD_PATH)),)
BUILD_PATH        := .
endif

# Define the name of the configuration file.
CONFIG_MK_FILE     := config.mk

# Include the configuration file.
-include $(BUILD_PATH)/$(CONFIG_MK_FILE)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(CONFIG_MK_INCLUDED)),yes)
CONFIG_MK_PRESENT := yes
else
CONFIG_MK_PRESENT := no
endif




#
# --- Primary makefile variable definitions ------------------------------------
#

# The base name of the BLIS library that we will build.
BLIS_LIB_BASE_NAME := libblis

# All makefile fragments in the tree will have this name.
FRAGMENT_MK        := .fragment.mk

# Locations of important files.
CONFIG_DIR         := config
FRAME_DIR          := frame
REFKERN_DIR        := ref_kernels
KERNELS_DIR        := kernels
OBJ_DIR            := obj
LIB_DIR            := lib
INCLUDE_DIR        := include
TESTSUITE_DIR      := testsuite

# Other kernel-related definitions.
KERNEL_SUFS        := c s S
KERNELS_STR        := kernels
REF_SUF            := ref

# The names of the testsuite binary executable and related default names
# of its input/configuration files.
TESTSUITE_NAME     := test_$(BLIS_LIB_BASE_NAME)
TESTSUITE_CONF_GEN := input.general
TESTSUITE_CONF_OPS := input.operations
TESTSUITE_OUT_FILE := output.testsuite

# CHANGELOG file.
CHANGELOG          := CHANGELOG

# Something for OS X so that echo -n works as expected.
SHELL              := bash

# Construct paths to the four primary directories of source code:
# the config directory, general framework code, reference kernel code,
# and optimized kernel code. NOTE: We declare these as recursively
# expanded variables since DIST_PATH may be overridden later.
CONFIG_PATH        := $(DIST_PATH)/$(CONFIG_DIR)
FRAME_PATH         := $(DIST_PATH)/$(FRAME_DIR)
REFKERN_PATH       := $(DIST_PATH)/$(REFKERN_DIR)
KERNELS_PATH       := $(DIST_PATH)/$(KERNELS_DIR)


#
# --- Utility program definitions ----------------------------------------------
#

SH         := /bin/sh
MV         := mv
MKDIR      := mkdir -p
RM_F       := rm -f
RM_RF      := rm -rf
SYMLINK    := ln -sf
FIND       := find
GREP       := grep
EGREP      := grep -E
XARGS      := xargs
RANLIB     := ranlib
INSTALL    := install -c

# Script for creating a monolithic header file.
#FLATTEN_H  := $(DIST_PATH)/build/flatten-headers.sh
FLATTEN_H  := $(DIST_PATH)/build/flatten-headers.py

# Default archiver flags.
AR         := ar
ARFLAGS    := cr

# Used to refresh CHANGELOG.
GIT        := git
GIT_LOG    := $(GIT) log --decorate



#
# --- Determine the compiler vendor --------------------------------------------
#

ifneq ($(CC),)

VENDOR_STRING := $(shell $(CC) --version 2>/dev/null)
ifeq ($(VENDOR_STRING),)
VENDOR_STRING := $(shell $(CC) -qversion 2>/dev/null)
endif
ifeq ($(VENDOR_STRING),)
$(error Unable to determine compiler vendor.)
endif

CC_VENDOR := $(firstword $(shell echo '$(VENDOR_STRING)' | $(EGREP) -o 'icc|gcc|clang|ibm'))
ifeq ($(CC_VENDOR),)
$(error Unable to determine compiler vendor. Have you run './configure' yet?)
endif

endif



#
# --- Include makefile definitions file ----------------------------------------
#

# Define the name of the file containing build and architecture-specific
# makefile definitions.
MAKE_DEFS_FILE     := make_defs.mk

# Construct the paths to the makefile definitions files, each of which resides
# in a separate configuration sub-directory. We include CONFIG_NAME in this
# list since we might need
ALL_CONFIGS        := $(sort $(strip $(CONFIG_LIST) $(CONFIG_NAME)))
CONFIG_PATHS       := $(addprefix $(CONFIG_PATH)/, $(ALL_CONFIGS))
MAKE_DEFS_MK_PATHS := $(addsuffix /$(MAKE_DEFS_FILE), $(CONFIG_PATHS))

# Initialize the list of included (found) configurations to empty.
CONFIGS_INCL       :=

# Include the makefile definitions files implied by the list of configurations.
-include $(MAKE_DEFS_MK_PATHS)

# Detect whether we actually got all of the make definitions files. If
# we didn't, then maybe a configuration is mislabeled or missing. The
# check-env-make-defs target checks ALL_MAKE_DEFS_MK_PRESENT and outputs
# an error message if it is set to 'no'.
# NOTE: We combine the CONFIG_NAME and CONFIG_LIST for situations where
# the CONFIG_NAME is absent from the CONFIG_LIST (e.g., 'intel64' is a
# configuration family name with its own configuration directory and its
# own make_defs.mk file, but not a sub-configuration itself). If
# CONFIG_NAME is present in CONFIG_LIST, as with singleton configuration
# families, then the sort() function will remove duplicates from both
# strings being compared.
CONFIGS_EXPECTED := $(CONFIG_LIST) $(CONFIG_NAME)
ifeq ($(sort $(strip $(CONFIGS_INCL))), \
      $(sort $(strip $(CONFIGS_EXPECTED))))
ALL_MAKE_DEFS_MK_PRESENT := yes
else
ALL_MAKE_DEFS_MK_PRESENT := no
endif



#
# --- Default linker definitions -----------------------------------------------
#

# Default linker, flags.
# NOTE: -lpthread is needed unconditionally because BLIS uses pthread_once()
# to initialize itself in a thread-safe manner.
LINKER     := $(CC)
LDFLAGS    := -lpthread

# Never use libm with Intel compilers.
ifneq ($(CC_VENDOR),icc)
LDFLAGS    += -lm
endif

SOFLAGS    := -shared



#
# --- Configuration-agnostic flags ---------------------------------------------
#

# --- C Preprocessor flags ---

# Enable clock_gettime() in time.h.
CPPROCFLAGS := -D_POSIX_C_SOURCE=200112L
$(foreach conf, $(CONFIG_LIST), $(eval $(call append-var-for,CPPROCFLAGS,$(conf))))

# --- Shared library (position-independent code) flags ---

# Emit position-independent code for dynamic linking.
CPICFLAGS := -fPIC
$(foreach conf, $(CONFIG_LIST), $(eval $(call append-var-for,CPICFLAGS,$(conf))))

# --- Miscellaneous flags ---

# Enable C99.
CMISCFLAGS := -std=c99
$(foreach conf, $(CONFIG_LIST), $(eval $(call append-var-for,CMISCFLAGS,$(conf))))

# Disable tautological comparision warnings in clang.
ifeq ($(CC_VENDOR),clang)
CMISCFLAGS := -Wno-tautological-compare
$(foreach conf, $(CONFIG_LIST), $(eval $(call append-var-for,CMISCFLAGS,$(conf))))
endif

# --- Warning flags ---

# Disable unused function warnings and stop compiling on first error for
# all compilers that accept such options: gcc, clang, and icc.
ifneq ($(CC_VENDOR),ibm)
CWARNFLAGS := -Wall -Wno-unused-function -Wfatal-errors
$(foreach conf, $(CONFIG_LIST), $(eval $(call append-var-for,CWARNFLAGS,$(conf))))
endif

# --- Threading flags ---

ifeq ($(CC_VENDOR),gcc)
ifeq ($(THREADING_MODEL),auto)
THREADING_MODEL := openmp
endif
ifeq ($(THREADING_MODEL),openmp)
CTHREADFLAGS := -fopenmp
LDFLAGS      += -fopenmp
endif
ifeq ($(THREADING_MODEL),pthreads)
CTHREADFLAGS := -pthread
LDFLAGS      += -lpthread
endif
endif

ifeq ($(CC_VENDOR),icc)
ifeq ($(THREADING_MODEL),auto)
THREADING_MODEL := openmp
endif
ifeq ($(THREADING_MODEL),openmp)
CTHREADFLAGS := -fopenmp
LDFLAGS      += -fopenmp
endif
ifeq ($(THREADING_MODEL),pthreads)
CTHREADFLAGS := -pthread
LDFLAGS      += -lpthread
endif
endif

ifeq ($(CC_VENDOR),clang)
ifeq ($(THREADING_MODEL),auto)
THREADING_MODEL := pthreads
endif
ifeq ($(THREADING_MODEL),openmp)
CTHREADFLAGS := -fopenmp
LDFLAGS      += -fopenmp
endif
ifeq ($(THREADING_MODEL),pthreads)
CTHREADFLAGS := -pthread
LDFLAGS      += -lpthread
endif
endif


#
# --- Adjust verbosity level manually using make V=[0,1] -----------------------
#

ifeq ($(V),1)
BLIS_ENABLE_VERBOSE_MAKE_OUTPUT := yes
BLIS_ENABLE_TEST_OUTPUT := yes
endif

ifeq ($(V),0)
BLIS_ENABLE_VERBOSE_MAKE_OUTPUT := no
BLIS_ENABLE_TEST_OUTPUT := no
endif



#
# --- Append OS-specific libraries to LDFLAGS ----------------------------------
#

ifeq ($(OS_NAME),Linux)
LDFLAGS += -lrt
endif

# Remove duplicate flags/options in LDFLAGS (such as -lpthread) by sorting.
LDFLAGS := $(sort $(LDFLAGS))



#
# --- Include makefile fragments -----------------------------------------------
#

# Initialize our list of directory paths to makefile fragments with the empty
# list. This variable will accumulate all of the directory paths in which
# makefile fragments reside.
FRAGMENT_DIR_PATHS :=

# Initialize our makefile variables that source code files will be accumulated
# into by the makefile fragments. This initialization is very important! These
# variables will end up with weird contents if we don't initialize them to
# empty prior to recursively including the makefile fragments.
MK_CONFIG_SRC          :=
MK_FRAME_SRC           :=
MK_REFKERN_SRC         :=
MK_KERNELS_SRC         :=


# Construct paths to each of the sub-configurations specified in the
# configuration list. If CONFIG_NAME is not in CONFIG_LIST, include it in
# CONFIG_PATHS since we'll need access to its header files.
ifeq ($(findstring $(CONFIG_NAME),$(CONFIG_LIST)),)
CONFIG_PATHS       := $(addprefix $(CONFIG_PATH)/, $(CONFIG_NAME) $(CONFIG_LIST))
else
CONFIG_PATHS       := $(addprefix $(CONFIG_PATH)/, $(CONFIG_LIST))
endif

# This variable is used by the include statements as they recursively include
# one another. For the 'config' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_PATH        := $(DIST_PATH)/$(CONFIG_DIR)

# Recursively include the makefile fragments in each of the sub-configuration
# directories.
-include $(addsuffix /$(FRAGMENT_MK), $(CONFIG_PATHS))


# Construct paths to each of the kernel sets required by the sub-configurations
# in the configuration list.
KERNEL_PATHS       := $(addprefix $(KERNELS_PATH)/, $(KERNEL_LIST))

# This variable is used by the include statements as they recursively include
# one another. For the 'kernels' directory, we initialize it to that directory
# in preparation to include the fragments in the configuration sub-directory.
PARENT_PATH        := $(DIST_PATH)/$(KERNELS_DIR)

# Recursively include the makefile fragments in each of the kernels sub-
# directories.
-include $(addsuffix /$(FRAGMENT_MK), $(KERNEL_PATHS))


# This variable is used by the include statements as they recursively include
# one another. For the framework and reference kernel source trees (ie: the
# 'frame' and 'ref_kernels' directories), we initialize it to the top-level
# directory since that is its parent. Same for the kernels directory, since it
# resides in the same top-level directory.
PARENT_PATH        := $(DIST_PATH)

# Recursively include all the makefile fragments in the directories for the
# reference kernels and portable framework.
-include $(addsuffix /$(FRAGMENT_MK), $(REFKERN_PATH))
-include $(addsuffix /$(FRAGMENT_MK), $(FRAME_PATH))


# Create a list of the makefile fragments.
MAKEFILE_FRAGMENTS := $(addsuffix /$(FRAGMENT_MK), $(FRAGMENT_DIR_PATHS))

# Detect whether we actually got any makefile fragments. If we didn't, then it
# is likely that the user has not yet generated them (via configure).
ifeq ($(strip $(MAKEFILE_FRAGMENTS)),)
MAKEFILE_FRAGMENTS_PRESENT := no
else
MAKEFILE_FRAGMENTS_PRESENT := yes
endif
#$(error fragment dir paths: $(FRAGMENT_DIR_PATHS))


#
# --- Compiler include path definitions ----------------------------------------
#

# Expand the fragment paths that contain .h files to attain the set of header
# files present in all fragment paths. Then strip all leading, internal, and
# trailing whitespace from the list.
MK_HEADER_FILES := $(strip $(foreach frag_path, \
                                     . $(FRAGMENT_DIR_PATHS), \
                                     $(wildcard $(frag_path)/*.h)))

# Expand the fragment paths that contain .h files, and take the first
# expansion. Then, strip the header filename to leave the path to each header
# location. Notice this process even weeds out duplicates!
MK_HEADER_DIR_PATHS := $(dir $(foreach frag_path, \
                                       . $(FRAGMENT_DIR_PATHS), \
                                       $(firstword $(wildcard $(frag_path)/*.h))))

# Add -I to each header path so we can specify our include search paths to the
# C compiler.
INCLUDE_PATHS   := $(strip $(patsubst %, -I%, $(MK_HEADER_DIR_PATHS)))

# Construct the base path for the intermediate include directory.
BASE_INC_PATH   := $(BUILD_PATH)/$(INCLUDE_DIR)/$(CONFIG_NAME)

# Isolate the path to blis.h by filtering the file from the list of headers.
BLIS_H          := blis.h
BLIS_H_SRC_PATH := $(filter %/$(BLIS_H), $(MK_HEADER_FILES))

# Construct the path to the intermediate flattened/monolithic blis.h file.
BLIS_H_FLAT     := $(BASE_INC_PATH)/$(BLIS_H)

# Obtain a list of header files #included inside of the bli_cntx_ref.c file.
# Paths to these files will be needed when compiling with the monolithic
# header.
REF_KER_SRC     := $(DIST_PATH)/$(REFKERN_DIR)/bli_cntx_ref.c
REF_KER_HEADERS := $(shell grep "\#include" $(REF_KER_SRC) | sed -e "s/\#include [\"<]\([a-zA-Z0-9\_\.\/\-]*\)[\">].*/\1/g" | grep -v blis.h)

# Match each header found above with the path to that header, and then strip
# leading, trailing, and internal whitespace.
REF_KER_H_PATHS := $(strip $(foreach header, $(REF_KER_HEADERS), \
                               $(dir $(filter %/$(header), \
                                              $(MK_HEADER_FILES)))))

# Add -I to each header path so we can specify our include search paths to the
# C compiler. Then add frame/include since it's needed for bli_oapi_w[o]_cntx.h.
REF_KER_I_PATHS := $(strip $(patsubst %, -I%, $(REF_KER_H_PATHS)))
REF_KER_I_PATHS += -I$(DIST_PATH)/frame/include

# Finally, prefix the paths above with the base include path.
INCLUDE_PATHS   := -I$(BASE_INC_PATH) $(REF_KER_I_PATHS)


#
# --- CBLAS header definitions -------------------------------------------------
#

CBLAS_H          := cblas.h
CBLAS_H_SRC_PATH := $(filter %/$(CBLAS_H), $(MK_HEADER_FILES))

# Construct the path to the intermediate flattened/monolithic cblas.h file.
CBLAS_H_FLAT    := $(BASE_INC_PATH)/$(CBLAS_H)


#
# --- Special preprocessor macro definitions -----------------------------------
#

# Define a C preprocessor macro to communicate the current version so that it
# can be embedded into the library and queried later.
VERS_DEF       := -DBLIS_VERSION_STRING=\"$(VERSION)\"



# end of ifndef COMMON_MK_INCLUDED conditional block
endif

