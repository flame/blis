#!/bin/bash
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
# --- Include makefile configuration file --------------------------------------
#

ifeq ($(strip $(RELPATH)),)
RELPATH := .
endif

# Define the name of the configuration file.
CONFIG_MK_FILE     := config.mk

# Include the configuration file.
-include $(RELPATH)/$(CONFIG_MK_FILE)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(CONFIG_MK_INCLUDED)),yes)
CONFIG_MK_PRESENT := yes
else
CONFIG_MK_PRESENT := no
endif

# Locations of important files.
CONFIG_DIR         := config

# Now we have access to CONFIG_NAME, which tells us which sub-directory of the
# config directory to use as our configuration. Also using CONFIG_NAME, we
# construct the path to the general framework source tree.
CONFIG_PATH       := $(DIST_PATH)/$(CONFIG_DIR)/$(CONFIG_NAME)

# If CONFIG_PATH is not an absolute path (does not begin with /) then prepend
# RELPATH to it.
ifeq ($(strip $(filter /%,$(CONFIG_PATH))),)
CONFIG_PATH := $(RELPATH)/$(CONFIG_PATH)
endif



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

CC_VENDOR := $(firstword $(shell echo '$(VENDOR_STRING)' | $(EGREP) -o 'icc|gcc|clang|emcc|pnacl|IBM'))
ifeq ($(CC_VENDOR),)
$(error Unable to determine compiler vendor.)
endif

endif



#
# --- Include makefile definitions file ----------------------------------------
#

# Define the name of the file containing build and architecture-specific
# makefile definitions.
MAKE_DEFS_FILE     := make_defs.mk

# Construct the path to the makefile definitions file residing inside of
# the configuration sub-directory.
MAKE_DEFS_MK_PATH := $(CONFIG_PATH)/$(MAKE_DEFS_FILE)

# Include the makefile definitions file.
-include $(MAKE_DEFS_MK_PATH)

# Detect whether we actually got the make definitios file. If we didn't, then
# it is likely that the configuration is invalid (or incomplete).
ifeq ($(strip $(MAKE_DEFS_MK_INCLUDED)),yes)
MAKE_DEFS_MK_PRESENT := yes
else
MAKE_DEFS_MK_PRESENT := no
endif



#
# --- Configuration-agnostic flags ---------------------------------------------
#

ifeq ($(CC_VENDOR),gcc)
ifeq ($(THREADING_MODEL),auto)
THREADING_MODEL := omp
endif
ifeq ($(THREADING_MODEL),omp)
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
THREADING_MODEL := omp
endif
ifeq ($(THREADING_MODEL),omp)
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
ifeq ($(THREADING_MODEL),omp)
CTHREADFLAGS := -fopenmp
LDFLAGS      += -fopenmp
endif
ifeq ($(THREADING_MODEL),pthreads)
CTHREADFLAGS := -pthread
LDFLAGS      += -lpthread
endif
endif

# Aggregate all of the flags into multiple groups: one for standard compilation,
# and one for each of the supported "special" compilation modes.

CFLAGS_NOOPT   := $(CDBGFLAGS) $(CWARNFLAGS) $(CPICFLAGS) $(CTHREADFLAGS) $(CMISCFLAGS) $(CPPROCFLAGS)
CFLAGS         := $(COPTFLAGS)  $(CVECFLAGS) $(CFLAGS_NOOPT)
CFLAGS_KERNELS := $(CKOPTFLAGS) $(CVECFLAGS) $(CFLAGS_NOOPT)



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



# end of ifndef COMMON_MK_INCLUDED conditional block
endif


