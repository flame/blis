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

#
# Makefile
#
# Field G. Van Zee
# 
# Top-level makefile for libflame linear algebra library.
#
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all libs test install uninstall clean \
        check-env check-env-mk check-env-fragments check-env-make-defs \
        testsuite testsuite-run testsuite-bin \
        install-libs install-headers install-lib-symlinks \
        showconfig \
        cleanlib distclean cleanmk cleanleaves \
        changelog \
        uninstall-libs uninstall-headers uninstall-lib-symlinks \
        uninstall-old



#
# --- Makefile initialization --------------------------------------------------
#

# The base name of the BLIS library that we will build.
BLIS_LIB_BASE_NAME := libblis

# Define the name of the configuration file.
CONFIG_MK_FILE     := config.mk

# Define the name of the file containing build and architecture-specific
# makefile definitions.
MAKE_DEFS_FILE     := make_defs.mk

# All makefile fragments in the tree will have this name.
FRAGMENT_MK        := .fragment.mk

# Locations of important files.
BUILD_DIR          := build
CONFIG_DIR         := config
FRAME_DIR          := frame
OBJ_DIR            := obj
LIB_DIR            := lib
TESTSUITE_DIR      := testsuite

# The names of the testsuite binary executable and related default names
# of its input/configuration files.
TESTSUITE_NAME     := test_$(BLIS_LIB_BASE_NAME)
TESTSUITE_CONF_GEN := input.general
TESTSUITE_CONF_OPS := input.operations
TESTSUITE_OUT_FILE := output.testsuite

# The name of the file where the version string is stored.
VERSION_FILE       := version

# The name of the "special" directories, which contain source code that
# use non-standard compiler flags.
NOOPT_DIR          := noopt
KERNELS_DIR        := kernels

# Text strings that alert the user to the fact that special source code is
# being compiled.
NOOPT_TEXT         := "(NOTE: using flags for no optimization)"
KERNELS_TEXT       := "(NOTE: using flags for kernels)"

# CHANGELOG file.
CHANGELOG          := CHANGELOG



#
# --- Include makefile configuration file --------------------------------------
#

# Include the configuration file.
-include $(CONFIG_MK_FILE)

# Detect whether we actually got the configuration file. If we didn't, then
# it is likely that the user has not yet generated it (via configure).
ifeq ($(strip $(CONFIG_MK_INCLUDED)),yes)
CONFIG_MK_PRESENT := yes
else
CONFIG_MK_PRESENT := no
endif

# Now we have access to CONFIG_NAME, which tells us which sub-directory of the
# config directory to use as our configuration. Also using CONFIG_NAME, we
# construct the path to the general framework source tree.
CONFIG_PATH       := $(DIST_PATH)/$(CONFIG_DIR)/$(CONFIG_NAME)
FRAME_PATH        := $(DIST_PATH)/$(FRAME_DIR)

# Construct base paths for the object file tree.
BASE_OBJ_PATH           := ./$(OBJ_DIR)/$(CONFIG_NAME)
BASE_OBJ_CONFIG_PATH    := $(BASE_OBJ_PATH)/$(CONFIG_DIR)
BASE_OBJ_FRAME_PATH     := $(BASE_OBJ_PATH)/$(FRAME_DIR)

# Construct base path for the library.
BASE_LIB_PATH           := ./$(LIB_DIR)/$(CONFIG_NAME)



#
# --- Include makefile definitions file ----------------------------------------
#

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
# --- Main target variable definitions -----------------------------------------
#


# Construct the architecture-version string, which will be used to name the
# library upon installation.
VERSION                := $(shell cat $(DIST_PATH)/$(VERSION_FILE))
VERS_CONF              := $(VERSION)-$(CONFIG_NAME)

# --- Library names ---

# Note: These names will be modified later to include the configuration and
# version strings.
BLIS_LIB_NAME      := $(BLIS_LIB_BASE_NAME).a
BLIS_DLL_NAME      := $(BLIS_LIB_BASE_NAME).so

# --- BLIS framework source and object variable names ---

# These are the makefile variables that source code files will be accumulated
# into by the makefile fragments. Notice that we include separate variables
# for regular and "special" source.
MK_FRAME_SRC           :=
MK_FRAME_NOOPT_SRC     :=
MK_FRAME_KERNELS_SRC   :=
MK_CONFIG_SRC          :=
MK_CONFIG_NOOPT_SRC    :=
MK_CONFIG_KERNELS_SRC  :=

# These hold object filenames corresponding to above.
MK_FRAME_OBJS          :=
MK_FRAME_NOOPT_OBJS    :=
MK_FRAME_KERNELS_OBJS  :=
MK_CONFIG_OBJS         :=
MK_CONFIG_NOOPT_OBJS   :=
MK_CONFIG_KERNELS_OBJS :=

# Append the base library path to the library names.
MK_ALL_BLIS_LIB        := $(BASE_LIB_PATH)/$(BLIS_LIB_NAME)
MK_ALL_BLIS_DLL        := $(BASE_LIB_PATH)/$(BLIS_DLL_NAME)

# --- Define install target names for static libraries ---

MK_BLIS_LIB                  := $(MK_ALL_BLIS_LIB)
MK_BLIS_LIB_INST             := $(patsubst $(BASE_LIB_PATH)/%.a, \
                                           $(INSTALL_PREFIX)/lib/%.a, \
                                           $(MK_BLIS_LIB))
MK_BLIS_LIB_INST_W_VERS_CONF := $(patsubst $(BASE_LIB_PATH)/%.a, \
                                           $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a, \
                                           $(MK_BLIS_LIB))

# --- Define install target names for shared libraries ---

MK_BLIS_DLL                  := $(MK_ALL_BLIS_DLL)
MK_BLIS_DLL_INST             := $(patsubst $(BASE_LIB_PATH)/%.so, \
                                           $(INSTALL_PREFIX)/lib/%.so, \
                                           $(MK_BLIS_DLL))
MK_BLIS_DLL_INST_W_VERS_CONF := $(patsubst $(BASE_LIB_PATH)/%.so, \
                                           $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).so, \
                                           $(MK_BLIS_DLL))

# --- Determine which libraries to build ---

MK_LIBS                           :=
MK_LIBS_INST                      :=
MK_LIBS_INST_W_VERS_CONF          :=

ifeq ($(BLIS_ENABLE_STATIC_BUILD),yes)
MK_LIBS                           += $(MK_BLIS_LIB)
MK_LIBS_INST                      += $(MK_BLIS_LIB_INST)
MK_LIBS_INST_W_VERS_CONF          += $(MK_BLIS_LIB_INST_W_VERS_CONF)
endif

ifeq ($(BLIS_ENABLE_DYNAMIC_BUILD),yes)
MK_LIBS                           += $(MK_BLIS_DLL)
MK_LIBS_INST                      += $(MK_BLIS_DLL_INST)
MK_LIBS_INST_W_VERS_CONF          += $(MK_BLIS_DLL_INST_W_VERS_CONF)
endif

# Strip leading, internal, and trailing whitespace.
MK_LIBS_INST                      := $(strip $(MK_LIBS_INST))
MK_LIBS_INST_W_VERS_CONF          := $(strip $(MK_LIBS_INST_W_VERS_CONF))

# Set the include directory names
MK_INCL_DIR_INST                  := $(INSTALL_PREFIX)/include/blis



#
# --- Include makefile fragments -----------------------------------------------
#

# Initialize our list of directory paths to makefile fragments with the empty
# list. This variable will accumulate all of the directory paths in which
# makefile fragments reside.
FRAGMENT_DIR_PATHS :=

# This variable is used by the include statements as they recursively include
# one another. For the framework source tree ('frame' directory), we initialize
# it to the top-level directory since that is its parent.
PARENT_PATH        := $(DIST_PATH)

# Recursively include all the makefile fragments in the framework itself.
-include $(addsuffix /$(FRAGMENT_MK), $(FRAME_PATH))

# Now set PARENT_PATH to $(DIST_PATH)/config in preparation to include the
# fragments in the configuration sub-directory.
PARENT_PATH        := $(DIST_PATH)/$(CONFIG_DIR)

# Recursively include all the makefile fragments in the configuration
# sub-directory.
-include $(addsuffix /$(FRAGMENT_MK), $(CONFIG_PATH))

# Create a list of the makefile fragments.
MAKEFILE_FRAGMENTS := $(addsuffix /$(FRAGMENT_MK), $(FRAGMENT_DIR_PATHS))

# Detect whether we actually got any makefile fragments. If we didn't, then it
# is likely that the user has not yet generated them (via configure).
ifeq ($(strip $(MAKEFILE_FRAGMENTS)),)
MAKEFILE_FRAGMENTS_PRESENT := no
else
MAKEFILE_FRAGMENTS_PRESENT := yes
endif



#
# --- Compiler include path definitions ----------------------------------------
#

# Expand the fragment paths that contain .h files to attain the set of header
# files present in all fragment paths.
MK_HEADER_FILES := $(foreach frag_path, $(FRAGMENT_DIR_PATHS), \
                                        $(wildcard $(frag_path)/*.h))

# Strip the leading, internal, and trailing whitespace from our list of header
# files. This makes the "make install-headers" much more readable.
MK_HEADER_FILES := $(strip $(MK_HEADER_FILES))

# Expand the fragment paths that contain .h files, and take the first
# expansion. Then, strip the header filename to leave the path to each header
# location. Notice this process even weeds out duplicates! Add the config
# directory manually since it contains FLA_config.h.
MK_HEADER_DIR_PATHS := $(dir $(foreach frag_path, $(FRAGMENT_DIR_PATHS), \
                                       $(firstword $(wildcard $(frag_path)/*.h))))

# Add -I to each header path so we can specify our include search paths to the
# C compiler.
INCLUDE_PATHS   := $(strip $(patsubst %, -I%, $(MK_HEADER_DIR_PATHS)))
CFLAGS          := $(CFLAGS) $(INCLUDE_PATHS)
CFLAGS_NOOPT    := $(CFLAGS_NOOPT) $(INCLUDE_PATHS)
CFLAGS_KERNELS  := $(CFLAGS_KERNELS) $(INCLUDE_PATHS)



#
# --- Special preprocessor macro definitions -----------------------------------
#

# Define a C preprocessor macro to communicate the current version so that it
# can be embedded into the library and queried later.
VERS_DEF       := -DBLIS_VERSION_STRING=\"$(VERSION)\"
CFLAGS         := $(CFLAGS) $(VERS_DEF)
CFLAGS_NOOPT   := $(CFLAGS_NOOPT) $(VERS_DEF)
CFLAGS_KERNELS := $(CFLAGS_KERNELS) $(VERS_DEF)



#
# --- Library object definitions -----------------------------------------------
#

# Convert source file paths to object file paths by replacing the base source
# directories with the base object directories, and also replacing the source
# file suffix (eg: '.c') with '.o'.
MK_BLIS_FRAME_OBJS         := $(patsubst $(FRAME_PATH)/%.c, $(BASE_OBJ_FRAME_PATH)/%.o, \
                                          $(filter %.c, $(MK_FRAME_SRC)))
MK_BLIS_FRAME_NOOPT_OBJS   := $(patsubst $(FRAME_PATH)/%.c, $(BASE_OBJ_FRAME_PATH)/%.o, \
                                          $(filter %.c, $(MK_FRAME_NOOPT_SRC)))
MK_BLIS_FRAME_KERNELS_OBJS := $(patsubst $(FRAME_PATH)/%.c, $(BASE_OBJ_FRAME_PATH)/%.o, \
                                          $(filter %.c, $(MK_FRAME_KERNELS_SRC)))

MK_BLIS_CONFIG_OBJS          := $(patsubst $(CONFIG_PATH)/%.S, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                          $(filter %.S, $(MK_CONFIG_SRC)))
MK_BLIS_CONFIG_OBJS          += $(patsubst $(CONFIG_PATH)/%.c, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                          $(filter %.c, $(MK_CONFIG_SRC)))

MK_BLIS_CONFIG_NOOPT_OBJS    := $(patsubst $(CONFIG_PATH)/%.S, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                          $(filter %.S, $(MK_CONFIG_NOOPT_SRC)))
MK_BLIS_CONFIG_NOOPT_OBJS    += $(patsubst $(CONFIG_PATH)/%.c, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                          $(filter %.c, $(MK_CONFIG_NOOPT_SRC)))

MK_BLIS_CONFIG_KERNELS_OBJS  := $(patsubst $(CONFIG_PATH)/%.S, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                          $(filter %.S, $(MK_CONFIG_KERNELS_SRC)))
MK_BLIS_CONFIG_KERNELS_OBJS  += $(patsubst $(CONFIG_PATH)/%.c, $(BASE_OBJ_CONFIG_PATH)/%.o, \
                                          $(filter %.c, $(MK_CONFIG_KERNELS_SRC)))

# Combine all of the object files into some readily-accessible variables.
MK_ALL_BLIS_OPT_OBJS      := $(MK_BLIS_CONFIG_OBJS) \
                             $(MK_BLIS_FRAME_OBJS)

MK_ALL_BLIS_NOOPT_OBJS    := $(MK_BLIS_CONFIG_NOOPT_OBJS) \
                             $(MK_BLIS_FRAME_NOOPT_OBJS)

MK_ALL_BLIS_KERNELS_OBJS  := $(MK_BLIS_CONFIG_KERNELS_OBJS) \
                             $(MK_BLIS_FRAME_KERNELS_OBJS)

MK_ALL_BLIS_OBJS          := $(MK_ALL_BLIS_OPT_OBJS) \
                             $(MK_ALL_BLIS_NOOPT_OBJS) \
                             $(MK_ALL_BLIS_KERNELS_OBJS)



#
# --- Test suite definitions ---------------------------------------------------
#

# The location of the test suite's general and operations-specific
# input/configuration files.
TESTSUITE_CONF_GEN_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_CONF_GEN)
TESTSUITE_CONF_OPS_PATH := $(DIST_PATH)/$(TESTSUITE_DIR)/$(TESTSUITE_CONF_OPS)

# The locations of the test suite source directory and the local object
# directory.
TESTSUITE_SRC_PATH      := $(DIST_PATH)/$(TESTSUITE_DIR)/src
BASE_OBJ_TESTSUITE_PATH := $(BASE_OBJ_PATH)/$(TESTSUITE_DIR)

# Convert source file paths to object file paths by replacing the base source
# directories with the base object directories, and also replacing the source
# file suffix (eg: '.c') with '.o'.
MK_TESTSUITE_OBJS       := $(patsubst $(TESTSUITE_SRC_PATH)/%.c, \
                                      $(BASE_OBJ_TESTSUITE_PATH)/%.o, \
                                      $(wildcard $(TESTSUITE_SRC_PATH)/*.c))

# The test suite binary executable filename.
ifeq ($(CONFIG_NAME),pnacl)
# Linked executable
MK_TESTSUITE_BIN_UNSTABLE := $(BASE_OBJ_TESTSUITE_PATH)/test_libblis.unstable.pexe
# Finalized executable
MK_TESTSUITE_BIN_PNACL    := $(BASE_OBJ_TESTSUITE_PATH)/test_libblis.pexe
# Translated executable (for x86-64)
TESTSUITE_BIN             := test_libblis.x86-64.nexe
else
ifeq ($(CONFIG_NAME),emscripten)
# JS script name.
TESTSUITE_BIN             := test_libblis.js
else
# Binary executable name.
TESTSUITE_BIN             := test_libblis.x
endif # emscripten
endif # pnacl



#
# --- Uninstall definitions ----------------------------------------------------
#

# This shell command grabs all files named "libblis-*.a" or "libblis-*.so" in
# the installation directory and then filters out the name of the library
# archive for the current version/configuration. We consider this remaining set
# of libraries to be "old" and eligible for removal upon running of the
# uninstall-old target.
UNINSTALL_LIBS   := $(shell $(FIND) $(INSTALL_PREFIX)/lib/ -name "$(BLIS_LIB_BASE_NAME)-*.[a|so]" 2> /dev/null | $(GREP) -v "$(BLIS_LIB_BASE_NAME)-$(VERS_CONF).[a|so]" | $(GREP) -v $(BLIS_LIB_NAME))



#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: libs

libs: blis-lib

test: testsuite

install: libs install-libs install-headers install-lib-symlinks

uninstall: uninstall-libs uninstall-lib-symlinks uninstall-headers

clean: cleanlib cleantest


# --- General source code / object code rules ---

# Define two functions, each of which takes one argument (an object file
# path). The functions determine which CFLAGS and text string are needed to
# compile the object file. Note that we match with a preceding forward slash,
# so the directory name must begin with the special directory name, but it
# can have trailing characters (e.g. 'kernels_x86').
get_cflags_for_obj = $(if $(findstring /$(NOOPT_DIR),$1),$(CFLAGS_NOOPT),\
                     $(if $(findstring /$(KERNELS_DIR),$1),$(CFLAGS_KERNELS),\
                     $(CFLAGS)))

get_ctext_for_obj = $(if $(findstring /$(NOOPT_DIR),$1),$(NOOPT_TEXT),\
                    $(if $(findstring /$(KERNELS_DIR),$1),$(KERNELS_TEXT),))

$(BASE_OBJ_FRAME_PATH)/%.o: $(FRAME_PATH)/%.c $(MK_HEADER_FILES) $(MAKE_DEFS_MK_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get_cflags_for_obj,$@) -c $< -o $@
else
	@echo "Compiling $<" $(call get_ctext_for_obj,$@)
	@$(CC) $(call get_cflags_for_obj,$@) -c $< -o $@
endif

$(BASE_OBJ_CONFIG_PATH)/%.o: $(CONFIG_PATH)/%.c $(MK_HEADER_FILES) $(MAKE_DEFS_MK_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get_cflags_for_obj,$@) -c $< -o $@
else
	@echo "Compiling $<" $(call get_ctext_for_obj,$@)
	@$(CC) $(call get_cflags_for_obj,$@) -c $< -o $@
endif

$(BASE_OBJ_CONFIG_PATH)/%.o: $(CONFIG_PATH)/%.S $(MK_HEADER_FILES) $(MAKE_DEFS_MK_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(call get_cflags_for_obj,$@) -c $< -o $@
else
	@echo "Compiling $<" $(call get_ctext_for_obj,$@)
	@$(CC) $(call get_cflags_for_obj,$@) -c $< -o $@
endif


# --- Environment check rules ---

check-env: check-env-make-defs check-env-fragments check-env-mk

check-env-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
	$(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-env-fragments: check-env-mk
ifeq ($(MAKEFILE_FRAGMENTS_PRESENT),no)
	$(error Cannot proceed: makefile fragments not detected! Run configure first)
endif

check-env-make-defs: check-env-fragments
ifeq ($(MAKE_DEFS_MK_PRESENT),no)
	$(error Cannot proceed: make_defs.mk not detected! Invalid configuration)
endif


# --- All-purpose library rule (static and shared) ---

blis-lib: check-env $(MK_LIBS)


# --- Static library archiver rules ---

$(MK_ALL_BLIS_LIB): $(MK_ALL_BLIS_OBJS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(AR) $(ARFLAGS) $@ $?
	$(RANLIB) $@
else
	@echo "Archiving $@"
	@$(AR) $(ARFLAGS) $@ $?
	@$(RANLIB) $@
endif


# --- Dynamic library linker rules ---

$(MK_ALL_BLIS_DLL): $(MK_ALL_BLIS_OBJS)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(LINKER) $(SOFLAGS) $(LDFLAGS) -o $@ $?
else 
	@echo "Dynamically linking $@"
	@$(LINKER) $(SOFLAGS) $(LDFLAGS) -o $@ $?
endif


# --- Test suite rules ---

testsuite: testsuite-run

testsuite-bin: check-env $(TESTSUITE_BIN)

$(BASE_OBJ_TESTSUITE_PATH)/%.o: $(TESTSUITE_SRC_PATH)/%.c
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(CC) $(CFLAGS) -c $< -o $@
else
	@echo "Compiling $<"
	@$(CC) $(CFLAGS) -c $< -o $@
endif

ifeq ($(CONFIG_NAME),pnacl)

# Link executable (produces unstable LLVM bitcode)
$(MK_TESTSUITE_BIN_UNSTABLE): $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(MK_BLIS_LIB) $(LDFLAGS)'"
	@$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@
endif

# Finalize PNaCl executable (i.e. convert from LLVM bitcode to PNaCl bitcode)
$(MK_TESTSUITE_BIN_PNACL): $(MK_TESTSUITE_BIN_UNSTABLE)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(FINALIZER) $(FINFLAGS) -o $@ $<
else
	@echo "Finalizing $@"
	@$(FINALIZER) $(FINFLAGS) -o $@ $<
endif

# Translate PNaCl executable to x86-64 NaCl executable
$(TESTSUITE_BIN): $(MK_TESTSUITE_BIN_PNACL)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(TRANSLATOR) $(TRNSFLAGS) $(TRNSAMD64FLAGS) $< -o $@
else
	@echo "Translating $< -> $@"
	@$(TRANSLATOR) $(TRNSFLAGS) $(TRNSAMD64FLAGS) $< -o $@
endif

else # Non-PNaCl case

ifeq ($(CONFIG_NAME),emscripten)
# Generate JavaScript and embed testsuite resources normally
$(TESTSUITE_BIN): $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(TESTSUITE_CONF_GEN_PATH) $(TESTSUITE_CONF_OPS_PATH)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@ \
		--embed-file $(TESTSUITE_CONF_GEN_PATH)@input.general \
		--embed-file $(TESTSUITE_CONF_OPS_PATH)@input.operations
else
	@echo "Linking $@ against '$(MK_BLIS_LIB) $(LDFLAGS)'"
	@$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@ \
		--embed-file $(TESTSUITE_CONF_GEN_PATH)@input.general \
		--embed-file $(TESTSUITE_CONF_OPS_PATH)@input.operations
endif
else
# Link executable normally
$(TESTSUITE_BIN): $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(MK_BLIS_LIB) $(LDFLAGS)'"
	@$(LINKER) $(MK_TESTSUITE_OBJS) $(MK_BLIS_LIB) $(LDFLAGS) -o $@
endif
endif

endif

testsuite-run: testsuite-bin
ifeq ($(CONFIG_NAME),pnacl)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(NACL_SDK_ROOT)/tools/sel_ldr_x86_64 -a -c -q \
	    -B $(NACL_SDK_ROOT)/tools/irt_core_x86_64.nexe -- \
	    $(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                     -o $(TESTSUITE_CONF_OPS_PATH) \
                         > $(TESTSUITE_OUT_FILE)
else
	@echo "Running $(TESTSUITE_BIN) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@$(NACL_SDK_ROOT)/tools/sel_ldr_x86_64 -a -c -q \
	    -B $(NACL_SDK_ROOT)/tools/irt_core_x86_64.nexe -- \
	    $(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                     -o $(TESTSUITE_CONF_OPS_PATH) \
                         > $(TESTSUITE_OUT_FILE)
endif
else
ifeq ($(CONFIG_NAME),emscripten)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(JSINT) $(TESTSUITE_BIN)
else
	@echo "Running $(TESTSUITE_BIN)"
	@$(JSINT) $(TESTSUITE_BIN)
endif
else
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	./$(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                   -o $(TESTSUITE_CONF_OPS_PATH) \
                        > $(TESTSUITE_OUT_FILE)

else ifeq ($(BLIS_ENABLE_TEST_OUTPUT), yes)
	./$(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                   -o $(TESTSUITE_CONF_OPS_PATH) | \
                        tee $(TESTSUITE_OUT_FILE)
else
	@echo "Running $(TESTSUITE_BIN) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@./$(TESTSUITE_BIN) -g $(TESTSUITE_CONF_GEN_PATH) \
	                    -o $(TESTSUITE_CONF_OPS_PATH) \
                         > $(TESTSUITE_OUT_FILE)
endif
endif # emscripten
endif # pnacl

# --- Install rules ---

install-libs: check-env $(MK_LIBS_INST_W_VERS_CONF)

install-headers: check-env $(MK_INCL_DIR_INST)

$(MK_INCL_DIR_INST): $(MK_HEADER_FILES) $(CONFIG_MK_FILE)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(MKDIR) $(@)
	$(INSTALL) -m 0644 $(MK_HEADER_FILES) $(@)
else
	@$(MKDIR) $(@)
	@echo "Installing C header files into $(@)/"
	@$(INSTALL) -m 0644 $(MK_HEADER_FILES) $(@)
endif

$(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a: $(BASE_LIB_PATH)/%.a $(CONFIG_MK_FILE)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(MKDIR) $(@D)
	$(INSTALL) -m 0644 $< $@
else
	@echo "Installing $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $< $@
endif

$(INSTALL_PREFIX)/lib/%-$(VERS_CONF).so: $(BASE_LIB_PATH)/%.so $(CONFIG_MK_FILE)
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(MKDIR) $(@D)
	$(INSTALL) -m 0644 $< $@
else
	@echo "Installing $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(MKDIR) $(@D)
	@$(INSTALL) -m 0644 $< $@
endif


# --- Install-symlinks rules ---

install-lib-symlinks: check-env $(MK_LIBS_INST)

$(INSTALL_PREFIX)/lib/%.a: $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).a
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_PREFIX)/lib/
else
	@echo "Installing symlink $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_PREFIX)/lib/
endif

$(INSTALL_PREFIX)/lib/%.so: $(INSTALL_PREFIX)/lib/%-$(VERS_CONF).so
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	$(SYMLINK) $(<F) $(@F)
	$(MV) $(@F) $(INSTALL_PREFIX)/lib/
else
	@echo "Installing symlink $(@F) into $(INSTALL_PREFIX)/lib/"
	@$(SYMLINK) $(<F) $(@F)
	@$(MV) $(@F) $(INSTALL_PREFIX)/lib/
endif


# --- Query current configuration ---

showconfig: check-env
	@echo "Current configuration is '$(CONFIG_NAME)', located in '$(CONFIG_PATH)'"


# --- Clean rules ---

cleanlib: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(BASE_OBJ_CONFIG_PATH) -name "*.o" | $(XARGS) $(RM_F)
	- $(FIND) $(BASE_OBJ_FRAME_PATH) -name "*.o" | $(XARGS) $(RM_F)
	- $(FIND) $(BASE_LIB_PATH) -name "*.a" | $(XARGS) $(RM_F)
	- $(FIND) $(BASE_LIB_PATH) -name "*.so" | $(XARGS) $(RM_F)
else
	@echo "Removing .o files from $(BASE_OBJ_CONFIG_PATH)."
	@- $(FIND) $(BASE_OBJ_CONFIG_PATH) -name "*.o" | $(XARGS) $(RM_F)
	@echo "Removing .o files from $(BASE_OBJ_FRAME_PATH)."
	@- $(FIND) $(BASE_OBJ_FRAME_PATH) -name "*.o" | $(XARGS) $(RM_F)
	@echo "Removing .a files from $(BASE_LIB_PATH)."
	@- $(FIND) $(BASE_LIB_PATH) -name "*.a" | $(XARGS) $(RM_F)
	@echo "Removing .so files from $(BASE_LIB_PATH)."
	@- $(FIND) $(BASE_LIB_PATH) -name "*.so" | $(XARGS) $(RM_F)
endif

cleantest: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(BASE_OBJ_TESTSUITE_PATH) -name "*.o" -name "*.pexe" | $(XARGS) $(RM_F)
	- $(RM_RF) $(TESTSUITE_BIN)
else
	@echo "Removing object files from $(BASE_OBJ_TESTSUITE_PATH)."
	@- $(FIND) $(BASE_OBJ_TESTSUITE_PATH) -name "*.o" -name "*.pexe" | $(XARGS) $(RM_F)
	@echo "Removing $(TESTSUITE_BIN) binary."
	@- $(RM_RF) $(TESTSUITE_BIN)
endif

cleanmk: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(FIND) $(CONFIG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	- $(FIND) $(FRAME_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
else
	@echo "Removing makefile fragments from $(CONFIG_PATH)."
	@- $(FIND) $(CONFIG_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
	@echo "Removing makefile fragments from $(FRAME_PATH)."
	@- $(FIND) $(FRAME_PATH) -name "$(FRAGMENT_MK)" | $(XARGS) $(RM_F)
endif

distclean: check-env cleanmk cleanlib cleantest
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $(CONFIG_MK_FILE)
	- $(RM_RF) $(TESTSUITE_OUT_FILE)
	- $(RM_RF) $(OBJ_DIR)
	- $(RM_RF) $(LIB_DIR)
else
	@echo "Removing $(CONFIG_MK_FILE)."
	@- $(RM_F) $(CONFIG_MK_FILE)
	@echo "Removing $(TESTSUITE_OUT_FILE)."
	@- $(RM_F) $(TESTSUITE_OUT_FILE)
	@echo "Removing $(OBJ_DIR)."
	@- $(RM_RF) $(OBJ_DIR)
	@echo "Removing $(LIB_DIR)."
	@- $(RM_RF) $(LIB_DIR)
endif


# --- CHANGELOG rules ---

changelog: check-env
	@echo "Updating '$(DIST_PATH)/$(CHANGELOG)' via '$(GIT_LOG)'."
	@$(GIT_LOG) > $(DIST_PATH)/$(CHANGELOG) 


# --- Uninstall rules ---

uninstall-old: $(UNINSTALL_LIBS)

uninstall-libs: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $(MK_LIBS_INST_W_VERS_CONF)
else
	@echo "Removing $(MK_LIBS_INST_W_VERS_CONF)."
	@- $(RM_F) $(MK_LIBS_INST_W_VERS_CONF)
endif

uninstall-lib-symlinks: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $(MK_LIBS_INST)
else
	@echo "Removing $(MK_LIBS_INST)."
	@- $(RM_F) $(MK_LIBS_INST)
endif

uninstall-headers: check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_RF) $(MK_INCL_DIR_INST)
else
	@echo "Removing $(MK_INCL_DIR_INST)/."
	@- $(RM_RF) $(MK_INCL_DIR_INST)
endif

$(UNINSTALL_LIBS): check-env
ifeq ($(BLIS_ENABLE_VERBOSE_MAKE_OUTPUT),yes)
	- $(RM_F) $@
else
	@echo "Removing $(@F) from $(@D)/."
	@- $(RM_F) $@
endif


