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
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
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
# Makefile for standalone BLIS test drivers.
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

.PHONY: all \
        blis essl \
        clean cleanx



# Comments:
# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
ifneq ($(strip $(BLIS_INSTALL_PATH)),)
LIB_PATH   := $(BLIS_INSTALL_PATH)/lib
INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
SHARE_PATH := $(BLIS_INSTALL_PATH)/share/blis
else
DIST_PATH  := ..
LIB_PATH    = ../lib/$(CONFIG_NAME)
INC_PATH    = ../include/$(CONFIG_NAME)
SHARE_PATH := ..
endif



#
# --- Include common makefile definitions --------------------------------------
#

# Include the common makefile fragment.
-include $(SHARE_PATH)/common.mk



#
# --- BLAS and LAPACK implementations ------------------------------------------
#

# BLAS library path(s). This is where the BLAS libraries reside.
BLAS_LIB_PATH  := $(HOME)/flame/lib
MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64/
ESSL_LIB_PATH  := /soft/libraries/essl/current/lib64

# OpenBLAS
OPENBLAS_LIB   := $(BLAS_LIB_PATH)/libopenblas.a

# ATLAS
ATLAS_LIB      := $(BLAS_LIB_PATH)/libf77blas.a \
                  $(BLAS_LIB_PATH)/libatlas.a

# MKL
MKL_LIB        := -L$(MKL_LIB_PATH) \
                  -lmkl_sequential \
                  -lmkl_core \
                  -lmkl_intel_lp64

# ESSL
# Note: ESSL is named differently for SMP and/or BG
ESSL_LIB        := $(ESSL_LIB_PATH)/libesslsmpbg.a   \
                -L$(IBM_MAIN_DIR)/xlsmp/bg/3.1/bglib64/ \
				-L$(IBM_MAIN_DIR)/xlf/bg/14.1/bglib64/ \
				-lxlsmp -lxlf90_r -lxlfmath -lxl

# Accelerate
MAC_LIB        := -framework Accelerate



#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(patsubst $(TEST_SRC_PATH)/%.c, \
                             $(TEST_OBJ_PATH)/%.o, \
                             $(wildcard $(TEST_SRC_PATH)/*.c))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the "framework" CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)



#
# --- Targets/rules ------------------------------------------------------------
#

# Complete list of possible targets when defining 'all':
#
#   blis openblas atlas mkl mac essl
#
all: blis essl

blis: test_gemm_blis.x \
      test_hemm_blis.x \
      test_herk_blis.x \
      test_her2k_blis.x \
      test_trmm_blis.x \
      test_trsm_blis.x

essl: test_gemm_essl.x \
      test_hemm_essl.x \
      test_herk_essl.x \
      test_her2k_essl.x \
      test_trmm_essl.x \
      test_trsm_essl.x

openblas: test_gemv_openblas.x \
      test_ger_openblas.x \
      test_hemv_openblas.x \
      test_her_openblas.x \
      test_her2_openblas.x \
      test_trmv_openblas.x \
      test_trsv_openblas.x \
      \
      test_gemm_openblas.x \
      test_hemm_openblas.x \
      test_herk_openblas.x \
      test_her2k_openblas.x \
      test_trmm_openblas.x \
      test_trsm_openblas.x

atlas: test_gemv_atlas.x \
      test_ger_atlas.x \
      test_hemv_atlas.x \
      test_her_atlas.x \
      test_her2_atlas.x \
      test_trmv_atlas.x \
      test_trsv_atlas.x \
      \
      test_gemm_atlas.x \
      test_hemm_atlas.x \
      test_herk_atlas.x \
      test_her2k_atlas.x \
      test_trmm_atlas.x \
      test_trsm_atlas.x

mkl:  test_gemv_mkl.x \
      test_ger_mkl.x \
      test_hemv_mkl.x \
      test_her_mkl.x \
      test_her2_mkl.x \
      test_trmv_mkl.x \
      test_trsv_mkl.x \
      \
      test_gemm_mkl.x \
      test_hemm_mkl.x \
      test_herk_mkl.x \
      test_her2k_mkl.x \
      test_trmm_mkl.x \
      test_trsm_mkl.x

mac:  test_gemv_mac.x \
      test_ger_mac.x \
      test_hemv_mac.x \
      test_her_mac.x \
      test_her2_mac.x \
      test_trmv_mac.x \
      test_trsv_mac.x \
      \
      test_gemm_mac.x \
      test_hemm_mac.x \
      test_herk_mac.x \
      test_her2k_mac.x \
      test_trmm_mac.x \
      test_trsm_mac.x



# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test_%_openblas.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"openblas\" -c $< -o $@

test_%_atlas.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"atlas\" -c $< -o $@

test_%_mkl.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"mkl\" -c $< -o $@

test_%_essl.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"essl\" -c $< -o $@

test_%_mac.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"mac\" -c $< -o $@

test_%_blis.o: test_%.c
	$(CC) $(CFLAGS) -DBLIS -c $< -o $@


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

test_%_openblas.x: test_%_openblas.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(OPENBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_atlas.x: test_%_atlas.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(ATLAS_LIB)    $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_mkl.x: test_%_mkl.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(MKL_LIB)      $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_essl.x: test_%_essl.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(ESSL_LIB)     $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_mac.x: test_%_mac.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(MAC_LIB)      $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_blis.x: test_%_blis.o $(LIBBLIS_LINK)
	$(LINKER) $<                             $(LIBBLIS_LINK) $(LDFLAGS) -o $@


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x

