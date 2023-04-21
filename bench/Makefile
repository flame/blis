
#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2017 - 2022, Advanced Micro Devices, Inc. All rights reserved.
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
        blis openblas atlas mkl \
        check-env check-env-mk check-lib \
        clean cleanx



#
# --- Determine makefile fragment location -------------------------------------
#

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

# BLIS library and header path. This is simply wherever it was installed.
#BLIS_LIB_PATH  := $(INSTALL_PREFIX)/lib
#BLIS_INC_PATH  := $(INSTALL_PREFIX)/include/blis

# BLIS library.
#BLIS_LIB       := $(BLIS_LIB_PATH)/libblis-mt.a

# BLAS library path(s). This is where the BLAS libraries reside.
BLAS_LIB_PATH  := $(HOME)/mylibs/openblas/lib

MKL_LIB_PATH   := ${MKLROOT}/lib/intel64


# OpenBLAS
OPENBLAS_LIB   := $(BLAS_LIB_PATH)/libopenblas.a

# ATLAS
ATLAS_LIB      := $(BLAS_LIB_PATH)/libf77blas.a \
                  $(BLAS_LIB_PATH)/libatlas.a

# MKL
#MKL_LIB        := -L$(MKL_LIB_PATH) \
#                  -lmkl_intel_lp64 \
#                  -lmkl_core \
#                  -lmkl_sequential \
#                  -lpthread -lm -ldl

# Uncomment below lines & comment above lines to link with multi-threaded library.
MKL_LIB        := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_gnu_thread \
                  -lpthread -lm -ldl -liomp5


# ESSL
# Note: ESSL is named differently for SMP and/or BG
#ESSL_TYPE      :=       # This is the 32b library on POWER
#ESSL_TYPE      := 6464  # This is the 64b library on POWER
#ESSL_TYPE      := bg    # This is the 32b single-threaded library on Blue Gene
#ESSL_TYPE      := smpbg # This is the 32b multi-threaded library on Blue Gene
#ESSL_LIB       := $(ESSL_LIB_PATH)/libessl$(ESSL_TYPE).a

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


CBLAS_HEADER_PATH  = ../frame/compat/cblas/src

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH) -I$(CBLAS_HEADER_PATH)

# Use the CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)


# Define Number of Repeats
NRTS := -DN_REPEAT=1000

#
# --- Targets/rules ------------------------------------------------------------
#

# Complete list of possible targets when defining 'all':
#
#   blis openblas atlas mkl mac essl
#
#all: blis openblas atlas mkl
all: blis openblas mkl

blis: \
       bench_gemm_blis.x \
       bench_gemmt_blis.x \
       bench_trsm_blis.x \
       bench_gemv_blis.x \
       bench_syrk_blis.x \
       bench_ger_blis.x \
       bench_nrm2_blis.x \
       bench_scalv_blis.x \
       bench_dotv_blis.x \
       bench_trsv_blis.x \
       bench_amaxv_blis.x \
       bench_copyv_blis.x \
       bench_swapv_blis.x \
       bench_axpbyv_blis.x

openblas: \
      bench_gemm_openblas.x \
      bench_gemmt_openblas.x \
      bench_trsm_openblas.x \
      bench_gemv_openblas.x \
      bench_syrk_openblas.x \
      bench_ger_openblas.x \
      bench_nrm2_openblas.x \
      bench_scalv_openblas.x \
      bench_dotv_openblas.x \
      bench_trsv_openblas.x \
      bench_amaxv_openblas.x \
      bench_copyv_openblas.x \
      bench_swapv_openblas.x \
      bench_axpbyv_openblas.x

atlas: \
      bench_gemm_atlas.x \
      bench_gemmt_atlas.x \
      bench_trsm_atlas.x\
      bench_gemv_atlas.x \
      bench_syrk_atlas.x \
      bench_ger_atlas.x \
      bench_scalv_atlas.x \
      bench_dotv_atlas.x \
      bench_trsv_atlas.x \
      bench_amaxv_atlas.x \
      bench_copyv_atlas.x \
      bench_swapv_atlas.x \
      bench_axpbyv_atlax.x

mkl:  \
      bench_gemm_mkl.x \
      bench_gemmt_mkl.x \
      bench_trsm_mkl.x \
      bench_gemv_mkl.x \
      bench_syrk_mkl.x \
      bench_ger_mkl.x \
      bench_nrm2_mkl.x \
      bench_scalv_mkl.x \
      bench_dotv_mkl.x \
      bench_trsv_mkl.x \
      bench_amaxv_mkl.x \
      bench_copyv_mkl.x \
      bench_swapv_mkl.x \
      bench_axpbyv_mkl.x


# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

bench_%_openblas.o: bench_%.c
	$(CC) $(CFLAGS) -DBLAS=\"openblas\" $(NRTS) -DINT_FS=\"%ld\" -DUINT_FS=\"%lu\" -c $< -o $@

bench_%_atlas.o: bench_%.c
	$(CC) $(CFLAGS) -DBLAS=\"atlas\" $(NRTS) -DINT_FS=\"%ld\" -DUINT_FS=\"%lu\" -c $< -o $@

bench_%_mkl.o: bench_%.c
	$(CC) $(CFLAGS) -DBLAS=\"mkl\" $(NRTS) -DINT_FS=\"%ld\" -DUINT_FS=\"%lu\" -c $< -o $@


bench_%_blis.o: bench_%.c
	$(CC) $(CFLAGS) -DBLAS=\"aocl\" $(NRTS) -DINT_FS=\"%ld\" -DUINT_FS=\"%lu\" -c $< -o $@


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

bench_%_openblas.x: bench_%_openblas.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(OPENBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@

bench_%_atlas.x: bench_%_atlas.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(ATLAS_LIB)    $(LIBBLIS_LINK) $(LDFLAGS) -o $@

bench_%_mkl.x: bench_%_mkl.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(MKL_LIB)      $(LIBBLIS_LINK) $(LDFLAGS) -o $@


bench_%_blis.x: bench_%_blis.o $(LIBBLIS_LINK)
	$(LINKER) $<                             $(LIBBLIS_LINK) $(LDFLAGS) -o $@


# -- Environment check rules --

check-env: check-lib

check-env-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
	$(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-lib: check-env-mk
ifeq ($(wildcard $(LIBBLIS_LINK)),)
	$(error Cannot proceed: BLIS library not yet built! Run make first)
endif


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x
