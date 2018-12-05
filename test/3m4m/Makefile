#!/bin/bash
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2018, Advanced Micro Devices, Inc.
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
        blis-gemm-st openblas-gemm-st mkl-gemm-st acml-gemm-st \
        blis-gemm-mt openblas-gemm-mt mkl-gemm-mt acml-gemm-mt \
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
DIST_PATH  := ../..
LIB_PATH    = ../../lib/$(CONFIG_NAME)
INC_PATH    = ../../include/$(CONFIG_NAME)
SHARE_PATH := ../..
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
#BLIS_LIB       := $(BLIS_LIB_PATH)/libblis.a

# BLAS library path(s). This is where the BLAS libraries reside.
HOME_LIB_PATH  := $(HOME)/flame/lib
#MKL_LIB_PATH   := /opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64
MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64
#MKL_LIB_PATH   := ${MKLROOT}/lib/intel64
#ICC_LIB_PATH   := /opt/apps/intel/13/composer_xe_2013.2.146/compiler/lib/intel64
ACML_LIB_PATH  := $(HOME_LIB_PATH)/acml/5.3.1/gfortran64_fma4_int64/lib
ACMLP_LIB_PATH := $(HOME_LIB_PATH)/acml/5.3.1/gfortran64_fma4_mp_int64/lib

# OpenBLAS
OPENBLAS_LIB   := $(HOME_LIB_PATH)/libopenblas.a
OPENBLASP_LIB  := $(HOME_LIB_PATH)/libopenblasp.a

# ATLAS
ATLAS_LIB      := $(HOME_LIB_PATH)/libf77blas.a \
                  $(HOME_LIB_PATH)/libatlas.a

# MKL
MKL_LIB        := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_sequential \
                  -lpthread -lm -ldl
#MKLP_LIB       := -L$(MKL_LIB_PATH) \
#                  -lmkl_intel_thread \
#                  -lmkl_core \
#                  -lmkl_intel_ilp64 \
#                  -L$(ICC_LIB_PATH) \
#                  -liomp5
MKLP_LIB       := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_gnu_thread \
                  -lpthread -lm -ldl -fopenmp
                  #-L$(ICC_LIB_PATH) \
                  #-lgomp

# ACML
ACML_LIB       := -L$(ACML_LIB_PATH) \
                  -lgfortran -lm -lrt -ldl -lacml
ACMLP_LIB      := -L$(ACMLP_LIB_PATH) \
                  -lgfortran -lm -lrt -ldl -lacml_mp



#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(sort $(patsubst $(TEST_SRC_PATH)/%.c, \
                                    $(TEST_OBJ_PATH)/%.o, \
                                    $(wildcard $(TEST_SRC_PATH)/*.c)))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the "framework" CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS.
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)


# Datatype
DT_S     := -DDT=BLIS_FLOAT
DT_D     := -DDT=BLIS_DOUBLE
DT_C     := -DDT=BLIS_SCOMPLEX
DT_Z     := -DDT=BLIS_DCOMPLEX

# Which library?
BLI_DEF  := -DBLIS
BLA_DEF  := -DBLAS

# Complex implementation type
D3MHW    := -DIND=BLIS_3MH
D3M1     := -DIND=BLIS_3M1
D4MHW    := -DIND=BLIS_4MH
D4M1B    := -DIND=BLIS_4M1B
D4M1A    := -DIND=BLIS_4M1A
D1M      := -DIND=BLIS_1M
DNAT     := -DIND=BLIS_NAT

# Implementation string
STR_3MHW := -DSTR=\"3mhw\"
STR_3M1  := -DSTR=\"3m1\"
STR_4MHW := -DSTR=\"4mhw\"
STR_4M1B := -DSTR=\"4m1b\"
STR_4M1A := -DSTR=\"4m1a\"
STR_1M   := -DSTR=\"1m\"
STR_NAT  := -DSTR=\"asm\"
STR_OBL  := -DSTR=\"openblas\"
STR_MKL  := -DSTR=\"mkl\"
STR_ACML := -DSTR=\"acml\"

# Single or multithreaded string
STR_ST   := -DTHR_STR=\"st\"
STR_MT   := -DTHR_STR=\"mt\"

# Problem size specification
PDEF_ST  := -DP_BEGIN=56 \
            -DP_END=2800 \
            -DP_INC=56

PDEF_MT  := -DP_BEGIN=160 \
            -DP_END=8000 \
            -DP_INC=160



#
# --- Targets/rules ------------------------------------------------------------
#

all:        all-st all-mt
blis:       blis-st blis-mt
openblas:   openblas-st openblas-mt
mkl:        mkl-st mkl-mt

all-st:     blis-st openblas-st mkl-st
all-mt:     blis-mt openblas-mt mkl-mt

blis-st:    blis-nat-st
blis-mt:    blis-nat-mt

blis-ind:   blis-ind-st blis-ind-mt
blis-nat:   blis-nat-st blis-nat-mt

blis-ind-st: \
      test_cgemm_3mhw_blis_st.x \
      test_zgemm_3mhw_blis_st.x \
      test_cgemm_3m1_blis_st.x \
      test_zgemm_3m1_blis_st.x \
      test_cgemm_4mhw_blis_st.x \
      test_zgemm_4mhw_blis_st.x \
      test_cgemm_4m1b_blis_st.x \
      test_zgemm_4m1b_blis_st.x \
      test_cgemm_4m1a_blis_st.x \
      test_zgemm_4m1a_blis_st.x \
      test_cgemm_1m_blis_st.x \
      test_zgemm_1m_blis_st.x

blis-ind-mt: \
      test_cgemm_3mhw_blis_mt.x \
      test_zgemm_3mhw_blis_mt.x \
      test_cgemm_3m1_blis_mt.x \
      test_zgemm_3m1_blis_mt.x \
      test_cgemm_4mhw_blis_mt.x \
      test_zgemm_4mhw_blis_mt.x \
      test_cgemm_4m1b_blis_mt.x \
      test_zgemm_4m1b_blis_mt.x \
      test_cgemm_4m1a_blis_mt.x \
      test_zgemm_4m1a_blis_mt.x \
      test_cgemm_1m_blis_mt.x \
      test_zgemm_1m_blis_mt.x

blis-nat-st: \
      test_sgemm_asm_blis_st.x \
      test_dgemm_asm_blis_st.x \
      test_cgemm_asm_blis_st.x \
      test_zgemm_asm_blis_st.x \
      test_shemm_asm_blis_st.x \
      test_dhemm_asm_blis_st.x \
      test_chemm_asm_blis_st.x \
      test_zhemm_asm_blis_st.x \
      test_sherk_asm_blis_st.x \
      test_dherk_asm_blis_st.x \
      test_cherk_asm_blis_st.x \
      test_zherk_asm_blis_st.x \
      test_strmm_asm_blis_st.x \
      test_dtrmm_asm_blis_st.x \
      test_ctrmm_asm_blis_st.x \
      test_ztrmm_asm_blis_st.x \
      test_strsm_asm_blis_st.x \
      test_dtrsm_asm_blis_st.x \
      test_ctrsm_asm_blis_st.x \
      test_ztrsm_asm_blis_st.x

blis-nat-mt: \
      test_sgemm_asm_blis_mt.x \
      test_dgemm_asm_blis_mt.x \
      test_cgemm_asm_blis_mt.x \
      test_zgemm_asm_blis_mt.x \
      test_shemm_asm_blis_mt.x \
      test_dhemm_asm_blis_mt.x \
      test_chemm_asm_blis_mt.x \
      test_zhemm_asm_blis_mt.x \
      test_sherk_asm_blis_mt.x \
      test_dherk_asm_blis_mt.x \
      test_cherk_asm_blis_mt.x \
      test_zherk_asm_blis_mt.x \
      test_strmm_asm_blis_mt.x \
      test_dtrmm_asm_blis_mt.x \
      test_ctrmm_asm_blis_mt.x \
      test_ztrmm_asm_blis_mt.x \
      test_strsm_asm_blis_mt.x \
      test_dtrsm_asm_blis_mt.x \
      test_ctrsm_asm_blis_mt.x \
      test_ztrsm_asm_blis_mt.x

openblas-st: \
      test_sgemm_openblas_st.x \
      test_dgemm_openblas_st.x \
      test_cgemm_openblas_st.x \
      test_zgemm_openblas_st.x \
      test_shemm_openblas_st.x \
      test_dhemm_openblas_st.x \
      test_chemm_openblas_st.x \
      test_zhemm_openblas_st.x \
      test_sherk_openblas_st.x \
      test_dherk_openblas_st.x \
      test_cherk_openblas_st.x \
      test_zherk_openblas_st.x \
      test_strmm_openblas_st.x \
      test_dtrmm_openblas_st.x \
      test_ctrmm_openblas_st.x \
      test_ztrmm_openblas_st.x \
      test_strsm_openblas_st.x \
      test_dtrsm_openblas_st.x \
      test_ctrsm_openblas_st.x \
      test_ztrsm_openblas_st.x

openblas-mt: \
      test_sgemm_openblas_mt.x \
      test_dgemm_openblas_mt.x \
      test_cgemm_openblas_mt.x \
      test_zgemm_openblas_mt.x \
      test_shemm_openblas_mt.x \
      test_dhemm_openblas_mt.x \
      test_chemm_openblas_mt.x \
      test_zhemm_openblas_mt.x \
      test_sherk_openblas_mt.x \
      test_dherk_openblas_mt.x \
      test_cherk_openblas_mt.x \
      test_zherk_openblas_mt.x \
      test_strmm_openblas_mt.x \
      test_dtrmm_openblas_mt.x \
      test_ctrmm_openblas_mt.x \
      test_ztrmm_openblas_mt.x \
      test_strsm_openblas_mt.x \
      test_dtrsm_openblas_mt.x \
      test_ctrsm_openblas_mt.x \
      test_ztrsm_openblas_mt.x

mkl-st: \
      test_sgemm_mkl_st.x \
      test_dgemm_mkl_st.x \
      test_cgemm_mkl_st.x \
      test_zgemm_mkl_st.x \
      test_shemm_mkl_st.x \
      test_dhemm_mkl_st.x \
      test_chemm_mkl_st.x \
      test_zhemm_mkl_st.x \
      test_sherk_mkl_st.x \
      test_dherk_mkl_st.x \
      test_cherk_mkl_st.x \
      test_zherk_mkl_st.x \
      test_strmm_mkl_st.x \
      test_dtrmm_mkl_st.x \
      test_ctrmm_mkl_st.x \
      test_ztrmm_mkl_st.x \
      test_strsm_mkl_st.x \
      test_dtrsm_mkl_st.x \
      test_ctrsm_mkl_st.x \
      test_ztrsm_mkl_st.x

mkl-mt: \
      test_sgemm_mkl_mt.x \
      test_dgemm_mkl_mt.x \
      test_cgemm_mkl_mt.x \
      test_zgemm_mkl_mt.x \
      test_shemm_mkl_mt.x \
      test_dhemm_mkl_mt.x \
      test_chemm_mkl_mt.x \
      test_zhemm_mkl_mt.x \
      test_sherk_mkl_mt.x \
      test_dherk_mkl_mt.x \
      test_cherk_mkl_mt.x \
      test_zherk_mkl_mt.x \
      test_strmm_mkl_mt.x \
      test_dtrmm_mkl_mt.x \
      test_ctrmm_mkl_mt.x \
      test_ztrmm_mkl_mt.x \
      test_strsm_mkl_mt.x \
      test_dtrsm_mkl_mt.x \
      test_ctrsm_mkl_mt.x \
      test_ztrsm_mkl_mt.x




# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# blis 3mhw
test_z%_3mhw_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLI_DEF) $(D3MHW) $(STR_3MHW) $(STR_ST) -c $< -o $@

test_c%_3mhw_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLI_DEF) $(D3MHW) $(STR_3MHW) $(STR_ST) -c $< -o $@

test_z%_3mhw_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLI_DEF) $(D3MHW) $(STR_3MHW) $(STR_MT) -c $< -o $@

test_c%_3mhw_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLI_DEF) $(D3MHW) $(STR_3MHW) $(STR_MT) -c $< -o $@

# blis 3m1
test_z%_3m1_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLI_DEF) $(D3M1)  $(STR_3M1)  $(STR_ST) -c $< -o $@

test_c%_3m1_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLI_DEF) $(D3M1)  $(STR_3M1)  $(STR_ST) -c $< -o $@

test_z%_3m1_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLI_DEF) $(D3M1)  $(STR_3M1)  $(STR_MT) -c $< -o $@

test_c%_3m1_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLI_DEF) $(D3M1)  $(STR_3M1)  $(STR_MT) -c $< -o $@

# blis 4mhw
test_z%_4mhw_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLI_DEF) $(D4MHW) $(STR_4MHW) $(STR_ST) -c $< -o $@

test_c%_4mhw_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLI_DEF) $(D4MHW) $(STR_4MHW) $(STR_ST) -c $< -o $@

test_z%_4mhw_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLI_DEF) $(D4MHW) $(STR_4MHW) $(STR_MT) -c $< -o $@

test_c%_4mhw_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLI_DEF) $(D4MHW) $(STR_4MHW) $(STR_MT) -c $< -o $@

# blis 4m1b
test_z%_4m1b_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLI_DEF) $(D4M1B) $(STR_4M1B) $(STR_ST) -c $< -o $@

test_c%_4m1b_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLI_DEF) $(D4M1B) $(STR_4M1B) $(STR_ST) -c $< -o $@

test_z%_4m1b_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLI_DEF) $(D4M1B) $(STR_4M1B) $(STR_MT) -c $< -o $@

test_c%_4m1b_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLI_DEF) $(D4M1B) $(STR_4M1B) $(STR_MT) -c $< -o $@

# blis 4m1a
test_z%_4m1a_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLI_DEF) $(D4M1A) $(STR_4M1A) $(STR_ST) -c $< -o $@

test_c%_4m1a_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLI_DEF) $(D4M1A) $(STR_4M1A) $(STR_ST) -c $< -o $@

test_z%_4m1a_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLI_DEF) $(D4M1A) $(STR_4M1A) $(STR_MT) -c $< -o $@

test_c%_4m1a_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLI_DEF) $(D4M1A) $(STR_4M1A) $(STR_MT) -c $< -o $@

# blis 1m
test_z%_1m_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLI_DEF) $(D1M) $(STR_1M) $(STR_ST) -c $< -o $@

test_c%_1m_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLI_DEF) $(D1M) $(STR_1M) $(STR_ST) -c $< -o $@

test_z%_1m_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLI_DEF) $(D1M) $(STR_1M) $(STR_MT) -c $< -o $@

test_c%_1m_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLI_DEF) $(D1M) $(STR_1M) $(STR_MT) -c $< -o $@

# blis asm
test_d%_asm_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_D) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_ST) -c $< -o $@

test_s%_asm_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_S) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_ST) -c $< -o $@

test_z%_asm_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_ST) -c $< -o $@

test_c%_asm_blis_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_ST) -c $< -o $@

test_d%_asm_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_D) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_MT) -c $< -o $@

test_s%_asm_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_S) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_MT) -c $< -o $@

test_z%_asm_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_MT) -c $< -o $@

test_c%_asm_blis_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLI_DEF) $(DNAT)  $(STR_NAT)  $(STR_MT) -c $< -o $@

# openblas
test_d%_openblas_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_D) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_ST) -c $< -o $@

test_s%_openblas_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_S) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_ST) -c $< -o $@

test_z%_openblas_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_ST) -c $< -o $@

test_c%_openblas_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_ST) -c $< -o $@

test_d%_openblas_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_D) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_MT) -c $< -o $@

test_s%_openblas_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_S) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_MT) -c $< -o $@

test_z%_openblas_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_MT) -c $< -o $@

test_c%_openblas_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLA_DEF) $(DNAT)  $(STR_OBL)  $(STR_MT) -c $< -o $@

# mkl
test_d%_mkl_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_D) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_ST) -c $< -o $@

test_s%_mkl_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_S) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_ST) -c $< -o $@

test_z%_mkl_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_Z) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_ST) -c $< -o $@

test_c%_mkl_st.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_ST) $(DT_C) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_ST) -c $< -o $@

test_d%_mkl_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_D) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_MT) -c $< -o $@

test_s%_mkl_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_S) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_MT) -c $< -o $@

test_z%_mkl_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_Z) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_MT) -c $< -o $@

test_c%_mkl_mt.o: test_%.c Makefile
	$(CC) $(CFLAGS) $(PDEF_MT) $(DT_C) $(BLA_DEF) $(DNAT)  $(STR_MKL)  $(STR_MT) -c $< -o $@


# -- Executable file rules --

# NOTE: For the BLAS test drivers, we place the BLAS libraries before BLIS
# on the link command line in case BLIS was configured with the BLAS
# compatibility layer. This prevents BLIS from inadvertently getting called
# for the BLAS routines we are trying to test with.

test_%_openblas_st.x: test_%_openblas_st.o $(LIBBLIS_LINK)
	$(LINKER) $<  $(OPENBLAS_LIB)  $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_openblas_mt.x: test_%_openblas_mt.o $(LIBBLIS_LINK)
	$(LINKER) $<  $(OPENBLASP_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_mkl_st.x: test_%_mkl_st.o $(LIBBLIS_LINK)
	$(LINKER) $<  $(MKL_LIB)       $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_mkl_mt.x: test_%_mkl_mt.o $(LIBBLIS_LINK)
	$(LINKER) $<  $(MKLP_LIB)      $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_blis_st.x: test_%_blis_st.o $(LIBBLIS_LINK)
	$(LINKER) $<                   $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_blis_mt.x: test_%_blis_mt.o $(LIBBLIS_LINK)
	$(LINKER) $<                   $(LIBBLIS_LINK) $(LDFLAGS) -o $@


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x

