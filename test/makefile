#
# test directory makefile
#

BLIS_LIB       := libblis.a

LIB_PATH       := $(HOME)/flame/lib

#BLAS_LIB        := $(LIB_PATH)/libblas.a
#BLAS_LIB       := $(LIB_PATH)/libgoto.a
#BLAS_LIB       := $(LIB_PATH)/libgoto2.a
#BLAS_LIB       := $(LIB_PATH)/libopenblas.a
#BLAS_LIB        := $(LIB_PATH)/libf77blas.a \
#                   $(LIB_PATH)/libatlas.a
#BLAS_LIB        := -L/opt/intel/mkl/10.2.2.025/lib/em64t/ \
#                   -lmkl_sequential -lmkl_core -lmkl_intel_lp64
OPENBLAS_LIB   := $(LIB_PATH)/libopenblas.a
ATLAS_LIB      := $(LIB_PATH)/libf77blas.a \
                  $(LIB_PATH)/libatlas.a
MKL_LIB        := -L/opt/intel/mkl/10.2.2.025/lib/em64t/ \
                  -lmkl_sequential -lmkl_core -lmkl_intel_lp64

CC             := gcc
CFLAGS         := -O2 -march=nocona \
                  -msse3 -mfpmath=sse \
                  -Wall -Wno-comment -std=c99 -g #-fomit-frame-pointer #-pg #-g

LINKER         := $(CC)
LDFLAGS        := -L/usr/lib/gcc/x86_64-redhat-linux/4.1.2 -L/usr/lib/gcc/x86_64-redhat-linux/4.1.2/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -lgfortranbegin -lgfortran -lm
LDFLAGS        += -lpthread


AR             := ar
ARFLAGS        := cru
RANLIB         := ranlib


SRC_PATH         := ../src
OBJ_PATH         := .

INCLUDE_PATH     := $(SRC_PATH)/include
INCLUDE_PATH2    := $(SRC_PATH)/include/level0
BASE_PATH        := $(SRC_PATH)/base
BASECHK_PATH     := $(SRC_PATH)/base/check
CNTL_PATH        := $(SRC_PATH)/cntl
AXPYV_PATH       := $(SRC_PATH)/1/axpyv
COPYV_PATH       := $(SRC_PATH)/1/copyv
COPYNZV_PATH     := $(SRC_PATH)/1/copynzv
DOTV_PATH        := $(SRC_PATH)/1/dotv
DOTXV_PATH       := $(SRC_PATH)/1/dotxv
INVERTV_PATH     := $(SRC_PATH)/1/invertv
SCALV_PATH       := $(SRC_PATH)/1/scalv
SCAL2V_PATH      := $(SRC_PATH)/1/scal2v
SETV_PATH        := $(SRC_PATH)/1/setv
PACKV_PATH       := $(SRC_PATH)/1/packv
UNPACKV_PATH     := $(SRC_PATH)/1/unpackv
AXPYD_PATH       := $(SRC_PATH)/1d/axpyd
COPYD_PATH       := $(SRC_PATH)/1d/copyd
INVERTD_PATH     := $(SRC_PATH)/1d/invertd
SCALD_PATH       := $(SRC_PATH)/1d/scald
SCAL2D_PATH      := $(SRC_PATH)/1d/scal2d
SETD_PATH        := $(SRC_PATH)/1d/setd
AXPY2V_PATH      := $(SRC_PATH)/1f/axpy2v
AXPYF_PATH       := $(SRC_PATH)/1f/axpyf
DOTXF_PATH       := $(SRC_PATH)/1f/dotxf
DOTAXPYV_PATH    := $(SRC_PATH)/1f/dotaxpyv
DOTXAXPYF_PATH   := $(SRC_PATH)/1f/dotxaxpyf
AXPYM_PATH       := $(SRC_PATH)/1m/axpym
COPYM_PATH       := $(SRC_PATH)/1m/copym
COPYNZM_PATH     := $(SRC_PATH)/1m/copynzm
SCALM_PATH       := $(SRC_PATH)/1m/scalm
SCAL2M_PATH      := $(SRC_PATH)/1m/scal2m
SETM_PATH        := $(SRC_PATH)/1m/setm
PACKM_PATH       := $(SRC_PATH)/1m/packm
PACKM_UKR_PATH   := $(SRC_PATH)/1m/packm/ukernels
UNPACKM_PATH     := $(SRC_PATH)/1m/unpackm
UNPACKM_UKR_PATH := $(SRC_PATH)/1m/unpackm/ukernels
GEMV_PATH        := $(SRC_PATH)/2/gemv
GER_PATH         := $(SRC_PATH)/2/ger
HEMV_PATH        := $(SRC_PATH)/2/hemv
HER_PATH         := $(SRC_PATH)/2/her
HER2_PATH        := $(SRC_PATH)/2/her2
SYMV_PATH        := $(SRC_PATH)/2/symv
SYR_PATH         := $(SRC_PATH)/2/syr
SYR2_PATH        := $(SRC_PATH)/2/syr2
TRMV_PATH        := $(SRC_PATH)/2/trmv
TRSV_PATH        := $(SRC_PATH)/2/trsv
GEMM_PATH        := $(SRC_PATH)/3/gemm
GEMM_UKR_PATH    := $(SRC_PATH)/3/gemm/ukernels
HEMM_PATH        := $(SRC_PATH)/3/hemm
HERK_PATH        := $(SRC_PATH)/3/herk
HER2K_PATH       := $(SRC_PATH)/3/her2k
SYMM_PATH        := $(SRC_PATH)/3/symm
SYRK_PATH        := $(SRC_PATH)/3/syrk
SYR2K_PATH       := $(SRC_PATH)/3/syr2k
TRMM_PATH        := $(SRC_PATH)/3/trmm
TRMM3_PATH       := $(SRC_PATH)/3/trmm3
TRSM_PATH        := $(SRC_PATH)/3/trsm
TRSM_UKR_PATH    := $(SRC_PATH)/3/trsm/ukernels
COPYSC_PATH      := $(SRC_PATH)/util/copysc
DUPL_PATH        := $(SRC_PATH)/util/dupl
PRINTV_PATH      := $(SRC_PATH)/util/printv
PRINTM_PATH      := $(SRC_PATH)/util/printm
RANDV_PATH       := $(SRC_PATH)/util/randv
RANDM_PATH       := $(SRC_PATH)/util/randm
SETS_PATH        := $(SRC_PATH)/util/sets
PROFILE_PATH     := $(SRC_PATH)/profile

INCLUDE_DIRS  := -I$(INCLUDE_PATH) \
                 -I$(INCLUDE_PATH2) \
                 -I$(BASE_PATH) \
                 -I$(BASECHK_PATH) \
                 -I$(AXPYV_PATH) \
                 -I$(COPYV_PATH) \
                 -I$(COPYNZV_PATH) \
                 -I$(DOTV_PATH) \
                 -I$(DOTXV_PATH) \
                 -I$(INVERTV_PATH) \
                 -I$(SCALV_PATH) \
                 -I$(SCAL2V_PATH) \
                 -I$(SETV_PATH) \
                 -I$(PACKV_PATH) \
                 -I$(UNPACKV_PATH) \
                 -I$(AXPYD_PATH) \
                 -I$(COPYD_PATH) \
                 -I$(INVERTD_PATH) \
                 -I$(SCALD_PATH) \
                 -I$(SCAL2D_PATH) \
                 -I$(SETD_PATH) \
                 -I$(AXPYF_PATH) \
                 -I$(AXPY2V_PATH) \
                 -I$(DOTXF_PATH) \
                 -I$(DOTAXPYV_PATH) \
                 -I$(DOTXAXPYF_PATH) \
                 -I$(AXPYM_PATH) \
                 -I$(COPYM_PATH) \
                 -I$(COPYNZM_PATH) \
                 -I$(SCALM_PATH) \
                 -I$(SCAL2M_PATH) \
                 -I$(SETM_PATH) \
                 -I$(PACKM_PATH) \
                 -I$(PACKM_UKR_PATH) \
                 -I$(UNPACKM_PATH) \
                 -I$(UNPACKM_UKR_PATH) \
                 -I$(GEMV_PATH) \
                 -I$(GER_PATH) \
                 -I$(HEMV_PATH) \
                 -I$(HER_PATH) \
                 -I$(HER2_PATH) \
                 -I$(SYMV_PATH) \
                 -I$(SYR_PATH) \
                 -I$(SYR2_PATH) \
                 -I$(TRMV_PATH) \
                 -I$(TRSV_PATH) \
                 -I$(GEMM_PATH) \
                 -I$(GEMM_UKR_PATH) \
                 -I$(HEMM_PATH) \
                 -I$(HERK_PATH) \
                 -I$(HER2K_PATH) \
                 -I$(SYMM_PATH) \
                 -I$(SYRK_PATH) \
                 -I$(SYR2K_PATH) \
                 -I$(TRMM_PATH) \
                 -I$(TRMM3_PATH) \
                 -I$(TRSM_PATH) \
                 -I$(TRSM_UKR_PATH) \
                 -I$(COPYSC_PATH) \
                 -I$(DUPL_PATH) \
                 -I$(PRINTV_PATH) \
                 -I$(PRINTM_PATH) \
                 -I$(RANDV_PATH) \
                 -I$(RANDM_PATH) \
                 -I$(SETS_PATH) \
                 -I$(CNTL_PATH) \
                 -I$(PROFILE_PATH)

CFLAGS       += $(INCLUDE_DIRS)

LIB_OBJS     := $(patsubst $(BASE_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(BASE_PATH)/*.c))
LIB_OBJS     += $(patsubst $(BASECHK_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(BASECHK_PATH)/*.c))
LIB_OBJS     += $(patsubst $(CNTL_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(CNTL_PATH)/*.c))
LIB_OBJS     += $(patsubst $(AXPYV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(AXPYV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(COPYV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(COPYV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(COPYNZV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(COPYNZV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(DOTV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(DOTV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(DOTXV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(DOTXV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(INVERTV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(INVERTV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SCALV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SCALV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SCAL2V_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SCAL2V_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SETV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SETV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PACKV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PACKV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(UNPACKV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(UNPACKV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(AXPYD_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(AXPYD_PATH)/*.c))
LIB_OBJS     += $(patsubst $(COPYD_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(COPYD_PATH)/*.c))
LIB_OBJS     += $(patsubst $(INVERTD_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(INVERTD_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SCALD_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SCALD_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SCAL2D_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SCAL2D_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SETD_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SETD_PATH)/*.c))
LIB_OBJS     += $(patsubst $(AXPYF_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(AXPYF_PATH)/*.c))
LIB_OBJS     += $(patsubst $(AXPY2V_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(AXPY2V_PATH)/*.c))
LIB_OBJS     += $(patsubst $(DOTXF_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(DOTXF_PATH)/*.c))
LIB_OBJS     += $(patsubst $(DOTAXPYV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(DOTAXPYV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(DOTXAXPYF_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(DOTXAXPYF_PATH)/*.c))
LIB_OBJS     += $(patsubst $(AXPYM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(AXPYM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(COPYM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(COPYM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(COPYNZM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(COPYNZM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SCALM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SCALM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SCAL2M_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SCAL2M_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SETM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SETM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PACKM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PACKM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PACKM_UKR_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PACKM_UKR_PATH)/*.c))
LIB_OBJS     += $(patsubst $(UNPACKM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(UNPACKM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(UNPACKM_UKR_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(UNPACKM_UKR_PATH)/*.c))
LIB_OBJS     += $(patsubst $(GEMV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(GEMV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(GER_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(GER_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HEMV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HEMV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HER_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HER_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HER2_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HER2_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SYMV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SYMV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SYR_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SYR_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SYR2_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SYR2_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRMV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRMV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRSV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRSV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(GEMM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(GEMM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(GEMM_UKR_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(GEMM_UKR_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HEMM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HEMM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HERK_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HERK_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HER2K_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HER2K_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SYMM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SYMM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SYRK_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SYRK_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SYR2K_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SYR2K_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRMM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRMM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRMM3_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRMM3_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRSM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRSM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRSM_UKR_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRSM_UKR_PATH)/*.c))
LIB_OBJS     += $(patsubst $(COPYSC_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(COPYSC_PATH)/*.c))
LIB_OBJS     += $(patsubst $(DUPL_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(DUPL_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PRINTV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PRINTV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PRINTM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PRINTM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(RANDV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(RANDV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(RANDM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(RANDM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SETS_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SETS_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PROFILE_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PROFILE_PATH)/*.c))

$(OBJ_PATH)/%.o: $(BASE_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(BASECHK_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(CNTL_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(AXPYV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(COPYV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(COPYNZV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(DOTV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(DOTXV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(INVERTV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SCALV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SCAL2V_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SETV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(PACKV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(UNPACKV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(AXPYD_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(COPYD_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(INVERTD_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SCALD_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SCAL2D_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SETD_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(AXPYF_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(AXPY2V_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(DOTXF_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(DOTAXPYV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(DOTXAXPYF_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(AXPYM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(COPYM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(COPYNZM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SCALM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SCAL2M_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SETM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(PACKM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(PACKM_UKR_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(UNPACKM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(UNPACKM_UKR_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(GEMV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(GER_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(HEMV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(HER_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(HER2_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SYMV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SYR_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SYR2_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(TRMV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(TRSV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(GEMM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(GEMM_UKR_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(HEMM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(HERK_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(HER2K_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SYMM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SYRK_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SYR2K_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(TRMM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(TRMM3_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(TRSM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(TRSM_UKR_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(COPYSC_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(DUPL_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(PRINTV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(PRINTM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(RANDV_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(RANDM_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(SETS_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ_PATH)/%.o: $(PROFILE_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@


test_%_openblas.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"openblas\" -c $< -o $@

test_%_atlas.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"atlas\" -c $< -o $@

test_%_mkl.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"mkl\" -c $< -o $@

test_%_blis.o: test_%.c
	$(CC) $(CFLAGS) -DBLIS -c $< -o $@


all: $(BLIS_LIB) \
     blis \
     openblas \
     atlas \
     mkl

blis: $(BLIS_LIB) \
      test_gemv_blis.x \
      test_ger_blis.x \
      test_hemv_blis.x \
      test_her_blis.x \
      test_her2_blis.x \
      test_trmv_blis.x \
      test_trsv_blis.x \
      \
      test_gemm_blis.x \
      test_hemm_blis.x \
      test_herk_blis.x \
      test_her2k_blis.x \
      test_trmm_blis.x \
      test_trsm_blis.x

openblas: $(BLIS_LIB) \
      test_gemv_openblas.x \
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

atlas: $(BLIS_LIB) \
      test_gemv_atlas.x \
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

mkl: $(BLIS_LIB) \
      test_gemv_mkl.x \
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



$(BLIS_LIB): $(LIB_OBJS)
	$(AR) $(ARFLAGS) $@ $?
	@echo "$@"
	$(RANLIB) $@


test_%_openblas.x: test_%_openblas.o $(BLIS_LIB)
	$(LINKER) $<             $(BLIS_LIB) $(OPENBLAS_LIB) $(LDFLAGS) -o $@

test_%_atlas.x: test_%_atlas.o $(BLIS_LIB)
	$(LINKER) $<             $(BLIS_LIB) $(ATLAS_LIB) $(LDFLAGS) -o $@

test_%_mkl.x: test_%_mkl.o $(BLIS_LIB)
	$(LINKER) $<             $(BLIS_LIB) $(MKL_LIB) $(LDFLAGS) -o $@

#test_%_blas.x: test_%_blas.o $(BLIS_LIB)
#	$(LINKER) $<             $(BLIS_LIB) $(BLAS_LIB) $(LDFLAGS) -o $@

test_%_blis.x: test_%_blis.o $(BLIS_LIB)
	$(LINKER) $<             $(BLIS_LIB)             $(LDFLAGS) -o $@




clean:
	rm -f $(OBJ_PATH)/*.o *.x $(BLIS_LIB)

cleanx:
	rm -f *.x
