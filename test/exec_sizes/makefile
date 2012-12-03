#
# test directory makefile
#

MAKE_BLIS      := yes
MAKE_OPEN      := yes
MAKE_ATLAS     := yes
MAKE_MKL       := yes

BLIS_LIB       := ../libblis.a

LIB_PATH       := $(HOME)/flame/lib
MKL_PATH       := /opt/intel/mkl/10.2.2.025

#BLAS_LIB       := $(LIB_PATH)/libblas.a
#BLAS_LIB       := $(LIB_PATH)/libgoto.a
#BLAS_LIB       := $(LIB_PATH)/libgoto2.a
OBLAS_LIB       := $(LIB_PATH)/libopenblas.a
ABLAS_LIB       := $(LIB_PATH)/libf77blas.a \
                   $(LIB_PATH)/libatlas.a
#BLAS_LIB        := -L/opt/intel/mkl/10.2.2.025/lib/em64t/ \
#                   -lmkl_sequential -lmkl_core -lmkl_intel_lp64
MBLAS_LIB       := $(MKL_PATH)/lib/em64t/libmkl_solver_lp64_sequential.a \
                   -Wl,--start-group \
                   $(MKL_PATH)/lib/em64t/libmkl_intel_lp64.a \
                   $(MKL_PATH)/lib/em64t/libmkl_sequential.a \
                   $(MKL_PATH)/lib/em64t/libmkl_core.a \
                   -Wl,--end-group \
                   -lpthread -lm

CC             := gcc
CFLAGS         := -O2 -march=nocona \
                  -msse3 -mfpmath=sse \
                  -Wall -Wno-comment -std=c99 -fomit-frame-pointer #-pg #-g

LINKER         := $(CC)
LDFLAGS        := -L/usr/lib/gcc/x86_64-redhat-linux/4.1.2 -L/usr/lib/gcc/x86_64-redhat-linux/4.1.2/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -lgfortranbegin -lgfortran -lm
LDFLAGS        += -lpthread

TEST_SIZES_SRC  := test_size.c

TEST_BINS := 


AR             := ar
ARFLAGS        := cru
RANLIB         := ranlib



OBJ_PATH         := ..
INCLUDE_PATH     := ../../include
INCLUDE_PATH2    := ../../include/level0
BASE_PATH        := ../../base
BASECHK_PATH     := ../../base/check
CNTL_PATH        := ../../cntl
AXPYV_PATH       := ../../1/axpyv
COPYV_PATH       := ../../1/copyv
COPYNZV_PATH     := ../../1/copynzv
DOTV_PATH        := ../../1/dotv
DOTXV_PATH       := ../../1/dotxv
INVERTV_PATH     := ../../1/invertv
SCALV_PATH       := ../../1/scalv
SCAL2V_PATH      := ../../1/scal2v
SETV_PATH        := ../../1/setv
PACKV_PATH       := ../../1/packv
UNPACKV_PATH     := ../../1/unpackv
AXPYD_PATH       := ../../1d/axpyd
COPYD_PATH       := ../../1d/copyd
INVERTD_PATH     := ../../1d/invertd
SCALD_PATH       := ../../1d/scald
SCAL2D_PATH      := ../../1d/scal2d
SETD_PATH        := ../../1d/setd
AXPY2V_PATH      := ../../1f/axpy2v
AXPYF_PATH       := ../../1f/axpyf
DOTXF_PATH       := ../../1f/dotxf
DOTAXPYV_PATH    := ../../1f/dotaxpyv
DOTXAXPYF_PATH   := ../../1f/dotxaxpyf
AXPYM_PATH       := ../../1m/axpym
COPYM_PATH       := ../../1m/copym
COPYNZM_PATH     := ../../1m/copynzm
SCALM_PATH       := ../../1m/scalm
SCAL2M_PATH      := ../../1m/scal2m
SETM_PATH        := ../../1m/setm
PACKM_PATH       := ../../1m/packm
PACKM_UKR_PATH   := ../../1m/packm/ukernels
UNPACKM_PATH     := ../../1m/unpackm
UNPACKM_UKR_PATH := ../../1m/unpackm/ukernels
GEMV_PATH        := ../../2/gemv
GER_PATH         := ../../2/ger
HEMV_PATH        := ../../2/hemv
HER_PATH         := ../../2/her
HER2_PATH        := ../../2/her2
TRMV_PATH        := ../../2/trmv
TRSV_PATH        := ../../2/trsv
GEMM_PATH        := ../../3/gemm
GEMM_UKR_PATH    := ../../3/gemm/ukernels
HEMM_PATH        := ../../3/hemm
HERK_PATH        := ../../3/herk
HER2K_PATH       := ../../3/her2k
TRMM_PATH        := ../../3/trmm
TRMM3_PATH       := ../../3/trmm3
TRSM_PATH        := ../../3/trsm
TRSM_UKR_PATH    := ../../3/trsm/ukernels
COPYSC_PATH      := ../../util/copysc
PRINTV_PATH      := ../../util/printv
PRINTM_PATH      := ../../util/printm
RANDV_PATH       := ../../util/randv
RANDM_PATH       := ../../util/randm
SETS_PATH        := ../../util/sets

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
                 -I$(TRMV_PATH) \
                 -I$(TRSV_PATH) \
                 -I$(GEMM_PATH) \
                 -I$(GEMM_UKR_PATH) \
                 -I$(HEMM_PATH) \
                 -I$(HERK_PATH) \
                 -I$(HER2K_PATH) \
                 -I$(TRMM_PATH) \
                 -I$(TRMM3_PATH) \
                 -I$(TRSM_PATH) \
                 -I$(TRSM_UKR_PATH) \
                 -I$(COPYSC_PATH) \
                 -I$(PRINTV_PATH) \
                 -I$(PRINTM_PATH) \
                 -I$(RANDV_PATH) \
                 -I$(RANDM_PATH) \
                 -I$(SETS_PATH) \
                 -I$(CNTL_PATH)

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
LIB_OBJS     += $(patsubst $(TRMV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRMV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRSV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRSV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(GEMM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(GEMM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(GEMM_UKR_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(GEMM_UKR_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HEMM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HEMM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HERK_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HERK_PATH)/*.c))
LIB_OBJS     += $(patsubst $(HER2K_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(HER2K_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRMM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRMM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRMM3_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRMM3_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRSM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRSM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(TRSM_UKR_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(TRSM_UKR_PATH)/*.c))
LIB_OBJS     += $(patsubst $(COPYSC_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(COPYSC_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PRINTV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PRINTV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(PRINTM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(PRINTM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(RANDV_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(RANDV_PATH)/*.c))
LIB_OBJS     += $(patsubst $(RANDM_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(RANDM_PATH)/*.c))
LIB_OBJS     += $(patsubst $(SETS_PATH)/%.c, $(OBJ_PATH)/%.o, $(wildcard $(SETS_PATH)/*.c))


ifeq ($(MAKE_BLIS),yes)
TEST_BINS += test_blis1 \
             test_blis2 \
             test_blis3 \
             test_blis4 \
             test_blis5 \
             test_blis6
endif
ifeq ($(MAKE_OPEN),yes)
TEST_BINS += test_oblas1 \
             test_oblas2 \
             test_oblas3 \
             test_oblas4 \
             test_oblas5 \
             test_oblas6
endif
ifeq ($(MAKE_ATLAS),yes)
TEST_BINS += test_ablas1 \
             test_ablas2 \
             test_ablas3 \
             test_ablas4 \
             test_ablas5 \
             test_ablas6
endif
ifeq ($(MAKE_MKL),yes)
TEST_BINS += test_mblas1 \
             test_mblas2 \
             test_mblas3 \
             test_mblas4 \
             test_mblas5 \
             test_mblas6
endif


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

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: $(TEST_BINS)
#	@echo "$(TEST_BINS)"

$(BLIS_LIB): $(LIB_OBJS)
	$(AR) $(ARFLAGS) $@ $?
	@echo "$@"
	$(RANLIB) $@

# BLIS rules
test_blis1: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLIS=1 $(TEST_SIZES_SRC) $(BLIS_LIB) $(BLAS_LIB) $(LDFLAGS) -o $@.x
test_blis2: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLIS=2 $(TEST_SIZES_SRC) $(BLIS_LIB) $(BLAS_LIB) $(LDFLAGS) -o $@.x
test_blis3: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLIS=3 $(TEST_SIZES_SRC) $(BLIS_LIB) $(BLAS_LIB) $(LDFLAGS) -o $@.x
test_blis4: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLIS=4 $(TEST_SIZES_SRC) $(BLIS_LIB) $(BLAS_LIB) $(LDFLAGS) -o $@.x
test_blis5: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLIS=5 $(TEST_SIZES_SRC) $(BLIS_LIB) $(BLAS_LIB) $(LDFLAGS) -o $@.x
test_blis6: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLIS=6 $(TEST_SIZES_SRC) $(BLIS_LIB) $(BLAS_LIB) $(LDFLAGS) -o $@.x

# OpenBLAS rules
test_oblas1: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=1 $(TEST_SIZES_SRC) $(BLIS_LIB) $(OBLAS_LIB) $(LDFLAGS) -o $@.x
test_oblas2: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=2 $(TEST_SIZES_SRC) $(BLIS_LIB) $(OBLAS_LIB) $(LDFLAGS) -o $@.x
test_oblas3: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=3 $(TEST_SIZES_SRC) $(BLIS_LIB) $(OBLAS_LIB) $(LDFLAGS) -o $@.x
test_oblas4: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=4 $(TEST_SIZES_SRC) $(BLIS_LIB) $(OBLAS_LIB) $(LDFLAGS) -o $@.x
test_oblas5: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=5 $(TEST_SIZES_SRC) $(BLIS_LIB) $(OBLAS_LIB) $(LDFLAGS) -o $@.x
test_oblas6: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=6 $(TEST_SIZES_SRC) $(BLIS_LIB) $(OBLAS_LIB) $(LDFLAGS) -o $@.x

# ATLAS BLAS rules
test_ablas1: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=1 $(TEST_SIZES_SRC) $(BLIS_LIB) $(ABLAS_LIB) $(LDFLAGS) -o $@.x
test_ablas2: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=2 $(TEST_SIZES_SRC) $(BLIS_LIB) $(ABLAS_LIB) $(LDFLAGS) -o $@.x
test_ablas3: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=3 $(TEST_SIZES_SRC) $(BLIS_LIB) $(ABLAS_LIB) $(LDFLAGS) -o $@.x
test_ablas4: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=4 $(TEST_SIZES_SRC) $(BLIS_LIB) $(ABLAS_LIB) $(LDFLAGS) -o $@.x
test_ablas5: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=5 $(TEST_SIZES_SRC) $(BLIS_LIB) $(ABLAS_LIB) $(LDFLAGS) -o $@.x
test_ablas6: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=6 $(TEST_SIZES_SRC) $(BLIS_LIB) $(ABLAS_LIB) $(LDFLAGS) -o $@.x

# MKL BLAS rules
test_mblas1: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=1 $(TEST_SIZES_SRC) $(BLIS_LIB) $(MBLAS_LIB) $(LDFLAGS) -o $@.x
test_mblas2: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=2 $(TEST_SIZES_SRC) $(BLIS_LIB) $(MBLAS_LIB) $(LDFLAGS) -o $@.x
test_mblas3: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=3 $(TEST_SIZES_SRC) $(BLIS_LIB) $(MBLAS_LIB) $(LDFLAGS) -o $@.x
test_mblas4: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=4 $(TEST_SIZES_SRC) $(BLIS_LIB) $(MBLAS_LIB) $(LDFLAGS) -o $@.x
test_mblas5: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=5 $(TEST_SIZES_SRC) $(BLIS_LIB) $(MBLAS_LIB) $(LDFLAGS) -o $@.x
test_mblas6: $(BLIS_LIB)
	$(CC) $(CFLAGS) -DNBLAS=6 $(TEST_SIZES_SRC) $(BLIS_LIB) $(MBLAS_LIB) $(LDFLAGS) -o $@.x


clean:
	rm -f *.x

