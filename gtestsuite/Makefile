################################################################################################
## Compiler options
################################################################################################
ifeq ($(AOCL),1)
CXX             := clang++ #-stdlib=libc++
else
CXX             := g++ -std=c++11
endif
CFLAGS 	        := -c


# Flags passed to the C++ compiler.
CXXFLAGS        += -g -Wall -Wno-unused-function -Wfatal-errors -fPIC -fnon-call-exceptions
#Enable macro __GTEST_VALGRIND_TEST__ only when valgrind tool is used
CXXFLAGS        += #-D__GTEST_VALGRIND_TEST__
#Enable below flags only when Address Sanitizer tool is used
CXXFLAGS        += #-fsanitize=address -static-libsan
LIBPTHREAD      := -fopenmp #-lpthread
LIBM            := -lm
LDFLAGS         := $(LIBM) $(LIBPTHREAD)

BLIS_PATH       := ../blis_gcc
BLIS_LIB        := ${BLIS_PATH}/lib/libblis-mt.a
GTEST_LIB       := ./lib/libgtest.a
INC_PATH        := ./inc
LINUX_PATH      := ${BLIS_PATH}/include/blis
GTEST_HEADERS   := ./inc/gtest

ifeq ($(mkl),1)
CXXFLAGS        +=-D_POSIX_C_SOURCE=200112L
CXXFLAGS        += -std=c++11 -DBLAS=\"mkl\"
LDFLAGS         += -lrt
MKL_LIB_PATH   := ${MKLROOT}/intel64
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
endif
LIB_PATH        := $(MKL_LIB) $(BLIS_LIB) $(GTEST_LIB)
################################################################################################
## Variables
################################################################################################
BIN_PATH      := bin
TEST_OBJ_PATH := obj
TEST_SRC_PATH := src
CPPS          := $(shell ls $(TEST_SRC_PATH)/*.cpp)
TEMP          := $(subst $(TEST_SRC_PATH)/, $(TEST_OBJ_PATH)/, $(CPPS))
OBJS          := $(subst .cpp, .o, $(TEMP))
HEADERS       := $(shell ls inc/*.h)
INCLUDE_PATH  := -I$(INC_PATH) -I$(LINUX_PATH) -I$(GTEST_HEADERS)


TESTSUITE_OUT_FILE := output.testsuite
ifneq ($(mkl),1)
TEST_EXE  := $(BIN_PATH)/libblis_gtest
else
TEST_EXE  := $(BIN_PATH)/gtest_mkl
endif


TEST_OBJS := $(TEST_OBJ_PATH)/gtest_suite.o \
           $(TEST_OBJ_PATH)/gtest_pthread.o  \
           $(TEST_OBJ_PATH)/blis_api.o       \
           $(TEST_OBJ_PATH)/blis_inpfile.o   \
           $(TEST_OBJ_PATH)/blis_utils_int.o \
           $(TEST_OBJ_PATH)/blis_utils.o     \
           $(TEST_OBJ_PATH)/main.o           \
           $(TEST_OBJ_PATH)/test_process.o   \
           $(TEST_OBJ_PATH)/ref_addv.o       \
           $(TEST_OBJ_PATH)/ref_amaxv.o      \
           $(TEST_OBJ_PATH)/ref_axpbyv.o     \
           $(TEST_OBJ_PATH)/ref_axpyv.o      \
           $(TEST_OBJ_PATH)/ref_copyv.o      \
           $(TEST_OBJ_PATH)/ref_dotv.o       \
           $(TEST_OBJ_PATH)/ref_dotxv.o      \
           $(TEST_OBJ_PATH)/ref_normfv.o     \
           $(TEST_OBJ_PATH)/ref_scalv.o      \
           $(TEST_OBJ_PATH)/ref_scal2v.o     \
           $(TEST_OBJ_PATH)/ref_subv.o       \
           $(TEST_OBJ_PATH)/ref_xpbyv.o      \
           $(TEST_OBJ_PATH)/ref_addm.o       \
           $(TEST_OBJ_PATH)/ref_axpym.o      \
           $(TEST_OBJ_PATH)/ref_copym.o      \
           $(TEST_OBJ_PATH)/ref_normfm.o     \
           $(TEST_OBJ_PATH)/ref_scalm.o      \
           $(TEST_OBJ_PATH)/ref_scal2m.o     \
           $(TEST_OBJ_PATH)/ref_subm.o       \
           $(TEST_OBJ_PATH)/ref_xpbym.o      \
           $(TEST_OBJ_PATH)/ref_axpy2v.o     \
           $(TEST_OBJ_PATH)/ref_dotaxpyv.o   \
           $(TEST_OBJ_PATH)/ref_axpyf.o      \
           $(TEST_OBJ_PATH)/ref_dotxf.o      \
           $(TEST_OBJ_PATH)/ref_dotxaxpyf.o  \
           $(TEST_OBJ_PATH)/ref_gemv.o       \
           $(TEST_OBJ_PATH)/ref_ger.o        \
           $(TEST_OBJ_PATH)/ref_hemv.o       \
           $(TEST_OBJ_PATH)/ref_her.o        \
           $(TEST_OBJ_PATH)/ref_her2.o       \
           $(TEST_OBJ_PATH)/ref_symv.o       \
           $(TEST_OBJ_PATH)/ref_syr.o        \
           $(TEST_OBJ_PATH)/ref_syr2.o       \
           $(TEST_OBJ_PATH)/ref_trmv.o       \
           $(TEST_OBJ_PATH)/ref_trsv.o       \
           $(TEST_OBJ_PATH)/ref_gemm.o       \
           $(TEST_OBJ_PATH)/ref_gemmt.o      \
           $(TEST_OBJ_PATH)/ref_hemm.o       \
           $(TEST_OBJ_PATH)/ref_herk.o       \
           $(TEST_OBJ_PATH)/ref_her2k.o      \
           $(TEST_OBJ_PATH)/ref_symm.o       \
           $(TEST_OBJ_PATH)/ref_syrk.o       \
           $(TEST_OBJ_PATH)/ref_syr2k.o      \
           $(TEST_OBJ_PATH)/ref_trmm.o       \
           $(TEST_OBJ_PATH)/ref_trmm3.o      \
           $(TEST_OBJ_PATH)/ref_trsm.o       \
           $(TEST_OBJ_PATH)/test_randv.o     \
           $(TEST_OBJ_PATH)/test_randm.o     \
           $(TEST_OBJ_PATH)/test_addv.o      \
           $(TEST_OBJ_PATH)/test_amaxv.o     \
           $(TEST_OBJ_PATH)/test_axpbyv.o    \
           $(TEST_OBJ_PATH)/test_axpyv.o     \
           $(TEST_OBJ_PATH)/test_copyv.o     \
           $(TEST_OBJ_PATH)/test_dotv.o      \
           $(TEST_OBJ_PATH)/test_dotxv.o     \
           $(TEST_OBJ_PATH)/test_normfv.o    \
           $(TEST_OBJ_PATH)/test_scalv.o     \
           $(TEST_OBJ_PATH)/test_scal2v.o    \
           $(TEST_OBJ_PATH)/test_setv.o      \
           $(TEST_OBJ_PATH)/test_subv.o      \
           $(TEST_OBJ_PATH)/test_xpbyv.o     \
           $(TEST_OBJ_PATH)/test_addm.o      \
           $(TEST_OBJ_PATH)/test_axpym.o     \
           $(TEST_OBJ_PATH)/test_copym.o     \
           $(TEST_OBJ_PATH)/test_normfm.o    \
           $(TEST_OBJ_PATH)/test_scalm.o     \
           $(TEST_OBJ_PATH)/test_scal2m.o    \
           $(TEST_OBJ_PATH)/test_setm.o      \
           $(TEST_OBJ_PATH)/test_subm.o      \
           $(TEST_OBJ_PATH)/test_xpbym.o     \
           $(TEST_OBJ_PATH)/test_axpy2v.o    \
           $(TEST_OBJ_PATH)/test_dotaxpyv.o  \
           $(TEST_OBJ_PATH)/test_axpyf.o     \
           $(TEST_OBJ_PATH)/test_dotxf.o     \
           $(TEST_OBJ_PATH)/test_dotxaxpyf.o \
           $(TEST_OBJ_PATH)/test_gemv.o      \
           $(TEST_OBJ_PATH)/test_ger.o       \
           $(TEST_OBJ_PATH)/test_hemv.o      \
           $(TEST_OBJ_PATH)/test_her.o       \
           $(TEST_OBJ_PATH)/test_her2.o      \
           $(TEST_OBJ_PATH)/test_symv.o      \
           $(TEST_OBJ_PATH)/test_syr.o       \
           $(TEST_OBJ_PATH)/test_syr2.o      \
           $(TEST_OBJ_PATH)/test_trmv.o      \
           $(TEST_OBJ_PATH)/test_trsv.o      \
           $(TEST_OBJ_PATH)/test_gemm.o      \
           $(TEST_OBJ_PATH)/test_gemmt.o     \
           $(TEST_OBJ_PATH)/test_hemm.o      \
           $(TEST_OBJ_PATH)/test_herk.o      \
           $(TEST_OBJ_PATH)/test_her2k.o     \
           $(TEST_OBJ_PATH)/test_symm.o      \
           $(TEST_OBJ_PATH)/test_syrk.o      \
           $(TEST_OBJ_PATH)/test_syr2k.o     \
           $(TEST_OBJ_PATH)/test_trmm.o       \
           $(TEST_OBJ_PATH)/test_trmm3.o       \
           $(TEST_OBJ_PATH)/test_trsm.o         \
           $(TEST_OBJ_PATH)/lpgemm_utils.o       \
           $(TEST_OBJ_PATH)/test_gemm_u8s8s16os8.o \
           $(TEST_OBJ_PATH)/test_gemm_u8s8s32os8.o  \
           $(TEST_OBJ_PATH)/test_gemm_u8s8s16os16.o  \
           $(TEST_OBJ_PATH)/test_gemm_u8s8s32os32.o   \
           $(TEST_OBJ_PATH)/test_gemm_f32f32f32of32.o  \
           $(TEST_OBJ_PATH)/test_gemm_bf16bf16f32of32.o \
           $(TEST_OBJ_PATH)/test_gemm_bf16bf16f32obf16.o

#all: build run

build: $(TEST_EXE)

$(shell mkdir -p $(TEST_OBJ_PATH) $(BIN_PATH))

$(TEST_EXE): $(TEST_OBJS) $(HEADERS)
	@echo "------------------------------------------"
	@echo "Creating the executable for the Program"
	@echo "------------------------------------------"
	$(CXX) $(CXXFLAGS) $(INCLUDE_PATH) $(TEST_OBJS) $(LIB_PATH) $(LDFLAGS) -o $(TEST_EXE)

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.cpp
	@echo "------------------------------------------"
	@echo "Compiling the file $<"
	@echo "------------------------------------------"
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INCLUDE_PATH) $< $(LIBPTHREAD)  -o $@


# -- Test run/check rules --
run: $(TEST_EXE)
#	@echo "Running $(TEST_EXE) with output redirected to '$(TESTSUITE_OUT_FILE)'"
	@./$(TEST_EXE) > $(TESTSUITE_OUT_FILE)
	@./$(TEST_EXE)


clean:
	rm -rf $(TEST_OBJ_PATH)/*.o $(TEST_EXE)
	rm -rf $(TEST_OBJ_PATH) $(BIN_PATH)
