#[=[

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

add_definitions(-DBLAS="AOCL")

add_executable(TestAminv test_aminv.c)
target_link_libraries(TestAminv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestAminv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestAminv optimized "${LIB_NAME}.lib")

add_executable(TestAxpyv test_axpyv.c)
target_link_libraries(TestAxpyv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestAxpyv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestAxpyv optimized "${LIB_NAME}.lib")

add_executable(TestAxpbyv test_axpbyv.c)
target_link_libraries(TestAxpbyv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestAxpbyv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestAxpbyv optimized "${LIB_NAME}.lib")

add_executable(TestCopyv test_copyv.c)
target_link_libraries(TestCopyv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestCopyv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestCopyv optimized "${LIB_NAME}.lib")

add_executable(TestCabs1 test_cabs1.c)
target_link_libraries(TestCabs1 debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestCabs1 OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestCabs1 optimized "${LIB_NAME}.lib")

add_executable(TestDotv test_dotv.c)
target_link_libraries(TestDotv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestDotv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestDotv optimized "${LIB_NAME}.lib")

add_executable(TestGemm test_gemm.c)
target_link_libraries(TestGemm debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestGemm OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestGemm optimized "${LIB_NAME}.lib")

add_executable(TestGemmBatch test_gemm_batch.c)
target_link_libraries(TestGemmBatch debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestGemmBatch OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestGemmBatch optimized "${LIB_NAME}.lib")

add_executable(TestGemm3m test_gemm3m.c)
target_link_libraries(TestGemm3m debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestGemm3m OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestGemm3m optimized "${LIB_NAME}.lib")

add_executable(TestGemmt test_gemmt.c)
target_link_libraries(TestGemmt debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestGemmt OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestGemmt optimized "${LIB_NAME}.lib")

add_executable(TestGemv test_gemv.c)
target_link_libraries(TestGemv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestGemv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestGemv optimized "${LIB_NAME}.lib")

add_executable(TestGer test_ger.c)
target_link_libraries(TestGer debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestGer OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestGer optimized "${LIB_NAME}.lib")

add_executable(TestHemm test_hemm.c)
target_link_libraries(TestHemm debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestHemm OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestHemm optimized "${LIB_NAME}.lib")

add_executable(TestHemv test_hemv.c)
target_link_libraries(TestHemv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestHemv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestHemv optimized "${LIB_NAME}.lib")

add_executable(TestHer test_her.c)
target_link_libraries(TestHer debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestHer OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestHer optimized "${LIB_NAME}.lib")

add_executable(TestHer2 test_her2.c)
target_link_libraries(TestHer2 debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestHer2 OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestHer2 optimized "${LIB_NAME}.lib")

add_executable(TestHer2k test_her2k.c)
target_link_libraries(TestHer2k debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestHer2k OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestHer2k optimized "${LIB_NAME}.lib")

add_executable(TestHerk test_herk.c)
target_link_libraries(TestHerk debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestHerk OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestHerk optimized "${LIB_NAME}.lib")

add_executable(TestScalv test_scalv.c)
target_link_libraries(TestScalv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestScalv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestScalv optimized "${LIB_NAME}.lib")

add_executable(TestSwapv test_swapv.c)
target_link_libraries(TestSwapv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestSwapv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestSwapv optimized "${LIB_NAME}.lib")

add_executable(TestTrmm test_trmm.c)
target_link_libraries(TestTrmm debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestTrmm OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestTrmm optimized "${LIB_NAME}.lib")

add_executable(TestTrmv test_trmv.c)
target_link_libraries(TestTrmv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestTrmv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestTrmv optimized "${LIB_NAME}.lib")

add_executable(TestTrsm test_trsm.c)
target_link_libraries(TestTrsm debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestTrsm OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestTrsm optimized "${LIB_NAME}.lib")

add_executable(TestTrsv test_trsv.c)
target_link_libraries(TestTrsv debug "${LIB_NAME}.lib")
if(ENABLE_OPENMP)
    target_link_libraries(TestTrsv OpenMP::OpenMP_CXX)
endif()
target_link_libraries(TestTrsv optimized "${LIB_NAME}.lib")


