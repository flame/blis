/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc.

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

*/
/* square packed (sqp) kernels */
#define KLP 1// k loop partition.

/* sqp dgemm core kernels, targetted mainly for square sizes by default.
   sqp framework allows tunning for other shapes.*/
inc_t bli_sqp_dgemm_kernel_8mx6n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc);
inc_t bli_sqp_dgemm_kernel_8mx5n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc);
inc_t bli_sqp_dgemm_kernel_8mx4n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc);
inc_t bli_sqp_dgemm_kernel_8mx3n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc);
inc_t bli_sqp_dgemm_kernel_8mx2n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc);
inc_t bli_sqp_dgemm_kernel_8mx1n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc);
inc_t bli_sqp_dgemm_kernel_1mx1n(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc);
inc_t bli_sqp_dgemm_kernel_mxn(gint_t n, gint_t k, gint_t j, double* aPacked, guint_t lda, double* b, guint_t ldb, double* c, guint_t ldc, gint_t mx);

//add and sub kernels
void bli_add_m(gint_t m,gint_t n,double* w,double* c);
void bli_sub_m(gint_t m, gint_t n, double* w, double* c);

//packing kernels
//Pack A with alpha multiplication
void bli_sqp_prepackA(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha, gint_t mx);

void bli_prepackA_8(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha);
void bli_prepackA_4(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha);
void bli_prepackA_G4(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha, gint_t mx);
void bli_prepackA_L4(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha, gint_t mx);
void bli_prepackA_1(double* pa, double* aPacked, gint_t k, guint_t lda, bool isTransA, double alpha);

/* Pack real and imaginary parts in separate buffers and also multipy with multiplication factor */
void bli_3m_sqp_packC_real_imag(double* pb, guint_t n, guint_t k, guint_t ldb, double* pbr, double* pbi, double mul, gint_t mx);
void bli_3m_sqp_packB_real_imag_sum(double* pb, guint_t n, guint_t k, guint_t ldb, double* pbr, double* pbi, double* pbs, double mul, gint_t mx);
void bli_3m_sqp_packA_real_imag_sum(double *pa, gint_t i, guint_t k, guint_t lda, double *par, double *pai, double *pas, trans_t transa, gint_t mx, gint_t p);