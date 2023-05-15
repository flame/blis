/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
// -- level-1m --
// Removed - reference packm kernels are used


// -- level-1v --

// amaxv (intrinsics)
AMAXV_KER_PROT( float,    s, amaxv_zen_int )
AMAXV_KER_PROT( double,   d, amaxv_zen_int )

// axpbyv (intrinsics)
AXPBYV_KER_PROT( float,    s, axpbyv_zen_int )
AXPBYV_KER_PROT( double,   d, axpbyv_zen_int )
AXPBYV_KER_PROT( scomplex, c, axpbyv_zen_int )
AXPBYV_KER_PROT( dcomplex, z, axpbyv_zen_int )

// axpbyv (intrinsics, unrolled x10)
AXPBYV_KER_PROT( float,    s, axpbyv_zen_int10 )
AXPBYV_KER_PROT( double,   d, axpbyv_zen_int10 )

// axpyv (intrinsics)
AXPYV_KER_PROT( float,    s, axpyv_zen_int )
AXPYV_KER_PROT( double,   d, axpyv_zen_int )

// axpyv (intrinsics unrolled x10)
AXPYV_KER_PROT( float,    s, axpyv_zen_int10 )
AXPYV_KER_PROT( double,   d, axpyv_zen_int10 )
AXPYV_KER_PROT( scomplex, c, axpyv_zen_int5 )
AXPYV_KER_PROT( dcomplex, z, axpyv_zen_int5 )

// dotv (intrinsics)
DOTV_KER_PROT( float,    s, dotv_zen_int )
DOTV_KER_PROT( double,   d, dotv_zen_int )

// dotv (intrinsics, unrolled x10)
DOTV_KER_PROT( float,    s, dotv_zen_int10 )
DOTV_KER_PROT( double,   d, dotv_zen_int10 )
DOTV_KER_PROT( scomplex,  c, dotv_zen_int5 )
DOTV_KER_PROT( dcomplex,  z, dotv_zen_int5 )

// dotxv (intrinsics)
DOTXV_KER_PROT( float,    s, dotxv_zen_int )
DOTXV_KER_PROT( double,   d, dotxv_zen_int )
DOTXV_KER_PROT( dcomplex, z, dotxv_zen_int )
DOTXV_KER_PROT( scomplex, c, dotxv_zen_int )

// scalv (intrinsics)
SCALV_KER_PROT( float,    s, scalv_zen_int )
SCALV_KER_PROT( double,   d, scalv_zen_int )
SCALV_KER_PROT( dcomplex, z, scalv_zen_int )

// scalv (intrinsics unrolled x10)
SCALV_KER_PROT( float,      s, scalv_zen_int10 )
SCALV_KER_PROT( double,     d, scalv_zen_int10 )
SCALV_KER_PROT( dcomplex,   z, dscalv_zen_int10 )

// swapv (intrinsics)
SWAPV_KER_PROT(float,   s, swapv_zen_int8 )
SWAPV_KER_PROT(double,  d, swapv_zen_int8 )

// copyv (intrinsics)
COPYV_KER_PROT( float,      s, copyv_zen_int )
COPYV_KER_PROT( double,     d, copyv_zen_int )
COPYV_KER_PROT( dcomplex,   z, copyv_zen_int )

// scal2v (intrinsics)
SCAL2V_KER_PROT(dcomplex, z, scal2v_zen_int)

// setv (intrinsics)
SETV_KER_PROT(float,    s, setv_zen_int)
SETV_KER_PROT(double,   d, setv_zen_int)

// -- level-1f --

// axpyf (intrinsics)
AXPYF_KER_PROT( float,    s, axpyf_zen_int_8 )
AXPYF_KER_PROT( double,   d, axpyf_zen_int_8 )
AXPYF_KER_PROT( double,   d, axpyf_zen_int_16x4 )

AXPYF_KER_PROT( float,    s, axpyf_zen_int_5 )
AXPYF_KER_PROT( float,    s, axpyf_zen_int_6 )
AXPYF_KER_PROT( double,   d, axpyf_zen_int_5 )
AXPYF_KER_PROT( scomplex, c, axpyf_zen_int_5 )
AXPYF_KER_PROT( scomplex, c, axpyf_zen_int_4 )
AXPYF_KER_PROT( dcomplex, z, axpyf_zen_int_5 )
AXPYF_KER_PROT( dcomplex, z, axpyf_zen_int_4 )
// axpy2v (intrinsics)
AXPY2V_KER_PROT(double, d, axpy2v_zen_int )
AXPY2V_KER_PROT(dcomplex, z, axpy2v_zen_int )

// dotxf (intrinsics)
DOTXF_KER_PROT( float,    s, dotxf_zen_int_8 )
DOTXF_KER_PROT( double,   d, dotxf_zen_int_8 )
DOTXF_KER_PROT( double,   d, dotxf_zen_int_4 )
DOTXF_KER_PROT( double,   d, dotxf_zen_int_2 )
DOTXF_KER_PROT( dcomplex,   z, dotxf_zen_int_6 )
DOTXF_KER_PROT( scomplex,   c, dotxf_zen_int_6 )
// dotxaxpyf (intrinsics)
DOTXAXPYF_KER_PROT( double,   d, dotxaxpyf_zen_int_8 )
DOTXAXPYF_KER_PROT( scomplex, c, dotxaxpyf_zen_int_8 )
DOTXAXPYF_KER_PROT( dcomplex, z, dotxaxpyf_zen_int_8 )

// -- level-2 ----------------------------------------------------------------

//gemv(scalar code)
GEMV_KER_PROT( double,   d,  gemv_zen_ref_c )
GEMV_KER_PROT( scomplex, c,  gemv_zen_int_4x4 )
GEMV_KER_PROT( dcomplex, z,  gemv_zen_int_4x4 )

// her (intrinsics)
HER_KER_PROT( dcomplex, z,  her_zen_int_var1 )
HER_KER_PROT( dcomplex, z,  her_zen_int_var2 )

// -- level-3 sup --------------------------------------------------------------
// semmsup_rv

//GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x16 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x16 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x8 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x8 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x4 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x4 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x2 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x2 )

GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_6x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_5x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_4x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_3x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_2x1 )
GEMMSUP_KER_PROT( float,   s, gemmsup_r_zen_ref_1x1 )

// gemmsup_rv (mkernel in m dim)
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16m )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x8m )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x4m )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x2m )
// gemmsup_rv (mkernel in n dim)

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x16n )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x16n )

// gemmsup_rd
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x8)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x16)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x8)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x16)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x4)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x4)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x4)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x2)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x16m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x8m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x4m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x2m)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x16n)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x16n)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x16n)
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x16n)

GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_3x8m )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_3x4m )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_3x2m )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_2x8 )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_1x8 )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_2x4 )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_1x4 )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_2x2 )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_1x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_3x4m )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_3x2m )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_2x4 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_1x4 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_2x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_1x2 )

//gemmsup_rd

GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_3x4m )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_3x2m )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_2x4 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_1x4 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_2x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_1x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_3x4n )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rd_zen_asm_2x4n )

// gemmsup_rv (mkernel in n dim)


GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_3x8n )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_2x8n )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_1x8n )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_3x4 )
GEMMSUP_KER_PROT( scomplex,   c, gemmsup_rv_zen_asm_3x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_3x4n )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_2x4n )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_1x4n )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_3x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_rv_zen_asm_3x1 )

err_t bli_dgemm_small
    (
      obj_t*  alpha,
      obj_t*  a,
      obj_t*  b,
      obj_t*  beta,
      obj_t*  c,
      cntx_t* cntx,
      cntl_t* cntl
    );

err_t bli_dgemm_small_At
    (
      obj_t*  alpha,
      obj_t*  a,
      obj_t*  b,
      obj_t*  beta,
      obj_t*  c,
      cntx_t* cntx,
      cntl_t* cntl
    );

err_t bli_zgemm_small
    (
      obj_t*  alpha,
      obj_t*  a,
      obj_t*  b,
      obj_t*  beta,
      obj_t*  c,
      cntx_t* cntx,
      cntl_t* cntl
    );

err_t bli_zgemm_small_At
    (
      obj_t*  alpha,
      obj_t*  a,
      obj_t*  b,
      obj_t*  beta,
      obj_t*  c,
      cntx_t* cntx,
      cntl_t* cntl
    );

void bli_dgemm_8x6_avx2_k1_nn
    (
      dim_t m,
      dim_t n,
      dim_t k,
      double* alpha,
      double* a, const inc_t lda,
      double* b, const inc_t ldb,
      double* beta,
      double* c, const inc_t ldc
     );

void bli_zgemm_4x6_avx2_k1_nn
    (
      dim_t m,
      dim_t n,
      dim_t k,
      dcomplex* alpha,
      dcomplex* a, const inc_t lda,
      dcomplex* b, const inc_t ldb,
      dcomplex* beta,
      dcomplex* c, const inc_t ldc
     );

err_t bli_trsm_small
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl,
       bool is_parallel
     );

#ifdef BLIS_ENABLE_OPENMP
err_t bli_trsm_small_mt
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       cntl_t* cntl,
      bool     is_parallel
     );

void bli_multi_sgemv_4x2
    (
       conj_t           conjat,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       float*  restrict alpha,
       float*  restrict a, inc_t inca, inc_t lda,
       float*  restrict x, inc_t incx,
       float*  restrict beta,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx,
       dim_t            n_threads
     );

#endif

// threshold functions
bool bli_cntx_gemmtsup_thresh_is_met_zen
     (
       obj_t* a,
       obj_t* b,
       obj_t* c,
       cntx_t* cntx
     );

bool bli_cntx_syrksup_thresh_is_met_zen
     (
       obj_t* a,
       obj_t* b,
       obj_t* c,
       cntx_t* cntx
     );

/*
 * Check if the TRSM small path should be taken for this
 * input and threads combination
 */
bool bli_cntx_trsm_small_thresh_is_met_zen
     (
        obj_t* a,
        dim_t m,
        dim_t n
    );

void bli_snorm2fv_unb_var1_avx2
     (
       dim_t    n,
       float*   x, inc_t incx,
       float* norm,
       cntx_t*  cntx
     );

void bli_dnorm2fv_unb_var1_avx2
     (
       dim_t    n,
       double*   x, inc_t incx,
       double* norm,
       cntx_t*  cntx
     );

void bli_scnorm2fv_unb_var1_avx2
     (
       dim_t    n,
       scomplex*   x, inc_t incx,
       float* norm,
       cntx_t*  cntx
     );

void bli_dznorm2fv_unb_var1_avx2
     (
       dim_t    n,
       dcomplex*   x, inc_t incx,
       double* norm,
       cntx_t*  cntx
     );
