/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.

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

// -- level-1v --

// amaxv (intrinsics)
AMAXV_KER_PROT( float,    s, amaxv_zen_int_avx512 )
AMAXV_KER_PROT( double,   d, amaxv_zen_int_avx512 )

// scalv (AVX512 intrinsics)
SCALV_KER_PROT( float,   s, scalv_zen_int_avx512 )
SCALV_KER_PROT( double,  d, scalv_zen_int_avx512 )

// dotv (intrinsics)
DOTV_KER_PROT( float,    s, dotv_zen_int_avx512 )
DOTV_KER_PROT( double,   d, dotv_zen_int_avx512 )

// axpyv (intrinsics)
AXPYV_KER_PROT( float,    s, axpyv_zen_int_avx512 )
AXPYV_KER_PROT( double,   d, axpyv_zen_int_avx512 )

GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_l_zen_asm_16x14)
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_u_zen_asm_16x14)
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_l_zen4_asm_8x24)
GEMMTRSM_UKR_PROT( double,   d, gemmtrsm_u_zen4_asm_8x24)

//packing kernels
PACKM_KER_PROT( double,   d, packm_zen4_asm_16xk )
PACKM_KER_PROT( double,   d, packm_zen4_asm_8xk )
PACKM_KER_PROT( double,   d, packm_zen4_asm_24xk )
PACKM_KER_PROT( double,   d, packm_zen4_asm_32xk )
PACKM_KER_PROT( double,   d, packm_32xk_zen4_ref )
PACKM_KER_PROT( dcomplex, z, packm_zen4_asm_12xk )
PACKM_KER_PROT( dcomplex, z, packm_zen4_asm_4xk )

// native dgemm kernel
GEMM_UKR_PROT( double,   d, gemm_zen4_asm_32x6 )
GEMM_UKR_PROT( double,   d, gemm_zen4_asm_8x24 )
GEMM_UKR_PROT( dcomplex, z, gemm_zen4_asm_12x4 )

//sgemm rv sup
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x64m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x48m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x32m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x16m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x64m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x48m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x32m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x16m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x64m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x48m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x32m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x16m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x64m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x48m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x32m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x16m_avx512 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_6x64n_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x64n_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_4x64n_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x64n_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_2x64n_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_1x64n_avx512 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x48_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x32_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_5x16_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x48_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x32_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rv_zen_asm_3x16_avx512 )

// sgemm rd sup
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x64m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x48m_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x32m_avx512 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x64n_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x64n_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_6x64n_avx512 )

GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_5x64_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_4x64_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x64_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x64_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x64_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_5x48_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_4x48_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x48_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x48_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x48_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_5x32_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_4x32_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_3x32_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_2x32_avx512 )
GEMMSUP_KER_PROT( float,   s, gemmsup_rd_zen_asm_1x32_avx512 )

TRSMSMALL_PROT(trsm_small_AVX512)
TRSMSMALL_KER_PROT( d, trsm_small_AutXB_AlXB_AVX512 )
TRSMSMALL_KER_PROT( d, trsm_small_XAltB_XAuB_AVX512 )
TRSMSMALL_KER_PROT( d, trsm_small_XAutB_XAlB_AVX512 )
TRSMSMALL_KER_PROT( d, trsm_small_AltXB_AuXB_AVX512 )

#ifdef BLIS_ENABLE_OPENMP
TRSMSMALL_PROT(trsm_small_mt_AVX512)
#endif

// Dgemm sup RV kernels
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x8m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x7m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x6m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x5m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x4m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x3m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x2m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x1m)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x8)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x8)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x8)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x7)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x7)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x7)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x6)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x6)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x6)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x5)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x5)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x5)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x4)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x4)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x4)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x3)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x3)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x3)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x2)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x2)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x2)

GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_24x1)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_16x1)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen4_asm_8x1)

// Zgemm sup CV kernels
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_12x4m )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_12x3m )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_12x2m )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_12x1m )

GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_8x4 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_8x3 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_8x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_8x1 )

GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_4x4 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_4x3 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_4x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_4x1 )

GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_2x4 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_2x3 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_2x2 )
GEMMSUP_KER_PROT( dcomplex,   z, gemmsup_cv_zen4_asm_2x1 )

// threshold functions
bool bli_cntx_gemmsup_thresh_is_met_zen4
	 (
		obj_t*  a,
		obj_t*  b,
		obj_t*  c,
		cntx_t* cntx
	 );
