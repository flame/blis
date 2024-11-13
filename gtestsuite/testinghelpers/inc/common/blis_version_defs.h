/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

/* Define for each kernel available in each version. These definitions
   are used to exclude tests of missing kernels in other versions of BLIS.
   We assume for now that we only add kernels, so we can build the list for
   each release based on the previous plus new kernels added.
*/

#ifdef AOCL_DEV

    #define K_bli_zgemmsup_cd_zen4_asm_12x2m 1
    #define K_bli_zgemmsup_cd_zen4_asm_12x4m 1
    #define K_bli_zgemmsup_cd_zen4_asm_2x2 1
    #define K_bli_zgemmsup_cd_zen4_asm_2x4 1
    #define K_bli_zgemmsup_cd_zen4_asm_4x2 1
    #define K_bli_zgemmsup_cd_zen4_asm_4x4 1
    #define K_bli_zgemmsup_cd_zen4_asm_8x2 1
    #define K_bli_zgemmsup_cd_zen4_asm_8x4 1

    #define AOCL_50

#endif


#ifdef AOCL_50

    #define K_bli_caddv_zen_int 1
    #define K_bli_ccopyv_zen_int 1
    #define K_bli_cscal2v_zen_int 1
    #define K_bli_cscalv_zen_int 1
    #define K_bli_cscalv_zen_int_avx512 1
    #define K_bli_csetv_zen_int 1
    #define K_bli_daddv_zen_int 1
    #define K_bli_daddv_zen_int_avx512 1
    #define K_bli_daxpbyv_zen_int_avx512 1
    #define K_bli_daxpyf_zen_int_avx512 1
    #define K_bli_dcopyv_zen4_asm_avx512 1
    #define K_bli_dnorm2fv_unb_var1_avx512 1
    #define K_bli_dscal2v_zen_int 1
    #define K_bli_dscal2v_zen_int_avx512 1
    #define K_bli_dsetv_zen_int_avx512 1
    #define K_bli_saddv_zen_int 1
    #define K_bli_scopyv_zen4_asm_avx512 1
    #define K_bli_sscal2v_zen_int 1
    #define K_bli_ssetv_zen_int_avx512 1
    #define K_bli_zaddv_zen_int 1
    #define K_bli_zaxpyf_zen_int_8_avx512 1
    #define K_bli_zaxpyv_zen_int_avx512 1
    #define K_bli_zcopyv_zen4_asm_avx512 1
    #define K_bli_zdotv_zen4_asm_avx512 1
    #define K_bli_zdotv_zen_int_avx512 1
    #define K_bli_zgemm_16x4_avx512_k1_nn 1
    #define K_bli_zscalv_zen_int_avx512 1
    #define K_bli_zsetv_zen_int 1
    #define K_bli_zsetv_zen_int_avx512 1

    // In AOCL 4.2 but interface changed at 5.0
    #define K_bli_zgemm_4x4_avx2_k1_nn 1

    #define AOCL_42

#endif


#ifdef AOCL_42

    #define E_GEMM_COMPUTE

    #define K_bli_dgemm_24x8_avx512_k1_nn 1
    #define K_bli_zdscalv_zen_int_avx512 1
    #define K_bli_zgemm_zen4_asm_4x12 1
    #define K_bli_zgemm_zen_asm_2x6 1

    // In AOCL 4.1 but interface changed at 4.2
    #define K_bli_dgemm_8x6_avx2_k1_nn 1

    #define AOCL_41

#endif


#ifdef AOCL_41

    #define K_bli_caxpbyv_zen_int 1
    #define K_bli_caxpyv_zen_int5 1
    #define K_bli_cgemm_haswell_asm_3x8 1
    #define K_bli_cgemmsup_rv_zen_asm_1x2 1
    #define K_bli_cgemmsup_rv_zen_asm_1x4 1
    #define K_bli_cgemmsup_rv_zen_asm_1x8 1
    #define K_bli_cgemmsup_rv_zen_asm_1x8n 1
    #define K_bli_cgemmsup_rv_zen_asm_2x2 1
    #define K_bli_cgemmsup_rv_zen_asm_2x4 1
    #define K_bli_cgemmsup_rv_zen_asm_2x8 1
    #define K_bli_cgemmsup_rv_zen_asm_2x8n 1
    #define K_bli_cgemmsup_rv_zen_asm_3x2 1
    #define K_bli_cgemmsup_rv_zen_asm_3x2m 1
    #define K_bli_cgemmsup_rv_zen_asm_3x4 1
    #define K_bli_cgemmsup_rv_zen_asm_3x4m 1
    #define K_bli_cgemmsup_rv_zen_asm_3x8m 1
    #define K_bli_cgemmsup_rv_zen_asm_3x8n 1
    #define K_bli_damaxv_zen_int 1
    #define K_bli_damaxv_zen_int_avx512 1
    #define K_bli_daxpbyv_zen_int 1
    #define K_bli_daxpbyv_zen_int10 1
    #define K_bli_daxpyv_zen_int 1
    #define K_bli_daxpyv_zen_int10 1
    #define K_bli_daxpyv_zen_int_avx512 1
    #define K_bli_dcopyv_zen_int 1
    #define K_bli_ddotv_zen_int 1
    #define K_bli_ddotv_zen_int10 1
    #define K_bli_ddotv_zen_int_avx512 1
    #define K_bli_dgemm_haswell_asm_6x8 1
    #define K_bli_dgemm_zen4_asm_32x6 1
    #define K_bli_dgemm_zen4_asm_8x24 1
    #define K_bli_dgemmsup_rd_haswell_asm_6x8m 1
    #define K_bli_dgemmsup_rd_haswell_asm_6x8n 1
    #define K_bli_dgemmsup_rv_haswell_asm_6x8m 1
    #define K_bli_dgemmsup_rv_haswell_asm_6x8n 1
    #define K_bli_dgemmsup_rv_zen4_asm_24x8m 1
    #define K_bli_dgemmsup_rv_zen5_asm_24x8m 1
    #define K_bli_dgemmtrsm_l_haswell_asm_6x8 1
    #define K_bli_dgemmtrsm_l_zen4_asm_8x24 1
    #define K_bli_dgemmtrsm_u_haswell_asm_6x8 1
    #define K_bli_dgemmtrsm_u_zen4_asm_8x24 1
    #define K_bli_dnorm2fv_unb_var1_avx2 1
    #define K_bli_dscalv_zen_int 1
    #define K_bli_dscalv_zen_int10 1
    #define K_bli_dscalv_zen_int_avx512 1
    #define K_bli_dsetv_zen_int 1
    #define K_bli_dswapv_zen_int8 1
    #define K_bli_dznorm2fv_unb_var1_avx2 1
    #define K_bli_samaxv_zen_int 1
    #define K_bli_samaxv_zen_int_avx512 1
    #define K_bli_saxpbyv_zen_int 1
    #define K_bli_saxpbyv_zen_int10 1
    #define K_bli_saxpyv_zen_int 1
    #define K_bli_saxpyv_zen_int10 1
    #define K_bli_saxpyv_zen_int_avx512 1
    #define K_bli_scnorm2fv_unb_var1_avx2 1
    #define K_bli_scopyv_zen_int 1
    #define K_bli_sgemm_haswell_asm_6x16 1
    #define K_bli_sgemm_skx_asm_32x12_l2 1
    #define K_bli_sgemmsup_rd_zen_asm_6x16m 1
    #define K_bli_sgemmsup_rd_zen_asm_6x16n 1
    #define K_bli_sgemmsup_rd_zen_asm_6x64m_avx512 1
    #define K_bli_sgemmsup_rd_zen_asm_6x64n_avx512 1
    #define K_bli_sgemmsup_rv_zen_asm_6x16m 1
    #define K_bli_sgemmsup_rv_zen_asm_6x16n 1
    #define K_bli_sgemmsup_rv_zen_asm_6x64m_avx512 1
    #define K_bli_sgemmsup_rv_zen_asm_6x64n_avx512 1
    #define K_bli_sgemmtrsm_l_haswell_asm_6x16 1
    #define K_bli_sgemmtrsm_u_haswell_asm_6x16 1
    #define K_bli_snorm2fv_unb_var1_avx2 1
    #define K_bli_sscalv_zen_int 1
    #define K_bli_sscalv_zen_int10 1
    #define K_bli_ssetv_zen_int 1
    #define K_bli_sswapv_zen_int8 1
    #define K_bli_trsm_small 1
    #define K_bli_trsm_small_AVX512 1
    #define K_bli_zaxpbyv_zen_int 1
    #define K_bli_zaxpyv_zen_int5 1
    #define K_bli_zcopyv_zen_int 1
    #define K_bli_zdscalv_zen_int10 1
    #define K_bli_zgemm_haswell_asm_3x4 1
    #define K_bli_zgemm_zen4_asm_12x4 1
    #define K_bli_zgemmsup_cv_zen4_asm_12x1m 1
    #define K_bli_zgemmsup_cv_zen4_asm_12x2m 1
    #define K_bli_zgemmsup_cv_zen4_asm_12x3m 1
    #define K_bli_zgemmsup_cv_zen4_asm_12x4m 1
    #define K_bli_zgemmsup_cv_zen4_asm_2x1 1
    #define K_bli_zgemmsup_cv_zen4_asm_2x2 1
    #define K_bli_zgemmsup_cv_zen4_asm_2x3 1
    #define K_bli_zgemmsup_cv_zen4_asm_2x4 1
    #define K_bli_zgemmsup_cv_zen4_asm_4x1 1
    #define K_bli_zgemmsup_cv_zen4_asm_4x2 1
    #define K_bli_zgemmsup_cv_zen4_asm_4x3 1
    #define K_bli_zgemmsup_cv_zen4_asm_4x4 1
    #define K_bli_zgemmsup_cv_zen4_asm_8x1 1
    #define K_bli_zgemmsup_cv_zen4_asm_8x2 1
    #define K_bli_zgemmsup_cv_zen4_asm_8x3 1
    #define K_bli_zgemmsup_cv_zen4_asm_8x4 1
    #define K_bli_zgemmsup_rd_zen_asm_1x2 1
    #define K_bli_zgemmsup_rd_zen_asm_1x4 1
    #define K_bli_zgemmsup_rd_zen_asm_2x2 1
    #define K_bli_zgemmsup_rd_zen_asm_2x4 1
    #define K_bli_zgemmsup_rd_zen_asm_2x4n 1
    #define K_bli_zgemmsup_rd_zen_asm_3x2m 1
    #define K_bli_zgemmsup_rd_zen_asm_3x4m 1
    #define K_bli_zgemmsup_rd_zen_asm_3x4n 1
    #define K_bli_zgemmsup_rv_zen_asm_1x2 1
    #define K_bli_zgemmsup_rv_zen_asm_1x4 1
    #define K_bli_zgemmsup_rv_zen_asm_1x4n 1
    #define K_bli_zgemmsup_rv_zen_asm_2x2 1
    #define K_bli_zgemmsup_rv_zen_asm_2x4 1
    #define K_bli_zgemmsup_rv_zen_asm_2x4n 1
    #define K_bli_zgemmsup_rv_zen_asm_3x2 1
    #define K_bli_zgemmsup_rv_zen_asm_3x2m 1
    #define K_bli_zgemmsup_rv_zen_asm_3x4m 1
    #define K_bli_zgemmsup_rv_zen_asm_3x4n 1
    #define K_bli_zgemmtrsm_l_zen4_asm_4x12 1
    #define K_bli_zgemmtrsm_l_zen_asm_2x6 1
    #define K_bli_zgemmtrsm_u_zen4_asm_4x12 1
    #define K_bli_zgemmtrsm_u_zen_asm_2x6 1
    #define K_bli_zscal2v_zen_int 1
    #define K_bli_zscalv_zen_int 1

#endif

