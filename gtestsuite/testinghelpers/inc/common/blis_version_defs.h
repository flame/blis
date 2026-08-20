/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
    #define AOCL_53
#endif

#ifdef AOCL_53
    #define AOCL_52
#endif

#ifdef AOCL_52
    #define K_zen4_int_40x2_mt zen4_int_40x2_mt
    #define K_zen4_int_40x8_mt zen4_int_40x8_mt
    #define K_zen4_int_40x2_st zen4_int_40x2_st
    #define K_zen4_int_40x8_st zen4_int_40x8_st
    #define K_bli_zdotv_zen_int_5 bli_zdotv_zen_int_5
    #define K_bli_cdotv_zen_int_5 bli_cdotv_zen_int_5
    #define K_bli_zaxpyv_zen_int_5 bli_zaxpyv_zen_int_5
    #define K_bli_caxpyv_zen_int_5 bli_caxpyv_zen_int_5
    #define K_bli_sswapv_zen_int_8 bli_sswapv_zen_int_8
    #define K_bli_dswapv_zen_int_8 bli_dswapv_zen_int_8
    #define K_bli_zdscalv_zen_int_10 bli_zdscalv_zen_int_10
    #define K_bli_sscalv_zen_int_10 bli_sscalv_zen_int_10
    #define K_bli_dscalv_zen_int_10 bli_dscalv_zen_int_10
    #define K_bli_sdotv_zen_int_10 bli_sdotv_zen_int_10
    #define K_bli_ddotv_zen_int_10 bli_ddotv_zen_int_10
    #define K_bli_saxpyv_zen_int_10 bli_saxpyv_zen_int_10
    #define K_bli_daxpyv_zen_int_10 bli_daxpyv_zen_int_10
    #define K_bli_saxpbyv_zen_int_10 bli_saxpbyv_zen_int_10
    #define K_bli_daxpbyv_zen_int_10 bli_daxpbyv_zen_int_10
    #define K_bli_dgemmsup_cv_zen5_asm_24x8m bli_dgemmsup_cv_zen5_asm_24x8m
    #define K_bli_dgemmsup_cv_zen4_asm_24x8m bli_dgemmsup_cv_zen4_asm_24x8m
    #define K_bli_dgemmsup_cv_zen4_asm_24x8m_new bli_dgemmsup_cv_zen4_asm_24x8m_new
    #define K_bli_dgemm_tiny_zen4_24x8 bli_dgemm_tiny_zen4_24x8
    #define K_bli_dgemm_tiny_zen_6x8 bli_dgemm_tiny_zen_6x8
    #define K_bli_zaxpyf_zen4_int_8 bli_zaxpyf_zen4_int_8
    #define K_bli_daxpyf_zen4_int bli_daxpyf_zen4_int
    #define K_bli_ddotxf_zen4_int bli_ddotxf_zen4_int
    #define K_bli_dgemm_zen4_asm_8x24 bli_dgemm_zen4_asm_8x24
    #define K_bli_ztrsm_small_zen_int_pack bli_ztrsm_small_zen_int_pack
    #define K_bli_ctrsm_small_zen_int_pack bli_ctrsm_small_zen_int_pack
    #define K_bli_strsm_small_zen_int_pack bli_strsm_small_zen_int_pack
    #define K_bli_dtrsm_small_zen_int_pack bli_dtrsm_small_zen_int_pack
    #define K_bli_ztrsm_small_zen5 bli_ztrsm_small_zen5
    #define K_bli_dtrsm_small_zen4_int_pack bli_dtrsm_small_zen4_int_pack
    #define K_bli_trsm_small_ref bli_trsm_small_ref
    #define K_bli_trsm_small_zen bli_trsm_small_zen
    #define K_bli_trsm_small_zen bli_trsm_small_zen
    #define K_bli_trsm_small_zen5_mt bli_trsm_small_zen5_mt
    #define K_bli_trsm_small_zen5 bli_trsm_small_zen5
    #define K_bli_trsm_small_zen4_mt bli_trsm_small_zen4_mt
    #define K_bli_trsm_small_zen4 bli_trsm_small_zen4
    #define K_bli_zsetv_zen4_int bli_zsetv_zen4_int
    #define K_bli_dsetv_zen4_int bli_dsetv_zen4_int
    #define K_bli_ssetv_zen4_int bli_ssetv_zen4_int
    #define K_scalv_zen4_int scalv_zen4_int
    #define K_scalv_zen4_int scalv_zen4_int
    #define K_bli_zscalv_zen4_int bli_zscalv_zen4_int
    #define K_bli_cscalv_zen4_int bli_cscalv_zen4_int
    #define K_bli_zdscalv_zen4_int bli_zdscalv_zen4_int
    #define K_bli_dscalv_zen4_int bli_dscalv_zen4_int
    #define K_bli_sscalv_zen4_int bli_sscalv_zen4_int
    #define K_bli_dscal2v_zen4_int bli_dscal2v_zen4_int
    #define K_bli_zdotxv_zen4_int bli_zdotxv_zen4_int
    #define K_bli_zdotv_zen4_asm bli_zdotv_zen4_asm
    #define K_bli_zdotv_zen4_int bli_zdotv_zen4_int
    #define K_bli_ddotv_zen4_int bli_ddotv_zen4_int
    #define K_bli_sdotv_zen4_int bli_sdotv_zen4_int
    #define K_bli_dcopyv_zen5_asm bli_dcopyv_zen5_asm
    #define K_bli_zcopyv_zen4_int bli_zcopyv_zen4_int
    #define K_bli_dcopyv_zen4_int bli_dcopyv_zen4_int
    #define K_bli_scopyv_zen4_int bli_scopyv_zen4_int
    #define K_bli_zcopyv_zen4_asm bli_zcopyv_zen4_asm
    #define K_bli_dcopyv_zen4_asm bli_dcopyv_zen4_asm
    #define K_bli_scopyv_zen4_asm bli_scopyv_zen4_asm
    #define K_bli_zaxpyv_zen4_int bli_zaxpyv_zen4_int
    #define K_bli_daxpyv_zen4_int bli_daxpyv_zen4_int
    #define K_bli_saxpyv_zen4_int bli_saxpyv_zen4_int
    #define K_bli_daxpbyv_zen4_int bli_daxpbyv_zen4_int
    #define K_bli_damaxv_zen4_int bli_damaxv_zen4_int
    #define K_bli_samaxv_zen4_int bli_samaxv_zen4_int
    #define K_bli_daddv_zen4_int bli_daddv_zen4_int
    #define K_bli_dnorm2fv_zen4_int_unb_var1 bli_dnorm2fv_zen4_int_unb_var1
    #define K_bli_snorm2fv_zen_int_unb_var1 bli_snorm2fv_zen_int_unb_var1
    #define K_bli_scnorm2fv_zen_int_unb_var1 bli_scnorm2fv_zen_int_unb_var1
    #define K_bli_dznorm2fv_zen_int_unb_var1 bli_dznorm2fv_zen_int_unb_var1
    #define K_bli_dnorm2fv_zen_int_unb_var1 bli_dnorm2fv_zen_int_unb_var1
    #define K_bli_sgemmsup_rd_zen4_asm_6x64n bli_sgemmsup_rd_zen4_asm_6x64n
    #define K_bli_sgemmsup_rd_zen4_asm_6x64m bli_sgemmsup_rd_zen4_asm_6x64m
    #define K_bli_sgemmsup_rd_zen5_asm_6x64n bli_sgemmsup_rd_zen5_asm_6x64n
    #define K_bli_sgemmsup_rd_zen5_asm_6x64m bli_sgemmsup_rd_zen5_asm_6x64m
    #define K_bli_sgemmsup_rv_zen4_asm_6x64n bli_sgemmsup_rv_zen4_asm_6x64n
    #define K_bli_sgemmsup_rv_zen4_asm_6x64m bli_sgemmsup_rv_zen4_asm_6x64m
    #define K_bli_sgemmsup_rv_zen4_asm_6x64n bli_sgemmsup_rv_zen4_asm_6x64n
    #define K_bli_sgemmsup_rv_zen4_asm_6x64m bli_sgemmsup_rv_zen4_asm_6x64m
    #define K_bli_dgemmtrsm_u_zen4_asm_16x14 bli_dgemmtrsm_u_zen4_asm_16x14
    #define K_bli_dgemmtrsm_l_zen4_asm_16x14 bli_dgemmtrsm_l_zen4_asm_16x14
    // ZEN (AVX2) GEMV kernels
    #define K_bli_dgemv_t_zen_int bli_dgemv_t_zen_int
    #define K_bli_dgemv_t_zen_int_16x4 bli_dgemv_t_zen_int_16x4
    #define K_bli_dgemv_t_zen_int_16x4_mt bli_dgemv_t_zen_int_16x4_mt
    #define K_bli_dgemv_n_zen_int bli_dgemv_n_zen_int
    #define K_bli_dgemv_n_zen_int_20x4 bli_dgemv_n_zen_int_20x4
    #define K_bli_dgemv_n_zen_int_20x4_mt bli_dgemv_n_zen_int_20x4_mt
    #define K_bli_dgemv_m_zen_int_20x4 bli_dgemv_m_zen_int_20x4
    #define K_bli_sgemv_t_zen_int bli_sgemv_t_zen_int
    #define K_bli_sgemv_t_zen_int_24x4 bli_sgemv_t_zen_int_24x4
    #define K_bli_sgemv_t_zen_int_24x4_mt bli_sgemv_t_zen_int_24x4_mt
    #define K_bli_sgemv_n_zen_int bli_sgemv_n_zen_int
    #define K_bli_sgemv_n_zen_int_40x4 bli_sgemv_n_zen_int_40x4
    #define K_bli_sgemv_n_zen_int_40x4_mt bli_sgemv_n_zen_int_40x4_mt
    #define K_bli_sgemv_m_zen_int_40x4 bli_sgemv_m_zen_int_40x4
    #define K_bli_sgemv_m_zen_int_40x4_mt_Mdiv bli_sgemv_m_zen_int_40x4_mt_Mdiv
    #define K_bli_sgemv_m_zen_int_40x4_mt_Ndiv bli_sgemv_m_zen_int_40x4_mt_Ndiv
    #define K_bli_cgemv_t_zen_int bli_cgemv_t_zen_int
    #define K_bli_cgemv_t_zen_int_20x4 bli_cgemv_t_zen_int_20x4
    #define K_bli_cgemv_t_zen_int_20x4_mt bli_cgemv_t_zen_int_20x4_mt
    #define K_bli_cgemv_n_zen_int bli_cgemv_n_zen_int
    #define K_bli_cgemv_n_zen_int_20x5 bli_cgemv_n_zen_int_20x5
    #define K_bli_cgemv_n_zen_int_20x5_mt bli_cgemv_n_zen_int_20x5_mt
    #define K_bli_zgemv_t_zen_int bli_zgemv_t_zen_int
    #define K_bli_zgemv_t_zen_int_10x4 bli_zgemv_t_zen_int_10x4
    #define K_bli_zgemv_t_zen_int_10x4_mt bli_zgemv_t_zen_int_10x4_mt
    #define K_bli_zgemv_n_zen_int bli_zgemv_n_zen_int
    #define K_bli_zgemv_n_zen_int_10x5 bli_zgemv_n_zen_int_10x5
    #define K_bli_zgemv_n_zen_int_10x5_mt bli_zgemv_n_zen_int_10x5_mt
    // ZEN4 (AVX-512) GEMV kernels
    #define K_bli_dgemv_t_zen4_int bli_dgemv_t_zen4_int
    #define K_bli_dgemv_t_zen4_int_32x8 bli_dgemv_t_zen4_int_32x8
    #define K_bli_dgemv_t_zen4_int_32x8_mt bli_dgemv_t_zen4_int_32x8_mt
    #define K_bli_dgemv_n_zen4_int bli_dgemv_n_zen4_int
    #define K_bli_dgemv_n_zen4_int_40x8 bli_dgemv_n_zen4_int_40x8
    #define K_bli_dgemv_n_zen4_int_40x8_mt bli_dgemv_n_zen4_int_40x8_mt
    #define K_bli_dgemv_m_zen4_int_40x8 bli_dgemv_m_zen4_int_40x8
    #define K_bli_dgemv_m_zen4_int_40x8_mt_Ndiv bli_dgemv_m_zen4_int_40x8_mt_Ndiv
    #define K_bli_dgemv_m_zen4_int_40x8_mt_Mdiv bli_dgemv_m_zen4_int_40x8_mt_Mdiv
    #define K_bli_sgemv_t_zen4_int bli_sgemv_t_zen4_int
    #define K_bli_sgemv_t_zen4_int_48x8 bli_sgemv_t_zen4_int_48x8
    #define K_bli_sgemv_t_zen4_int_48x8_mt bli_sgemv_t_zen4_int_48x8_mt
    #define K_bli_sgemv_n_zen4_int bli_sgemv_n_zen4_int
    #define K_bli_sgemv_n_zen4_int_80x8 bli_sgemv_n_zen4_int_80x8
    #define K_bli_sgemv_n_zen4_int_80x8_mt bli_sgemv_n_zen4_int_80x8_mt
    #define K_bli_sgemv_m_zen4_int_80x8 bli_sgemv_m_zen4_int_80x8
    #define K_bli_sgemv_m_zen4_int_80x8_mt_Mdiv bli_sgemv_m_zen4_int_80x8_mt_Mdiv
    #define K_bli_sgemv_m_zen4_int_80x8_mt_Ndiv bli_sgemv_m_zen4_int_80x8_mt_Ndiv
    #define K_bli_cgemv_t_zen4_int bli_cgemv_t_zen4_int
    #define K_bli_cgemv_t_zen4_int_40x8 bli_cgemv_t_zen4_int_40x8
    #define K_bli_cgemv_t_zen4_int_40x8_mt bli_cgemv_t_zen4_int_40x8_mt
    #define K_bli_cgemv_n_zen4_int bli_cgemv_n_zen4_int
    #define K_bli_cgemv_n_zen4_int_40x10 bli_cgemv_n_zen4_int_40x10
    #define K_bli_cgemv_n_zen4_int_40x10_mt bli_cgemv_n_zen4_int_40x10_mt
    #define K_bli_zgemv_t_zen4_int bli_zgemv_t_zen4_int
    #define K_bli_zgemv_t_zen4_int_20x8 bli_zgemv_t_zen4_int_20x8
    #define K_bli_zgemv_t_zen4_int_20x8_mt bli_zgemv_t_zen4_int_20x8_mt
    #define K_bli_zgemv_n_zen4_int bli_zgemv_n_zen4_int
    #define K_bli_zgemv_n_zen4_int_20x10 bli_zgemv_n_zen4_int_20x10
    #define K_bli_zgemv_n_zen4_int_20x10_mt bli_zgemv_n_zen4_int_20x10_mt
    #define K_bli_cgemm_zen4_int_32x4_k1_nn bli_cgemm_zen4_int_32x4_k1_nn
    #define K_bli_zgemm_zen4_int_16x4_k1_nn bli_zgemm_zen4_int_16x4_k1_nn
    #define K_bli_dgemm_zen4_int_24x8_k1_nn bli_dgemm_zen4_int_24x8_k1_nn
    #define K_bli_zgemm_zen_int_4x4_k1_nn bli_zgemm_zen_int_4x4_k1_nn
    #define K_bli_dgemm_zen_int_8x6_k1_nn bli_dgemm_zen_int_8x6_k1_nn

    #define AOCL_51

#endif


#ifdef AOCL_51

    #define K_bli_zgemmsup_cd_zen4_asm_12x2m 1
    #define K_bli_zgemmsup_cd_zen4_asm_12x4m 1
    #define K_bli_zgemmsup_cd_zen4_asm_2x2 1
    #define K_bli_zgemmsup_cd_zen4_asm_2x4 1
    #define K_bli_zgemmsup_cd_zen4_asm_4x2 1
    #define K_bli_zgemmsup_cd_zen4_asm_4x4 1
    #define K_bli_zgemmsup_cd_zen4_asm_8x2 1
    #define K_bli_zgemmsup_cd_zen4_asm_8x4 1
    #ifndef K_bli_dgemmsup_cv_zen4_asm_24x8m_new
      #define K_bli_dgemmsup_cv_zen4_asm_24x8m_new bli_dgemmsup_rv_zen4_asm_24x8m_new
    #endif
    #ifndef K_bli_dgemv_t_zen_int
      #define K_bli_dgemv_t_zen_int bli_dgemv_t_zen_int
    #endif
    #ifndef K_bli_dgemv_t_zen_int_16x4
      #define K_bli_dgemv_t_zen_int_16x4 bli_dgemv_t_zen_int_16x4
    #endif
    #ifndef K_bli_dgemv_t_zen_int_16x4_mt
      #define K_bli_dgemv_t_zen_int_16x4_mt bli_dgemv_t_zen_int_16x4_mt
    #endif
    #ifndef K_bli_dgemv_n_zen_int
      #define K_bli_dgemv_n_zen_int bli_dgemv_n_zen_int
    #endif
    #ifndef K_bli_dgemv_n_zen_int_20x4
      #define K_bli_dgemv_n_zen_int_20x4 bli_dgemv_n_zen_int_20x4
    #endif
    #ifndef K_bli_dgemv_n_zen_int_20x4_mt
      #define K_bli_dgemv_n_zen_int_20x4_mt bli_dgemv_n_zen_int_20x4_mt
    #endif
    #ifndef K_bli_dgemv_m_zen_int_20x4
      #define K_bli_dgemv_m_zen_int_20x4 bli_dgemv_m_zen_int_20x4
    #endif
    #ifndef K_bli_sgemv_t_zen_int
      #define K_bli_sgemv_t_zen_int bli_sgemv_t_zen_int
    #endif
    #ifndef K_bli_sgemv_t_zen_int_24x4
      #define K_bli_sgemv_t_zen_int_24x4 bli_sgemv_t_zen_int_24x4
    #endif
    #ifndef K_bli_sgemv_t_zen_int_24x4_mt
      #define K_bli_sgemv_t_zen_int_24x4_mt bli_sgemv_t_zen_int_24x4_mt
    #endif
    #ifndef K_bli_sgemv_n_zen_int
      #define K_bli_sgemv_n_zen_int bli_sgemv_n_zen_int
    #endif
    #ifndef K_bli_sgemv_n_zen_int_40x4
      #define K_bli_sgemv_n_zen_int_40x4 bli_sgemv_n_zen_int_40x4
    #endif
    #ifndef K_bli_sgemv_n_zen_int_40x4_mt
      #define K_bli_sgemv_n_zen_int_40x4_mt bli_sgemv_n_zen_int_40x4_mt
    #endif
    #ifndef K_bli_sgemv_m_zen_int_40x4
      #define K_bli_sgemv_m_zen_int_40x4 bli_sgemv_m_zen_int_40x4
    #endif
    #ifndef K_bli_sgemv_m_zen_int_40x4_mt_Mdiv
      #define K_bli_sgemv_m_zen_int_40x4_mt_Mdiv bli_sgemv_m_zen_int_40x4_mt_Mdiv
    #endif
    #ifndef K_bli_sgemv_m_zen_int_40x4_mt_Ndiv
      #define K_bli_sgemv_m_zen_int_40x4_mt_Ndiv bli_sgemv_m_zen_int_40x4_mt_Ndiv
    #endif
    #ifndef K_bli_cgemv_t_zen_int
      #define K_bli_cgemv_t_zen_int bli_cgemv_t_zen_int
    #endif
    #ifndef K_bli_cgemv_t_zen_int_20x4
      #define K_bli_cgemv_t_zen_int_20x4 bli_cgemv_t_zen_int_20x4
    #endif
    #ifndef K_bli_cgemv_t_zen_int_20x4_mt
      #define K_bli_cgemv_t_zen_int_20x4_mt bli_cgemv_t_zen_int_20x4_mt
    #endif
    #ifndef K_bli_cgemv_n_zen_int
      #define K_bli_cgemv_n_zen_int bli_cgemv_n_zen_int
    #endif
    #ifndef K_bli_cgemv_n_zen_int_20x5
      #define K_bli_cgemv_n_zen_int_20x5 bli_cgemv_n_zen_int_20x5
    #endif
    #ifndef K_bli_cgemv_n_zen_int_20x5_mt
      #define K_bli_cgemv_n_zen_int_20x5_mt bli_cgemv_n_zen_int_20x5_mt
    #endif
    #ifndef K_bli_zgemv_t_zen_int
      #define K_bli_zgemv_t_zen_int bli_zgemv_t_zen_int
    #endif
    #ifndef K_bli_zgemv_t_zen_int_10x4
      #define K_bli_zgemv_t_zen_int_10x4 bli_zgemv_t_zen_int_10x4
    #endif
    #ifndef K_bli_zgemv_t_zen_int_10x4_mt
      #define K_bli_zgemv_t_zen_int_10x4_mt bli_zgemv_t_zen_int_10x4_mt
    #endif
    #ifndef K_bli_zgemv_n_zen_int
      #define K_bli_zgemv_n_zen_int bli_zgemv_n_zen_int
    #endif
    #ifndef K_bli_zgemv_n_zen_int_10x5
      #define K_bli_zgemv_n_zen_int_10x5 bli_zgemv_n_zen_int_10x5
    #endif
    #ifndef K_bli_zgemv_n_zen_int_10x5_mt
      #define K_bli_zgemv_n_zen_int_10x5_mt bli_zgemv_n_zen_int_10x5_mt
    #endif
    #ifndef K_bli_ztrsm_small_zen5
      #define K_bli_ztrsm_small_zen5 bli_ztrsm_small_ZEN5
    #endif
    #ifndef K_bli_dgemv_t_zen4_int
      #define K_bli_dgemv_t_zen4_int bli_dgemv_t_zen4_int
    #endif
    #ifndef K_bli_dgemv_t_zen4_int_32x8
      #define K_bli_dgemv_t_zen4_int_32x8 bli_dgemv_t_zen4_int_32x8
    #endif
    #ifndef K_bli_dgemv_t_zen4_int_32x8_mt
      #define K_bli_dgemv_t_zen4_int_32x8_mt bli_dgemv_t_zen4_int_32x8_mt
    #endif
    #ifndef K_bli_dgemv_n_zen4_int
      #define K_bli_dgemv_n_zen4_int bli_dgemv_n_zen4_int
    #endif
    #ifndef K_bli_dgemv_n_zen4_int_40x8
      #define K_bli_dgemv_n_zen4_int_40x8 bli_dgemv_n_zen4_int_40x8
    #endif
    #ifndef K_bli_dgemv_n_zen4_int_40x8_mt
      #define K_bli_dgemv_n_zen4_int_40x8_mt bli_dgemv_n_zen4_int_40x8_mt
    #endif
    #ifndef K_bli_dgemv_m_zen4_int_40x8
      #define K_bli_dgemv_m_zen4_int_40x8 bli_dgemv_m_zen4_int_40x8
    #endif
    #ifndef K_bli_dgemv_m_zen4_int_40x8_mt_Ndiv
      #define K_bli_dgemv_m_zen4_int_40x8_mt_Ndiv bli_dgemv_m_zen4_int_40x8_mt_Ndiv
    #endif
    #ifndef K_bli_dgemv_m_zen4_int_40x8_mt_Mdiv
      #define K_bli_dgemv_m_zen4_int_40x8_mt_Mdiv bli_dgemv_m_zen4_int_40x8_mt_Mdiv
    #endif
    #ifndef K_bli_sgemv_t_zen4_int
      #define K_bli_sgemv_t_zen4_int bli_sgemv_t_zen4_int
    #endif
    #ifndef K_bli_sgemv_t_zen4_int_48x8
      #define K_bli_sgemv_t_zen4_int_48x8 bli_sgemv_t_zen4_int_48x8
    #endif
    #ifndef K_bli_sgemv_t_zen4_int_48x8_mt
      #define K_bli_sgemv_t_zen4_int_48x8_mt bli_sgemv_t_zen4_int_48x8_mt
    #endif
    #ifndef K_bli_sgemv_n_zen4_int
      #define K_bli_sgemv_n_zen4_int bli_sgemv_n_zen4_int
    #endif
    #ifndef K_bli_sgemv_n_zen4_int_80x8
      #define K_bli_sgemv_n_zen4_int_80x8 bli_sgemv_n_zen4_int_80x8
    #endif
    #ifndef K_bli_sgemv_n_zen4_int_80x8_mt
      #define K_bli_sgemv_n_zen4_int_80x8_mt bli_sgemv_n_zen4_int_80x8_mt
    #endif
    #ifndef K_bli_sgemv_m_zen4_int_80x8
      #define K_bli_sgemv_m_zen4_int_80x8 bli_sgemv_m_zen4_int_80x8
    #endif
    #ifndef K_bli_sgemv_m_zen4_int_80x8_mt_Mdiv
      #define K_bli_sgemv_m_zen4_int_80x8_mt_Mdiv bli_sgemv_m_zen4_int_80x8_mt_Mdiv
    #endif
    #ifndef K_bli_sgemv_m_zen4_int_80x8_mt_Ndiv
      #define K_bli_sgemv_m_zen4_int_80x8_mt_Ndiv bli_sgemv_m_zen4_int_80x8_mt_Ndiv
    #endif
    #ifndef K_bli_cgemv_t_zen4_int
      #define K_bli_cgemv_t_zen4_int bli_cgemv_t_zen4_int
    #endif
    #ifndef K_bli_cgemv_t_zen4_int_40x8
      #define K_bli_cgemv_t_zen4_int_40x8 bli_cgemv_t_zen4_int_40x8
    #endif
    #ifndef K_bli_cgemv_t_zen4_int_40x8_mt
      #define K_bli_cgemv_t_zen4_int_40x8_mt bli_cgemv_t_zen4_int_40x8_mt
    #endif
    #ifndef K_bli_cgemv_n_zen4_int
      #define K_bli_cgemv_n_zen4_int bli_cgemv_n_zen4_int
    #endif
    #ifndef K_bli_cgemv_n_zen4_int_40x10
      #define K_bli_cgemv_n_zen4_int_40x10 bli_cgemv_n_zen4_int_40x10
    #endif
    #ifndef K_bli_cgemv_n_zen4_int_40x10_mt
      #define K_bli_cgemv_n_zen4_int_40x10_mt bli_cgemv_n_zen4_int_40x10_mt
    #endif
    #ifndef K_bli_zgemv_t_zen4_int
      #define K_bli_zgemv_t_zen4_int bli_zgemv_t_zen4_int
    #endif
    #ifndef K_bli_zgemv_t_zen4_int_20x8
      #define K_bli_zgemv_t_zen4_int_20x8 bli_zgemv_t_zen4_int_20x8
    #endif
    #ifndef K_bli_zgemv_t_zen4_int_20x8_mt
      #define K_bli_zgemv_t_zen4_int_20x8_mt bli_zgemv_t_zen4_int_20x8_mt
    #endif
    #ifndef K_bli_zgemv_n_zen4_int
      #define K_bli_zgemv_n_zen4_int bli_zgemv_n_zen4_int
    #endif
    #ifndef K_bli_zgemv_n_zen4_int_20x10
      #define K_bli_zgemv_n_zen4_int_20x10 bli_zgemv_n_zen4_int_20x10
    #endif
    #ifndef K_bli_zgemv_n_zen4_int_20x10_mt
      #define K_bli_zgemv_n_zen4_int_20x10_mt bli_zgemv_n_zen4_int_20x10_mt
    #endif

    #define AOCL_50

#endif


#ifdef AOCL_50

    #define K_bli_caddv_zen_int 1
    #define K_bli_ccopyv_zen_int 1
    #define K_bli_cscal2v_zen_int 1
    #define K_bli_cscalv_zen_int 1
    #ifndef K_bli_cscalv_zen4_int
      #define K_bli_cscalv_zen4_int bli_cscalv_zen_int_avx512
    #endif
    #define K_bli_csetv_zen_int 1
    #define K_bli_daddv_zen_int 1
    #ifndef K_bli_daddv_zen4_int
      #define K_bli_daddv_zen4_int bli_daddv_zen_int_avx512
    #endif
    #ifndef K_bli_daxpbyv_zen4_int
      #define K_bli_daxpbyv_zen4_int bli_daxpbyv_zen_int_avx512
    #endif
    #ifndef K_bli_daxpyf_zen4_int
      #define K_bli_daxpyf_zen4_int bli_daxpyf_zen_int_avx512
    #endif
    #ifndef K_bli_dcopyv_zen4_asm
      #define K_bli_dcopyv_zen4_asm bli_dcopyv_zen4_asm_avx512
    #endif
    #ifndef K_bli_dgemm_zen4_asm_8x24
      #define K_bli_dgemm_zen4_asm_8x24 bli_dgemm_avx512_asm_8x24
    #endif
    #ifndef K_bli_dnorm2fv_zen4_int_unb_var1
      #define K_bli_dnorm2fv_zen4_int_unb_var1 bli_dnorm2fv_unb_var1_avx512
    #endif
    #define K_bli_dscal2v_zen_int 1
    #ifndef K_bli_dscal2v_zen4_int
      #define K_bli_dscal2v_zen4_int bli_dscal2v_zen_int_avx512
    #endif
    #ifndef K_bli_dsetv_zen4_int
      #define K_bli_dsetv_zen4_int bli_dsetv_zen_int_avx512
    #endif
    #define K_bli_saddv_zen_int 1
    #ifndef K_bli_scopyv_zen4_asm
      #define K_bli_scopyv_zen4_asm bli_scopyv_zen4_asm_avx512
    #endif
    #define K_bli_sscal2v_zen_int 1
    #ifndef K_bli_ssetv_zen4_int
      #define K_bli_ssetv_zen4_int bli_ssetv_zen_int_avx512
    #endif
    #define K_bli_zaddv_zen_int 1
    #ifndef K_bli_zaxpyf_zen4_int_8
      #define K_bli_zaxpyf_zen4_int_8 bli_zaxpyf_zen_int_8_avx512
    #endif
    #ifndef K_bli_zaxpyv_zen4_int
      #define K_bli_zaxpyv_zen4_int bli_zaxpyv_zen_int_avx512
    #endif
    #ifndef K_bli_zcopyv_zen4_asm
      #define K_bli_zcopyv_zen4_asm bli_zcopyv_zen4_asm_avx512
    #endif
    #ifndef K_bli_zdotv_zen4_asm
      #define K_bli_zdotv_zen4_asm bli_zdotv_zen4_asm_avx512
    #endif
    #ifndef K_bli_zdotv_zen4_int
      #define K_bli_zdotv_zen4_int bli_zdotv_zen_int_avx512
    #endif
    #ifndef K_bli_zgemm_zen4_int_16x4_k1_nn
      #define K_bli_zgemm_zen4_int_16x4_k1_nn bli_zgemm_16x4_avx512_k1_nn
    #endif
    #ifndef K_bli_zscalv_zen4_int
      #define K_bli_zscalv_zen4_int bli_zscalv_zen_int_avx512
    #endif
    #define K_bli_zsetv_zen_int 1
    #ifndef K_bli_zsetv_zen4_int
      #define K_bli_zsetv_zen4_int bli_zsetv_zen_int_avx512
    #endif

    // In AOCL 4.2 but interface changed at 5.0
    #ifndef K_bli_zgemm_zen_int_4x4_k1_nn
      #define K_bli_zgemm_zen_int_4x4_k1_nn bli_zgemm_4x4_avx2_k1_nn
    #endif

    #define AOCL_42

#endif


#ifdef AOCL_42

    #define E_GEMM_COMPUTE

    #ifndef K_bli_dgemm_zen4_int_24x8_k1_nn
      #define K_bli_dgemm_zen4_int_24x8_k1_nn bli_dgemm_24x8_avx512_k1_nn
    #endif
    #ifndef K_bli_zdscalv_zen4_int
      #define K_bli_zdscalv_zen4_int bli_zdscalv_zen_int_avx512
    #endif
    #define K_bli_zgemm_zen4_asm_4x12 1
    #define K_bli_zgemm_zen_asm_2x6 1

    // In AOCL 4.1 but interface changed at 4.2
    #ifndef K_bli_dgemm_zen_int_8x6_k1_nn
      #define K_bli_dgemm_zen_int_8x6_k1_nn bli_dgemm_8x6_avx2_k1_nn
    #endif

    #define AOCL_41

#endif


#ifdef AOCL_41

    #define K_bli_caxpbyv_zen_int 1
    #ifndef K_bli_caxpyv_zen_int_5
      #define K_bli_caxpyv_zen_int_5 bli_caxpyv_zen_int5
    #endif
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
    #ifndef K_bli_damaxv_zen4_int
      #define K_bli_damaxv_zen4_int bli_damaxv_zen_int_avx512
    #endif
    #define K_bli_daxpbyv_zen_int 1
    #ifndef K_bli_daxpbyv_zen_int_10
      #define K_bli_daxpbyv_zen_int_10 bli_daxpbyv_zen_int10
    #endif
    #define K_bli_daxpyv_zen_int 1
    #ifndef K_bli_daxpyv_zen_int_10
      #define K_bli_daxpyv_zen_int_10 bli_daxpyv_zen_int10
    #endif
    #ifndef K_bli_daxpyv_zen4_int
      #define K_bli_daxpyv_zen4_int bli_daxpyv_zen_int_avx512
    #endif
    #define K_bli_dcopyv_zen_int 1
    #define K_bli_ddotv_zen_int 1
    #ifndef K_bli_ddotv_zen_int_10
      #define K_bli_ddotv_zen_int_10 bli_ddotv_zen_int10
    #endif
    #ifndef K_bli_ddotv_zen4_int
      #define K_bli_ddotv_zen4_int bli_ddotv_zen_int_avx512
    #endif
    #define K_bli_dgemm_haswell_asm_6x8 1
    #define K_bli_dgemm_zen4_asm_32x6 1
    #ifndef K_bli_dgemm_zen4_asm_8x24
      #define K_bli_dgemm_zen4_asm_8x24 bli_dgemm_zen4_asm_8x24
    #endif
    #define K_bli_dgemmsup_rd_haswell_asm_6x8m 1
    #define K_bli_dgemmsup_rd_haswell_asm_6x8n 1
    #define K_bli_dgemmsup_rv_haswell_asm_6x8m 1
    #define K_bli_dgemmsup_rv_haswell_asm_6x8n 1
    #ifndef K_bli_dgemmsup_cv_zen4_asm_24x8m
      #define K_bli_dgemmsup_cv_zen4_asm_24x8m bli_dgemmsup_rv_zen4_asm_24x8m
    #endif
    #ifndef K_bli_dgemmsup_cv_zen5_asm_24x8m
      #define K_bli_dgemmsup_cv_zen5_asm_24x8m bli_dgemmsup_rv_zen5_asm_24x8m
    #endif
    #define K_bli_dgemmtrsm_l_haswell_asm_6x8 1
    #define K_bli_dgemmtrsm_l_zen4_asm_8x24 1
    #define K_bli_dgemmtrsm_u_haswell_asm_6x8 1
    #define K_bli_dgemmtrsm_u_zen4_asm_8x24 1
    #ifndef K_bli_dnorm2fv_zen_int_unb_var1
      #define K_bli_dnorm2fv_zen_int_unb_var1 bli_dnorm2fv_unb_var1_avx2
    #endif
    #define K_bli_dscalv_zen_int 1
    #ifndef K_bli_dscalv_zen_int_10
      #define K_bli_dscalv_zen_int_10 bli_dscalv_zen_int10
    #endif
    #ifndef K_bli_dscalv_zen4_int
      #define K_bli_dscalv_zen4_int bli_dscalv_zen_int_avx512
    #endif
    #define K_bli_dsetv_zen_int 1
    #ifndef K_bli_dswapv_zen_int_8
      #define K_bli_dswapv_zen_int_8 bli_dswapv_zen_int8
    #endif
    #ifndef K_bli_dznorm2fv_zen_int_unb_var1
      #define K_bli_dznorm2fv_zen_int_unb_var1 bli_dznorm2fv_unb_var1_avx2
    #endif
    #define K_bli_samaxv_zen_int 1
    #ifndef K_bli_samaxv_zen4_int
      #define K_bli_samaxv_zen4_int bli_samaxv_zen_int_avx512
    #endif
    #define K_bli_saxpbyv_zen_int 1
    #ifndef K_bli_saxpbyv_zen_int_10
      #define K_bli_saxpbyv_zen_int_10 bli_saxpbyv_zen_int10
    #endif
    #define K_bli_saxpyv_zen_int 1
    #ifndef K_bli_saxpyv_zen_int_10
      #define K_bli_saxpyv_zen_int_10 bli_saxpyv_zen_int10
    #endif
    #ifndef K_bli_saxpyv_zen4_int
      #define K_bli_saxpyv_zen4_int bli_saxpyv_zen_int_avx512
    #endif
    #ifndef K_bli_scnorm2fv_zen_int_unb_var1
      #define K_bli_scnorm2fv_zen_int_unb_var1 bli_scnorm2fv_unb_var1_avx2
    #endif
    #define K_bli_scopyv_zen_int 1
    #define K_bli_sgemm_haswell_asm_6x16 1
    #define K_bli_sgemm_skx_asm_32x12_l2 1
    #define K_bli_sgemmsup_rd_zen_asm_6x16m 1
    #define K_bli_sgemmsup_rd_zen_asm_6x16n 1
    #ifndef K_bli_sgemmsup_rd_zen4_asm_6x64m
      #define K_bli_sgemmsup_rd_zen4_asm_6x64m bli_sgemmsup_rd_zen_asm_6x64m_avx512
    #endif
    #ifndef K_bli_sgemmsup_rd_zen4_asm_6x64n
      #define K_bli_sgemmsup_rd_zen4_asm_6x64n bli_sgemmsup_rd_zen_asm_6x64n_avx512
    #endif
    #define K_bli_sgemmsup_rv_zen_asm_6x16m 1
    #define K_bli_sgemmsup_rv_zen_asm_6x16n 1
    #ifndef K_bli_sgemmsup_rv_zen4_asm_6x64m
      #define K_bli_sgemmsup_rv_zen4_asm_6x64m bli_sgemmsup_rv_zen_asm_6x64m_avx512
    #endif
    #ifndef K_bli_sgemmsup_rv_zen4_asm_6x64n
      #define K_bli_sgemmsup_rv_zen4_asm_6x64n bli_sgemmsup_rv_zen_asm_6x64n_avx512
    #endif
    #define K_bli_sgemmtrsm_l_haswell_asm_6x16 1
    #define K_bli_sgemmtrsm_u_haswell_asm_6x16 1
    #ifndef K_bli_snorm2fv_zen_int_unb_var1
      #define K_bli_snorm2fv_zen_int_unb_var1 bli_snorm2fv_unb_var1_avx2
    #endif
    #define K_bli_sscalv_zen_int 1
    #ifndef K_bli_sscalv_zen_int_10
      #define K_bli_sscalv_zen_int_10 bli_sscalv_zen_int10
    #endif
    #define K_bli_ssetv_zen_int 1
    #ifndef K_bli_sswapv_zen_int_8
      #define K_bli_sswapv_zen_int_8 bli_sswapv_zen_int8
    #endif
    #ifndef K_bli_trsm_small_zen
      #define K_bli_trsm_small_zen bli_trsm_small
    #endif
    #ifndef K_bli_trsm_small_zen4
      #define K_bli_trsm_small_zen4 bli_trsm_small_AVX512
    #endif
    #define K_bli_zaxpbyv_zen_int 1
    #ifndef K_bli_zaxpyv_zen_int_5
      #define K_bli_zaxpyv_zen_int_5 bli_zaxpyv_zen_int5
    #endif
    #define K_bli_zcopyv_zen_int 1
    #ifndef K_bli_zdscalv_zen_int_10
      #define K_bli_zdscalv_zen_int_10 bli_zdscalv_zen_int10
    #endif
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

// If kernels have been removed, we need to undefine them here.

//#ifdef AOCL_51
//    #undef K_bli_dgemm_zen4_asm_8x24
//#endif
