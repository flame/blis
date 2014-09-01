/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#ifndef BLIS_KERNEL_H
#define BLIS_KERNEL_H


// -- LEVEL-3 MICRO-KERNEL CONSTANTS -------------------------------------------

// -- Cache blocksizes --

// Constraints:
//
// (1) MC must be a multiple of:
//     (a) MR (for zero-padding purposes)
//     (b) NR (for zero-padding purposes when MR and NR are "swapped")
// (2) NC must be a multiple of
//     (a) NR (for zero-padding purposes)
//     (b) MR (for zero-padding purposes when MR and NR are "swapped")
// (3) KC must be a multiple of
//     (a) MR and
//     (b) NR (for triangular operations such as trmm and trsm).

#define BLIS_DEFAULT_MC_S              128
#define BLIS_DEFAULT_KC_S              256
#define BLIS_DEFAULT_NC_S              2048

#define BLIS_DEFAULT_MC_D              128
#define BLIS_DEFAULT_KC_D              256
#define BLIS_DEFAULT_NC_D              2048

#define BLIS_DEFAULT_MC_C              128
#define BLIS_DEFAULT_KC_C              256
#define BLIS_DEFAULT_NC_C              2048

#define BLIS_DEFAULT_MC_Z              128
#define BLIS_DEFAULT_KC_Z              256
#define BLIS_DEFAULT_NC_Z              2048

// -- Register blocksizes --

#define BLIS_DEFAULT_MR_S              8
#define BLIS_DEFAULT_NR_S              4

#define BLIS_DEFAULT_MR_D              8
#define BLIS_DEFAULT_NR_D              4

#define BLIS_DEFAULT_MR_C              8
#define BLIS_DEFAULT_NR_C              4

#define BLIS_DEFAULT_MR_Z              8
#define BLIS_DEFAULT_NR_Z              4

// NOTE: If the micro-kernel, which is typically unrolled to a factor
// of f, handles leftover edge cases (ie: when k % f > 0) then these
// register blocksizes in the k dimension can be defined to 1.

//#define BLIS_DEFAULT_KR_S              1
//#define BLIS_DEFAULT_KR_D              1
//#define BLIS_DEFAULT_KR_C              1
//#define BLIS_DEFAULT_KR_Z              1

// -- Maximum cache blocksizes (for optimizing edge cases) --

// NOTE: These cache blocksize "extensions" have the same constraints as
// the corresponding default blocksizes above. When these values are
// larger than the default blocksizes, blocksizes used at edge cases are
// enlarged if such an extension would encompass the remaining portion of
// the matrix dimension.

//#define BLIS_MAXIMUM_MC_S              (BLIS_DEFAULT_MC_S + BLIS_DEFAULT_MC_S/4)
//#define BLIS_MAXIMUM_KC_S              (BLIS_DEFAULT_KC_S + BLIS_DEFAULT_KC_S/4)
//#define BLIS_MAXIMUM_NC_S              (BLIS_DEFAULT_NC_S + BLIS_DEFAULT_NC_S/4)

//#define BLIS_MAXIMUM_MC_D              (BLIS_DEFAULT_MC_D + BLIS_DEFAULT_MC_D/4)
//#define BLIS_MAXIMUM_KC_D              (BLIS_DEFAULT_KC_D + BLIS_DEFAULT_KC_D/4)
//#define BLIS_MAXIMUM_NC_D              (BLIS_DEFAULT_NC_D + BLIS_DEFAULT_NC_D/4)

//#define BLIS_MAXIMUM_MC_C              (BLIS_DEFAULT_MC_C + BLIS_DEFAULT_MC_C/4)
//#define BLIS_MAXIMUM_KC_C              (BLIS_DEFAULT_KC_C + BLIS_DEFAULT_KC_C/4)
//#define BLIS_MAXIMUM_NC_C              (BLIS_DEFAULT_NC_C + BLIS_DEFAULT_NC_C/4)

//#define BLIS_MAXIMUM_MC_Z              (BLIS_DEFAULT_MC_Z + BLIS_DEFAULT_MC_Z/4)
//#define BLIS_MAXIMUM_KC_Z              (BLIS_DEFAULT_KC_Z + BLIS_DEFAULT_KC_Z/4)
//#define BLIS_MAXIMUM_NC_Z              (BLIS_DEFAULT_NC_Z + BLIS_DEFAULT_NC_Z/4)

// -- Packing register blocksize (for packed micro-panels) --

// NOTE: These register blocksize "extensions" determine whether the
// leading dimensions used within the packed micro-panels are equal to
// or greater than their corresponding register blocksizes above.

//#define BLIS_PACKDIM_MR_S              (BLIS_DEFAULT_MR_S + ...)
//#define BLIS_PACKDIM_NR_S              (BLIS_DEFAULT_NR_S + ...)

//#define BLIS_PACKDIM_MR_D              (BLIS_DEFAULT_MR_D + ...)
//#define BLIS_PACKDIM_NR_D              (BLIS_DEFAULT_NR_D + ...)

//#define BLIS_PACKDIM_MR_C              (BLIS_DEFAULT_MR_C + ...)
//#define BLIS_PACKDIM_NR_C              (BLIS_DEFAULT_NR_C + ...)

//#define BLIS_PACKDIM_MR_Z              (BLIS_DEFAULT_MR_Z + ...)
//#define BLIS_PACKDIM_NR_Z              (BLIS_DEFAULT_NR_Z + ...)




// -- LEVEL-3 MICRO-KERNELS ---------------------------------------------------

// -- gemm --

#define BLIS_SGEMM_UKERNEL         bli_sgemm_opt_mxn
#define BLIS_DGEMM_UKERNEL         bli_dgemm_opt_mxn
#define BLIS_CGEMM_UKERNEL         bli_cgemm_opt_mxn
#define BLIS_ZGEMM_UKERNEL         bli_zgemm_opt_mxn

// -- trsm-related --

#define BLIS_SGEMMTRSM_L_UKERNEL   bli_sgemmtrsm_l_opt_mxn
#define BLIS_DGEMMTRSM_L_UKERNEL   bli_dgemmtrsm_l_opt_mxn
#define BLIS_CGEMMTRSM_L_UKERNEL   bli_cgemmtrsm_l_opt_mxn
#define BLIS_ZGEMMTRSM_L_UKERNEL   bli_zgemmtrsm_l_opt_mxn

#define BLIS_SGEMMTRSM_U_UKERNEL   bli_sgemmtrsm_u_opt_mxn
#define BLIS_DGEMMTRSM_U_UKERNEL   bli_dgemmtrsm_u_opt_mxn
#define BLIS_CGEMMTRSM_U_UKERNEL   bli_cgemmtrsm_u_opt_mxn
#define BLIS_ZGEMMTRSM_U_UKERNEL   bli_zgemmtrsm_u_opt_mxn

#define BLIS_STRSM_L_UKERNEL       bli_strsm_l_opt_mxn
#define BLIS_DTRSM_L_UKERNEL       bli_dtrsm_l_opt_mxn
#define BLIS_CTRSM_L_UKERNEL       bli_ctrsm_l_opt_mxn
#define BLIS_ZTRSM_L_UKERNEL       bli_ztrsm_l_opt_mxn

#define BLIS_STRSM_U_UKERNEL       bli_strsm_u_opt_mxn
#define BLIS_DTRSM_U_UKERNEL       bli_dtrsm_u_opt_mxn
#define BLIS_CTRSM_U_UKERNEL       bli_ctrsm_u_opt_mxn
#define BLIS_ZTRSM_U_UKERNEL       bli_ztrsm_u_opt_mxn




// -- LEVEL-2 KERNEL CONSTANTS -------------------------------------------------

// NOTE: These values determine high-level cache blocking for level-2
// operations ONLY. So, if gemv is performed with a 2000x2000 matrix A and
// MC = NC = 1000, then a total of four unblocked (or unblocked fused)
// gemv subproblems are called. The blocked algorithms are only useful in
// that they provide the opportunity for packing vectors. (Matrices can also
// be packed here, but this tends to be much too expensive in practice to
// actually employ.)

//#define BLIS_DEFAULT_L2_MC_S           1000
//#define BLIS_DEFAULT_L2_NC_S           1000

//#define BLIS_DEFAULT_L2_MC_D           1000
//#define BLIS_DEFAULT_L2_NC_D           1000

//#define BLIS_DEFAULT_L2_MC_C           1000
//#define BLIS_DEFAULT_L2_NC_C           1000

//#define BLIS_DEFAULT_L2_MC_Z           1000
//#define BLIS_DEFAULT_L2_NC_Z           1000




// -- LEVEL-1F KERNEL CONSTANTS ------------------------------------------------

// -- Default fusing factors for level-1f operations --

//#define BLIS_L1F_FUSE_FAC_S            8
//#define BLIS_L1F_FUSE_FAC_D            4
//#define BLIS_L1F_FUSE_FAC_C            4
//#define BLIS_L1F_FUSE_FAC_Z            2

//#define BLIS_AXPYF_FUSE_FAC_S          BLIS_L1F_FUSE_FAC_S
//#define BLIS_AXPYF_FUSE_FAC_D          BLIS_L1F_FUSE_FAC_D
//#define BLIS_AXPYF_FUSE_FAC_C          BLIS_L1F_FUSE_FAC_C
//#define BLIS_AXPYF_FUSE_FAC_Z          BLIS_L1F_FUSE_FAC_Z

//#define BLIS_DOTXF_FUSE_FAC_S          BLIS_L1F_FUSE_FAC_S
//#define BLIS_DOTXF_FUSE_FAC_D          BLIS_L1F_FUSE_FAC_D
//#define BLIS_DOTXF_FUSE_FAC_C          BLIS_L1F_FUSE_FAC_C
//#define BLIS_DOTXF_FUSE_FAC_Z          BLIS_L1F_FUSE_FAC_Z

//#define BLIS_DOTXAXPYF_FUSE_FAC_S      BLIS_L1F_FUSE_FAC_S
//#define BLIS_DOTXAXPYF_FUSE_FAC_D      BLIS_L1F_FUSE_FAC_D
//#define BLIS_DOTXAXPYF_FUSE_FAC_C      BLIS_L1F_FUSE_FAC_C
//#define BLIS_DOTXAXPYF_FUSE_FAC_Z      BLIS_L1F_FUSE_FAC_Z




// -- LEVEL-1F KERNEL DEFINITIONS ----------------------------------------------

// -- axpy2v --

#define BLIS_SAXPY2V_KERNEL        bli_saxpy2v_opt_var1
#define BLIS_DAXPY2V_KERNEL        bli_daxpy2v_opt_var1
#define BLIS_CAXPY2V_KERNEL        bli_caxpy2v_opt_var1
#define BLIS_ZAXPY2V_KERNEL        bli_zaxpy2v_opt_var1

// -- dotaxpyv --

#define BLIS_SDOTAXPYV_KERNEL      bli_sdotaxpyv_opt_var1
#define BLIS_DDOTAXPYV_KERNEL      bli_ddotaxpyv_opt_var1
#define BLIS_CDOTAXPYV_KERNEL      bli_cdotaxpyv_opt_var1
#define BLIS_ZDOTAXPYV_KERNEL      bli_zdotaxpyv_opt_var1

// -- axpyf --

#define BLIS_SAXPYF_KERNEL         bli_saxpyf_opt_var1
#define BLIS_DAXPYF_KERNEL         bli_daxpyf_opt_var1
#define BLIS_CAXPYF_KERNEL         bli_caxpyf_opt_var1
#define BLIS_ZAXPYF_KERNEL         bli_zaxpyf_opt_var1

// -- dotxf --

#define BLIS_SDOTXF_KERNEL         bli_sdotxf_opt_var1
#define BLIS_DDOTXF_KERNEL         bli_ddotxf_opt_var1
#define BLIS_CDOTXF_KERNEL         bli_cdotxf_opt_var1
#define BLIS_ZDOTXF_KERNEL         bli_zdotxf_opt_var1


// -- dotxaxpyf --

#define BLIS_SDOTXAXPYF_KERNEL     bli_sdotxaxpyf_opt_var1
#define BLIS_DDOTXAXPYF_KERNEL     bli_ddotxaxpyf_opt_var1
#define BLIS_CDOTXAXPYF_KERNEL     bli_cdotxaxpyf_opt_var1
#define BLIS_ZDOTXAXPYF_KERNEL     bli_zdotxaxpyf_opt_var1




// -- LEVEL-1M KERNEL DEFINITIONS ----------------------------------------------

// -- packm --

//#define BLIS_SPACKM_2XK_KERNEL     bli_spackm_ref_2xk
//#define BLIS_DPACKM_2XK_KERNEL     bli_dpackm_ref_2xk
//#define BLIS_CPACKM_2XK_KERNEL     bli_cpackm_ref_2xk
//#define BLIS_ZPACKM_2XK_KERNEL     bli_zpackm_ref_2xk

//#define BLIS_SPACKM_4XK_KERNEL     bli_spackm_ref_4xk
//#define BLIS_DPACKM_4XK_KERNEL     bli_dpackm_ref_4xk
//#define BLIS_CPACKM_4XK_KERNEL     bli_cpackm_ref_4xk
//#define BLIS_ZPACKM_4XK_KERNEL     bli_zpackm_ref_4xk

//#define BLIS_SPACKM_6XK_KERNEL     bli_spackm_ref_6xk
//#define BLIS_DPACKM_6XK_KERNEL     bli_dpackm_ref_6xk
//#define BLIS_CPACKM_6XK_KERNEL     bli_cpackm_ref_6xk
//#define BLIS_ZPACKM_6XK_KERNEL     bli_zpackm_ref_6xk

//#define BLIS_SPACKM_8XK_KERNEL     bli_spackm_ref_8xk
//#define BLIS_DPACKM_8XK_KERNEL     bli_dpackm_ref_8xk
//#define BLIS_CPACKM_8XK_KERNEL     bli_cpackm_ref_8xk
//#define BLIS_ZPACKM_8XK_KERNEL     bli_zpackm_ref_8xk

// ...

// (Commented definitions for 10, 12, 14, and 16 not shown).




// -- LEVEL-1V KERNEL DEFINITIONS ----------------------------------------------

// -- addv --

//#define BLIS_SADDV_KERNEL          bli_saddv_unb_var1
//#define BLIS_DADDV_KERNEL          bli_daddv_unb_var1
//#define BLIS_CADDV_KERNEL          bli_caddv_unb_var1
//#define BLIS_ZADDV_KERNEL          bli_zaddv_unb_var1

// -- axpyv --

#define BLIS_SAXPYV_KERNEL         bli_saxpyv_opt_var1
#define BLIS_DAXPYV_KERNEL         bli_daxpyv_opt_var1
#define BLIS_CAXPYV_KERNEL         bli_caxpyv_opt_var1
#define BLIS_ZAXPYV_KERNEL         bli_zaxpyv_opt_var1

// -- copyv --

//#define BLIS_SCOPYV_KERNEL         bli_scopyv_unb_var1
//#define BLIS_DCOPYV_KERNEL         bli_dcopyv_unb_var1
//#define BLIS_CCOPYV_KERNEL         bli_ccopyv_unb_var1
//#define BLIS_ZCOPYV_KERNEL         bli_zcopyv_unb_var1

// -- dotv --

#define BLIS_SDOTV_KERNEL          bli_sdotv_opt_var1
#define BLIS_DDOTV_KERNEL          bli_ddotv_opt_var1
#define BLIS_CDOTV_KERNEL          bli_cdotv_opt_var1
#define BLIS_ZDOTV_KERNEL          bli_zdotv_opt_var1

// -- dotxv --

//#define BLIS_SDOTXV_KERNEL         bli_sdotxv_unb_var1
//#define BLIS_DDOTXV_KERNEL         bli_ddotxv_unb_var1
//#define BLIS_CDOTXV_KERNEL         bli_cdotxv_unb_var1
//#define BLIS_ZDOTXV_KERNEL         bli_zdotxv_unb_var1

// -- invertv --

//#define BLIS_SINVERTV_KERNEL       bli_sinvertv_unb_var1
//#define BLIS_DINVERTV_KERNEL       bli_dinvertv_unb_var1
//#define BLIS_CINVERTV_KERNEL       bli_cinvertv_unb_var1
//#define BLIS_ZINVERTV_KERNEL       bli_zinvertv_unb_var1

// -- scal2v --

//#define BLIS_SSCAL2V_KERNEL        bli_sscal2v_unb_var1
//#define BLIS_DSCAL2V_KERNEL        bli_dscal2v_unb_var1
//#define BLIS_CSCAL2V_KERNEL        bli_cscal2v_unb_var1
//#define BLIS_ZSCAL2V_KERNEL        bli_zscal2v_unb_var1

// -- scalv --

//#define BLIS_SSCALV_KERNEL         bli_sscalv_unb_var1
//#define BLIS_DSCALV_KERNEL         bli_dscalv_unb_var1
//#define BLIS_CSCALV_KERNEL         bli_cscalv_unb_var1
//#define BLIS_ZSCALV_KERNEL         bli_zscalv_unb_var1

// -- setv --

//#define BLIS_SSETV_KERNEL          bli_ssetv_unb_var1
//#define BLIS_DSETV_KERNEL          bli_dsetv_unb_var1
//#define BLIS_CSETV_KERNEL          bli_csetv_unb_var1
//#define BLIS_ZSETV_KERNEL          bli_zsetv_unb_var1

// -- subv --

//#define BLIS_SSUBV_KERNEL          bli_ssubv_unb_var1
//#define BLIS_DSUBV_KERNEL          bli_dsubv_unb_var1
//#define BLIS_CSUBV_KERNEL          bli_csubv_unb_var1
//#define BLIS_ZSUBV_KERNEL          bli_zsubv_unb_var1

// -- swapv --

//#define BLIS_SSWAPV_KERNEL         bli_sswapv_unb_var1
//#define BLIS_DSWAPV_KERNEL         bli_dswapv_unb_var1
//#define BLIS_CSWAPV_KERNEL         bli_cswapv_unb_var1
//#define BLIS_ZSWAPV_KERNEL         bli_zswapv_unb_var1




#endif

