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

#ifndef BLIS_KERNEL_PRE_MACRO_DEFS_H
#define BLIS_KERNEL_PRE_MACRO_DEFS_H

// -- Reference kernel definitions ---------------------------------------------

//
// Level-3
//

// gemm micro-kernels

#define BLIS_SGEMM_UKERNEL_REF           bli_sgemm_ukr_ref
#define BLIS_DGEMM_UKERNEL_REF           bli_dgemm_ukr_ref
#define BLIS_CGEMM_UKERNEL_REF           bli_cgemm_ukr_ref
#define BLIS_ZGEMM_UKERNEL_REF           bli_zgemm_ukr_ref

// gemmtrsm_l micro-kernels

#define BLIS_SGEMMTRSM_L_UKERNEL_REF     bli_sgemmtrsm_l_ukr_ref
#define BLIS_DGEMMTRSM_L_UKERNEL_REF     bli_dgemmtrsm_l_ukr_ref
#define BLIS_CGEMMTRSM_L_UKERNEL_REF     bli_cgemmtrsm_l_ukr_ref
#define BLIS_ZGEMMTRSM_L_UKERNEL_REF     bli_zgemmtrsm_l_ukr_ref

// gemmtrsm_u micro-kernels

#define BLIS_SGEMMTRSM_U_UKERNEL_REF     bli_sgemmtrsm_u_ukr_ref
#define BLIS_DGEMMTRSM_U_UKERNEL_REF     bli_dgemmtrsm_u_ukr_ref
#define BLIS_CGEMMTRSM_U_UKERNEL_REF     bli_cgemmtrsm_u_ukr_ref
#define BLIS_ZGEMMTRSM_U_UKERNEL_REF     bli_zgemmtrsm_u_ukr_ref

// trsm_l micro-kernels

#define BLIS_STRSM_L_UKERNEL_REF         bli_strsm_l_ukr_ref
#define BLIS_DTRSM_L_UKERNEL_REF         bli_dtrsm_l_ukr_ref
#define BLIS_CTRSM_L_UKERNEL_REF         bli_ctrsm_l_ukr_ref
#define BLIS_ZTRSM_L_UKERNEL_REF         bli_ztrsm_l_ukr_ref

// trsm_u micro-kernels

#define BLIS_STRSM_U_UKERNEL_REF         bli_strsm_u_ukr_ref
#define BLIS_DTRSM_U_UKERNEL_REF         bli_dtrsm_u_ukr_ref
#define BLIS_CTRSM_U_UKERNEL_REF         bli_ctrsm_u_ukr_ref
#define BLIS_ZTRSM_U_UKERNEL_REF         bli_ztrsm_u_ukr_ref

//
// Level-1m
//

// packm_2xk kernels

#define BLIS_SPACKM_2XK_KERNEL_REF       bli_spackm_2xk_ref
#define BLIS_DPACKM_2XK_KERNEL_REF       bli_dpackm_2xk_ref
#define BLIS_CPACKM_2XK_KERNEL_REF       bli_cpackm_2xk_ref
#define BLIS_ZPACKM_2XK_KERNEL_REF       bli_zpackm_2xk_ref

// packm_3xk kernels

#define BLIS_SPACKM_3XK_KERNEL_REF       bli_spackm_3xk_ref
#define BLIS_DPACKM_3XK_KERNEL_REF       bli_dpackm_3xk_ref
#define BLIS_CPACKM_3XK_KERNEL_REF       bli_cpackm_3xk_ref
#define BLIS_ZPACKM_3XK_KERNEL_REF       bli_zpackm_3xk_ref

// packm_4xk kernels

#define BLIS_SPACKM_4XK_KERNEL_REF       bli_spackm_4xk_ref
#define BLIS_DPACKM_4XK_KERNEL_REF       bli_dpackm_4xk_ref
#define BLIS_CPACKM_4XK_KERNEL_REF       bli_cpackm_4xk_ref
#define BLIS_ZPACKM_4XK_KERNEL_REF       bli_zpackm_4xk_ref

// packm_6xk kernels

#define BLIS_SPACKM_6XK_KERNEL_REF       bli_spackm_6xk_ref
#define BLIS_DPACKM_6XK_KERNEL_REF       bli_dpackm_6xk_ref
#define BLIS_CPACKM_6XK_KERNEL_REF       bli_cpackm_6xk_ref
#define BLIS_ZPACKM_6XK_KERNEL_REF       bli_zpackm_6xk_ref

// packm_8xk kernels

#define BLIS_SPACKM_8XK_KERNEL_REF       bli_spackm_8xk_ref
#define BLIS_DPACKM_8XK_KERNEL_REF       bli_dpackm_8xk_ref
#define BLIS_CPACKM_8XK_KERNEL_REF       bli_cpackm_8xk_ref
#define BLIS_ZPACKM_8XK_KERNEL_REF       bli_zpackm_8xk_ref

// packm_10xk kernels

#define BLIS_SPACKM_10XK_KERNEL_REF      bli_spackm_10xk_ref
#define BLIS_DPACKM_10XK_KERNEL_REF      bli_dpackm_10xk_ref
#define BLIS_CPACKM_10XK_KERNEL_REF      bli_cpackm_10xk_ref
#define BLIS_ZPACKM_10XK_KERNEL_REF      bli_zpackm_10xk_ref

// packm_12xk kernels

#define BLIS_SPACKM_12XK_KERNEL_REF      bli_spackm_12xk_ref
#define BLIS_DPACKM_12XK_KERNEL_REF      bli_dpackm_12xk_ref
#define BLIS_CPACKM_12XK_KERNEL_REF      bli_cpackm_12xk_ref
#define BLIS_ZPACKM_12XK_KERNEL_REF      bli_zpackm_12xk_ref

// packm_14xk kernels

#define BLIS_SPACKM_14XK_KERNEL_REF      bli_spackm_14xk_ref
#define BLIS_DPACKM_14XK_KERNEL_REF      bli_dpackm_14xk_ref
#define BLIS_CPACKM_14XK_KERNEL_REF      bli_cpackm_14xk_ref
#define BLIS_ZPACKM_14XK_KERNEL_REF      bli_zpackm_14xk_ref

// packm_16xk kernels

#define BLIS_SPACKM_16XK_KERNEL_REF      bli_spackm_16xk_ref
#define BLIS_DPACKM_16XK_KERNEL_REF      bli_dpackm_16xk_ref
#define BLIS_CPACKM_16XK_KERNEL_REF      bli_cpackm_16xk_ref
#define BLIS_ZPACKM_16XK_KERNEL_REF      bli_zpackm_16xk_ref

// packm_24xk kernels

#define BLIS_SPACKM_24XK_KERNEL_REF      bli_spackm_24xk_ref
#define BLIS_DPACKM_24XK_KERNEL_REF      bli_dpackm_24xk_ref
#define BLIS_CPACKM_24XK_KERNEL_REF      bli_cpackm_24xk_ref
#define BLIS_ZPACKM_24XK_KERNEL_REF      bli_zpackm_24xk_ref

// packm_30xk kernels

#define BLIS_SPACKM_30XK_KERNEL_REF      bli_spackm_30xk_ref
#define BLIS_DPACKM_30XK_KERNEL_REF      bli_dpackm_30xk_ref
#define BLIS_CPACKM_30XK_KERNEL_REF      bli_cpackm_30xk_ref
#define BLIS_ZPACKM_30XK_KERNEL_REF      bli_zpackm_30xk_ref

// unpack_2xk kernels

#define BLIS_SUNPACKM_2XK_KERNEL_REF     bli_sunpackm_2xk_ref
#define BLIS_DUNPACKM_2XK_KERNEL_REF     bli_dunpackm_2xk_ref
#define BLIS_CUNPACKM_2XK_KERNEL_REF     bli_cunpackm_2xk_ref
#define BLIS_ZUNPACKM_2XK_KERNEL_REF     bli_zunpackm_2xk_ref

// unpack_4xk kernels

#define BLIS_SUNPACKM_4XK_KERNEL_REF     bli_sunpackm_4xk_ref
#define BLIS_DUNPACKM_4XK_KERNEL_REF     bli_dunpackm_4xk_ref
#define BLIS_CUNPACKM_4XK_KERNEL_REF     bli_cunpackm_4xk_ref
#define BLIS_ZUNPACKM_4XK_KERNEL_REF     bli_zunpackm_4xk_ref

// unpack_6xk kernels

#define BLIS_SUNPACKM_6XK_KERNEL_REF     bli_sunpackm_6xk_ref
#define BLIS_DUNPACKM_6XK_KERNEL_REF     bli_dunpackm_6xk_ref
#define BLIS_CUNPACKM_6XK_KERNEL_REF     bli_cunpackm_6xk_ref
#define BLIS_ZUNPACKM_6XK_KERNEL_REF     bli_zunpackm_6xk_ref

// unpack_8xk kernels

#define BLIS_SUNPACKM_8XK_KERNEL_REF     bli_sunpackm_8xk_ref
#define BLIS_DUNPACKM_8XK_KERNEL_REF     bli_dunpackm_8xk_ref
#define BLIS_CUNPACKM_8XK_KERNEL_REF     bli_cunpackm_8xk_ref
#define BLIS_ZUNPACKM_8XK_KERNEL_REF     bli_zunpackm_8xk_ref

// unpack_10xk kernels

#define BLIS_SUNPACKM_10XK_KERNEL_REF    bli_sunpackm_10xk_ref
#define BLIS_DUNPACKM_10XK_KERNEL_REF    bli_dunpackm_10xk_ref
#define BLIS_CUNPACKM_10XK_KERNEL_REF    bli_cunpackm_10xk_ref
#define BLIS_ZUNPACKM_10XK_KERNEL_REF    bli_zunpackm_10xk_ref

// unpack_12xk kernels

#define BLIS_SUNPACKM_12XK_KERNEL_REF    bli_sunpackm_12xk_ref
#define BLIS_DUNPACKM_12XK_KERNEL_REF    bli_dunpackm_12xk_ref
#define BLIS_CUNPACKM_12XK_KERNEL_REF    bli_cunpackm_12xk_ref
#define BLIS_ZUNPACKM_12XK_KERNEL_REF    bli_zunpackm_12xk_ref

// unpack_14xk kernels

#define BLIS_SUNPACKM_14XK_KERNEL_REF    bli_sunpackm_14xk_ref
#define BLIS_DUNPACKM_14XK_KERNEL_REF    bli_dunpackm_14xk_ref
#define BLIS_CUNPACKM_14XK_KERNEL_REF    bli_cunpackm_14xk_ref
#define BLIS_ZUNPACKM_14XK_KERNEL_REF    bli_zunpackm_14xk_ref

// unpack_16xk kernels

#define BLIS_SUNPACKM_16XK_KERNEL_REF    bli_sunpackm_16xk_ref
#define BLIS_DUNPACKM_16XK_KERNEL_REF    bli_dunpackm_16xk_ref
#define BLIS_CUNPACKM_16XK_KERNEL_REF    bli_cunpackm_16xk_ref
#define BLIS_ZUNPACKM_16XK_KERNEL_REF    bli_zunpackm_16xk_ref

//
// Level-1f
//

// axpy2v kernels

#define BLIS_SAXPY2V_KERNEL_REF          bli_saxpy2v_ref
#define BLIS_DAXPY2V_KERNEL_REF          bli_daxpy2v_ref
#define BLIS_CAXPY2V_KERNEL_REF          bli_caxpy2v_ref
#define BLIS_ZAXPY2V_KERNEL_REF          bli_zaxpy2v_ref

// dotaxpyv kernels

#define BLIS_SDOTAXPYV_KERNEL_REF        bli_sdotaxpyv_ref
#define BLIS_DDOTAXPYV_KERNEL_REF        bli_ddotaxpyv_ref
#define BLIS_CDOTAXPYV_KERNEL_REF        bli_cdotaxpyv_ref
#define BLIS_ZDOTAXPYV_KERNEL_REF        bli_zdotaxpyv_ref

// axpyf kernels

#define BLIS_SAXPYF_KERNEL_REF           bli_saxpyf_ref
#define BLIS_DAXPYF_KERNEL_REF           bli_daxpyf_ref
#define BLIS_CAXPYF_KERNEL_REF           bli_caxpyf_ref
#define BLIS_ZAXPYF_KERNEL_REF           bli_zaxpyf_ref

// dotxf kernels

#define BLIS_SDOTXF_KERNEL_REF           bli_sdotxf_ref
#define BLIS_DDOTXF_KERNEL_REF           bli_ddotxf_ref
#define BLIS_CDOTXF_KERNEL_REF           bli_cdotxf_ref
#define BLIS_ZDOTXF_KERNEL_REF           bli_zdotxf_ref

// dotxaxpyf kernels

//#define BLIS_SDOTXAXPYF_KERNEL_REF       bli_sdotxaxpyf_ref_var1
//#define BLIS_DDOTXAXPYF_KERNEL_REF       bli_ddotxaxpyf_ref_var1
//#define BLIS_CDOTXAXPYF_KERNEL_REF       bli_cdotxaxpyf_ref_var1
//#define BLIS_ZDOTXAXPYF_KERNEL_REF       bli_zdotxaxpyf_ref_var1
#define BLIS_SDOTXAXPYF_KERNEL_REF       bli_sdotxaxpyf_ref_var2
#define BLIS_DDOTXAXPYF_KERNEL_REF       bli_ddotxaxpyf_ref_var2
#define BLIS_CDOTXAXPYF_KERNEL_REF       bli_cdotxaxpyf_ref_var2
#define BLIS_ZDOTXAXPYF_KERNEL_REF       bli_zdotxaxpyf_ref_var2

//
// Level-1v
//

// addv kernels

#define BLIS_SADDV_KERNEL_REF            bli_saddv_ref
#define BLIS_DADDV_KERNEL_REF            bli_daddv_ref
#define BLIS_CADDV_KERNEL_REF            bli_caddv_ref
#define BLIS_ZADDV_KERNEL_REF            bli_zaddv_ref

// amaxv kernels

#define BLIS_SAMAXV_KERNEL_REF           bli_samaxv_ref
#define BLIS_DAMAXV_KERNEL_REF           bli_damaxv_ref
#define BLIS_CAMAXV_KERNEL_REF           bli_camaxv_ref
#define BLIS_ZAMAXV_KERNEL_REF           bli_zamaxv_ref

// axpbyv kernels

#define BLIS_SAXPBYV_KERNEL_REF          bli_saxpbyv_ref
#define BLIS_DAXPBYV_KERNEL_REF          bli_daxpbyv_ref
#define BLIS_CAXPBYV_KERNEL_REF          bli_caxpbyv_ref
#define BLIS_ZAXPBYV_KERNEL_REF          bli_zaxpbyv_ref

// axpyv kernels

#define BLIS_SAXPYV_KERNEL_REF           bli_saxpyv_ref
#define BLIS_DAXPYV_KERNEL_REF           bli_daxpyv_ref
#define BLIS_CAXPYV_KERNEL_REF           bli_caxpyv_ref
#define BLIS_ZAXPYV_KERNEL_REF           bli_zaxpyv_ref

// copyv kernels

#define BLIS_SCOPYV_KERNEL_REF           bli_scopyv_ref
#define BLIS_DCOPYV_KERNEL_REF           bli_dcopyv_ref
#define BLIS_CCOPYV_KERNEL_REF           bli_ccopyv_ref
#define BLIS_ZCOPYV_KERNEL_REF           bli_zcopyv_ref

// dotv kernels

#define BLIS_SDOTV_KERNEL_REF            bli_sdotv_ref
#define BLIS_DDOTV_KERNEL_REF            bli_ddotv_ref
#define BLIS_CDOTV_KERNEL_REF            bli_cdotv_ref
#define BLIS_ZDOTV_KERNEL_REF            bli_zdotv_ref

// dotxv kernels

#define BLIS_SDOTXV_KERNEL_REF           bli_sdotxv_ref
#define BLIS_DDOTXV_KERNEL_REF           bli_ddotxv_ref
#define BLIS_CDOTXV_KERNEL_REF           bli_cdotxv_ref
#define BLIS_ZDOTXV_KERNEL_REF           bli_zdotxv_ref

// invertv kernels

#define BLIS_SINVERTV_KERNEL_REF         bli_sinvertv_ref
#define BLIS_DINVERTV_KERNEL_REF         bli_dinvertv_ref
#define BLIS_CINVERTV_KERNEL_REF         bli_cinvertv_ref
#define BLIS_ZINVERTV_KERNEL_REF         bli_zinvertv_ref

// scal2v kernels

#define BLIS_SSCAL2V_KERNEL_REF          bli_sscal2v_ref
#define BLIS_DSCAL2V_KERNEL_REF          bli_dscal2v_ref
#define BLIS_CSCAL2V_KERNEL_REF          bli_cscal2v_ref
#define BLIS_ZSCAL2V_KERNEL_REF          bli_zscal2v_ref

// scalv kernels

#define BLIS_SSCALV_KERNEL_REF           bli_sscalv_ref
#define BLIS_DSCALV_KERNEL_REF           bli_dscalv_ref
#define BLIS_CSCALV_KERNEL_REF           bli_cscalv_ref
#define BLIS_ZSCALV_KERNEL_REF           bli_zscalv_ref

// setv kernels

#define BLIS_SSETV_KERNEL_REF            bli_ssetv_ref
#define BLIS_DSETV_KERNEL_REF            bli_dsetv_ref
#define BLIS_CSETV_KERNEL_REF            bli_csetv_ref
#define BLIS_ZSETV_KERNEL_REF            bli_zsetv_ref

// subv kernels

#define BLIS_SSUBV_KERNEL_REF            bli_ssubv_ref
#define BLIS_DSUBV_KERNEL_REF            bli_dsubv_ref
#define BLIS_CSUBV_KERNEL_REF            bli_csubv_ref
#define BLIS_ZSUBV_KERNEL_REF            bli_zsubv_ref

// swapv kernels

#define BLIS_SSWAPV_KERNEL_REF           bli_sswapv_ref
#define BLIS_DSWAPV_KERNEL_REF           bli_dswapv_ref
#define BLIS_CSWAPV_KERNEL_REF           bli_cswapv_ref
#define BLIS_ZSWAPV_KERNEL_REF           bli_zswapv_ref

// xpbyv kernels

#define BLIS_SXPBYV_KERNEL_REF           bli_sxpbyv_ref
#define BLIS_DXPBYV_KERNEL_REF           bli_dxpbyv_ref
#define BLIS_CXPBYV_KERNEL_REF           bli_cxpbyv_ref
#define BLIS_ZXPBYV_KERNEL_REF           bli_zxpbyv_ref



#endif 

