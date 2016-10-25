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

#ifndef BLIS_KERNEL_PROTOTYPES_H
#define BLIS_KERNEL_PROTOTYPES_H

// Generate prototypes for level-3 micro-kernels.

//
// Level-3
//

#define bli_sgemm_ukr_name       BLIS_SGEMM_UKERNEL
#define bli_dgemm_ukr_name       BLIS_DGEMM_UKERNEL
#define bli_cgemm_ukr_name       BLIS_CGEMM_UKERNEL
#define bli_zgemm_ukr_name       BLIS_ZGEMM_UKERNEL

#define bli_sgemmtrsm_l_ukr_name BLIS_SGEMMTRSM_L_UKERNEL
#define bli_dgemmtrsm_l_ukr_name BLIS_DGEMMTRSM_L_UKERNEL
#define bli_cgemmtrsm_l_ukr_name BLIS_CGEMMTRSM_L_UKERNEL
#define bli_zgemmtrsm_l_ukr_name BLIS_ZGEMMTRSM_L_UKERNEL

#define bli_sgemmtrsm_u_ukr_name BLIS_SGEMMTRSM_U_UKERNEL
#define bli_dgemmtrsm_u_ukr_name BLIS_DGEMMTRSM_U_UKERNEL
#define bli_cgemmtrsm_u_ukr_name BLIS_CGEMMTRSM_U_UKERNEL
#define bli_zgemmtrsm_u_ukr_name BLIS_ZGEMMTRSM_U_UKERNEL

#define bli_strsm_l_ukr_name     BLIS_STRSM_L_UKERNEL
#define bli_dtrsm_l_ukr_name     BLIS_DTRSM_L_UKERNEL
#define bli_ctrsm_l_ukr_name     BLIS_CTRSM_L_UKERNEL
#define bli_ztrsm_l_ukr_name     BLIS_ZTRSM_L_UKERNEL

#define bli_strsm_u_ukr_name     BLIS_STRSM_U_UKERNEL
#define bli_dtrsm_u_ukr_name     BLIS_DTRSM_U_UKERNEL
#define bli_ctrsm_u_ukr_name     BLIS_CTRSM_U_UKERNEL
#define bli_ztrsm_u_ukr_name     BLIS_ZTRSM_U_UKERNEL

#include "bli_l3_ukr.h"

//
// Level-1m
//

#define bli_spackm_2xk_ker_name  BLIS_SPACKM_2XK_KERNEL
#define bli_dpackm_2xk_ker_name  BLIS_DPACKM_2XK_KERNEL
#define bli_cpackm_2xk_ker_name  BLIS_CPACKM_2XK_KERNEL
#define bli_zpackm_2xk_ker_name  BLIS_ZPACKM_2XK_KERNEL

#define bli_spackm_3xk_ker_name  BLIS_SPACKM_3XK_KERNEL
#define bli_dpackm_3xk_ker_name  BLIS_DPACKM_3XK_KERNEL
#define bli_cpackm_3xk_ker_name  BLIS_CPACKM_3XK_KERNEL
#define bli_zpackm_3xk_ker_name  BLIS_ZPACKM_3XK_KERNEL

#define bli_spackm_4xk_ker_name  BLIS_SPACKM_4XK_KERNEL
#define bli_dpackm_4xk_ker_name  BLIS_DPACKM_4XK_KERNEL
#define bli_cpackm_4xk_ker_name  BLIS_CPACKM_4XK_KERNEL
#define bli_zpackm_4xk_ker_name  BLIS_ZPACKM_4XK_KERNEL

#define bli_spackm_6xk_ker_name  BLIS_SPACKM_6XK_KERNEL
#define bli_dpackm_6xk_ker_name  BLIS_DPACKM_6XK_KERNEL
#define bli_cpackm_6xk_ker_name  BLIS_CPACKM_6XK_KERNEL
#define bli_zpackm_6xk_ker_name  BLIS_ZPACKM_6XK_KERNEL

#define bli_spackm_8xk_ker_name  BLIS_SPACKM_8XK_KERNEL
#define bli_dpackm_8xk_ker_name  BLIS_DPACKM_8XK_KERNEL
#define bli_cpackm_8xk_ker_name  BLIS_CPACKM_8XK_KERNEL
#define bli_zpackm_8xk_ker_name  BLIS_ZPACKM_8XK_KERNEL

#define bli_spackm_10xk_ker_name BLIS_SPACKM_10XK_KERNEL
#define bli_dpackm_10xk_ker_name BLIS_DPACKM_10XK_KERNEL
#define bli_cpackm_10xk_ker_name BLIS_CPACKM_10XK_KERNEL
#define bli_zpackm_10xk_ker_name BLIS_ZPACKM_10XK_KERNEL

#define bli_spackm_12xk_ker_name BLIS_SPACKM_12XK_KERNEL
#define bli_dpackm_12xk_ker_name BLIS_DPACKM_12XK_KERNEL
#define bli_cpackm_12xk_ker_name BLIS_CPACKM_12XK_KERNEL
#define bli_zpackm_12xk_ker_name BLIS_ZPACKM_12XK_KERNEL

#define bli_spackm_14xk_ker_name BLIS_SPACKM_14XK_KERNEL
#define bli_dpackm_14xk_ker_name BLIS_DPACKM_14XK_KERNEL
#define bli_cpackm_14xk_ker_name BLIS_CPACKM_14XK_KERNEL
#define bli_zpackm_14xk_ker_name BLIS_ZPACKM_14XK_KERNEL

#define bli_spackm_16xk_ker_name BLIS_SPACKM_16XK_KERNEL
#define bli_dpackm_16xk_ker_name BLIS_DPACKM_16XK_KERNEL
#define bli_cpackm_16xk_ker_name BLIS_CPACKM_16XK_KERNEL
#define bli_zpackm_16xk_ker_name BLIS_ZPACKM_16XK_KERNEL

#define bli_spackm_24xk_ker_name BLIS_SPACKM_24XK_KERNEL
#define bli_dpackm_24xk_ker_name BLIS_DPACKM_24XK_KERNEL
#define bli_cpackm_24xk_ker_name BLIS_CPACKM_24XK_KERNEL
#define bli_zpackm_24xk_ker_name BLIS_ZPACKM_24XK_KERNEL

#define bli_spackm_30xk_ker_name BLIS_SPACKM_30XK_KERNEL
#define bli_dpackm_30xk_ker_name BLIS_DPACKM_30XK_KERNEL
#define bli_cpackm_30xk_ker_name BLIS_CPACKM_30XK_KERNEL
#define bli_zpackm_30xk_ker_name BLIS_ZPACKM_30XK_KERNEL

#include "bli_l1m_ker.h"

//
// Level-1f
//

#define bli_saxpy2v_ker_name    BLIS_SAXPY2V_KERNEL
#define bli_daxpy2v_ker_name    BLIS_DAXPY2V_KERNEL
#define bli_caxpy2v_ker_name    BLIS_CAXPY2V_KERNEL
#define bli_zaxpy2v_ker_name    BLIS_ZAXPY2V_KERNEL

#define bli_sdotaxpyv_ker_name  BLIS_SDOTAXPYV_KERNEL
#define bli_ddotaxpyv_ker_name  BLIS_DDOTAXPYV_KERNEL
#define bli_cdotaxpyv_ker_name  BLIS_CDOTAXPYV_KERNEL
#define bli_zdotaxpyv_ker_name  BLIS_ZDOTAXPYV_KERNEL

#define bli_sdotxf_ker_name     BLIS_SDOTXF_KERNEL
#define bli_ddotxf_ker_name     BLIS_DDOTXF_KERNEL
#define bli_cdotxf_ker_name     BLIS_CDOTXF_KERNEL
#define bli_zdotxf_ker_name     BLIS_ZDOTXF_KERNEL

#define bli_saxpyf_ker_name     BLIS_SAXPYF_KERNEL
#define bli_daxpyf_ker_name     BLIS_DAXPYF_KERNEL
#define bli_caxpyf_ker_name     BLIS_CAXPYF_KERNEL
#define bli_zaxpyf_ker_name     BLIS_ZAXPYF_KERNEL

#define bli_sdotxaxpyf_ker_name BLIS_SDOTXAXPYF_KERNEL
#define bli_ddotxaxpyf_ker_name BLIS_DDOTXAXPYF_KERNEL
#define bli_cdotxaxpyf_ker_name BLIS_CDOTXAXPYF_KERNEL
#define bli_zdotxaxpyf_ker_name BLIS_ZDOTXAXPYF_KERNEL

#include "bli_l1f_ker.h"

//
// Level-1v
//

#define bli_saddv_ker_name      BLIS_SADDV_KERNEL
#define bli_daddv_ker_name      BLIS_DADDV_KERNEL
#define bli_caddv_ker_name      BLIS_CADDV_KERNEL
#define bli_zaddv_ker_name      BLIS_ZADDV_KERNEL

#define bli_samaxv_ker_name     BLIS_SAMAXV_KERNEL
#define bli_damaxv_ker_name     BLIS_DAMAXV_KERNEL
#define bli_camaxv_ker_name     BLIS_CAMAXV_KERNEL
#define bli_zamaxv_ker_name     BLIS_ZAMAXV_KERNEL

#define bli_saxpbyv_ker_name    BLIS_SAXPBYV_KERNEL
#define bli_daxpbyv_ker_name    BLIS_DAXPBYV_KERNEL
#define bli_caxpbyv_ker_name    BLIS_CAXPBYV_KERNEL
#define bli_zaxpbyv_ker_name    BLIS_ZAXPBYV_KERNEL

#define bli_saxpyv_ker_name     BLIS_SAXPYV_KERNEL
#define bli_daxpyv_ker_name     BLIS_DAXPYV_KERNEL
#define bli_caxpyv_ker_name     BLIS_CAXPYV_KERNEL
#define bli_zaxpyv_ker_name     BLIS_ZAXPYV_KERNEL

#define bli_scopyv_ker_name     BLIS_SCOPYV_KERNEL
#define bli_dcopyv_ker_name     BLIS_DCOPYV_KERNEL
#define bli_ccopyv_ker_name     BLIS_CCOPYV_KERNEL
#define bli_zcopyv_ker_name     BLIS_ZCOPYV_KERNEL

#define bli_sdotv_ker_name      BLIS_SDOTV_KERNEL
#define bli_ddotv_ker_name      BLIS_DDOTV_KERNEL
#define bli_cdotv_ker_name      BLIS_CDOTV_KERNEL
#define bli_zdotv_ker_name      BLIS_ZDOTV_KERNEL

#define bli_sdotxv_ker_name     BLIS_SDOTXV_KERNEL
#define bli_ddotxv_ker_name     BLIS_DDOTXV_KERNEL
#define bli_cdotxv_ker_name     BLIS_CDOTXV_KERNEL
#define bli_zdotxv_ker_name     BLIS_ZDOTXV_KERNEL

#define bli_sinvertv_ker_name   BLIS_SINVERTV_KERNEL
#define bli_dinvertv_ker_name   BLIS_DINVERTV_KERNEL
#define bli_cinvertv_ker_name   BLIS_CINVERTV_KERNEL
#define bli_zinvertv_ker_name   BLIS_ZINVERTV_KERNEL

#define bli_sscalv_ker_name     BLIS_SSCALV_KERNEL
#define bli_dscalv_ker_name     BLIS_DSCALV_KERNEL
#define bli_cscalv_ker_name     BLIS_CSCALV_KERNEL
#define bli_zscalv_ker_name     BLIS_ZSCALV_KERNEL

#define bli_sscal2v_ker_name    BLIS_SSCAL2V_KERNEL
#define bli_dscal2v_ker_name    BLIS_DSCAL2V_KERNEL
#define bli_cscal2v_ker_name    BLIS_CSCAL2V_KERNEL
#define bli_zscal2v_ker_name    BLIS_ZSCAL2V_KERNEL

#define bli_ssetv_ker_name      BLIS_SSETV_KERNEL
#define bli_dsetv_ker_name      BLIS_DSETV_KERNEL
#define bli_csetv_ker_name      BLIS_CSETV_KERNEL
#define bli_zsetv_ker_name      BLIS_ZSETV_KERNEL

#define bli_ssubv_ker_name      BLIS_SSUBV_KERNEL
#define bli_dsubv_ker_name      BLIS_DSUBV_KERNEL
#define bli_csubv_ker_name      BLIS_CSUBV_KERNEL
#define bli_zsubv_ker_name      BLIS_ZSUBV_KERNEL

#define bli_sswapv_ker_name     BLIS_SSWAPV_KERNEL
#define bli_dswapv_ker_name     BLIS_DSWAPV_KERNEL
#define bli_cswapv_ker_name     BLIS_CSWAPV_KERNEL
#define bli_zswapv_ker_name     BLIS_ZSWAPV_KERNEL

#define bli_sxpbyv_ker_name     BLIS_SXPBYV_KERNEL
#define bli_dxpbyv_ker_name     BLIS_DXPBYV_KERNEL
#define bli_cxpbyv_ker_name     BLIS_CXPBYV_KERNEL
#define bli_zxpbyv_ker_name     BLIS_ZXPBYV_KERNEL

#include "bli_l1v_ker.h"


#endif

