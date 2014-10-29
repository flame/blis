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
// Level-3 4m
//

// gemm4m micro-kernels

#define BLIS_CGEMM4M_UKERNEL_REF         bli_cgemm4m_ukr_ref
#define BLIS_ZGEMM4M_UKERNEL_REF         bli_zgemm4m_ukr_ref

// gemmtrsm4m_l micro-kernels

#define BLIS_CGEMMTRSM4M_L_UKERNEL_REF   bli_cgemmtrsm4m_l_ukr_ref
#define BLIS_ZGEMMTRSM4M_L_UKERNEL_REF   bli_zgemmtrsm4m_l_ukr_ref

// gemmtrsm4m_u micro-kernels

#define BLIS_CGEMMTRSM4M_U_UKERNEL_REF   bli_cgemmtrsm4m_u_ukr_ref
#define BLIS_ZGEMMTRSM4M_U_UKERNEL_REF   bli_zgemmtrsm4m_u_ukr_ref

// trsm4m_l micro-kernels

#define BLIS_CTRSM4M_L_UKERNEL_REF       bli_ctrsm4m_l_ukr_ref
#define BLIS_ZTRSM4M_L_UKERNEL_REF       bli_ztrsm4m_l_ukr_ref

// trsm4m_u micro-kernels

#define BLIS_CTRSM4M_U_UKERNEL_REF       bli_ctrsm4m_u_ukr_ref
#define BLIS_ZTRSM4M_U_UKERNEL_REF       bli_ztrsm4m_u_ukr_ref

//
// Level-3 3m
//

// gemm3m micro-kernels

#define BLIS_CGEMM3M_UKERNEL_REF         bli_cgemm3m_ukr_ref
#define BLIS_ZGEMM3M_UKERNEL_REF         bli_zgemm3m_ukr_ref

// gemmtrsm3m_l micro-kernels

#define BLIS_CGEMMTRSM3M_L_UKERNEL_REF   bli_cgemmtrsm3m_l_ukr_ref
#define BLIS_ZGEMMTRSM3M_L_UKERNEL_REF   bli_zgemmtrsm3m_l_ukr_ref

// gemmtrsm3m_u micro-kernels

#define BLIS_CGEMMTRSM3M_U_UKERNEL_REF   bli_cgemmtrsm3m_u_ukr_ref
#define BLIS_ZGEMMTRSM3M_U_UKERNEL_REF   bli_zgemmtrsm3m_u_ukr_ref

// trsm3m_l micro-kernels

#define BLIS_CTRSM3M_L_UKERNEL_REF       bli_ctrsm3m_l_ukr_ref
#define BLIS_ZTRSM3M_L_UKERNEL_REF       bli_ztrsm3m_l_ukr_ref

// trsm3m_u micro-kernels

#define BLIS_CTRSM3M_U_UKERNEL_REF       bli_ctrsm3m_u_ukr_ref
#define BLIS_ZTRSM3M_U_UKERNEL_REF       bli_ztrsm3m_u_ukr_ref

//
// Level-3 4mh
//

// gemm4mh micro-kernels

#define BLIS_CGEMM4MH_UKERNEL_REF        bli_cgemm4mh_ukr_ref
#define BLIS_ZGEMM4MH_UKERNEL_REF        bli_zgemm4mh_ukr_ref

//
//
// Level-3 3mh
//

// gemm3mh micro-kernels

#define BLIS_CGEMM3MH_UKERNEL_REF        bli_cgemm3mh_ukr_ref
#define BLIS_ZGEMM3MH_UKERNEL_REF        bli_zgemm3mh_ukr_ref

// Level-1m
//

// packm_2xk kernels

#define BLIS_SPACKM_2XK_KERNEL_REF       bli_spackm_ref_2xk
#define BLIS_DPACKM_2XK_KERNEL_REF       bli_dpackm_ref_2xk
#define BLIS_CPACKM_2XK_KERNEL_REF       bli_cpackm_ref_2xk
#define BLIS_ZPACKM_2XK_KERNEL_REF       bli_zpackm_ref_2xk

// packm_3xk kernels

#define BLIS_SPACKM_3XK_KERNEL_REF       bli_spackm_ref_3xk
#define BLIS_DPACKM_3XK_KERNEL_REF       bli_dpackm_ref_3xk
#define BLIS_CPACKM_3XK_KERNEL_REF       bli_cpackm_ref_3xk
#define BLIS_ZPACKM_3XK_KERNEL_REF       bli_zpackm_ref_3xk

// packm_4xk kernels

#define BLIS_SPACKM_4XK_KERNEL_REF       bli_spackm_ref_4xk
#define BLIS_DPACKM_4XK_KERNEL_REF       bli_dpackm_ref_4xk
#define BLIS_CPACKM_4XK_KERNEL_REF       bli_cpackm_ref_4xk
#define BLIS_ZPACKM_4XK_KERNEL_REF       bli_zpackm_ref_4xk

// packm_6xk kernels

#define BLIS_SPACKM_6XK_KERNEL_REF       bli_spackm_ref_6xk
#define BLIS_DPACKM_6XK_KERNEL_REF       bli_dpackm_ref_6xk
#define BLIS_CPACKM_6XK_KERNEL_REF       bli_cpackm_ref_6xk
#define BLIS_ZPACKM_6XK_KERNEL_REF       bli_zpackm_ref_6xk

// packm_8xk kernels

#define BLIS_SPACKM_8XK_KERNEL_REF       bli_spackm_ref_8xk
#define BLIS_DPACKM_8XK_KERNEL_REF       bli_dpackm_ref_8xk
#define BLIS_CPACKM_8XK_KERNEL_REF       bli_cpackm_ref_8xk
#define BLIS_ZPACKM_8XK_KERNEL_REF       bli_zpackm_ref_8xk

// packm_10xk kernels

#define BLIS_SPACKM_10XK_KERNEL_REF      bli_spackm_ref_10xk
#define BLIS_DPACKM_10XK_KERNEL_REF      bli_dpackm_ref_10xk
#define BLIS_CPACKM_10XK_KERNEL_REF      bli_cpackm_ref_10xk
#define BLIS_ZPACKM_10XK_KERNEL_REF      bli_zpackm_ref_10xk

// packm_12xk kernels

#define BLIS_SPACKM_12XK_KERNEL_REF      bli_spackm_ref_12xk
#define BLIS_DPACKM_12XK_KERNEL_REF      bli_dpackm_ref_12xk
#define BLIS_CPACKM_12XK_KERNEL_REF      bli_cpackm_ref_12xk
#define BLIS_ZPACKM_12XK_KERNEL_REF      bli_zpackm_ref_12xk

// packm_14xk kernels

#define BLIS_SPACKM_14XK_KERNEL_REF      bli_spackm_ref_14xk
#define BLIS_DPACKM_14XK_KERNEL_REF      bli_dpackm_ref_14xk
#define BLIS_CPACKM_14XK_KERNEL_REF      bli_cpackm_ref_14xk
#define BLIS_ZPACKM_14XK_KERNEL_REF      bli_zpackm_ref_14xk

// packm_16xk kernels

#define BLIS_SPACKM_16XK_KERNEL_REF      bli_spackm_ref_16xk
#define BLIS_DPACKM_16XK_KERNEL_REF      bli_dpackm_ref_16xk
#define BLIS_CPACKM_16XK_KERNEL_REF      bli_cpackm_ref_16xk
#define BLIS_ZPACKM_16XK_KERNEL_REF      bli_zpackm_ref_16xk

// packm_30xk kernels

#define BLIS_SPACKM_30XK_KERNEL_REF      bli_spackm_ref_30xk
#define BLIS_DPACKM_30XK_KERNEL_REF      bli_dpackm_ref_30xk
#define BLIS_CPACKM_30XK_KERNEL_REF      bli_cpackm_ref_30xk
#define BLIS_ZPACKM_30XK_KERNEL_REF      bli_zpackm_ref_30xk

// packm_2xk_4m kernels

#define BLIS_CPACKM_2XK_4M_KERNEL_REF    bli_cpackm_ref_2xk_4m
#define BLIS_ZPACKM_2XK_4M_KERNEL_REF    bli_zpackm_ref_2xk_4m

// packm_4xk_4m kernels

#define BLIS_CPACKM_4XK_4M_KERNEL_REF    bli_cpackm_ref_4xk_4m
#define BLIS_ZPACKM_4XK_4M_KERNEL_REF    bli_zpackm_ref_4xk_4m

// packm_6xk_4m kernels

#define BLIS_CPACKM_6XK_4M_KERNEL_REF    bli_cpackm_ref_6xk_4m
#define BLIS_ZPACKM_6XK_4M_KERNEL_REF    bli_zpackm_ref_6xk_4m

// packm_8xk_4m kernels

#define BLIS_CPACKM_8XK_4M_KERNEL_REF    bli_cpackm_ref_8xk_4m
#define BLIS_ZPACKM_8XK_4M_KERNEL_REF    bli_zpackm_ref_8xk_4m

// packm_10xk_4m kernels

#define BLIS_CPACKM_10XK_4M_KERNEL_REF   bli_cpackm_ref_10xk_4m
#define BLIS_ZPACKM_10XK_4M_KERNEL_REF   bli_zpackm_ref_10xk_4m

// packm_12xk_4m kernels

#define BLIS_CPACKM_12XK_4M_KERNEL_REF   bli_cpackm_ref_12xk_4m
#define BLIS_ZPACKM_12XK_4M_KERNEL_REF   bli_zpackm_ref_12xk_4m

// packm_14xk_4m kernels

#define BLIS_CPACKM_14XK_4M_KERNEL_REF   bli_cpackm_ref_14xk_4m
#define BLIS_ZPACKM_14XK_4M_KERNEL_REF   bli_zpackm_ref_14xk_4m

// packm_16xk_4m kernels

#define BLIS_CPACKM_16XK_4M_KERNEL_REF   bli_cpackm_ref_16xk_4m
#define BLIS_ZPACKM_16XK_4M_KERNEL_REF   bli_zpackm_ref_16xk_4m

// packm_30xk_4m kernels

#define BLIS_CPACKM_30XK_4M_KERNEL_REF   bli_cpackm_ref_30xk_4m
#define BLIS_ZPACKM_30XK_4M_KERNEL_REF   bli_zpackm_ref_30xk_4m

// packm_2xk_3m kernels

#define BLIS_CPACKM_2XK_3M_KERNEL_REF    bli_cpackm_ref_2xk_3m
#define BLIS_ZPACKM_2XK_3M_KERNEL_REF    bli_zpackm_ref_2xk_3m

// packm_4xk_3m kernels

#define BLIS_CPACKM_4XK_3M_KERNEL_REF    bli_cpackm_ref_4xk_3m
#define BLIS_ZPACKM_4XK_3M_KERNEL_REF    bli_zpackm_ref_4xk_3m

// packm_6xk_3m kernels

#define BLIS_CPACKM_6XK_3M_KERNEL_REF    bli_cpackm_ref_6xk_3m
#define BLIS_ZPACKM_6XK_3M_KERNEL_REF    bli_zpackm_ref_6xk_3m

// packm_8xk_3m kernels

#define BLIS_CPACKM_8XK_3M_KERNEL_REF    bli_cpackm_ref_8xk_3m
#define BLIS_ZPACKM_8XK_3M_KERNEL_REF    bli_zpackm_ref_8xk_3m

// packm_10xk_3m kernels

#define BLIS_CPACKM_10XK_3M_KERNEL_REF   bli_cpackm_ref_10xk_3m
#define BLIS_ZPACKM_10XK_3M_KERNEL_REF   bli_zpackm_ref_10xk_3m

// packm_12xk_3m kernels

#define BLIS_CPACKM_12XK_3M_KERNEL_REF   bli_cpackm_ref_12xk_3m
#define BLIS_ZPACKM_12XK_3M_KERNEL_REF   bli_zpackm_ref_12xk_3m

// packm_14xk_3m kernels

#define BLIS_CPACKM_14XK_3M_KERNEL_REF   bli_cpackm_ref_14xk_3m
#define BLIS_ZPACKM_14XK_3M_KERNEL_REF   bli_zpackm_ref_14xk_3m

// packm_16xk_3m kernels

#define BLIS_CPACKM_16XK_3M_KERNEL_REF   bli_cpackm_ref_16xk_3m
#define BLIS_ZPACKM_16XK_3M_KERNEL_REF   bli_zpackm_ref_16xk_3m

// packm_30xk_3m kernels

#define BLIS_CPACKM_30XK_3M_KERNEL_REF   bli_cpackm_ref_30xk_3m
#define BLIS_ZPACKM_30XK_3M_KERNEL_REF   bli_zpackm_ref_30xk_3m

// packm_2xk_rih kernels

#define BLIS_CPACKM_2XK_RIH_KERNEL_REF   bli_cpackm_ref_2xk_rih
#define BLIS_ZPACKM_2XK_RIH_KERNEL_REF   bli_zpackm_ref_2xk_rih

// packm_4xk_rih kernels

#define BLIS_CPACKM_4XK_RIH_KERNEL_REF   bli_cpackm_ref_4xk_rih
#define BLIS_ZPACKM_4XK_RIH_KERNEL_REF   bli_zpackm_ref_4xk_rih

// packm_6xk_rih kernels

#define BLIS_CPACKM_6XK_RIH_KERNEL_REF   bli_cpackm_ref_6xk_rih
#define BLIS_ZPACKM_6XK_RIH_KERNEL_REF   bli_zpackm_ref_6xk_rih

// packm_8xk_rih kernels

#define BLIS_CPACKM_8XK_RIH_KERNEL_REF   bli_cpackm_ref_8xk_rih
#define BLIS_ZPACKM_8XK_RIH_KERNEL_REF   bli_zpackm_ref_8xk_rih

// packm_10xk_rih kernels

#define BLIS_CPACKM_10XK_RIH_KERNEL_REF  bli_cpackm_ref_10xk_rih
#define BLIS_ZPACKM_10XK_RIH_KERNEL_REF  bli_zpackm_ref_10xk_rih

// packm_12xk_rih kernels

#define BLIS_CPACKM_12XK_RIH_KERNEL_REF  bli_cpackm_ref_12xk_rih
#define BLIS_ZPACKM_12XK_RIH_KERNEL_REF  bli_zpackm_ref_12xk_rih

// packm_14xk_rih kernels

#define BLIS_CPACKM_14XK_RIH_KERNEL_REF  bli_cpackm_ref_14xk_rih
#define BLIS_ZPACKM_14XK_RIH_KERNEL_REF  bli_zpackm_ref_14xk_rih

// packm_16xk_rih kernels

#define BLIS_CPACKM_16XK_RIH_KERNEL_REF  bli_cpackm_ref_16xk_rih
#define BLIS_ZPACKM_16XK_RIH_KERNEL_REF  bli_zpackm_ref_16xk_rih

// packm_30xk_rih kernels

#define BLIS_CPACKM_30XK_RIH_KERNEL_REF  bli_cpackm_ref_30xk_rih
#define BLIS_ZPACKM_30XK_RIH_KERNEL_REF  bli_zpackm_ref_30xk_rih

// unpack_2xk kernels

#define BLIS_SUNPACKM_2XK_KERNEL_REF     bli_sunpackm_ref_2xk
#define BLIS_DUNPACKM_2XK_KERNEL_REF     bli_dunpackm_ref_2xk
#define BLIS_CUNPACKM_2XK_KERNEL_REF     bli_cunpackm_ref_2xk
#define BLIS_ZUNPACKM_2XK_KERNEL_REF     bli_zunpackm_ref_2xk

// unpack_4xk kernels

#define BLIS_SUNPACKM_4XK_KERNEL_REF     bli_sunpackm_ref_4xk
#define BLIS_DUNPACKM_4XK_KERNEL_REF     bli_dunpackm_ref_4xk
#define BLIS_CUNPACKM_4XK_KERNEL_REF     bli_cunpackm_ref_4xk
#define BLIS_ZUNPACKM_4XK_KERNEL_REF     bli_zunpackm_ref_4xk

// unpack_6xk kernels

#define BLIS_SUNPACKM_6XK_KERNEL_REF     bli_sunpackm_ref_6xk
#define BLIS_DUNPACKM_6XK_KERNEL_REF     bli_dunpackm_ref_6xk
#define BLIS_CUNPACKM_6XK_KERNEL_REF     bli_cunpackm_ref_6xk
#define BLIS_ZUNPACKM_6XK_KERNEL_REF     bli_zunpackm_ref_6xk

// unpack_8xk kernels

#define BLIS_SUNPACKM_8XK_KERNEL_REF     bli_sunpackm_ref_8xk
#define BLIS_DUNPACKM_8XK_KERNEL_REF     bli_dunpackm_ref_8xk
#define BLIS_CUNPACKM_8XK_KERNEL_REF     bli_cunpackm_ref_8xk
#define BLIS_ZUNPACKM_8XK_KERNEL_REF     bli_zunpackm_ref_8xk

// unpack_10xk kernels

#define BLIS_SUNPACKM_10XK_KERNEL_REF    bli_sunpackm_ref_10xk
#define BLIS_DUNPACKM_10XK_KERNEL_REF    bli_dunpackm_ref_10xk
#define BLIS_CUNPACKM_10XK_KERNEL_REF    bli_cunpackm_ref_10xk
#define BLIS_ZUNPACKM_10XK_KERNEL_REF    bli_zunpackm_ref_10xk

// unpack_12xk kernels

#define BLIS_SUNPACKM_12XK_KERNEL_REF    bli_sunpackm_ref_12xk
#define BLIS_DUNPACKM_12XK_KERNEL_REF    bli_dunpackm_ref_12xk
#define BLIS_CUNPACKM_12XK_KERNEL_REF    bli_cunpackm_ref_12xk
#define BLIS_ZUNPACKM_12XK_KERNEL_REF    bli_zunpackm_ref_12xk

// unpack_14xk kernels

#define BLIS_SUNPACKM_14XK_KERNEL_REF    bli_sunpackm_ref_14xk
#define BLIS_DUNPACKM_14XK_KERNEL_REF    bli_dunpackm_ref_14xk
#define BLIS_CUNPACKM_14XK_KERNEL_REF    bli_cunpackm_ref_14xk
#define BLIS_ZUNPACKM_14XK_KERNEL_REF    bli_zunpackm_ref_14xk

// unpack_16xk kernels

#define BLIS_SUNPACKM_16XK_KERNEL_REF    bli_sunpackm_ref_16xk
#define BLIS_DUNPACKM_16XK_KERNEL_REF    bli_dunpackm_ref_16xk
#define BLIS_CUNPACKM_16XK_KERNEL_REF    bli_cunpackm_ref_16xk
#define BLIS_ZUNPACKM_16XK_KERNEL_REF    bli_zunpackm_ref_16xk

//
// Level-1f
//

// axpy2v kernels

#define BLIS_SAXPY2V_KERNEL_REF          bli_sssaxpy2v_ref
#define BLIS_DAXPY2V_KERNEL_REF          bli_dddaxpy2v_ref
#define BLIS_CAXPY2V_KERNEL_REF          bli_cccaxpy2v_ref
#define BLIS_ZAXPY2V_KERNEL_REF          bli_zzzaxpy2v_ref

// dotaxpyv kernels

#define BLIS_SDOTAXPYV_KERNEL_REF        bli_sssdotaxpyv_ref
#define BLIS_DDOTAXPYV_KERNEL_REF        bli_ddddotaxpyv_ref
#define BLIS_CDOTAXPYV_KERNEL_REF        bli_cccdotaxpyv_ref
#define BLIS_ZDOTAXPYV_KERNEL_REF        bli_zzzdotaxpyv_ref

// axpyf kernels

#define BLIS_SAXPYF_KERNEL_REF           bli_sssaxpyf_ref
#define BLIS_DAXPYF_KERNEL_REF           bli_dddaxpyf_ref
#define BLIS_CAXPYF_KERNEL_REF           bli_cccaxpyf_ref
#define BLIS_ZAXPYF_KERNEL_REF           bli_zzzaxpyf_ref

// dotxf kernels

#define BLIS_SDOTXF_KERNEL_REF           bli_sssdotxf_ref
#define BLIS_DDOTXF_KERNEL_REF           bli_ddddotxf_ref
#define BLIS_CDOTXF_KERNEL_REF           bli_cccdotxf_ref
#define BLIS_ZDOTXF_KERNEL_REF           bli_zzzdotxf_ref

// dotxaxpyf kernels

//#define BLIS_SDOTXAXPYF_KERNEL_REF       bli_sssdotxaxpyf_ref_var1
//#define BLIS_DDOTXAXPYF_KERNEL_REF       bli_ddddotxaxpyf_ref_var1
//#define BLIS_CDOTXAXPYF_KERNEL_REF       bli_cccdotxaxpyf_ref_var1
//#define BLIS_ZDOTXAXPYF_KERNEL_REF       bli_zzzdotxaxpyf_ref_var1
#define BLIS_SDOTXAXPYF_KERNEL_REF       bli_sssdotxaxpyf_ref_var2
#define BLIS_DDOTXAXPYF_KERNEL_REF       bli_ddddotxaxpyf_ref_var2
#define BLIS_CDOTXAXPYF_KERNEL_REF       bli_cccdotxaxpyf_ref_var2
#define BLIS_ZDOTXAXPYF_KERNEL_REF       bli_zzzdotxaxpyf_ref_var2

//
// Level-1v
//

// addv kernels

#define BLIS_SADDV_KERNEL_REF            bli_ssaddv_ref
#define BLIS_DADDV_KERNEL_REF            bli_ddaddv_ref
#define BLIS_CADDV_KERNEL_REF            bli_ccaddv_ref
#define BLIS_ZADDV_KERNEL_REF            bli_zzaddv_ref

// axpyv kernels

#define BLIS_SAXPYV_KERNEL_REF           bli_sssaxpyv_ref
#define BLIS_DAXPYV_KERNEL_REF           bli_dddaxpyv_ref
#define BLIS_CAXPYV_KERNEL_REF           bli_cccaxpyv_ref
#define BLIS_ZAXPYV_KERNEL_REF           bli_zzzaxpyv_ref

// copyv kernels

#define BLIS_SCOPYV_KERNEL_REF           bli_sscopyv_ref
#define BLIS_DCOPYV_KERNEL_REF           bli_ddcopyv_ref
#define BLIS_CCOPYV_KERNEL_REF           bli_cccopyv_ref
#define BLIS_ZCOPYV_KERNEL_REF           bli_zzcopyv_ref

// dotv kernels

#define BLIS_SDOTV_KERNEL_REF            bli_sssdotv_ref
#define BLIS_DDOTV_KERNEL_REF            bli_ddddotv_ref
#define BLIS_CDOTV_KERNEL_REF            bli_cccdotv_ref
#define BLIS_ZDOTV_KERNEL_REF            bli_zzzdotv_ref

// dotxv kernels

#define BLIS_SDOTXV_KERNEL_REF           bli_sssdotxv_ref
#define BLIS_DDOTXV_KERNEL_REF           bli_ddddotxv_ref
#define BLIS_CDOTXV_KERNEL_REF           bli_cccdotxv_ref
#define BLIS_ZDOTXV_KERNEL_REF           bli_zzzdotxv_ref

// invertv kernels

#define BLIS_SINVERTV_KERNEL_REF         bli_sinvertv_ref
#define BLIS_DINVERTV_KERNEL_REF         bli_dinvertv_ref
#define BLIS_CINVERTV_KERNEL_REF         bli_cinvertv_ref
#define BLIS_ZINVERTV_KERNEL_REF         bli_zinvertv_ref

// scal2v kernels

#define BLIS_SSCAL2V_KERNEL_REF          bli_sssscal2v_ref
#define BLIS_DSCAL2V_KERNEL_REF          bli_dddscal2v_ref
#define BLIS_CSCAL2V_KERNEL_REF          bli_cccscal2v_ref
#define BLIS_ZSCAL2V_KERNEL_REF          bli_zzzscal2v_ref

// scalv kernels

#define BLIS_SSCALV_KERNEL_REF           bli_ssscalv_ref
#define BLIS_DSCALV_KERNEL_REF           bli_ddscalv_ref
#define BLIS_CSCALV_KERNEL_REF           bli_ccscalv_ref
#define BLIS_ZSCALV_KERNEL_REF           bli_zzscalv_ref

// setv kernels

#define BLIS_SSETV_KERNEL_REF            bli_sssetv_ref
#define BLIS_DSETV_KERNEL_REF            bli_ddsetv_ref
#define BLIS_CSETV_KERNEL_REF            bli_ccsetv_ref
#define BLIS_ZSETV_KERNEL_REF            bli_zzsetv_ref

// subv kernels

#define BLIS_SSUBV_KERNEL_REF            bli_sssubv_ref
#define BLIS_DSUBV_KERNEL_REF            bli_ddsubv_ref
#define BLIS_CSUBV_KERNEL_REF            bli_ccsubv_ref
#define BLIS_ZSUBV_KERNEL_REF            bli_zzsubv_ref

// swapv kernels

#define BLIS_SSWAPV_KERNEL_REF           bli_ssswapv_ref
#define BLIS_DSWAPV_KERNEL_REF           bli_ddswapv_ref
#define BLIS_CSWAPV_KERNEL_REF           bli_ccswapv_ref
#define BLIS_ZSWAPV_KERNEL_REF           bli_zzswapv_ref



#endif 

