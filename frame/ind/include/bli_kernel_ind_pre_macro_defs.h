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

#include "bli_packm_ind_pre_macro_defs.h"

#ifndef BLIS_KERNEL_IND_PRE_MACRO_DEFS_H
#define BLIS_KERNEL_IND_PRE_MACRO_DEFS_H

//
// Level-3 3mh
//

// gemm3mh micro-kernels

#define BLIS_CGEMM3MH_UKERNEL_REF        bli_cgemm3mh_ukr_ref
#define BLIS_ZGEMM3MH_UKERNEL_REF        bli_zgemm3mh_ukr_ref

//
// Level-3 3m3
//

// gemm3m3 micro-kernels

#define BLIS_CGEMM3M3_UKERNEL_REF        bli_cgemm3m3_ukr_ref
#define BLIS_ZGEMM3M3_UKERNEL_REF        bli_zgemm3m3_ukr_ref

//
// Level-3 3m2
//

// gemm3m2 micro-kernels

#define BLIS_CGEMM3M2_UKERNEL_REF        bli_cgemm3m2_ukr_ref
#define BLIS_ZGEMM3M2_UKERNEL_REF        bli_zgemm3m2_ukr_ref

//
// Level-3 3m1
//

// gemm3m1 micro-kernels

#define BLIS_CGEMM3M1_UKERNEL_REF        bli_cgemm3m1_ukr_ref
#define BLIS_ZGEMM3M1_UKERNEL_REF        bli_zgemm3m1_ukr_ref

// gemmtrsm3m1_l micro-kernels

#define BLIS_CGEMMTRSM3M1_L_UKERNEL_REF  bli_cgemmtrsm3m1_l_ukr_ref
#define BLIS_ZGEMMTRSM3M1_L_UKERNEL_REF  bli_zgemmtrsm3m1_l_ukr_ref

// gemmtrsm3m1_u micro-kernels

#define BLIS_CGEMMTRSM3M1_U_UKERNEL_REF  bli_cgemmtrsm3m1_u_ukr_ref
#define BLIS_ZGEMMTRSM3M1_U_UKERNEL_REF  bli_zgemmtrsm3m1_u_ukr_ref

// trsm3m1_l micro-kernels

#define BLIS_CTRSM3M1_L_UKERNEL_REF      bli_ctrsm3m1_l_ukr_ref
#define BLIS_ZTRSM3M1_L_UKERNEL_REF      bli_ztrsm3m1_l_ukr_ref

// trsm3m1_u micro-kernels

#define BLIS_CTRSM3M1_U_UKERNEL_REF      bli_ctrsm3m1_u_ukr_ref
#define BLIS_ZTRSM3M1_U_UKERNEL_REF      bli_ztrsm3m1_u_ukr_ref

//
// Level-3 4mh
//

// gemm4mh micro-kernels

#define BLIS_CGEMM4MH_UKERNEL_REF        bli_cgemm4mh_ukr_ref
#define BLIS_ZGEMM4MH_UKERNEL_REF        bli_zgemm4mh_ukr_ref

//
// Level-3 4mb
//

// gemm4mb micro-kernels

#define BLIS_CGEMM4MB_UKERNEL_REF        bli_cgemm4mb_ukr_ref
#define BLIS_ZGEMM4MB_UKERNEL_REF        bli_zgemm4mb_ukr_ref

//
// Level-3 4m1
//

// gemm4m1 micro-kernels

#define BLIS_CGEMM4M1_UKERNEL_REF        bli_cgemm4m1_ukr_ref
#define BLIS_ZGEMM4M1_UKERNEL_REF        bli_zgemm4m1_ukr_ref

// gemmtrsm4m1_l micro-kernels

#define BLIS_CGEMMTRSM4M1_L_UKERNEL_REF  bli_cgemmtrsm4m1_l_ukr_ref
#define BLIS_ZGEMMTRSM4M1_L_UKERNEL_REF  bli_zgemmtrsm4m1_l_ukr_ref

// gemmtrsm4m1_u micro-kernels

#define BLIS_CGEMMTRSM4M1_U_UKERNEL_REF  bli_cgemmtrsm4m1_u_ukr_ref
#define BLIS_ZGEMMTRSM4M1_U_UKERNEL_REF  bli_zgemmtrsm4m1_u_ukr_ref

// trsm4m1_l micro-kernels

#define BLIS_CTRSM4M1_L_UKERNEL_REF      bli_ctrsm4m1_l_ukr_ref
#define BLIS_ZTRSM4M1_L_UKERNEL_REF      bli_ztrsm4m1_l_ukr_ref

// trsm4m1_u micro-kernels

#define BLIS_CTRSM4M1_U_UKERNEL_REF      bli_ctrsm4m1_u_ukr_ref
#define BLIS_ZTRSM4M1_U_UKERNEL_REF      bli_ztrsm4m1_u_ukr_ref

//
// Level-3 1m
//

// gemm1m micro-kernels

#define BLIS_CGEMM1M_UKERNEL_REF         bli_cgemm1m_ukr_ref
#define BLIS_ZGEMM1M_UKERNEL_REF         bli_zgemm1m_ukr_ref

// gemmtrsm1m_l micro-kernels

#define BLIS_CGEMMTRSM1M_L_UKERNEL_REF   bli_cgemmtrsm1m_l_ukr_ref
#define BLIS_ZGEMMTRSM1M_L_UKERNEL_REF   bli_zgemmtrsm1m_l_ukr_ref

// gemmtrsm1m_u micro-kernels

#define BLIS_CGEMMTRSM1M_U_UKERNEL_REF   bli_cgemmtrsm1m_u_ukr_ref
#define BLIS_ZGEMMTRSM1M_U_UKERNEL_REF   bli_zgemmtrsm1m_u_ukr_ref

// trsm1m_l micro-kernels

#define BLIS_CTRSM1M_L_UKERNEL_REF       bli_ctrsm1m_l_ukr_ref
#define BLIS_ZTRSM1M_L_UKERNEL_REF       bli_ztrsm1m_l_ukr_ref

// trsm1m_u micro-kernels

#define BLIS_CTRSM1M_U_UKERNEL_REF       bli_ctrsm1m_u_ukr_ref
#define BLIS_ZTRSM1M_U_UKERNEL_REF       bli_ztrsm1m_u_ukr_ref



#endif 

