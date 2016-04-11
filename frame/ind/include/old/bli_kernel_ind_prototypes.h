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

#ifndef BLIS_KERNEL_IND_PROTOTYPES_H
#define BLIS_KERNEL_IND_PROTOTYPES_H


// -- Define PASTEMAC-friendly kernel function name macros ---------------------


//
// 3m1
//

// gemm3m micro-kernels

#define bli_cGEMM3M1_UKERNEL BLIS_CGEMM3M1_UKERNEL
#define bli_zGEMM3M1_UKERNEL BLIS_ZGEMM3M1_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t           k, \
       ctype* restrict alpha, \
       ctype* restrict a, \
       ctype* restrict b, \
       ctype* restrict beta, \
       ctype* restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROTCO_BASIC( GEMM3M1_UKERNEL )

// gemmtrsm3m1_l micro-kernels

#define bli_cGEMMTRSM3M1_L_UKERNEL BLIS_CGEMMTRSM3M1_L_UKERNEL
#define bli_zGEMMTRSM3M1_L_UKERNEL BLIS_ZGEMMTRSM3M1_L_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t           k, \
       ctype* restrict alpha, \
       ctype* restrict a10, \
       ctype* restrict a11, \
       ctype* restrict b01, \
       ctype* restrict b11, \
       ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROTCO_BASIC( GEMMTRSM3M1_L_UKERNEL )

// gemmtrsm3m1_u micro-kernels

#define bli_cGEMMTRSM3M1_U_UKERNEL BLIS_CGEMMTRSM3M1_U_UKERNEL
#define bli_zGEMMTRSM3M1_U_UKERNEL BLIS_ZGEMMTRSM3M1_U_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t           k, \
       ctype* restrict alpha, \
       ctype* restrict a12, \
       ctype* restrict a11, \
       ctype* restrict b21, \
       ctype* restrict b11, \
       ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROTCO_BASIC( GEMMTRSM3M1_U_UKERNEL )

// trsm3m1_l micro-kernels

#define bli_cTRSM3M1_L_UKERNEL BLIS_CTRSM3M1_L_UKERNEL
#define bli_zTRSM3M1_L_UKERNEL BLIS_ZTRSM3M1_L_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       ctype_r* restrict a11r, \
       ctype_r* restrict b11r, \
       ctype*   restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*        data  \
     );

INSERT_GENTPROTCO_BASIC( TRSM3M1_L_UKERNEL )

// trsm3m1_u micro-kernels

#define bli_cTRSM3M1_U_UKERNEL BLIS_CTRSM3M1_U_UKERNEL
#define bli_zTRSM3M1_U_UKERNEL BLIS_ZTRSM3M1_U_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       ctype_r* restrict a11r, \
       ctype_r* restrict b11r, \
       ctype*   restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*        data  \
     );

INSERT_GENTPROTCO_BASIC( TRSM3M1_U_UKERNEL )


//
// 4m1
//

// gemm4m micro-kernels

#define bli_cGEMM4M1_UKERNEL BLIS_CGEMM4M1_UKERNEL
#define bli_zGEMM4M1_UKERNEL BLIS_ZGEMM4M1_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t           k, \
       ctype* restrict alpha, \
       ctype* restrict a, \
       ctype* restrict b, \
       ctype* restrict beta, \
       ctype* restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROTCO_BASIC( GEMM4M1_UKERNEL )

// gemmtrsm4m1_l micro-kernels

#define bli_cGEMMTRSM4M1_L_UKERNEL BLIS_CGEMMTRSM4M1_L_UKERNEL
#define bli_zGEMMTRSM4M1_L_UKERNEL BLIS_ZGEMMTRSM4M1_L_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t           k, \
       ctype* restrict alpha, \
       ctype* restrict a10, \
       ctype* restrict a11, \
       ctype* restrict b01, \
       ctype* restrict b11, \
       ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROTCO_BASIC( GEMMTRSM4M1_L_UKERNEL )

// gemmtrsm4m1_u micro-kernels

#define bli_cGEMMTRSM4M1_U_UKERNEL BLIS_CGEMMTRSM4M1_U_UKERNEL
#define bli_zGEMMTRSM4M1_U_UKERNEL BLIS_ZGEMMTRSM4M1_U_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t           k, \
       ctype* restrict alpha, \
       ctype* restrict a12, \
       ctype* restrict a11, \
       ctype* restrict b21, \
       ctype* restrict b11, \
       ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROTCO_BASIC( GEMMTRSM4M1_U_UKERNEL )

// trsm4m1_l micro-kernels

#define bli_cTRSM4M1_L_UKERNEL BLIS_CTRSM4M1_L_UKERNEL
#define bli_zTRSM4M1_L_UKERNEL BLIS_ZTRSM4M1_L_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       ctype_r* restrict a11r, \
       ctype_r* restrict b11r, \
       ctype*   restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*        data  \
     );

INSERT_GENTPROTCO_BASIC( TRSM4M1_L_UKERNEL )

// trsm4m1_u micro-kernels

#define bli_cTRSM4M1_U_UKERNEL BLIS_CTRSM4M1_U_UKERNEL
#define bli_zTRSM4M1_U_UKERNEL BLIS_ZTRSM4M1_U_UKERNEL

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       ctype_r* restrict a11r, \
       ctype_r* restrict b11r, \
       ctype*   restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*        data  \
     );

INSERT_GENTPROTCO_BASIC( TRSM4M1_U_UKERNEL )



#endif

