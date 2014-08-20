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


// -- Define PASTEMAC-friendly kernel function name macros ---------------------

//
// Level-3
//

// gemm micro-kernels

#define bli_sGEMM_UKERNEL BLIS_SGEMM_UKERNEL
#define bli_dGEMM_UKERNEL BLIS_DGEMM_UKERNEL
#define bli_cGEMM_UKERNEL BLIS_CGEMM_UKERNEL
#define bli_zGEMM_UKERNEL BLIS_ZGEMM_UKERNEL

#undef  GENTPROT
#define GENTPROT( ctype, ch, kername ) \
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

INSERT_GENTPROT_BASIC( GEMM_UKERNEL )

// gemmtrsm_l micro-kernels

#define bli_sGEMMTRSM_L_UKERNEL BLIS_SGEMMTRSM_L_UKERNEL
#define bli_dGEMMTRSM_L_UKERNEL BLIS_DGEMMTRSM_L_UKERNEL
#define bli_cGEMMTRSM_L_UKERNEL BLIS_CGEMMTRSM_L_UKERNEL
#define bli_zGEMMTRSM_L_UKERNEL BLIS_ZGEMMTRSM_L_UKERNEL

#undef  GENTPROT
#define GENTPROT( ctype, ch, kername ) \
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

INSERT_GENTPROT_BASIC( GEMMTRSM_L_UKERNEL )

// gemmtrsm_u micro-kernels

#define bli_sGEMMTRSM_U_UKERNEL BLIS_SGEMMTRSM_U_UKERNEL
#define bli_dGEMMTRSM_U_UKERNEL BLIS_DGEMMTRSM_U_UKERNEL
#define bli_cGEMMTRSM_U_UKERNEL BLIS_CGEMMTRSM_U_UKERNEL
#define bli_zGEMMTRSM_U_UKERNEL BLIS_ZGEMMTRSM_U_UKERNEL

#undef  GENTPROT
#define GENTPROT( ctype, ch, kername ) \
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

INSERT_GENTPROT_BASIC( GEMMTRSM_U_UKERNEL )

// trsm_l micro-kernels

#define bli_sTRSM_L_UKERNEL BLIS_STRSM_L_UKERNEL
#define bli_dTRSM_L_UKERNEL BLIS_DTRSM_L_UKERNEL
#define bli_cTRSM_L_UKERNEL BLIS_CTRSM_L_UKERNEL
#define bli_zTRSM_L_UKERNEL BLIS_ZTRSM_L_UKERNEL

#undef  GENTPROT
#define GENTPROT( ctype, ch, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       ctype* restrict a11, \
       ctype* restrict b11, \
       ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROT_BASIC( TRSM_L_UKERNEL )

// trsm_u micro-kernels

#define bli_sTRSM_U_UKERNEL BLIS_STRSM_U_UKERNEL
#define bli_dTRSM_U_UKERNEL BLIS_DTRSM_U_UKERNEL
#define bli_cTRSM_U_UKERNEL BLIS_CTRSM_U_UKERNEL
#define bli_zTRSM_U_UKERNEL BLIS_ZTRSM_U_UKERNEL

#undef  GENTPROT
#define GENTPROT( ctype, ch, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       ctype* restrict a11, \
       ctype* restrict b11, \
       ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t*      data  \
     );

INSERT_GENTPROT_BASIC( TRSM_U_UKERNEL )


//
// Level-3 4m
//

// gemm4m micro-kernels

#define bli_cGEMM4M_UKERNEL BLIS_CGEMM4M_UKERNEL
#define bli_zGEMM4M_UKERNEL BLIS_ZGEMM4M_UKERNEL

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

INSERT_GENTPROTCO_BASIC( GEMM4M_UKERNEL )

// gemmtrsm4m_l micro-kernels

#define bli_cGEMMTRSM4M_L_UKERNEL BLIS_CGEMMTRSM4M_L_UKERNEL
#define bli_zGEMMTRSM4M_L_UKERNEL BLIS_ZGEMMTRSM4M_L_UKERNEL

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

INSERT_GENTPROTCO_BASIC( GEMMTRSM4M_L_UKERNEL )

// gemmtrsm4m_u micro-kernels

#define bli_cGEMMTRSM4M_U_UKERNEL BLIS_CGEMMTRSM4M_U_UKERNEL
#define bli_zGEMMTRSM4M_U_UKERNEL BLIS_ZGEMMTRSM4M_U_UKERNEL

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

INSERT_GENTPROTCO_BASIC( GEMMTRSM4M_U_UKERNEL )

// trsm4m_l micro-kernels

#define bli_cTRSM4M_L_UKERNEL BLIS_CTRSM4M_L_UKERNEL
#define bli_zTRSM4M_L_UKERNEL BLIS_ZTRSM4M_L_UKERNEL

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

INSERT_GENTPROTCO_BASIC( TRSM4M_L_UKERNEL )

// trsm4m_u micro-kernels

#define bli_cTRSM4M_U_UKERNEL BLIS_CTRSM4M_U_UKERNEL
#define bli_zTRSM4M_U_UKERNEL BLIS_ZTRSM4M_U_UKERNEL

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

INSERT_GENTPROTCO_BASIC( TRSM4M_U_UKERNEL )


//
// Level-3 3m
//

// gemm3m micro-kernels

#define bli_cGEMM3M_UKERNEL BLIS_CGEMM3M_UKERNEL
#define bli_zGEMM3M_UKERNEL BLIS_ZGEMM3M_UKERNEL

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

INSERT_GENTPROTCO_BASIC( GEMM3M_UKERNEL )

// gemmtrsm3m_l micro-kernels

#define bli_cGEMMTRSM3M_L_UKERNEL BLIS_CGEMMTRSM3M_L_UKERNEL
#define bli_zGEMMTRSM3M_L_UKERNEL BLIS_ZGEMMTRSM3M_L_UKERNEL

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

INSERT_GENTPROTCO_BASIC( GEMMTRSM3M_L_UKERNEL )

// gemmtrsm3m_u micro-kernels

#define bli_cGEMMTRSM3M_U_UKERNEL BLIS_CGEMMTRSM3M_U_UKERNEL
#define bli_zGEMMTRSM3M_U_UKERNEL BLIS_ZGEMMTRSM3M_U_UKERNEL

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

INSERT_GENTPROTCO_BASIC( GEMMTRSM3M_U_UKERNEL )

// trsm3m_l micro-kernels

#define bli_cTRSM3M_L_UKERNEL BLIS_CTRSM3M_L_UKERNEL
#define bli_zTRSM3M_L_UKERNEL BLIS_ZTRSM3M_L_UKERNEL

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

INSERT_GENTPROTCO_BASIC( TRSM3M_L_UKERNEL )

// trsm3m_u micro-kernels

#define bli_cTRSM3M_U_UKERNEL BLIS_CTRSM3M_U_UKERNEL
#define bli_zTRSM3M_U_UKERNEL BLIS_ZTRSM3M_U_UKERNEL

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

INSERT_GENTPROTCO_BASIC( TRSM3M_U_UKERNEL )


//
// Level-1m
//

// NOTE: We don't need any PASTEMAC-friendly aliases to packm kernel
// macros because they are used directly in the initialization of the
// function pointer array, rather than via a templatizing wrapper macro.


//
// Level-1f
//

// axpy2v kernels

#define bli_sssAXPY2V_KERNEL BLIS_SAXPY2V_KERNEL
#define bli_dddAXPY2V_KERNEL BLIS_DAXPY2V_KERNEL
#define bli_cccAXPY2V_KERNEL BLIS_CAXPY2V_KERNEL
#define bli_zzzAXPY2V_KERNEL BLIS_ZAXPY2V_KERNEL

#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_x, ctype_y, ctype_z, ctype_xy, chx, chy, chz, chxy, kername ) \
\
void PASTEMAC3(chx,chy,chz,kername) \
     ( \
       conj_t             conjx, \
       conj_t             conjy, \
       dim_t              n, \
       ctype_xy* restrict alpha1, \
       ctype_xy* restrict alpha2, \
       ctype_x*  restrict x, inc_t incx, \
       ctype_y*  restrict y, inc_t incy, \
       ctype_z*  restrict z, inc_t incz  \
     );

INSERT_GENTPROT3U12_BASIC( AXPY2V_KERNEL )

// dotaxpyv kernels

#define bli_sssDOTAXPYV_KERNEL BLIS_SDOTAXPYV_KERNEL
#define bli_dddDOTAXPYV_KERNEL BLIS_DDOTAXPYV_KERNEL
#define bli_cccDOTAXPYV_KERNEL BLIS_CDOTAXPYV_KERNEL
#define bli_zzzDOTAXPYV_KERNEL BLIS_ZDOTAXPYV_KERNEL

#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_x, ctype_y, ctype_z, ctype_xy, chx, chy, chz, chxy, kername ) \
\
void PASTEMAC3(chx,chy,chz,kername) \
     ( \
       conj_t             conjxt, \
       conj_t             conjx, \
       conj_t             conjy, \
       dim_t              m, \
       ctype_x*  restrict alpha, \
       ctype_x*  restrict x, inc_t incx, \
       ctype_y*  restrict y, inc_t incy, \
       ctype_xy* restrict rho, \
       ctype_z*  restrict z, inc_t incz  \
     );

INSERT_GENTPROT3U12_BASIC( DOTAXPYV_KERNEL )

// axpyf kernels

#define bli_sssAXPYF_KERNEL BLIS_SAXPYF_KERNEL
#define bli_dddAXPYF_KERNEL BLIS_DAXPYF_KERNEL
#define bli_cccAXPYF_KERNEL BLIS_CAXPYF_KERNEL
#define bli_zzzAXPYF_KERNEL BLIS_ZAXPYF_KERNEL

#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_a, ctype_x, ctype_y, ctype_ax, cha, chx, chy, chax, kername ) \
\
void PASTEMAC3(cha,chx,chy,kername) \
     ( \
       conj_t             conja, \
       conj_t             conjx, \
       dim_t              m, \
       dim_t              b_n, \
       ctype_ax* restrict alpha, \
       ctype_a*  restrict a, inc_t inca, inc_t lda, \
       ctype_x*  restrict x, inc_t incx, \
       ctype_y*  restrict y, inc_t incy  \
     );

INSERT_GENTPROT3U12_BASIC( AXPYF_KERNEL )

// dotxf kernels

#define bli_sssDOTXF_KERNEL BLIS_SDOTXF_KERNEL
#define bli_dddDOTXF_KERNEL BLIS_DDOTXF_KERNEL
#define bli_cccDOTXF_KERNEL BLIS_CDOTXF_KERNEL
#define bli_zzzDOTXF_KERNEL BLIS_ZDOTXF_KERNEL

#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_a, ctype_x, ctype_y, ctype_ax, cha, chx, chy, chax, kername ) \
\
void PASTEMAC3(cha,chx,chy,kername) \
     ( \
       conj_t             conjat, \
       conj_t             conjx, \
       dim_t              m, \
       dim_t              b_n, \
       ctype_ax* restrict alpha, \
       ctype_a*  restrict a, inc_t inca, inc_t lda, \
       ctype_x*  restrict x, inc_t incx, \
       ctype_y*  restrict beta, \
       ctype_y*  restrict y, inc_t incy  \
     );

INSERT_GENTPROT3U12_BASIC( DOTXF_KERNEL )

// dotxaxpyf kernels

#define bli_sssDOTXAXPYF_KERNEL BLIS_SDOTXAXPYF_KERNEL
#define bli_dddDOTXAXPYF_KERNEL BLIS_DDOTXAXPYF_KERNEL
#define bli_cccDOTXAXPYF_KERNEL BLIS_CDOTXAXPYF_KERNEL
#define bli_zzzDOTXAXPYF_KERNEL BLIS_ZDOTXAXPYF_KERNEL

#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_a, ctype_b, ctype_c, ctype_ab, cha, chb, chc, chab, kername ) \
\
void PASTEMAC3(cha,chb,chc,kername) \
     ( \
       conj_t             conjat, \
       conj_t             conja, \
       conj_t             conjw, \
       conj_t             conjx, \
       dim_t              m, \
       dim_t              b_n, \
       ctype_ab* restrict alpha, \
       ctype_a*  restrict a, inc_t inca, inc_t lda, \
       ctype_b*  restrict w, inc_t incw, \
       ctype_b*  restrict x, inc_t incx, \
       ctype_c*  restrict beta, \
       ctype_c*  restrict y, inc_t incy, \
       ctype_c*  restrict z, inc_t incz  \
     );

INSERT_GENTPROT3U12_BASIC( DOTXAXPYF_KERNEL )


//
// Level-1v
//

// addv kernels

#define bli_ssADDV_KERNEL BLIS_SADDV_KERNEL
#define bli_ddADDV_KERNEL BLIS_DADDV_KERNEL
#define bli_ccADDV_KERNEL BLIS_CADDV_KERNEL
#define bli_zzADDV_KERNEL BLIS_ZADDV_KERNEL

#undef  GENTPROT2
#define GENTPROT2( ctype_x, ctype_y, chx, chy, kername ) \
\
void PASTEMAC2(chx,chy,kername) \
     ( \
       conj_t            conjx, \
       dim_t             n, \
       ctype_x* restrict x, inc_t incx, \
       ctype_y* restrict y, inc_t incy  \
     );

INSERT_GENTPROT2_BASIC( ADDV_KERNEL )

// axpyv kernels

#define bli_sssAXPYV_KERNEL BLIS_SAXPYV_KERNEL
#define bli_dddAXPYV_KERNEL BLIS_DAXPYV_KERNEL
#define bli_cccAXPYV_KERNEL BLIS_CAXPYV_KERNEL
#define bli_zzzAXPYV_KERNEL BLIS_ZAXPYV_KERNEL

#undef  GENTPROT3
#define GENTPROT3( ctype_a, ctype_x, ctype_y, cha, chx, chy, kername ) \
\
void PASTEMAC3(cha,chx,chy,kername) \
     ( \
       conj_t            conjx, \
       dim_t             n, \
       ctype_a* restrict alpha, \
       ctype_x* restrict x, inc_t incx, \
       ctype_y* restrict y, inc_t incy  \
     );

INSERT_GENTPROT3_BASIC( AXPYV_KERNEL )

// copyv kernels

#define bli_ssCOPYV_KERNEL BLIS_SCOPYV_KERNEL
#define bli_ddCOPYV_KERNEL BLIS_DCOPYV_KERNEL
#define bli_ccCOPYV_KERNEL BLIS_CCOPYV_KERNEL
#define bli_zzCOPYV_KERNEL BLIS_ZCOPYV_KERNEL

#undef  GENTPROT2
#define GENTPROT2( ctype_x, ctype_y, chx, chy, kername ) \
\
void PASTEMAC2(chx,chy,kername) \
     ( \
       conj_t            conjx, \
       dim_t             n, \
       ctype_x* restrict x, inc_t incx, \
       ctype_y* restrict y, inc_t incy  \
     );

INSERT_GENTPROT2_BASIC( COPYV_KERNEL )

// dotv kernels

#define bli_sssDOTV_KERNEL BLIS_SDOTV_KERNEL
#define bli_dddDOTV_KERNEL BLIS_DDOTV_KERNEL
#define bli_cccDOTV_KERNEL BLIS_CDOTV_KERNEL
#define bli_zzzDOTV_KERNEL BLIS_ZDOTV_KERNEL

#undef  GENTPROT3
#define GENTPROT3( ctype_x, ctype_y, ctype_r, chx, chy, chr, kername ) \
\
void PASTEMAC3(chx,chy,chr,kername) \
     ( \
       conj_t            conjx, \
       conj_t            conjy, \
       dim_t             n, \
       ctype_x* restrict x, inc_t incx, \
       ctype_y* restrict y, inc_t incy, \
       ctype_r* restrict rho  \
     );

INSERT_GENTPROT3_BASIC( DOTV_KERNEL )

// dotxv kernels

#define bli_sssDOTXV_KERNEL BLIS_SDOTXV_KERNEL
#define bli_dddDOTXV_KERNEL BLIS_DDOTXV_KERNEL
#define bli_cccDOTXV_KERNEL BLIS_CDOTXV_KERNEL
#define bli_zzzDOTXV_KERNEL BLIS_ZDOTXV_KERNEL

#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_x, ctype_y, ctype_r, ctype_xy, chx, chy, chr, chxy, kername ) \
\
void PASTEMAC3(chx,chy,chr,kername) \
     ( \
       conj_t             conjx, \
       conj_t             conjy, \
       dim_t              n, \
       ctype_xy* restrict alpha, \
       ctype_x*  restrict x, inc_t incx, \
       ctype_y*  restrict y, inc_t incy, \
       ctype_r*  restrict beta, \
       ctype_r*  restrict rho  \
     );

INSERT_GENTPROT3U12_BASIC( DOTXV_KERNEL )

// invertv kernels

#define bli_sINVERTV_KERNEL BLIS_SINVERTV_KERNEL
#define bli_dINVERTV_KERNEL BLIS_DINVERTV_KERNEL
#define bli_cINVERTV_KERNEL BLIS_CINVERTV_KERNEL
#define bli_zINVERTV_KERNEL BLIS_ZINVERTV_KERNEL

#undef  GENTPROT
#define GENTPROT( ctype, ch, kername ) \
\
void PASTEMAC(ch,kername) \
     ( \
       dim_t           n, \
       ctype* restrict x, inc_t incx  \
     );

INSERT_GENTPROT_BASIC( INVERTV_KERNEL )

// scal2v kernels

#define bli_sssSCAL2V_KERNEL BLIS_SSCAL2V_KERNEL
#define bli_dddSCAL2V_KERNEL BLIS_DSCAL2V_KERNEL
#define bli_cccSCAL2V_KERNEL BLIS_CSCAL2V_KERNEL
#define bli_zzzSCAL2V_KERNEL BLIS_ZSCAL2V_KERNEL

#undef  GENTPROT3
#define GENTPROT3( ctype_b, ctype_x, ctype_y, chb, chx, chy, kername ) \
\
void PASTEMAC3(chb,chx,chy,kername) \
     ( \
       conj_t            conjx, \
       dim_t             n, \
       ctype_b* restrict beta, \
       ctype_x* restrict x, inc_t incx, \
       ctype_y* restrict y, inc_t incy  \
     );

INSERT_GENTPROT3_BASIC( SCAL2V_KERNEL )

// scalv kernels

#define bli_ssSCALV_KERNEL BLIS_SSCALV_KERNEL
#define bli_ddSCALV_KERNEL BLIS_DSCALV_KERNEL
#define bli_ccSCALV_KERNEL BLIS_CSCALV_KERNEL
#define bli_zzSCALV_KERNEL BLIS_ZSCALV_KERNEL

#undef  GENTPROT2
#define GENTPROT2( ctype_b, ctype_x, chb, chx, kername ) \
\
void PASTEMAC2(chb,chx,kername) \
     ( \
       conj_t            conjbeta, \
       dim_t             n, \
       ctype_b* restrict beta, \
       ctype_x* restrict x, inc_t incx \
     );

INSERT_GENTPROT2_BASIC( SCALV_KERNEL )

// setv kernels

#define bli_ssSETV_KERNEL BLIS_SSETV_KERNEL
#define bli_ddSETV_KERNEL BLIS_DSETV_KERNEL
#define bli_ccSETV_KERNEL BLIS_CSETV_KERNEL
#define bli_zzSETV_KERNEL BLIS_ZSETV_KERNEL

#undef  GENTPROT2
#define GENTPROT2( ctype_b, ctype_x, chb, chx, kername ) \
\
void PASTEMAC2(chb,chx,kername) \
     ( \
       dim_t             n, \
       ctype_b* restrict beta, \
       ctype_x* restrict x, inc_t incx \
     );

INSERT_GENTPROT2_BASIC( SETV_KERNEL )

// subv kernels

#define bli_ssSUBV_KERNEL BLIS_SSUBV_KERNEL
#define bli_ddSUBV_KERNEL BLIS_DSUBV_KERNEL
#define bli_ccSUBV_KERNEL BLIS_CSUBV_KERNEL
#define bli_zzSUBV_KERNEL BLIS_ZSUBV_KERNEL

#undef  GENTPROT2
#define GENTPROT2( ctype_x, ctype_y, chx, chy, kername ) \
\
void PASTEMAC2(chx,chy,kername) \
     ( \
       conj_t            conjx, \
       dim_t             n, \
       ctype_x* restrict x, inc_t incx, \
       ctype_y* restrict y, inc_t incy  \
     );

INSERT_GENTPROT2_BASIC( SUBV_KERNEL )

// swapv kernels

#define bli_ssSWAPV_KERNEL BLIS_SSWAPV_KERNEL
#define bli_ddSWAPV_KERNEL BLIS_DSWAPV_KERNEL
#define bli_ccSWAPV_KERNEL BLIS_CSWAPV_KERNEL
#define bli_zzSWAPV_KERNEL BLIS_ZSWAPV_KERNEL

#undef  GENTPROT2
#define GENTPROT2( ctype_x, ctype_y, chx, chy, kername ) \
\
void PASTEMAC2(chx,chy,kername) \
     ( \
       dim_t             n, \
       ctype_x* restrict x, inc_t incx, \
       ctype_y* restrict y, inc_t incy  \
     );

INSERT_GENTPROT2_BASIC( SWAPV_KERNEL )



#endif

