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

#ifndef BLIS_KERNEL_MACRO_DEFS_H
#define BLIS_KERNEL_MACRO_DEFS_H


// -- Define row access bools --------------------------------------------------

// In this section we consider each datatype-specific "prefers contiguous rows"
// macro. If it is defined, we re-define it to be 1 (TRUE); otherwise, we
// define it to be 0 (FALSE).

// gemm micro-kernels

#ifdef  BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef  BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS 1 
#else
#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS 0 
#endif

#ifdef  BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef  BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS 1 
#else
#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS 0 
#endif

#ifdef  BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef  BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS 1 
#else
#define BLIS_CGEMM_UKERNEL_PREFERS_CONTIG_ROWS 0 
#endif

#ifdef  BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#undef  BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS 1 
#else
#define BLIS_ZGEMM_UKERNEL_PREFERS_CONTIG_ROWS 0 
#endif


// -- Define default kernel names ----------------------------------------------

// In this section we consider each datatype-specific micro-kernel macro;
// if it is undefined, we define it to be the corresponding reference kernel.
// In the case of complex gemm micro-kernels, we also define special macros so
// that later on we can tell whether or not to employ the 4m implementations.
// Note that in order to properly determine whether/ 4m is a viable option, we
// need to be able to test the existence of the real gemm micro-kernels, which
// means we must consider the complex gemm micro-kernel cases *BEFORE* the
// real cases.

//
// Level-3
//

// gemm micro-kernels

#ifndef BLIS_CGEMM_UKERNEL
#define BLIS_CGEMM_UKERNEL BLIS_CGEMM_UKERNEL_REF
#ifdef  BLIS_SGEMM_UKERNEL
#define BLIS_ENABLE_VIRTUAL_SCOMPLEX
#endif
#else
#endif

#ifndef BLIS_ZGEMM_UKERNEL
#define BLIS_ZGEMM_UKERNEL BLIS_ZGEMM_UKERNEL_REF
#ifdef  BLIS_DGEMM_UKERNEL
#define BLIS_ENABLE_VIRTUAL_DCOMPLEX
#endif
#endif

#ifndef BLIS_SGEMM_UKERNEL
#define BLIS_SGEMM_UKERNEL BLIS_SGEMM_UKERNEL_REF
#endif

#ifndef BLIS_DGEMM_UKERNEL
#define BLIS_DGEMM_UKERNEL BLIS_DGEMM_UKERNEL_REF
#endif

// gemmtrsm_l micro-kernels

#ifndef BLIS_SGEMMTRSM_L_UKERNEL
#define BLIS_SGEMMTRSM_L_UKERNEL BLIS_SGEMMTRSM_L_UKERNEL_REF
#endif

#ifndef BLIS_DGEMMTRSM_L_UKERNEL
#define BLIS_DGEMMTRSM_L_UKERNEL BLIS_DGEMMTRSM_L_UKERNEL_REF
#endif

#ifndef BLIS_CGEMMTRSM_L_UKERNEL
#define BLIS_CGEMMTRSM_L_UKERNEL BLIS_CGEMMTRSM_L_UKERNEL_REF
#endif

#ifndef BLIS_ZGEMMTRSM_L_UKERNEL
#define BLIS_ZGEMMTRSM_L_UKERNEL BLIS_ZGEMMTRSM_L_UKERNEL_REF
#endif

// gemmtrsm_u micro-kernels

#ifndef BLIS_SGEMMTRSM_U_UKERNEL
#define BLIS_SGEMMTRSM_U_UKERNEL BLIS_SGEMMTRSM_U_UKERNEL_REF
#endif

#ifndef BLIS_DGEMMTRSM_U_UKERNEL
#define BLIS_DGEMMTRSM_U_UKERNEL BLIS_DGEMMTRSM_U_UKERNEL_REF
#endif

#ifndef BLIS_CGEMMTRSM_U_UKERNEL
#define BLIS_CGEMMTRSM_U_UKERNEL BLIS_CGEMMTRSM_U_UKERNEL_REF
#endif

#ifndef BLIS_ZGEMMTRSM_U_UKERNEL
#define BLIS_ZGEMMTRSM_U_UKERNEL BLIS_ZGEMMTRSM_U_UKERNEL_REF
#endif

// trsm_l micro-kernels

#ifndef BLIS_STRSM_L_UKERNEL
#define BLIS_STRSM_L_UKERNEL BLIS_STRSM_L_UKERNEL_REF
#endif

#ifndef BLIS_DTRSM_L_UKERNEL
#define BLIS_DTRSM_L_UKERNEL BLIS_DTRSM_L_UKERNEL_REF
#endif

#ifndef BLIS_CTRSM_L_UKERNEL
#define BLIS_CTRSM_L_UKERNEL BLIS_CTRSM_L_UKERNEL_REF
#endif

#ifndef BLIS_ZTRSM_L_UKERNEL
#define BLIS_ZTRSM_L_UKERNEL BLIS_ZTRSM_L_UKERNEL_REF
#endif

// trsm_u micro-kernels

#ifndef BLIS_STRSM_U_UKERNEL
#define BLIS_STRSM_U_UKERNEL BLIS_STRSM_U_UKERNEL_REF
#endif

#ifndef BLIS_DTRSM_U_UKERNEL
#define BLIS_DTRSM_U_UKERNEL BLIS_DTRSM_U_UKERNEL_REF
#endif

#ifndef BLIS_CTRSM_U_UKERNEL
#define BLIS_CTRSM_U_UKERNEL BLIS_CTRSM_U_UKERNEL_REF
#endif

#ifndef BLIS_ZTRSM_U_UKERNEL
#define BLIS_ZTRSM_U_UKERNEL BLIS_ZTRSM_U_UKERNEL_REF
#endif

//
// Level-1m
//

// packm_2xk kernels

#ifndef BLIS_SPACKM_2XK_KERNEL
#define BLIS_SPACKM_2XK_KERNEL BLIS_SPACKM_2XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_2XK_KERNEL
#define BLIS_DPACKM_2XK_KERNEL BLIS_DPACKM_2XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_2XK_KERNEL
#define BLIS_CPACKM_2XK_KERNEL BLIS_CPACKM_2XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_2XK_KERNEL
#define BLIS_ZPACKM_2XK_KERNEL BLIS_ZPACKM_2XK_KERNEL_REF
#endif

// packm_3xk kernels

#ifndef BLIS_SPACKM_3XK_KERNEL
#define BLIS_SPACKM_3XK_KERNEL BLIS_SPACKM_3XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_3XK_KERNEL
#define BLIS_DPACKM_3XK_KERNEL BLIS_DPACKM_3XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_3XK_KERNEL
#define BLIS_CPACKM_3XK_KERNEL BLIS_CPACKM_3XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_3XK_KERNEL
#define BLIS_ZPACKM_3XK_KERNEL BLIS_ZPACKM_3XK_KERNEL_REF
#endif

// packm_4xk kernels

#ifndef BLIS_SPACKM_4XK_KERNEL
#define BLIS_SPACKM_4XK_KERNEL BLIS_SPACKM_4XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_4XK_KERNEL
#define BLIS_DPACKM_4XK_KERNEL BLIS_DPACKM_4XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_4XK_KERNEL
#define BLIS_CPACKM_4XK_KERNEL BLIS_CPACKM_4XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_4XK_KERNEL
#define BLIS_ZPACKM_4XK_KERNEL BLIS_ZPACKM_4XK_KERNEL_REF
#endif

// packm_6xk kernels

#ifndef BLIS_SPACKM_6XK_KERNEL
#define BLIS_SPACKM_6XK_KERNEL BLIS_SPACKM_6XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_6XK_KERNEL
#define BLIS_DPACKM_6XK_KERNEL BLIS_DPACKM_6XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_6XK_KERNEL
#define BLIS_CPACKM_6XK_KERNEL BLIS_CPACKM_6XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_6XK_KERNEL
#define BLIS_ZPACKM_6XK_KERNEL BLIS_ZPACKM_6XK_KERNEL_REF
#endif

// packm_8xk kernels

#ifndef BLIS_SPACKM_8XK_KERNEL
#define BLIS_SPACKM_8XK_KERNEL BLIS_SPACKM_8XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_8XK_KERNEL
#define BLIS_DPACKM_8XK_KERNEL BLIS_DPACKM_8XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_8XK_KERNEL
#define BLIS_CPACKM_8XK_KERNEL BLIS_CPACKM_8XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_8XK_KERNEL
#define BLIS_ZPACKM_8XK_KERNEL BLIS_ZPACKM_8XK_KERNEL_REF
#endif

// packm_10xk kernels

#ifndef BLIS_SPACKM_10XK_KERNEL
#define BLIS_SPACKM_10XK_KERNEL BLIS_SPACKM_10XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_10XK_KERNEL
#define BLIS_DPACKM_10XK_KERNEL BLIS_DPACKM_10XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_10XK_KERNEL
#define BLIS_CPACKM_10XK_KERNEL BLIS_CPACKM_10XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_10XK_KERNEL
#define BLIS_ZPACKM_10XK_KERNEL BLIS_ZPACKM_10XK_KERNEL_REF
#endif

// packm_12xk kernels

#ifndef BLIS_SPACKM_12XK_KERNEL
#define BLIS_SPACKM_12XK_KERNEL BLIS_SPACKM_12XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_12XK_KERNEL
#define BLIS_DPACKM_12XK_KERNEL BLIS_DPACKM_12XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_12XK_KERNEL
#define BLIS_CPACKM_12XK_KERNEL BLIS_CPACKM_12XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_12XK_KERNEL
#define BLIS_ZPACKM_12XK_KERNEL BLIS_ZPACKM_12XK_KERNEL_REF
#endif

// packm_14xk kernels

#ifndef BLIS_SPACKM_14XK_KERNEL
#define BLIS_SPACKM_14XK_KERNEL BLIS_SPACKM_14XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_14XK_KERNEL
#define BLIS_DPACKM_14XK_KERNEL BLIS_DPACKM_14XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_14XK_KERNEL
#define BLIS_CPACKM_14XK_KERNEL BLIS_CPACKM_14XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_14XK_KERNEL
#define BLIS_ZPACKM_14XK_KERNEL BLIS_ZPACKM_14XK_KERNEL_REF
#endif

// packm_16xk kernels

#ifndef BLIS_SPACKM_16XK_KERNEL
#define BLIS_SPACKM_16XK_KERNEL BLIS_SPACKM_16XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_16XK_KERNEL
#define BLIS_DPACKM_16XK_KERNEL BLIS_DPACKM_16XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_16XK_KERNEL
#define BLIS_CPACKM_16XK_KERNEL BLIS_CPACKM_16XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_16XK_KERNEL
#define BLIS_ZPACKM_16XK_KERNEL BLIS_ZPACKM_16XK_KERNEL_REF
#endif

// packm_30xk kernels

#ifndef BLIS_SPACKM_30XK_KERNEL
#define BLIS_SPACKM_30XK_KERNEL BLIS_SPACKM_30XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_30XK_KERNEL
#define BLIS_DPACKM_30XK_KERNEL BLIS_DPACKM_30XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_30XK_KERNEL
#define BLIS_CPACKM_30XK_KERNEL BLIS_CPACKM_30XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_30XK_KERNEL
#define BLIS_ZPACKM_30XK_KERNEL BLIS_ZPACKM_30XK_KERNEL_REF
#endif

// unpackm_2xk kernels

#ifndef BLIS_SUNPACKM_2XK_KERNEL
#define BLIS_SUNPACKM_2XK_KERNEL BLIS_SUNPACKM_2XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_2XK_KERNEL
#define BLIS_DUNPACKM_2XK_KERNEL BLIS_DUNPACKM_2XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_2XK_KERNEL
#define BLIS_CUNPACKM_2XK_KERNEL BLIS_CUNPACKM_2XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_2XK_KERNEL
#define BLIS_ZUNPACKM_2XK_KERNEL BLIS_ZUNPACKM_2XK_KERNEL_REF
#endif

// unpackm_4xk kernels

#ifndef BLIS_SUNPACKM_4XK_KERNEL
#define BLIS_SUNPACKM_4XK_KERNEL BLIS_SUNPACKM_4XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_4XK_KERNEL
#define BLIS_DUNPACKM_4XK_KERNEL BLIS_DUNPACKM_4XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_4XK_KERNEL
#define BLIS_CUNPACKM_4XK_KERNEL BLIS_CUNPACKM_4XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_4XK_KERNEL
#define BLIS_ZUNPACKM_4XK_KERNEL BLIS_ZUNPACKM_4XK_KERNEL_REF
#endif

// unpackm_6xk kernels

#ifndef BLIS_SUNPACKM_6XK_KERNEL
#define BLIS_SUNPACKM_6XK_KERNEL BLIS_SUNPACKM_6XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_6XK_KERNEL
#define BLIS_DUNPACKM_6XK_KERNEL BLIS_DUNPACKM_6XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_6XK_KERNEL
#define BLIS_CUNPACKM_6XK_KERNEL BLIS_CUNPACKM_6XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_6XK_KERNEL
#define BLIS_ZUNPACKM_6XK_KERNEL BLIS_ZUNPACKM_6XK_KERNEL_REF
#endif

// unpackm_8xk kernels

#ifndef BLIS_SUNPACKM_8XK_KERNEL
#define BLIS_SUNPACKM_8XK_KERNEL BLIS_SUNPACKM_8XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_8XK_KERNEL
#define BLIS_DUNPACKM_8XK_KERNEL BLIS_DUNPACKM_8XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_8XK_KERNEL
#define BLIS_CUNPACKM_8XK_KERNEL BLIS_CUNPACKM_8XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_8XK_KERNEL
#define BLIS_ZUNPACKM_8XK_KERNEL BLIS_ZUNPACKM_8XK_KERNEL_REF
#endif

// unpackm_10xk kernels

#ifndef BLIS_SUNPACKM_10XK_KERNEL
#define BLIS_SUNPACKM_10XK_KERNEL BLIS_SUNPACKM_10XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_10XK_KERNEL
#define BLIS_DUNPACKM_10XK_KERNEL BLIS_DUNPACKM_10XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_10XK_KERNEL
#define BLIS_CUNPACKM_10XK_KERNEL BLIS_CUNPACKM_10XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_10XK_KERNEL
#define BLIS_ZUNPACKM_10XK_KERNEL BLIS_ZUNPACKM_10XK_KERNEL_REF
#endif

// unpackm_12xk kernels

#ifndef BLIS_SUNPACKM_12XK_KERNEL
#define BLIS_SUNPACKM_12XK_KERNEL BLIS_SUNPACKM_12XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_12XK_KERNEL
#define BLIS_DUNPACKM_12XK_KERNEL BLIS_DUNPACKM_12XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_12XK_KERNEL
#define BLIS_CUNPACKM_12XK_KERNEL BLIS_CUNPACKM_12XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_12XK_KERNEL
#define BLIS_ZUNPACKM_12XK_KERNEL BLIS_ZUNPACKM_12XK_KERNEL_REF
#endif

// unpackm_14xk kernels

#ifndef BLIS_SUNPACKM_14XK_KERNEL
#define BLIS_SUNPACKM_14XK_KERNEL BLIS_SUNPACKM_14XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_14XK_KERNEL
#define BLIS_DUNPACKM_14XK_KERNEL BLIS_DUNPACKM_14XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_14XK_KERNEL
#define BLIS_CUNPACKM_14XK_KERNEL BLIS_CUNPACKM_14XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_14XK_KERNEL
#define BLIS_ZUNPACKM_14XK_KERNEL BLIS_ZUNPACKM_14XK_KERNEL_REF
#endif

// unpackm_16xk kernels

#ifndef BLIS_SUNPACKM_16XK_KERNEL
#define BLIS_SUNPACKM_16XK_KERNEL BLIS_SUNPACKM_16XK_KERNEL_REF
#endif

#ifndef BLIS_DUNPACKM_16XK_KERNEL
#define BLIS_DUNPACKM_16XK_KERNEL BLIS_DUNPACKM_16XK_KERNEL_REF
#endif

#ifndef BLIS_CUNPACKM_16XK_KERNEL
#define BLIS_CUNPACKM_16XK_KERNEL BLIS_CUNPACKM_16XK_KERNEL_REF
#endif

#ifndef BLIS_ZUNPACKM_16XK_KERNEL
#define BLIS_ZUNPACKM_16XK_KERNEL BLIS_ZUNPACKM_16XK_KERNEL_REF
#endif

//
// Level-1f
//

// axpy2v kernels

//#ifndef       AXPY2V_KERNEL
//#define       AXPY2V_KERNEL       AXPY2V_KERNEL_REF
//#endif

#ifndef BLIS_SAXPY2V_KERNEL
#define BLIS_SAXPY2V_KERNEL BLIS_SAXPY2V_KERNEL_REF
#endif

#ifndef BLIS_DAXPY2V_KERNEL
#define BLIS_DAXPY2V_KERNEL BLIS_DAXPY2V_KERNEL_REF
#endif

#ifndef BLIS_CAXPY2V_KERNEL
#define BLIS_CAXPY2V_KERNEL BLIS_CAXPY2V_KERNEL_REF
#endif

#ifndef BLIS_ZAXPY2V_KERNEL
#define BLIS_ZAXPY2V_KERNEL BLIS_ZAXPY2V_KERNEL_REF
#endif

// dotaxpyv kernels

//#ifndef       DOTAXPYV_KERNEL
//#define       DOTAXPYV_KERNEL       DOTAXPYV_KERNEL_REF
//#endif

#ifndef BLIS_SDOTAXPYV_KERNEL
#define BLIS_SDOTAXPYV_KERNEL BLIS_SDOTAXPYV_KERNEL_REF
#endif

#ifndef BLIS_DDOTAXPYV_KERNEL
#define BLIS_DDOTAXPYV_KERNEL BLIS_DDOTAXPYV_KERNEL_REF
#endif

#ifndef BLIS_CDOTAXPYV_KERNEL
#define BLIS_CDOTAXPYV_KERNEL BLIS_CDOTAXPYV_KERNEL_REF
#endif

#ifndef BLIS_ZDOTAXPYV_KERNEL
#define BLIS_ZDOTAXPYV_KERNEL BLIS_ZDOTAXPYV_KERNEL_REF
#endif

// axpyf kernels

//#ifndef       AXPYF_KERNEL
//#define       AXPYF_KERNEL       AXPYF_KERNEL_REF
//#endif

#ifndef BLIS_SAXPYF_KERNEL
#define BLIS_SAXPYF_KERNEL BLIS_SAXPYF_KERNEL_REF
#endif

#ifndef BLIS_DAXPYF_KERNEL
#define BLIS_DAXPYF_KERNEL BLIS_DAXPYF_KERNEL_REF
#endif

#ifndef BLIS_CAXPYF_KERNEL
#define BLIS_CAXPYF_KERNEL BLIS_CAXPYF_KERNEL_REF
#endif

#ifndef BLIS_ZAXPYF_KERNEL
#define BLIS_ZAXPYF_KERNEL BLIS_ZAXPYF_KERNEL_REF
#endif

// dotxf kernels

//#ifndef       DOTXF_KERNEL
//#define       DOTXF_KERNEL       DOTXF_KERNEL_REF
//#endif

#ifndef BLIS_SDOTXF_KERNEL
#define BLIS_SDOTXF_KERNEL BLIS_SDOTXF_KERNEL_REF
#endif

#ifndef BLIS_DDOTXF_KERNEL
#define BLIS_DDOTXF_KERNEL BLIS_DDOTXF_KERNEL_REF
#endif

#ifndef BLIS_CDOTXF_KERNEL
#define BLIS_CDOTXF_KERNEL BLIS_CDOTXF_KERNEL_REF
#endif

#ifndef BLIS_ZDOTXF_KERNEL
#define BLIS_ZDOTXF_KERNEL BLIS_ZDOTXF_KERNEL_REF
#endif

// dotxaxpyf kernels

//#ifndef       DOTXAXPYF_KERNEL
//#define       DOTXAXPYF_KERNEL       DOTXAXPYF_KERNEL_REF
//#endif

#ifndef BLIS_SDOTXAXPYF_KERNEL
#define BLIS_SDOTXAXPYF_KERNEL BLIS_SDOTXAXPYF_KERNEL_REF
#endif

#ifndef BLIS_DDOTXAXPYF_KERNEL
#define BLIS_DDOTXAXPYF_KERNEL BLIS_DDOTXAXPYF_KERNEL_REF
#endif

#ifndef BLIS_CDOTXAXPYF_KERNEL
#define BLIS_CDOTXAXPYF_KERNEL BLIS_CDOTXAXPYF_KERNEL_REF
#endif

#ifndef BLIS_ZDOTXAXPYF_KERNEL
#define BLIS_ZDOTXAXPYF_KERNEL BLIS_ZDOTXAXPYF_KERNEL_REF
#endif

//
// Level-1v
//

// addv kernels

//#ifndef       ADDV_KERNEL
//#define       ADDV_KERNEL       ADDV_KERNEL_REF
//#endif

#ifndef BLIS_SADDV_KERNEL
#define BLIS_SADDV_KERNEL BLIS_SADDV_KERNEL_REF
#endif

#ifndef BLIS_DADDV_KERNEL
#define BLIS_DADDV_KERNEL BLIS_DADDV_KERNEL_REF
#endif

#ifndef BLIS_CADDV_KERNEL
#define BLIS_CADDV_KERNEL BLIS_CADDV_KERNEL_REF
#endif

#ifndef BLIS_ZADDV_KERNEL
#define BLIS_ZADDV_KERNEL BLIS_ZADDV_KERNEL_REF
#endif

// axpyv kernels

//#ifndef       AXPYV_KERNEL
//#define       AXPYV_KERNEL       AXPYV_KERNEL_REF
//#endif

#ifndef BLIS_SAXPYV_KERNEL
#define BLIS_SAXPYV_KERNEL BLIS_SAXPYV_KERNEL_REF
#endif

#ifndef BLIS_DAXPYV_KERNEL
#define BLIS_DAXPYV_KERNEL BLIS_DAXPYV_KERNEL_REF
#endif

#ifndef BLIS_CAXPYV_KERNEL
#define BLIS_CAXPYV_KERNEL BLIS_CAXPYV_KERNEL_REF
#endif

#ifndef BLIS_ZAXPYV_KERNEL
#define BLIS_ZAXPYV_KERNEL BLIS_ZAXPYV_KERNEL_REF
#endif

// copyv kernels

//#ifndef       COPYV_KERNEL
//#define       COPYV_KERNEL       COPYV_KERNEL_REF
//#endif

#ifndef BLIS_SCOPYV_KERNEL
#define BLIS_SCOPYV_KERNEL BLIS_SCOPYV_KERNEL_REF
#endif

#ifndef BLIS_DCOPYV_KERNEL
#define BLIS_DCOPYV_KERNEL BLIS_DCOPYV_KERNEL_REF
#endif

#ifndef BLIS_CCOPYV_KERNEL
#define BLIS_CCOPYV_KERNEL BLIS_CCOPYV_KERNEL_REF
#endif

#ifndef BLIS_ZCOPYV_KERNEL
#define BLIS_ZCOPYV_KERNEL BLIS_ZCOPYV_KERNEL_REF
#endif

// dotv kernels

//#ifndef       DOTV_KERNEL
//#define       DOTV_KERNEL       DOTV_KERNEL_REF
//#endif

#ifndef BLIS_SDOTV_KERNEL
#define BLIS_SDOTV_KERNEL BLIS_SDOTV_KERNEL_REF
#endif

#ifndef BLIS_DDOTV_KERNEL
#define BLIS_DDOTV_KERNEL BLIS_DDOTV_KERNEL_REF
#endif

#ifndef BLIS_CDOTV_KERNEL
#define BLIS_CDOTV_KERNEL BLIS_CDOTV_KERNEL_REF
#endif

#ifndef BLIS_ZDOTV_KERNEL
#define BLIS_ZDOTV_KERNEL BLIS_ZDOTV_KERNEL_REF
#endif

// dotxv kernels

//#ifndef       DOTXV_KERNEL
//#define       DOTXV_KERNEL       DOTXV_KERNEL_REF
//#endif

#ifndef BLIS_SDOTXV_KERNEL
#define BLIS_SDOTXV_KERNEL BLIS_SDOTXV_KERNEL_REF
#endif

#ifndef BLIS_DDOTXV_KERNEL
#define BLIS_DDOTXV_KERNEL BLIS_DDOTXV_KERNEL_REF
#endif

#ifndef BLIS_CDOTXV_KERNEL
#define BLIS_CDOTXV_KERNEL BLIS_CDOTXV_KERNEL_REF
#endif

#ifndef BLIS_ZDOTXV_KERNEL
#define BLIS_ZDOTXV_KERNEL BLIS_ZDOTXV_KERNEL_REF
#endif

// invertv kernels

//#ifndef       INVERTV_KERNEL
//#define       INVERTV_KERNEL       INVERTV_KERNEL_REF
//#endif

#ifndef BLIS_SINVERTV_KERNEL
#define BLIS_SINVERTV_KERNEL BLIS_SINVERTV_KERNEL_REF
#endif

#ifndef BLIS_DINVERTV_KERNEL
#define BLIS_DINVERTV_KERNEL BLIS_DINVERTV_KERNEL_REF
#endif

#ifndef BLIS_CINVERTV_KERNEL
#define BLIS_CINVERTV_KERNEL BLIS_CINVERTV_KERNEL_REF
#endif

#ifndef BLIS_ZINVERTV_KERNEL
#define BLIS_ZINVERTV_KERNEL BLIS_ZINVERTV_KERNEL_REF
#endif

// scal2v kernels

//#ifndef       SCAL2V_KERNEL
//#define       SCAL2V_KERNEL       SCAL2V_KERNEL_REF
//#endif

#ifndef BLIS_SSCAL2V_KERNEL
#define BLIS_SSCAL2V_KERNEL BLIS_SSCAL2V_KERNEL_REF
#endif

#ifndef BLIS_DSCAL2V_KERNEL
#define BLIS_DSCAL2V_KERNEL BLIS_DSCAL2V_KERNEL_REF
#endif

#ifndef BLIS_CSCAL2V_KERNEL
#define BLIS_CSCAL2V_KERNEL BLIS_CSCAL2V_KERNEL_REF
#endif

#ifndef BLIS_ZSCAL2V_KERNEL
#define BLIS_ZSCAL2V_KERNEL BLIS_ZSCAL2V_KERNEL_REF
#endif

// scalv kernels

//#ifndef       SCALV_KERNEL
//#define       SCALV_KERNEL       SCALV_KERNEL_REF
//#endif

#ifndef BLIS_SSCALV_KERNEL
#define BLIS_SSCALV_KERNEL BLIS_SSCALV_KERNEL_REF
#endif

#ifndef BLIS_DSCALV_KERNEL
#define BLIS_DSCALV_KERNEL BLIS_DSCALV_KERNEL_REF
#endif

#ifndef BLIS_CSCALV_KERNEL
#define BLIS_CSCALV_KERNEL BLIS_CSCALV_KERNEL_REF
#endif

#ifndef BLIS_ZSCALV_KERNEL
#define BLIS_ZSCALV_KERNEL BLIS_ZSCALV_KERNEL_REF
#endif

// setv kernels

//#ifndef       SETV_KERNEL
//#define       SETV_KERNEL       SETV_KERNEL_REF
//#endif

#ifndef BLIS_SSETV_KERNEL
#define BLIS_SSETV_KERNEL BLIS_SSETV_KERNEL_REF
#endif

#ifndef BLIS_DSETV_KERNEL
#define BLIS_DSETV_KERNEL BLIS_DSETV_KERNEL_REF
#endif

#ifndef BLIS_CSETV_KERNEL
#define BLIS_CSETV_KERNEL BLIS_CSETV_KERNEL_REF
#endif

#ifndef BLIS_ZSETV_KERNEL
#define BLIS_ZSETV_KERNEL BLIS_ZSETV_KERNEL_REF
#endif

// subv kernels

//#ifndef       SUBV_KERNEL
//#define       SUBV_KERNEL       SUBV_KERNEL_REF
//#endif

#ifndef BLIS_SSUBV_KERNEL
#define BLIS_SSUBV_KERNEL BLIS_SSUBV_KERNEL_REF
#endif

#ifndef BLIS_DSUBV_KERNEL
#define BLIS_DSUBV_KERNEL BLIS_DSUBV_KERNEL_REF
#endif

#ifndef BLIS_CSUBV_KERNEL
#define BLIS_CSUBV_KERNEL BLIS_CSUBV_KERNEL_REF
#endif

#ifndef BLIS_ZSUBV_KERNEL
#define BLIS_ZSUBV_KERNEL BLIS_ZSUBV_KERNEL_REF
#endif

// swapv kernels

//#ifndef       SWAPV_KERNEL
//#define       SWAPV_KERNEL       SWAPV_KERNEL_REF
//#endif

#ifndef BLIS_SSWAPV_KERNEL
#define BLIS_SSWAPV_KERNEL BLIS_SSWAPV_KERNEL_REF
#endif

#ifndef BLIS_DSWAPV_KERNEL
#define BLIS_DSWAPV_KERNEL BLIS_DSWAPV_KERNEL_REF
#endif

#ifndef BLIS_CSWAPV_KERNEL
#define BLIS_CSWAPV_KERNEL BLIS_CSWAPV_KERNEL_REF
#endif

#ifndef BLIS_ZSWAPV_KERNEL
#define BLIS_ZSWAPV_KERNEL BLIS_ZSWAPV_KERNEL_REF
#endif


// -- Define default blocksize macros ------------------------------------------

//
// Define level-3 cache blocksizes.
//

// Define MC minimum

#ifndef BLIS_DEFAULT_MC_S
#define BLIS_DEFAULT_MC_S  512
#endif

#ifndef BLIS_DEFAULT_MC_D
#define BLIS_DEFAULT_MC_D  256
#endif

#ifndef BLIS_DEFAULT_MC_C
#define BLIS_DEFAULT_MC_C  256
#endif

#ifndef BLIS_DEFAULT_MC_Z
#define BLIS_DEFAULT_MC_Z  128
#endif

// Define KC minimum

#ifndef BLIS_DEFAULT_KC_S
#define BLIS_DEFAULT_KC_S  256
#endif

#ifndef BLIS_DEFAULT_KC_D
#define BLIS_DEFAULT_KC_D  256
#endif

#ifndef BLIS_DEFAULT_KC_C
#define BLIS_DEFAULT_KC_C  256
#endif

#ifndef BLIS_DEFAULT_KC_Z
#define BLIS_DEFAULT_KC_Z  256
#endif

// Define NC minimum

#ifndef BLIS_DEFAULT_NC_S
#define BLIS_DEFAULT_NC_S  4096
#endif

#ifndef BLIS_DEFAULT_NC_D
#define BLIS_DEFAULT_NC_D  4096
#endif

#ifndef BLIS_DEFAULT_NC_C
#define BLIS_DEFAULT_NC_C  4096
#endif

#ifndef BLIS_DEFAULT_NC_Z
#define BLIS_DEFAULT_NC_Z  4096
#endif

// Define MC maximum

#ifndef BLIS_MAXIMUM_MC_S
#define BLIS_MAXIMUM_MC_S  BLIS_DEFAULT_MC_S
#endif

#ifndef BLIS_MAXIMUM_MC_D
#define BLIS_MAXIMUM_MC_D  BLIS_DEFAULT_MC_D
#endif

#ifndef BLIS_MAXIMUM_MC_C
#define BLIS_MAXIMUM_MC_C  BLIS_DEFAULT_MC_C
#endif

#ifndef BLIS_MAXIMUM_MC_Z
#define BLIS_MAXIMUM_MC_Z  BLIS_DEFAULT_MC_Z
#endif

// Define KC maximum

#ifndef BLIS_MAXIMUM_KC_S
#define BLIS_MAXIMUM_KC_S  BLIS_DEFAULT_KC_S
#endif

#ifndef BLIS_MAXIMUM_KC_D
#define BLIS_MAXIMUM_KC_D  BLIS_DEFAULT_KC_D
#endif

#ifndef BLIS_MAXIMUM_KC_C
#define BLIS_MAXIMUM_KC_C  BLIS_DEFAULT_KC_C
#endif

#ifndef BLIS_MAXIMUM_KC_Z
#define BLIS_MAXIMUM_KC_Z  BLIS_DEFAULT_KC_Z
#endif

// Define NC maximum

#ifndef BLIS_MAXIMUM_NC_S
#define BLIS_MAXIMUM_NC_S  BLIS_DEFAULT_NC_S
#endif

#ifndef BLIS_MAXIMUM_NC_D
#define BLIS_MAXIMUM_NC_D  BLIS_DEFAULT_NC_D
#endif

#ifndef BLIS_MAXIMUM_NC_C
#define BLIS_MAXIMUM_NC_C  BLIS_DEFAULT_NC_C
#endif

#ifndef BLIS_MAXIMUM_NC_Z
#define BLIS_MAXIMUM_NC_Z  BLIS_DEFAULT_NC_Z
#endif

//
// Define level-3 register blocksizes.
//

// Define MR

#ifndef BLIS_DEFAULT_MR_S
#define BLIS_DEFAULT_MR_S  8
#endif

#ifndef BLIS_DEFAULT_MR_D
#define BLIS_DEFAULT_MR_D  4
#endif

#ifndef BLIS_DEFAULT_MR_C
#define BLIS_DEFAULT_MR_C  4
#endif

#ifndef BLIS_DEFAULT_MR_Z
#define BLIS_DEFAULT_MR_Z  2
#endif

// Define NR

#ifndef BLIS_DEFAULT_NR_S
#define BLIS_DEFAULT_NR_S  4
#endif

#ifndef BLIS_DEFAULT_NR_D
#define BLIS_DEFAULT_NR_D  4
#endif

#ifndef BLIS_DEFAULT_NR_C
#define BLIS_DEFAULT_NR_C  2
#endif

#ifndef BLIS_DEFAULT_NR_Z
#define BLIS_DEFAULT_NR_Z  2
#endif

// Define KR

#ifndef BLIS_DEFAULT_KR_S
#define BLIS_DEFAULT_KR_S  1
#endif

#ifndef BLIS_DEFAULT_KR_D
#define BLIS_DEFAULT_KR_D  1
#endif

#ifndef BLIS_DEFAULT_KR_C
#define BLIS_DEFAULT_KR_C  1
#endif

#ifndef BLIS_DEFAULT_KR_Z
#define BLIS_DEFAULT_KR_Z  1
#endif

// Define MR packdim

#ifndef BLIS_PACKDIM_MR_S
#define BLIS_PACKDIM_MR_S  BLIS_DEFAULT_MR_S
#endif

#ifndef BLIS_PACKDIM_MR_D
#define BLIS_PACKDIM_MR_D  BLIS_DEFAULT_MR_D
#endif

#ifndef BLIS_PACKDIM_MR_C
#define BLIS_PACKDIM_MR_C  BLIS_DEFAULT_MR_C
#endif

#ifndef BLIS_PACKDIM_MR_Z
#define BLIS_PACKDIM_MR_Z  BLIS_DEFAULT_MR_Z
#endif

// Define NR packdim

#ifndef BLIS_PACKDIM_NR_S
#define BLIS_PACKDIM_NR_S  BLIS_DEFAULT_NR_S
#endif

#ifndef BLIS_PACKDIM_NR_D
#define BLIS_PACKDIM_NR_D  BLIS_DEFAULT_NR_D
#endif

#ifndef BLIS_PACKDIM_NR_C
#define BLIS_PACKDIM_NR_C  BLIS_DEFAULT_NR_C
#endif

#ifndef BLIS_PACKDIM_NR_Z
#define BLIS_PACKDIM_NR_Z  BLIS_DEFAULT_NR_Z
#endif

// Define KR packdim

#ifndef BLIS_PACKDIM_KR_S
#define BLIS_PACKDIM_KR_S  BLIS_DEFAULT_KR_S
#endif

#ifndef BLIS_PACKDIM_KR_D
#define BLIS_PACKDIM_KR_D  BLIS_DEFAULT_KR_D
#endif

#ifndef BLIS_PACKDIM_KR_C
#define BLIS_PACKDIM_KR_C  BLIS_DEFAULT_KR_C
#endif

#ifndef BLIS_PACKDIM_KR_Z
#define BLIS_PACKDIM_KR_Z  BLIS_DEFAULT_KR_Z
#endif

//
// Define level-2 blocksizes.
//

// NOTE: These values determine high-level cache blocking for level-2
// operations ONLY. So, if gemv is performed with a 2000x2000 matrix A and
// MC = NC = 1000, then a total of four unblocked (or unblocked fused)
// gemv subproblems are called. The blocked algorithms are only useful in
// that they provide the opportunity for packing vectors. (Matrices can also
// be packed here, but this tends to be much too expensive in practice to
// actually employ.)

#ifndef BLIS_DEFAULT_L2_MC_S
#define BLIS_DEFAULT_L2_MC_S 1000
#endif

#ifndef BLIS_DEFAULT_L2_NC_S
#define BLIS_DEFAULT_L2_NC_S 1000
#endif

#ifndef BLIS_DEFAULT_L2_MC_D
#define BLIS_DEFAULT_L2_MC_D 1000
#endif

#ifndef BLIS_DEFAULT_L2_NC_D
#define BLIS_DEFAULT_L2_NC_D 1000
#endif

#ifndef BLIS_DEFAULT_L2_MC_C
#define BLIS_DEFAULT_L2_MC_C 1000
#endif

#ifndef BLIS_DEFAULT_L2_NC_C
#define BLIS_DEFAULT_L2_NC_C 1000
#endif

#ifndef BLIS_DEFAULT_L2_MC_Z
#define BLIS_DEFAULT_L2_MC_Z 1000
#endif

#ifndef BLIS_DEFAULT_L2_NC_Z
#define BLIS_DEFAULT_L2_NC_Z 1000
#endif

//
// Define level-1f fusing factors.
//

// Global level-1f fusing factors.

#ifndef BLIS_L1F_FUSE_FAC_S
#define BLIS_L1F_FUSE_FAC_S 8
#endif

#ifndef BLIS_L1F_FUSE_FAC_D
#define BLIS_L1F_FUSE_FAC_D 4
#endif

#ifndef BLIS_L1F_FUSE_FAC_C
#define BLIS_L1F_FUSE_FAC_C 4
#endif

#ifndef BLIS_L1F_FUSE_FAC_Z
#define BLIS_L1F_FUSE_FAC_Z 2
#endif

// axpyf

#ifndef BLIS_AXPYF_FUSE_FAC_S
#define BLIS_AXPYF_FUSE_FAC_S BLIS_L1F_FUSE_FAC_S
#endif

#ifndef BLIS_AXPYF_FUSE_FAC_D
#define BLIS_AXPYF_FUSE_FAC_D BLIS_L1F_FUSE_FAC_D
#endif

#ifndef BLIS_AXPYF_FUSE_FAC_C
#define BLIS_AXPYF_FUSE_FAC_C BLIS_L1F_FUSE_FAC_C
#endif

#ifndef BLIS_AXPYF_FUSE_FAC_Z
#define BLIS_AXPYF_FUSE_FAC_Z BLIS_L1F_FUSE_FAC_Z
#endif

// dotxf

#ifndef BLIS_DOTXF_FUSE_FAC_S
#define BLIS_DOTXF_FUSE_FAC_S BLIS_L1F_FUSE_FAC_S
#endif

#ifndef BLIS_DOTXF_FUSE_FAC_D
#define BLIS_DOTXF_FUSE_FAC_D BLIS_L1F_FUSE_FAC_D
#endif

#ifndef BLIS_DOTXF_FUSE_FAC_C
#define BLIS_DOTXF_FUSE_FAC_C BLIS_L1F_FUSE_FAC_C
#endif

#ifndef BLIS_DOTXF_FUSE_FAC_Z
#define BLIS_DOTXF_FUSE_FAC_Z BLIS_L1F_FUSE_FAC_Z
#endif

// dotxaxpyf

#ifndef BLIS_DOTXAXPYF_FUSE_FAC_S
#define BLIS_DOTXAXPYF_FUSE_FAC_S BLIS_L1F_FUSE_FAC_S
#endif

#ifndef BLIS_DOTXAXPYF_FUSE_FAC_D
#define BLIS_DOTXAXPYF_FUSE_FAC_D BLIS_L1F_FUSE_FAC_D
#endif

#ifndef BLIS_DOTXAXPYF_FUSE_FAC_C
#define BLIS_DOTXAXPYF_FUSE_FAC_C BLIS_L1F_FUSE_FAC_C
#endif

#ifndef BLIS_DOTXAXPYF_FUSE_FAC_Z
#define BLIS_DOTXAXPYF_FUSE_FAC_Z BLIS_L1F_FUSE_FAC_Z
#endif

//
// Define level-1v blocksizes.
//

// NOTE: Register blocksizes for vectors are used when packing
// non-contiguous vectors. Similar to that of KR, they can
// typically be set to 1.

#ifndef BLIS_DEFAULT_VR_S
#define BLIS_DEFAULT_VR_S   1
#endif

#ifndef BLIS_DEFAULT_VR_D
#define BLIS_DEFAULT_VR_D   1
#endif

#ifndef BLIS_DEFAULT_VR_C
#define BLIS_DEFAULT_VR_C   1
#endif

#ifndef BLIS_DEFAULT_VR_Z
#define BLIS_DEFAULT_VR_Z   1
#endif


// -- Define micro-panel alignment ---------------------------------------------

// In this section, we consider each datatype-specific alignment sizes for
// micro-panels of A and B. If any definition is undefined, we define it to
// a safe default value (the size of the datatype).

// Alignment for micro-panels of A
#ifndef BLIS_UPANEL_A_ALIGN_SIZE_S
#define BLIS_UPANEL_A_ALIGN_SIZE_S     BLIS_SIZEOF_S
#endif
#ifndef BLIS_UPANEL_A_ALIGN_SIZE_D
#define BLIS_UPANEL_A_ALIGN_SIZE_D     BLIS_SIZEOF_D
#endif
#ifndef BLIS_UPANEL_A_ALIGN_SIZE_C
#define BLIS_UPANEL_A_ALIGN_SIZE_C     BLIS_SIZEOF_C
#endif
#ifndef BLIS_UPANEL_A_ALIGN_SIZE_Z
#define BLIS_UPANEL_A_ALIGN_SIZE_Z     BLIS_SIZEOF_Z
#endif

// Alignment for micro-panels of B
#ifndef BLIS_UPANEL_B_ALIGN_SIZE_S
#define BLIS_UPANEL_B_ALIGN_SIZE_S     BLIS_SIZEOF_S
#endif
#ifndef BLIS_UPANEL_B_ALIGN_SIZE_D
#define BLIS_UPANEL_B_ALIGN_SIZE_D     BLIS_SIZEOF_D
#endif
#ifndef BLIS_UPANEL_B_ALIGN_SIZE_C
#define BLIS_UPANEL_B_ALIGN_SIZE_C     BLIS_SIZEOF_C
#endif
#ifndef BLIS_UPANEL_B_ALIGN_SIZE_Z
#define BLIS_UPANEL_B_ALIGN_SIZE_Z     BLIS_SIZEOF_Z
#endif


// -- Kernel blocksize checks --------------------------------------------------

// Verify that cache blocksizes are whole multiples of register blocksizes.
// Specifically, verify that:
//   - MC is a whole multiple of MR *AND* NR.
//   - NC is a whole multiple of NR *AND* MR.
//   - KC is a whole multiple of KR *AND* both MR, NR.
// These constraints are enforced because it makes it easier to handle diagonals
// in the macro-kernel implementations. 

//
// MC must be a whole multiple of MR and NR.
//

#if ( \
      ( BLIS_DEFAULT_MC_S % BLIS_DEFAULT_MR_S != 0 ) || \
      ( BLIS_DEFAULT_MC_D % BLIS_DEFAULT_MR_D != 0 ) || \
      ( BLIS_DEFAULT_MC_C % BLIS_DEFAULT_MR_C != 0 ) || \
      ( BLIS_DEFAULT_MC_Z % BLIS_DEFAULT_MR_Z != 0 )    \
    )
  #error "MC must be multiple of MR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_MC_S % BLIS_DEFAULT_NR_S != 0 ) || \
      ( BLIS_DEFAULT_MC_D % BLIS_DEFAULT_NR_D != 0 ) || \
      ( BLIS_DEFAULT_MC_C % BLIS_DEFAULT_NR_C != 0 ) || \
      ( BLIS_DEFAULT_MC_Z % BLIS_DEFAULT_NR_Z != 0 )    \
    )
  #error "MC must be multiple of NR for all datatypes."
#endif

//
// NC must be a whole multiple of NR and MR.
//

#if ( \
      ( BLIS_DEFAULT_NC_S % BLIS_DEFAULT_NR_S != 0 ) || \
      ( BLIS_DEFAULT_NC_D % BLIS_DEFAULT_NR_D != 0 ) || \
      ( BLIS_DEFAULT_NC_C % BLIS_DEFAULT_NR_C != 0 ) || \
      ( BLIS_DEFAULT_NC_Z % BLIS_DEFAULT_NR_Z != 0 )    \
    )
  #error "NC must be multiple of NR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_NC_S % BLIS_DEFAULT_MR_S != 0 ) || \
      ( BLIS_DEFAULT_NC_D % BLIS_DEFAULT_MR_D != 0 ) || \
      ( BLIS_DEFAULT_NC_C % BLIS_DEFAULT_MR_C != 0 ) || \
      ( BLIS_DEFAULT_NC_Z % BLIS_DEFAULT_MR_Z != 0 )    \
    )
  #error "NC must be multiple of MR for all datatypes."
#endif

//
// KC must be a whole multiple of KR.
//

#if ( \
      ( BLIS_DEFAULT_KC_S % BLIS_DEFAULT_KR_S != 0 ) || \
      ( BLIS_DEFAULT_KC_D % BLIS_DEFAULT_KR_D != 0 ) || \
      ( BLIS_DEFAULT_KC_C % BLIS_DEFAULT_KR_C != 0 ) || \
      ( BLIS_DEFAULT_KC_Z % BLIS_DEFAULT_KR_Z != 0 )    \
    )
  #error "KC must be multiple of KR for all datatypes."
#endif


// -- Abbreiviated kernel blocksize macros -------------------------------------

// Here, we shorten the blocksizes defined in bli_kernel.h so that they can
// derived via the PASTEMAC macro.

// Default (minimum) cache blocksizes

#define bli_smc      BLIS_DEFAULT_MC_S 
#define bli_skc      BLIS_DEFAULT_KC_S
#define bli_snc      BLIS_DEFAULT_NC_S

#define bli_dmc      BLIS_DEFAULT_MC_D 
#define bli_dkc      BLIS_DEFAULT_KC_D
#define bli_dnc      BLIS_DEFAULT_NC_D

#define bli_cmc      BLIS_DEFAULT_MC_C 
#define bli_ckc      BLIS_DEFAULT_KC_C
#define bli_cnc      BLIS_DEFAULT_NC_C

#define bli_zmc      BLIS_DEFAULT_MC_Z 
#define bli_zkc      BLIS_DEFAULT_KC_Z
#define bli_znc      BLIS_DEFAULT_NC_Z

// Register blocksizes

#define bli_smr      BLIS_DEFAULT_MR_S 
#define bli_skr      BLIS_DEFAULT_KR_S
#define bli_snr      BLIS_DEFAULT_NR_S

#define bli_dmr      BLIS_DEFAULT_MR_D 
#define bli_dkr      BLIS_DEFAULT_KR_D
#define bli_dnr      BLIS_DEFAULT_NR_D

#define bli_cmr      BLIS_DEFAULT_MR_C 
#define bli_ckr      BLIS_DEFAULT_KR_C
#define bli_cnr      BLIS_DEFAULT_NR_C

#define bli_zmr      BLIS_DEFAULT_MR_Z 
#define bli_zkr      BLIS_DEFAULT_KR_Z
#define bli_znr      BLIS_DEFAULT_NR_Z

// Extended (maximum) cache blocksizes

#define bli_smaxmc   BLIS_MAXIMUM_MC_S
#define bli_smaxkc   BLIS_MAXIMUM_KC_S
#define bli_smaxnc   BLIS_MAXIMUM_NC_S

#define bli_dmaxmc   BLIS_MAXIMUM_MC_D
#define bli_dmaxkc   BLIS_MAXIMUM_KC_D
#define bli_dmaxnc   BLIS_MAXIMUM_NC_D

#define bli_cmaxmc   BLIS_MAXIMUM_MC_C
#define bli_cmaxkc   BLIS_MAXIMUM_KC_C
#define bli_cmaxnc   BLIS_MAXIMUM_NC_C

#define bli_zmaxmc   BLIS_MAXIMUM_MC_Z
#define bli_zmaxkc   BLIS_MAXIMUM_KC_Z
#define bli_zmaxnc   BLIS_MAXIMUM_NC_Z

// Extended (packing) register blocksizes

#define bli_spackmr  BLIS_PACKDIM_MR_S
#define bli_spackkr  BLIS_PACKDIM_KR_S
#define bli_spacknr  BLIS_PACKDIM_NR_S

#define bli_dpackmr  BLIS_PACKDIM_MR_D
#define bli_dpackkr  BLIS_PACKDIM_KR_D
#define bli_dpacknr  BLIS_PACKDIM_NR_D

#define bli_cpackmr  BLIS_PACKDIM_MR_C
#define bli_cpackkr  BLIS_PACKDIM_KR_C
#define bli_cpacknr  BLIS_PACKDIM_NR_C

#define bli_zpackmr  BLIS_PACKDIM_MR_Z
#define bli_zpackkr  BLIS_PACKDIM_KR_Z
#define bli_zpacknr  BLIS_PACKDIM_NR_Z

// Level-1f fusing factors

#define bli_saxpyf_fusefac       BLIS_AXPYF_FUSE_FAC_S
#define bli_daxpyf_fusefac       BLIS_AXPYF_FUSE_FAC_D
#define bli_caxpyf_fusefac       BLIS_AXPYF_FUSE_FAC_C
#define bli_zaxpyf_fusefac       BLIS_AXPYF_FUSE_FAC_Z

#define bli_sdotxf_fusefac       BLIS_DOTXF_FUSE_FAC_S
#define bli_ddotxf_fusefac       BLIS_DOTXF_FUSE_FAC_D
#define bli_cdotxf_fusefac       BLIS_DOTXF_FUSE_FAC_C
#define bli_zdotxf_fusefac       BLIS_DOTXF_FUSE_FAC_Z

#define bli_sdotxaxpyf_fusefac   BLIS_DOTXAXPYF_FUSE_FAC_S
#define bli_ddotxaxpyf_fusefac   BLIS_DOTXAXPYF_FUSE_FAC_D
#define bli_cdotxaxpyf_fusefac   BLIS_DOTXAXPYF_FUSE_FAC_C
#define bli_zdotxaxpyf_fusefac   BLIS_DOTXAXPYF_FUSE_FAC_Z

#endif 
