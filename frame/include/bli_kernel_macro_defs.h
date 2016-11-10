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


// -- MEMORY ALLOCATION --------------------------------------------------------

// Memory allocation functions. These macros define the three types of
// malloc()-style functions, and their free() counterparts: one for each
// type of memory to be allocated.
// NOTE: ANY ALTERNATIVE TO malloc()/free() USED FOR ANY OF THE FOLLOWING
// THREE PAIRS OF MACROS MUST USE THE SAME FUNCTION PROTOTYPE AS malloc()
// and free():
//
//   void* malloc( size_t size );
//   void  free( void* p );
//

// This allocation function is called to allocate memory for blocks within
// BLIS's internal memory pools.
#ifndef BLIS_MALLOC_POOL
#define BLIS_MALLOC_POOL                 malloc
#endif

#ifndef BLIS_FREE_POOL
#define BLIS_FREE_POOL                   free
#endif

// This allocation function is called to allocate memory for internally-
// used objects and structures, such as control tree nodes.
#ifndef BLIS_MALLOC_INTL
#define BLIS_MALLOC_INTL                 malloc
#endif

#ifndef BLIS_FREE_INTL
#define BLIS_FREE_INTL                   free
#endif

// This allocation function is called to allocate memory for objects
// created by user-level API functions, such as bli_obj_create().
#ifndef BLIS_MALLOC_USER
#define BLIS_MALLOC_USER                 malloc
#endif

#ifndef BLIS_FREE_USER
#define BLIS_FREE_USER                   free
#endif

// Size of a virtual memory page. This is used to align blocks within the
// memory pools.
#ifndef BLIS_PAGE_SIZE
#define BLIS_PAGE_SIZE                   4096
#endif

// Number of named SIMD vector registers available for use.
#ifndef BLIS_SIMD_NUM_REGISTERS
#define BLIS_SIMD_NUM_REGISTERS          16
#endif

// Size (in bytes) of each SIMD vector.
#ifndef BLIS_SIMD_SIZE
#define BLIS_SIMD_SIZE                   32
#endif

// Alignment size (in bytes) needed by the instruction set for aligned
// SIMD/vector instructions.
#ifndef BLIS_SIMD_ALIGN_SIZE
#define BLIS_SIMD_ALIGN_SIZE             BLIS_SIMD_SIZE
#endif

// The maximum size in bytes of local stack buffers within macro-kernel
// functions. These buffers are usually used to store a temporary copy
// of a single microtile. The reason we multiply by 2 is to handle induced
// methods, where we use real domain register blocksizes in units of
// complex elements. Specifically, the macro-kernels will need this larger
// micro-tile footprint, even though the virtual micro-kernels will only
// ever be writing to half (real or imaginary part) at a time.
#ifndef BLIS_STACK_BUF_MAX_SIZE
#define BLIS_STACK_BUF_MAX_SIZE          ( BLIS_SIMD_NUM_REGISTERS * \
                                           BLIS_SIMD_SIZE * 2 )
#endif

// Alignment size used to align local stack buffers within macro-kernel
// functions.
#define BLIS_STACK_BUF_ALIGN_SIZE        BLIS_SIMD_ALIGN_SIZE

// Alignment size used when allocating memory via BLIS_MALLOC_USER.
// To disable heap alignment, set this to 1.
#define BLIS_HEAP_ADDR_ALIGN_SIZE        BLIS_SIMD_ALIGN_SIZE

// Alignment size used when sizing leading dimensions of memory allocated
// via BLIS_MALLOC_USER.
#define BLIS_HEAP_STRIDE_ALIGN_SIZE      BLIS_SIMD_ALIGN_SIZE

// Alignment size used when allocating blocks to the internal memory
// pool, via BLIS_MALLOC_POOL.
#define BLIS_POOL_ADDR_ALIGN_SIZE        BLIS_PAGE_SIZE


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
// In the case of complex gemm micro-kernels, we also define special macros
// so that later on we can tell whether or not to employ the induced
// implementations. Note that in order to properly determine whether the
// induced method is a viable option, we need to be able to test the
// existence of the real gemm micro-kernels, which means we must consider
// the complex gemm micro-kernel cases *BEFORE* the real cases.

//
// Level-3
//

// gemm micro-kernels

#ifndef BLIS_CGEMM_UKERNEL
#define BLIS_CGEMM_UKERNEL BLIS_CGEMM_UKERNEL_REF
#ifdef  BLIS_SGEMM_UKERNEL
#define BLIS_ENABLE_INDUCED_SCOMPLEX
#endif
#else
#endif

#ifndef BLIS_ZGEMM_UKERNEL
#define BLIS_ZGEMM_UKERNEL BLIS_ZGEMM_UKERNEL_REF
#ifdef  BLIS_DGEMM_UKERNEL
#define BLIS_ENABLE_INDUCED_DCOMPLEX
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

// packm_24xk kernels

#ifndef BLIS_SPACKM_24XK_KERNEL
#define BLIS_SPACKM_24XK_KERNEL BLIS_SPACKM_24XK_KERNEL_REF
#endif

#ifndef BLIS_DPACKM_24XK_KERNEL
#define BLIS_DPACKM_24XK_KERNEL BLIS_DPACKM_24XK_KERNEL_REF
#endif

#ifndef BLIS_CPACKM_24XK_KERNEL
#define BLIS_CPACKM_24XK_KERNEL BLIS_CPACKM_24XK_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_24XK_KERNEL
#define BLIS_ZPACKM_24XK_KERNEL BLIS_ZPACKM_24XK_KERNEL_REF
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

// amaxv kernels

#ifndef BLIS_SAMAXV_KERNEL
#define BLIS_SAMAXV_KERNEL BLIS_SAMAXV_KERNEL_REF
#endif

#ifndef BLIS_DAMAXV_KERNEL
#define BLIS_DAMAXV_KERNEL BLIS_DAMAXV_KERNEL_REF
#endif

#ifndef BLIS_CAMAXV_KERNEL
#define BLIS_CAMAXV_KERNEL BLIS_CAMAXV_KERNEL_REF
#endif

#ifndef BLIS_ZAMAXV_KERNEL
#define BLIS_ZAMAXV_KERNEL BLIS_ZAMAXV_KERNEL_REF
#endif

// addv kernels

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

// axpbyv kernels

#ifndef BLIS_SAXPBYV_KERNEL
#define BLIS_SAXPBYV_KERNEL BLIS_SAXPBYV_KERNEL_REF
#endif

#ifndef BLIS_DAXPBYV_KERNEL
#define BLIS_DAXPBYV_KERNEL BLIS_DAXPBYV_KERNEL_REF
#endif

#ifndef BLIS_CAXPBYV_KERNEL
#define BLIS_CAXPBYV_KERNEL BLIS_CAXPBYV_KERNEL_REF
#endif

#ifndef BLIS_ZAXPBYV_KERNEL
#define BLIS_ZAXPBYV_KERNEL BLIS_ZAXPBYV_KERNEL_REF
#endif

// axpyv kernels

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

// xpbyv kernels

#ifndef BLIS_SXPBYV_KERNEL
#define BLIS_SXPBYV_KERNEL BLIS_SXPBYV_KERNEL_REF
#endif

#ifndef BLIS_DXPBYV_KERNEL
#define BLIS_DXPBYV_KERNEL BLIS_DXPBYV_KERNEL_REF
#endif

#ifndef BLIS_CXPBYV_KERNEL
#define BLIS_CXPBYV_KERNEL BLIS_CXPBYV_KERNEL_REF
#endif

#ifndef BLIS_ZXPBYV_KERNEL
#define BLIS_ZXPBYV_KERNEL BLIS_ZXPBYV_KERNEL_REF
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
// M2 = N2 = 1000, then a total of four unblocked (or unblocked fused)
// gemv subproblems are called. The blocked algorithms are only useful in
// that they provide the opportunity for packing vectors. (Matrices can also
// be packed here, but this tends to be much too expensive in practice to
// actually employ.)

#ifndef BLIS_DEFAULT_M2_S
#define BLIS_DEFAULT_M2_S 1000
#endif

#ifndef BLIS_DEFAULT_N2_S
#define BLIS_DEFAULT_N2_S 1000
#endif

#ifndef BLIS_DEFAULT_M2_D
#define BLIS_DEFAULT_M2_D 1000
#endif

#ifndef BLIS_DEFAULT_N2_D
#define BLIS_DEFAULT_N2_D 1000
#endif

#ifndef BLIS_DEFAULT_M2_C
#define BLIS_DEFAULT_M2_C 1000
#endif

#ifndef BLIS_DEFAULT_N2_C
#define BLIS_DEFAULT_N2_C 1000
#endif

#ifndef BLIS_DEFAULT_M2_Z
#define BLIS_DEFAULT_M2_Z 1000
#endif

#ifndef BLIS_DEFAULT_N2_Z
#define BLIS_DEFAULT_N2_Z 1000
#endif

//
// Define level-1f fusing factors.
//

// Global level-1f fusing factors.

#ifndef BLIS_DEFAULT_1F_S
#define BLIS_DEFAULT_1F_S 8
#endif

#ifndef BLIS_DEFAULT_1F_D
#define BLIS_DEFAULT_1F_D 4
#endif

#ifndef BLIS_DEFAULT_1F_C
#define BLIS_DEFAULT_1F_C 4
#endif

#ifndef BLIS_DEFAULT_1F_Z
#define BLIS_DEFAULT_1F_Z 2
#endif

// axpyf

#ifndef BLIS_DEFAULT_AF_S
#define BLIS_DEFAULT_AF_S BLIS_DEFAULT_1F_S
#endif

#ifndef BLIS_DEFAULT_AF_D
#define BLIS_DEFAULT_AF_D BLIS_DEFAULT_1F_D
#endif

#ifndef BLIS_DEFAULT_AF_C
#define BLIS_DEFAULT_AF_C BLIS_DEFAULT_1F_C
#endif

#ifndef BLIS_DEFAULT_AF_Z
#define BLIS_DEFAULT_AF_Z BLIS_DEFAULT_1F_Z
#endif

// dotxf

#ifndef BLIS_DEFAULT_DF_S
#define BLIS_DEFAULT_DF_S BLIS_DEFAULT_1F_S
#endif

#ifndef BLIS_DEFAULT_DF_D
#define BLIS_DEFAULT_DF_D BLIS_DEFAULT_1F_D
#endif

#ifndef BLIS_DEFAULT_DF_C
#define BLIS_DEFAULT_DF_C BLIS_DEFAULT_1F_C
#endif

#ifndef BLIS_DEFAULT_DF_Z
#define BLIS_DEFAULT_DF_Z BLIS_DEFAULT_1F_Z
#endif

// dotxaxpyf

#ifndef BLIS_DEFAULT_XF_S
#define BLIS_DEFAULT_XF_S BLIS_DEFAULT_1F_S
#endif

#ifndef BLIS_DEFAULT_XF_D
#define BLIS_DEFAULT_XF_D BLIS_DEFAULT_1F_D
#endif

#ifndef BLIS_DEFAULT_XF_C
#define BLIS_DEFAULT_XF_C BLIS_DEFAULT_1F_C
#endif

#ifndef BLIS_DEFAULT_XF_Z
#define BLIS_DEFAULT_XF_Z BLIS_DEFAULT_1F_Z
#endif

//
// Define level-1v blocksizes.
//

// NOTE: Register blocksizes for vectors are used when packing
// non-contiguous vectors. Similar to that of KR, they can
// typically be set to 1.

#ifndef BLIS_DEFAULT_VF_S
#define BLIS_DEFAULT_VF_S   1
#endif

#ifndef BLIS_DEFAULT_VF_D
#define BLIS_DEFAULT_VF_D   1
#endif

#ifndef BLIS_DEFAULT_VF_C
#define BLIS_DEFAULT_VF_C   1
#endif

#ifndef BLIS_DEFAULT_VF_Z
#define BLIS_DEFAULT_VF_Z   1
#endif


// -- Define default threading parameters --------------------------------------


#ifndef BLIS_DEFAULT_M_THREAD_RATIO
#define BLIS_DEFAULT_M_THREAD_RATIO 2
#endif

#ifndef BLIS_DEFAULT_N_THREAD_RATIO
#define BLIS_DEFAULT_N_THREAD_RATIO 1
#endif

#ifndef BLIS_DEFAULT_MR_THREAD_MAX
#define BLIS_DEFAULT_MR_THREAD_MAX 1
#endif

#ifndef BLIS_DEFAULT_NR_THREAD_MAX
#define BLIS_DEFAULT_NR_THREAD_MAX 4
#endif


// -- Kernel blocksize checks --------------------------------------------------

// Verify that cache blocksizes are whole multiples of register blocksizes.
// Specifically, verify that:
//   - MC is a whole multiple of MR.
//   - NC is a whole multiple of NR.
//   - KC is a whole multiple of KR.
// These constraints are enforced because it makes it easier to handle diagonals
// in the macro-kernel implementations. Additionally, we optionally verify that:
//   - MC is a whole multiple of NR.
//   - NC is a whole multiple of MR.
// These latter constraints, guarded by #ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
// below, are only enforced when we wish to be able to handle the trsm right-
// side case handling that swaps A and B, so that B is the triangular matrix,
// with NR blocking used to pack A and MR blocking used to pack B, with the
// arguments to the gemmtrsm microkernel swapped at the last minute, as the
// kernel is called.

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

#ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
#if ( \
      ( BLIS_DEFAULT_MC_S % BLIS_DEFAULT_NR_S != 0 ) || \
      ( BLIS_DEFAULT_MC_D % BLIS_DEFAULT_NR_D != 0 ) || \
      ( BLIS_DEFAULT_MC_C % BLIS_DEFAULT_NR_C != 0 ) || \
      ( BLIS_DEFAULT_MC_Z % BLIS_DEFAULT_NR_Z != 0 )    \
    )
  #error "MC must be multiple of NR for all datatypes."
#endif
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

#ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
#if ( \
      ( BLIS_DEFAULT_NC_S % BLIS_DEFAULT_MR_S != 0 ) || \
      ( BLIS_DEFAULT_NC_D % BLIS_DEFAULT_MR_D != 0 ) || \
      ( BLIS_DEFAULT_NC_C % BLIS_DEFAULT_MR_C != 0 ) || \
      ( BLIS_DEFAULT_NC_Z % BLIS_DEFAULT_MR_Z != 0 )    \
    )
  #error "NC must be multiple of MR for all datatypes."
#endif
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


#endif 
