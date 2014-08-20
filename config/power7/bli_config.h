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

#ifndef BLIS_CONFIG_H
#define BLIS_CONFIG_H


// -- OPERATING SYSTEM ---------------------------------------------------------



// The bit size of the integer type used to track values such as dimensions,
// strides, diagonal offsets. A value of 32 results in BLIS using 32-bit signed
// integers while 64 results in 64-bit integers. Any other value results in use
// of the C99 type "long int". Note that this ONLY affects integers used
// internally within BLIS as well as those exposed in the native BLAS-like BLIS
// interface.
#define BLIS_INT_TYPE_SIZE               64



// -- FLOATING-POINT PROPERTIES ------------------------------------------------

// Define the number of floating-point types supported, and the size of the
// largest type.
#define BLIS_NUM_FP_TYPES                4
#define BLIS_MAX_TYPE_SIZE               sizeof(dcomplex)

// Enable use of built-in C99 "float complex" and "double complex" types and
// associated overloaded operations and functions? Disabling results in
// scomplex and dcomplex being defined in terms of simple structs.
//#define BLIS_ENABLE_C99_COMPLEX



// -- MULTITHREADING -----------------------------------------------------------

// The maximum number of BLIS threads that will run concurrently.
#define BLIS_MAX_NUM_THREADS             24



// -- MEMORY ALLOCATION --------------------------------------------------------

// -- Contiguous (static) memory allocator --

// The number of MC x KC, KC x NC, and MC x NC blocks to reserve in the
// contiguous memory pools.
#define BLIS_NUM_MC_X_KC_BLOCKS          BLIS_MAX_NUM_THREADS
#define BLIS_NUM_KC_X_NC_BLOCKS          1
#define BLIS_NUM_MC_X_NC_BLOCKS          0

// The maximum preload byte offset is used to pad the end of the contiguous
// memory pools so that the micro-kernel, when computing with the end of the
// last block, can exceed the bounds of the usable portion of the memory
// region without causing a segmentation fault.
#define BLIS_MAX_PRELOAD_BYTE_OFFSET     128

// -- Memory alignment --

// It is sometimes useful to define the various memory alignments in terms
// of some other characteristics of the system, such as the cache line size
// and the page size.
#define BLIS_CACHE_LINE_SIZE             64
#define BLIS_PAGE_SIZE                   4096

// Alignment size needed by the instruction set for aligned SIMD/vector
// instructions.
#define BLIS_SIMD_ALIGN_SIZE             16

// Alignment size used to align local stack buffers within macro-kernel
// functions.
#define BLIS_STACK_BUF_ALIGN_SIZE        16

// Alignment size used when allocating memory dynamically from the operating
// system (eg: posix_memalign()). To disable heap alignment and just use
// malloc() instead, set this to 1.
#define BLIS_HEAP_ADDR_ALIGN_SIZE        16

// Alignment size used when sizing leading dimensions of dynamically
// allocated memory.
#define BLIS_HEAP_STRIDE_ALIGN_SIZE      BLIS_CACHE_LINE_SIZE

// Alignment size used when allocating entire blocks of contiguous memory
// from the contiguous memory allocator.
#define BLIS_CONTIG_ADDR_ALIGN_SIZE      BLIS_PAGE_SIZE



// -- MIXED DATATYPE SUPPORT ---------------------------------------------------

// Basic (homogeneous) datatype support always enabled.

// Enable mixed domain operations?
//#define BLIS_ENABLE_MIXED_DOMAIN_SUPPORT

// Enable extra mixed precision operations?
//#define BLIS_ENABLE_MIXED_PRECISION_SUPPORT



// -- MISCELLANEOUS OPTIONS ----------------------------------------------------

// Stay initialized after auto-initialization, unless and until the user
// explicitly calls bli_finalize().
#define BLIS_ENABLE_STAY_AUTO_INITIALIZED



// -- BLAS COMPATIBILITY LAYER -------------------------------------------------

// Enable the BLAS compatibility layer?
#define BLIS_ENABLE_BLAS2BLIS

// The bit size of the integer type used to track values such as dimensions and
// leading dimensions (ie: column strides) within the BLAS compatibility layer.
// A value of 32 results in the compatibility layer using 32-bit signed integers
// while 64 results in 64-bit integers. Any other value results in use of the
// C99 type "long int". Note that this ONLY affects integers used within the
// BLAS compatibility layer.
#define BLIS_BLAS2BLIS_INT_TYPE_SIZE     64

// Fortran-77 name-mangling macros.
#define PASTEF770(name)                        name ## _
#define PASTEF77(ch1,name)       ch1        ## name ## _
#define PASTEF772(ch1,ch2,name)  ch1 ## ch2 ## name ## _



// -- CBLAS COMPATIBILITY LAYER ------------------------------------------------

// Enable the CBLAS compatibility layer?
// NOTE: Enabling CBLAS will automatically enable the BLAS compatibility layer
// regardless of whether or not it was explicitly enabled above. Furthermore,
// the CBLAS compatibility layer will use the integer type size definition
// specified above when defining the size of its own integers (regardless of
// whether the BLAS layer was enabled directly or indirectly).
//#define BLIS_ENABLE_CBLAS




#endif

