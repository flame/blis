/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

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

// Enable IEEE Standard 1003.1-2004 (POSIX.1d). 
// NOTE: This is needed to enable posix_memalign().
#define _POSIX_C_SOURCE 200112L



// -- FLOATING-POINT PROPERTIES ------------------------------------------------

#define BLIS_NUM_FP_TYPES                4
#define BLIS_MAX_TYPE_SIZE               sizeof(dcomplex)



// -- MEMORY ALLOCATOR ---------------------------------------------------------

// Enable memory alignment?
#define BLIS_ENABLE_MEMORY_ALIGNMENT     1

// If memory alignment is enabled, set the alignment boundary.
#ifndef BLIS_MEMORY_ALIGNMENT_BOUNDARY
  #define BLIS_MEMORY_ALIGNMENT_BOUNDARY 16
#endif

// The number of MC x KC and KC x NC blocks to reserve in the static memory
// pool.
#define BLIS_NUM_MC_X_KC_BLOCKS          2
#define BLIS_NUM_KC_X_NC_BLOCKS          1

// The page size is used by the memory allocator so that static memory
// can be allocated with alignment to the beginning of a page boundary.
#define BLIS_PAGE_SIZE                   4096

// The maximum prefetch byte offset is used to pad the end of any static
// memory allocation request so that the micro-kernel can exceed the
// bounds of the usable portion of a memory region without causing a
// segmentation fault.
#define BLIS_MAX_PREFETCH_BYTE_OFFSET    128



// -- MIXED DATATYPE SUPPORT ---------------------------------------------------

// Basic (homogeneous) datatype support always enabled.

// Enable mixed domain operations?
//#define BLIS_ENABLE_MIXED_DOMAIN_SUPPORT

// Enable extra mixed precision operations?
//#define BLIS_ENABLE_MIXED_PRECISION_SUPPORT



#endif
