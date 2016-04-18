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

#ifndef BLIS_KERNEL_POST_MACRO_DEFS_H
#define BLIS_KERNEL_POST_MACRO_DEFS_H


// -- Maximum register blocksize search ----------------------------------------

// The macro-kernels oftentimes need to statically allocate a temporary
// MR x NR micro-tile of C. This micro-tile must be sized such that it will
// work for both native and induced implementations, since the user can switch
// between them at runtime. In order to facilitate the sizing of those
// micro-tiles, we must determine the largest the register blocksizes would
// need to be to accommodate both native and induced-based complex
// micro-kernels. For real datatypes, the maximum is never larger than the
// actual s and d register blocksizes. However, for complex datatypes, the
// "native" register blocksizes may differ from the "virtual" register
// blocksizes used by the induced implementations. Usually, it is the register
// blocksizes used for induced-based complex micro-kernels that would be
// larger, and thus determine the maximum for c and z datatypes. But, we
// prefer not to assume this, therefore, we always take the larger of the
// two values.

#define BLIS_DEFAULT_IND_MR_C BLIS_DEFAULT_MR_S
#define BLIS_DEFAULT_IND_NR_C BLIS_DEFAULT_NR_S
#define BLIS_DEFAULT_IND_MR_Z BLIS_DEFAULT_MR_D
#define BLIS_DEFAULT_IND_NR_Z BLIS_DEFAULT_NR_D

//
// Find the largest register blocksize MR.
//

#define BLIS_MAX_DEFAULT_MR_S BLIS_DEFAULT_MR_S
#define BLIS_MAX_DEFAULT_MR_D BLIS_DEFAULT_MR_D

// Choose between the native and induced blocksize for scomplex.
#define BLIS_MAX_DEFAULT_MR_C BLIS_DEFAULT_MR_C
#if     BLIS_DEFAULT_IND_MR_C > BLIS_MAX_DEFAULT_MR_C
#undef  BLIS_MAX_DEFAULT_MR_C
#define BLIS_MAX_DEFAULT_MR_C BLIS_DEFAULT_IND_MR_C
#endif

// Choose between the native and induced blocksize for dcomplex.
#define BLIS_MAX_DEFAULT_MR_Z BLIS_DEFAULT_MR_Z
#if     BLIS_DEFAULT_IND_MR_Z > BLIS_MAX_DEFAULT_MR_Z
#undef  BLIS_MAX_DEFAULT_MR_Z
#define BLIS_MAX_DEFAULT_MR_Z BLIS_DEFAULT_IND_MR_Z
#endif

//
// Find the largest register blocksize NR.
//

#define BLIS_MAX_DEFAULT_NR_S BLIS_DEFAULT_NR_S
#define BLIS_MAX_DEFAULT_NR_D BLIS_DEFAULT_NR_D

// Choose between the native and induced blocksize for scomplex.
#define BLIS_MAX_DEFAULT_NR_C BLIS_DEFAULT_NR_C
#if     BLIS_DEFAULT_IND_NR_C > BLIS_MAX_DEFAULT_NR_C
#undef  BLIS_MAX_DEFAULT_NR_C
#define BLIS_MAX_DEFAULT_NR_C BLIS_DEFAULT_IND_NR_C
#endif

// Choose between the native and induced blocksize for dcomplex.
#define BLIS_MAX_DEFAULT_NR_Z BLIS_DEFAULT_NR_Z
#if     BLIS_DEFAULT_IND_NR_Z > BLIS_MAX_DEFAULT_NR_Z
#undef  BLIS_MAX_DEFAULT_NR_Z
#define BLIS_MAX_DEFAULT_NR_Z BLIS_DEFAULT_IND_NR_Z
#endif


// -- Abbreiviated macros ------------------------------------------------------

// Here, we shorten the maximum blocksizes found above so that they can be
// derived via the PASTEMAC macro.

// Maximum MR blocksizes

#define bli_smaxmr   BLIS_MAX_DEFAULT_MR_S
#define bli_dmaxmr   BLIS_MAX_DEFAULT_MR_D
#define bli_cmaxmr   BLIS_MAX_DEFAULT_MR_C
#define bli_zmaxmr   BLIS_MAX_DEFAULT_MR_Z

// Maximum NR blocksizes

#define bli_smaxnr   BLIS_MAX_DEFAULT_NR_S
#define bli_dmaxnr   BLIS_MAX_DEFAULT_NR_D
#define bli_cmaxnr   BLIS_MAX_DEFAULT_NR_C
#define bli_zmaxnr   BLIS_MAX_DEFAULT_NR_Z


#endif 

