/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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

#ifndef BLIS_KERNEL_POST_MACRO_DEFS_H
#define BLIS_KERNEL_POST_MACRO_DEFS_H

// -- Maximum register blocksize search ----------------------------------------

//
// Find the largest register blocksize MR.
//

#define BLIS_MAX_DEFAULT_MR_S BLIS_DEFAULT_MR_S
#define BLIS_MAX_DEFAULT_MR_D BLIS_DEFAULT_MR_D
#define BLIS_MAX_DEFAULT_MR_C BLIS_DEFAULT_MR_C
#define BLIS_MAX_DEFAULT_MR_Z BLIS_DEFAULT_MR_Z

#define BLIS_MAX_DEFAULT_NR_S BLIS_DEFAULT_NR_S
#define BLIS_MAX_DEFAULT_NR_D BLIS_DEFAULT_NR_D
#define BLIS_MAX_DEFAULT_NR_C BLIS_DEFAULT_NR_C
#define BLIS_MAX_DEFAULT_NR_Z BLIS_DEFAULT_NR_Z


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

