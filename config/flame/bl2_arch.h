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

#ifndef BLIS_ARCH_H
#define BLIS_ARCH_H

// -- General floating-point constants --

#define BLIS_NUM_FP_TYPES              4
#define BLIS_MAX_TYPE_SIZE             sizeof(dcomplex)


// -- Maximum offset of an element that might be pre-loaded/prefetched --

#define BLIS_MAX_PREFETCH_BYTE_OFFSET  128


// -- Page size --

#define BLIS_PAGE_SIZE                 4096


// -- Number of elements per vector register --

#define BLIS_NUM_ELEM_PER_REG_S        4
#define BLIS_NUM_ELEM_PER_REG_D        2
#define BLIS_NUM_ELEM_PER_REG_C        2
#define BLIS_NUM_ELEM_PER_REG_Z        1


// -- Default fusing factors for level-1 fused operations --

#define BLIS_DEFAULT_FUSING_FACTOR_S   8
#define BLIS_DEFAULT_FUSING_FACTOR_D   4
#define BLIS_DEFAULT_FUSING_FACTOR_C   4
#define BLIS_DEFAULT_FUSING_FACTOR_Z   2


// -- Default cache blocksizes --

// Constraints:
//
// (1) MC must be a multiple of:
//     (a) MR (for zero-padding purposes) and
//     (b) NR.
// (2) NC must be a multiple of
//     (a) NR (for zero-padding purposes) and
//     (b) MR.
// (3) KC does not need to be multiple of anything, unless the micro-kernel
//     specifically requires it (and typically it does not).
// 
// NOTE: For BLIS libraries built on block-panel macro-kernels, constraint
// (2b) is relaxed. In this case, (1b) is needed for operation implementations
// involving matrices with diagonals (trmm, trsm). In these cases, we want the
// diagonal offset of any panel of packed matrix A to have a diagonal offset
// that is a multiple of MR. If, instead, the library were to be built on
// block-panel macro-kernels, matrix B would be the one with structure, not A,
// and thus it would be constraint (2b) that would be needed instead of (1b).
//

#define BLIS_DEFAULT_MC_S              128
#define BLIS_DEFAULT_KC_S              256
#define BLIS_DEFAULT_NC_S              8192

#define BLIS_DEFAULT_MC_D              128
#define BLIS_DEFAULT_KC_D              256
#define BLIS_DEFAULT_NC_D              8192

#define BLIS_DEFAULT_MC_C              128
#define BLIS_DEFAULT_KC_C              256
#define BLIS_DEFAULT_NC_C              8192

#define BLIS_DEFAULT_MC_Z              128
#define BLIS_DEFAULT_KC_Z              256
#define BLIS_DEFAULT_NC_Z              8192


// -- Default register blocksizes for inner kernel --

#define BLIS_DEFAULT_MR_S              8
#define BLIS_DEFAULT_NR_S              2

#define BLIS_DEFAULT_MR_D              4
#define BLIS_DEFAULT_NR_D              4

#define BLIS_DEFAULT_MR_C              4
#define BLIS_DEFAULT_NR_C              1

#define BLIS_DEFAULT_MR_Z              2
#define BLIS_DEFAULT_NR_Z              1

// NOTE: If the micro-kernel, which is typically unrolled to a factor
// of f, handles leftover edge cases (ie: when k % f > 0) then these
// register blocksizes in the k dimension can be defined to 1.

#define BLIS_DEFAULT_KR_S              1
#define BLIS_DEFAULT_KR_D              1
#define BLIS_DEFAULT_KR_C              1
#define BLIS_DEFAULT_KR_Z              1


// -- Default switch for duplication of B --

// NOTE: If BLIS_DEFAULT_DUPLICATE_B is set to FALSE, then the
// NUM_DUPL definitions are not used.

#define BLIS_DEFAULT_DUPLICATE_B       TRUE
#define BLIS_DEFAULT_NUM_DUPL_S        BLIS_NUM_ELEM_PER_REG_S
#define BLIS_DEFAULT_NUM_DUPL_D        BLIS_NUM_ELEM_PER_REG_D
#define BLIS_DEFAULT_NUM_DUPL_C        BLIS_NUM_ELEM_PER_REG_C
#define BLIS_DEFAULT_NUM_DUPL_Z        BLIS_NUM_ELEM_PER_REG_Z


// -- Default incremental packing blocksizes (n dimension) --

// NOTE: These incremental packing blocksizes (for the n dimension) are only
// used by certain blocked variants. But when the *are* used, they MUST be
// be an integer multiple of NR!

#define BLIS_DEFAULT_NI_FAC            16
#define BLIS_DEFAULT_NI_S              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_S)
#define BLIS_DEFAULT_NI_D              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_D)
#define BLIS_DEFAULT_NI_C              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_C)
#define BLIS_DEFAULT_NI_Z              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_Z)


// -- Default register blocksizes for vectors --

// NOTE: Register blocksizes for vectors are used when packing
// non-contiguous vectors. Similar to that of KR, they can
// typically be set to 1.

#define BLIS_DEFAULT_VR_S              1
#define BLIS_DEFAULT_VR_D              1
#define BLIS_DEFAULT_VR_C              1
#define BLIS_DEFAULT_VR_Z              1




#endif
