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

#ifndef BLIS_POOL_BLOCKS_MACRO_DEFS_H
#define BLIS_POOL_BLOCKS_MACRO_DEFS_H


// -- Memory pool block sizing macros ------------------------------------------

// In this file, we compute the memory pool block sizes for A, B, and C for
// each floating-point datatype, and then search for and save the maximum.
// The reason we settle on the largest is to prevent a developer from
// implementing a micro-kernel for one datatype (say, single real) and then
// end up in a situation where the memory pool is not large enough because
// the cache blocksize value of the datatype used to size the pool (e.g.
// double) was not updated accordingly.

// First we compute possibly scaling factors for each datatype. These
// scaling factors actually take the form of numerator and denominator
// since we want stay in integer arithmetic. The purpose of the scaling
// factors is to increase the amount of space we reserve for the memory
// pool blocks if one of the packed micro-panels has a "leading dimension"
// that is larger than the register blocksize. (In this case, the leading
// dimension of a micro-panel is the default register blocksize plus its
// corresponding extension.)

// Note that when computing the scaling factor, we have to determine which
// of PACKDIM_MR/DEFAULT_MR and PACKDIM_NR/DEFAULT_NR is greater so that
// the pair of values can be used to scale MAXIMUM_MC and MAXIMUM_NC. This
// is needed ONLY because the amount of space allocated for a block of A
// and a panel of B needs to be such that MR and NR can be swapped (ie: A
// is packed with NR and B is packed with MR). This transformation is
// needed for right-side trsm when inducing an algorithm that (a) has
// favorable access patterns for column-stored C and (b) allows the
// macro-kernel to reuse the existing left-side fused gemmtrsm micro-kernels.
// We cross-multiply so that the comparison can stay in integer arithmetic.

//
// Compute scaling factors for single real.
//
#if ( BLIS_PACKDIM_MR_S * BLIS_DEFAULT_NR_S ) >= \
    ( BLIS_PACKDIM_NR_S * BLIS_DEFAULT_MR_S )
  #define BLIS_PACKDIM_MAXR_S BLIS_PACKDIM_MR_S
  #define BLIS_DEFAULT_MAXR_S BLIS_DEFAULT_MR_S
#else
  #define BLIS_PACKDIM_MAXR_S BLIS_PACKDIM_NR_S
  #define BLIS_DEFAULT_MAXR_S BLIS_DEFAULT_NR_S
#endif

//
// Compute scaling factors for double real.
//
#if ( BLIS_PACKDIM_MR_D * BLIS_DEFAULT_NR_D ) >= \
    ( BLIS_PACKDIM_NR_D * BLIS_DEFAULT_MR_D )
  #define BLIS_PACKDIM_MAXR_D BLIS_PACKDIM_MR_D
  #define BLIS_DEFAULT_MAXR_D BLIS_DEFAULT_MR_D
#else
  #define BLIS_PACKDIM_MAXR_D BLIS_PACKDIM_NR_D
  #define BLIS_DEFAULT_MAXR_D BLIS_DEFAULT_NR_D
#endif

//
// Compute scaling factors for single complex.
//
#if ( BLIS_PACKDIM_MR_C * BLIS_DEFAULT_NR_C ) >= \
    ( BLIS_PACKDIM_NR_C * BLIS_DEFAULT_MR_C )
  #define BLIS_PACKDIM_MAXR_C BLIS_PACKDIM_MR_C
  #define BLIS_DEFAULT_MAXR_C BLIS_DEFAULT_MR_C
#else
  #define BLIS_PACKDIM_MAXR_C BLIS_PACKDIM_NR_C
  #define BLIS_DEFAULT_MAXR_C BLIS_DEFAULT_NR_C
#endif

//
// Compute scaling factors for double complex.
//
#if ( BLIS_PACKDIM_MR_Z * BLIS_DEFAULT_NR_Z ) >= \
    ( BLIS_PACKDIM_NR_Z * BLIS_DEFAULT_MR_Z )
  #define BLIS_PACKDIM_MAXR_Z BLIS_PACKDIM_MR_Z
  #define BLIS_DEFAULT_MAXR_Z BLIS_DEFAULT_MR_Z
#else
  #define BLIS_PACKDIM_MAXR_Z BLIS_PACKDIM_NR_Z
  #define BLIS_DEFAULT_MAXR_Z BLIS_DEFAULT_NR_Z
#endif

// Next, we define the dimensions of the pool blocks for each datatype.

//
// Compute pool dimensions for single real
//
#define BLIS_POOL_MC_S     ( ( BLIS_MAXIMUM_MC_S * BLIS_PACKDIM_MAXR_S ) \
                                                 / BLIS_DEFAULT_MAXR_S )
#define BLIS_POOL_NC_S     ( ( BLIS_MAXIMUM_NC_S * BLIS_PACKDIM_MAXR_S ) \
                                                 / BLIS_DEFAULT_MAXR_S )
#define BLIS_POOL_KC_S     ( ( BLIS_MAXIMUM_KC_S * BLIS_PACKDIM_KR_S   ) \
                                                 / BLIS_DEFAULT_KR_S   )

//
// Compute pool dimensions for double real
//
#define BLIS_POOL_MC_D     ( ( BLIS_MAXIMUM_MC_D * BLIS_PACKDIM_MAXR_D ) \
                                                 / BLIS_DEFAULT_MAXR_D )
#define BLIS_POOL_NC_D     ( ( BLIS_MAXIMUM_NC_D * BLIS_PACKDIM_MAXR_D ) \
                                                 / BLIS_DEFAULT_MAXR_D )
#define BLIS_POOL_KC_D     ( ( BLIS_MAXIMUM_KC_D * BLIS_PACKDIM_KR_D   ) \
                                                 / BLIS_DEFAULT_KR_D   )

//
// Compute pool dimensions for single complex
//
#define BLIS_POOL_MC_C     ( ( BLIS_MAXIMUM_MC_C * BLIS_PACKDIM_MAXR_C ) \
                                                 / BLIS_DEFAULT_MAXR_C )
#define BLIS_POOL_NC_C     ( ( BLIS_MAXIMUM_NC_C * BLIS_PACKDIM_MAXR_C ) \
                                                 / BLIS_DEFAULT_MAXR_C )
#define BLIS_POOL_KC_C     ( ( BLIS_MAXIMUM_KC_C * BLIS_PACKDIM_KR_C   ) \
                                                 / BLIS_DEFAULT_KR_C   )

//
// Compute pool dimensions for double complex
//
#define BLIS_POOL_MC_Z     ( ( BLIS_MAXIMUM_MC_Z * BLIS_PACKDIM_MAXR_Z ) \
                                                 / BLIS_DEFAULT_MAXR_Z )
#define BLIS_POOL_NC_Z     ( ( BLIS_MAXIMUM_NC_Z * BLIS_PACKDIM_MAXR_Z ) \
                                                 / BLIS_DEFAULT_MAXR_Z )
#define BLIS_POOL_KC_Z     ( ( BLIS_MAXIMUM_KC_Z * BLIS_PACKDIM_KR_Z   ) \
                                                 / BLIS_DEFAULT_KR_Z   )

// Now, we compute the size of each block/panel of A, B, and C for each
// datatype.

// NOTE: In defining each BLIS_*_BLOCK_SIZE_? macro below, we assume the
// "worst case" of the register blocking being unit, in which case every row
// of A and column of B would need padding to allow for alignment of every
// packed micro-panel. (This is the worst case since for MR,NR > 1, padding
// is only needed for every few rows of A and columns of B.)

//
// Compute memory pool block sizes for single real.
//
#define BLIS_MK_BLOCK_SIZE_S ( BLIS_POOL_MC_S * \
                               ( BLIS_POOL_KC_S + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_S \
                                 ) \
                               ) * \
                               BLIS_SIZEOF_S \
                             )
#define BLIS_KN_BLOCK_SIZE_S ( ( BLIS_POOL_KC_S + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_S \
                                 ) \
                               ) * \
                               BLIS_POOL_NC_S * \
                               BLIS_SIZEOF_S \
                             )
#define BLIS_MN_BLOCK_SIZE_S ( BLIS_POOL_MC_S * \
                               BLIS_POOL_NC_S * \
                               BLIS_SIZEOF_S \
                             )

//
// Compute memory pool block sizes for double real.
//
#define BLIS_MK_BLOCK_SIZE_D ( BLIS_POOL_MC_D * \
                               ( BLIS_POOL_KC_D + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_D \
                                 ) \
                               ) * \
                               BLIS_SIZEOF_D \
                             )
#define BLIS_KN_BLOCK_SIZE_D ( ( BLIS_POOL_KC_D + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_D \
                                 ) \
                               ) * \
                               BLIS_POOL_NC_D * \
                               BLIS_SIZEOF_D \
                             )
#define BLIS_MN_BLOCK_SIZE_D ( BLIS_POOL_MC_D * \
                               BLIS_POOL_NC_D * \
                               BLIS_SIZEOF_D \
                             )

//
// Compute memory pool block sizes for single complex.
//
#define BLIS_MK_BLOCK_SIZE_C ( BLIS_POOL_MC_C * \
                               ( BLIS_POOL_KC_C + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_C \
                                 ) \
                               ) * \
                               BLIS_SIZEOF_C \
                             )
#define BLIS_KN_BLOCK_SIZE_C ( ( BLIS_POOL_KC_C + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_C \
                                 ) \
                               ) * \
                               BLIS_POOL_NC_C * \
                               BLIS_SIZEOF_C \
                             )
#define BLIS_MN_BLOCK_SIZE_C ( BLIS_POOL_MC_C * \
                               BLIS_POOL_NC_C * \
                               BLIS_SIZEOF_C \
                             )

//
// Compute memory pool block sizes for single complex.
//
#define BLIS_MK_BLOCK_SIZE_Z ( BLIS_POOL_MC_Z * \
                               ( BLIS_POOL_KC_Z + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_Z \
                                 ) \
                               ) * \
                               BLIS_SIZEOF_Z \
                             )
#define BLIS_KN_BLOCK_SIZE_Z ( ( BLIS_POOL_KC_Z + \
                                 ( BLIS_CONTIG_STRIDE_ALIGN_SIZE / \
                                   BLIS_SIZEOF_Z \
                                 ) \
                               ) * \
                               BLIS_POOL_NC_Z * \
                               BLIS_SIZEOF_Z \
                             )
#define BLIS_MN_BLOCK_SIZE_Z ( BLIS_POOL_MC_Z * \
                               BLIS_POOL_NC_Z * \
                               BLIS_SIZEOF_Z \
                             )


// -- Maximum block size search ------------------------------------------------

// In this section, we find the largest of each block size and save the result
// in a new macro for later use in bli_mem.c.

//
// Find the largest block size for blocks of A.
//
#define BLIS_MK_BLOCK_SIZE BLIS_MK_BLOCK_SIZE_S
#if     BLIS_MK_BLOCK_SIZE_D > BLIS_MK_BLOCK_SIZE
#undef  BLIS_MK_BLOCK_SIZE
#define BLIS_MK_BLOCK_SIZE BLIS_MK_BLOCK_SIZE_D
#endif
#if     BLIS_MK_BLOCK_SIZE_C > BLIS_MK_BLOCK_SIZE
#undef  BLIS_MK_BLOCK_SIZE
#define BLIS_MK_BLOCK_SIZE BLIS_MK_BLOCK_SIZE_C
#endif
#if     BLIS_MK_BLOCK_SIZE_Z > BLIS_MK_BLOCK_SIZE
#undef  BLIS_MK_BLOCK_SIZE
#define BLIS_MK_BLOCK_SIZE BLIS_MK_BLOCK_SIZE_Z
#endif

//
// Find the largest block size for panels of B.
//
#define BLIS_KN_BLOCK_SIZE BLIS_KN_BLOCK_SIZE_S
#if     BLIS_KN_BLOCK_SIZE_D > BLIS_KN_BLOCK_SIZE
#undef  BLIS_KN_BLOCK_SIZE
#define BLIS_KN_BLOCK_SIZE BLIS_KN_BLOCK_SIZE_D
#endif
#if     BLIS_KN_BLOCK_SIZE_C > BLIS_KN_BLOCK_SIZE
#undef  BLIS_KN_BLOCK_SIZE
#define BLIS_KN_BLOCK_SIZE BLIS_KN_BLOCK_SIZE_C
#endif
#if     BLIS_KN_BLOCK_SIZE_Z > BLIS_KN_BLOCK_SIZE
#undef  BLIS_KN_BLOCK_SIZE
#define BLIS_KN_BLOCK_SIZE BLIS_KN_BLOCK_SIZE_Z
#endif

//
// Find the largest block size for panels of C.
//
#define BLIS_MN_BLOCK_SIZE BLIS_MN_BLOCK_SIZE_S
#if     BLIS_MN_BLOCK_SIZE_D > BLIS_MN_BLOCK_SIZE
#undef  BLIS_MN_BLOCK_SIZE
#define BLIS_MN_BLOCK_SIZE BLIS_MN_BLOCK_SIZE_D
#endif
#if     BLIS_MN_BLOCK_SIZE_C > BLIS_MN_BLOCK_SIZE
#undef  BLIS_MN_BLOCK_SIZE
#define BLIS_MN_BLOCK_SIZE BLIS_MN_BLOCK_SIZE_C
#endif
#if     BLIS_MN_BLOCK_SIZE_Z > BLIS_MN_BLOCK_SIZE
#undef  BLIS_MN_BLOCK_SIZE
#define BLIS_MN_BLOCK_SIZE BLIS_MN_BLOCK_SIZE_Z
#endif


// -- Compute pool sizes -------------------------------------------------------


// Define each pool's total size using the block sizes determined above.

//
// Pool for MC x KC blocks of A.
//
#define BLIS_MK_POOL_SIZE  ( \
                             BLIS_NUM_MC_X_KC_BLOCKS * \
                             ( BLIS_MK_BLOCK_SIZE + \
                               BLIS_CONTIG_ADDR_ALIGN_SIZE \
                             ) + \
                             BLIS_MAX_PRELOAD_BYTE_OFFSET \
                           )

//
// Pool for KC x NC panels of B.
//
#define BLIS_KN_POOL_SIZE  ( \
                             BLIS_NUM_KC_X_NC_BLOCKS * \
                             ( BLIS_KN_BLOCK_SIZE + \
                               BLIS_CONTIG_ADDR_ALIGN_SIZE \
                             ) + \
                             BLIS_MAX_PRELOAD_BYTE_OFFSET \
                           )

//
// Pool for MC x NC panels of C.
//
#define BLIS_MN_POOL_SIZE  ( \
                             BLIS_NUM_MC_X_NC_BLOCKS * \
                             ( BLIS_MN_BLOCK_SIZE + \
                               BLIS_CONTIG_ADDR_ALIGN_SIZE \
                             ) + \
                             BLIS_MAX_PRELOAD_BYTE_OFFSET \
                           )


#endif 
