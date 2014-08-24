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

#ifndef BLIS_KERNEL_3M_MACRO_DEFS_H
#define BLIS_KERNEL_3M_MACRO_DEFS_H


// -- Define row access bools --------------------------------------------------

// gemm3m micro-kernels

#define BLIS_CGEMM3M_UKERNEL_PREFERS_CONTIG_ROWS \
        BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#define BLIS_ZGEMM3M_UKERNEL_PREFERS_CONTIG_ROWS \
        BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS


// -- Define default 3m-specific kernel names ----------------------------------

//
// Level-3
//

// gemm3m micro-kernels

#ifndef BLIS_CGEMM3M_UKERNEL
#define BLIS_CGEMM3M_UKERNEL BLIS_CGEMM3M_UKERNEL_REF
#endif

#ifndef BLIS_ZGEMM3M_UKERNEL
#define BLIS_ZGEMM3M_UKERNEL BLIS_ZGEMM3M_UKERNEL_REF
#endif

// gemmtrsm3m_l micro-kernels

#ifndef BLIS_CGEMMTRSM3M_L_UKERNEL
#define BLIS_CGEMMTRSM3M_L_UKERNEL BLIS_CGEMMTRSM3M_L_UKERNEL_REF
#endif

#ifndef BLIS_ZGEMMTRSM3M_L_UKERNEL
#define BLIS_ZGEMMTRSM3M_L_UKERNEL BLIS_ZGEMMTRSM3M_L_UKERNEL_REF
#endif

// gemmtrsm3m_u micro-kernels

#ifndef BLIS_CGEMMTRSM3M_U_UKERNEL
#define BLIS_CGEMMTRSM3M_U_UKERNEL BLIS_CGEMMTRSM3M_U_UKERNEL_REF
#endif

#ifndef BLIS_ZGEMMTRSM3M_U_UKERNEL
#define BLIS_ZGEMMTRSM3M_U_UKERNEL BLIS_ZGEMMTRSM3M_U_UKERNEL_REF
#endif

// trsm3m_l micro-kernels

#ifndef BLIS_CTRSM3M_L_UKERNEL
#define BLIS_CTRSM3M_L_UKERNEL BLIS_CTRSM3M_L_UKERNEL_REF
#endif

#ifndef BLIS_ZTRSM3M_L_UKERNEL
#define BLIS_ZTRSM3M_L_UKERNEL BLIS_ZTRSM3M_L_UKERNEL_REF
#endif

// trsm3m_u micro-kernels

#ifndef BLIS_CTRSM3M_U_UKERNEL
#define BLIS_CTRSM3M_U_UKERNEL BLIS_CTRSM3M_U_UKERNEL_REF
#endif

#ifndef BLIS_ZTRSM3M_U_UKERNEL
#define BLIS_ZTRSM3M_U_UKERNEL BLIS_ZTRSM3M_U_UKERNEL_REF
#endif

//
// Level-1m
//

// packm_2xk_3m kernels

#ifndef BLIS_CPACKM_2XK_3M_KERNEL
#define BLIS_CPACKM_2XK_3M_KERNEL BLIS_CPACKM_2XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_2XK_3M_KERNEL
#define BLIS_ZPACKM_2XK_3M_KERNEL BLIS_ZPACKM_2XK_3M_KERNEL_REF
#endif

// packm_4xk_3m kernels

#ifndef BLIS_CPACKM_4XK_3M_KERNEL
#define BLIS_CPACKM_4XK_3M_KERNEL BLIS_CPACKM_4XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_4XK_3M_KERNEL
#define BLIS_ZPACKM_4XK_3M_KERNEL BLIS_ZPACKM_4XK_3M_KERNEL_REF
#endif

// packm_6xk_3m kernels

#ifndef BLIS_CPACKM_6XK_3M_KERNEL
#define BLIS_CPACKM_6XK_3M_KERNEL BLIS_CPACKM_6XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_6XK_3M_KERNEL
#define BLIS_ZPACKM_6XK_3M_KERNEL BLIS_ZPACKM_6XK_3M_KERNEL_REF
#endif

// packm_8xk_3m kernels

#ifndef BLIS_CPACKM_8XK_3M_KERNEL
#define BLIS_CPACKM_8XK_3M_KERNEL BLIS_CPACKM_8XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_8XK_3M_KERNEL
#define BLIS_ZPACKM_8XK_3M_KERNEL BLIS_ZPACKM_8XK_3M_KERNEL_REF
#endif

// packm_10xk_3m kernels

#ifndef BLIS_CPACKM_10XK_3M_KERNEL
#define BLIS_CPACKM_10XK_3M_KERNEL BLIS_CPACKM_10XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_10XK_3M_KERNEL
#define BLIS_ZPACKM_10XK_3M_KERNEL BLIS_ZPACKM_10XK_3M_KERNEL_REF
#endif

// packm_12xk_3m kernels

#ifndef BLIS_CPACKM_12XK_3M_KERNEL
#define BLIS_CPACKM_12XK_3M_KERNEL BLIS_CPACKM_12XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_12XK_3M_KERNEL
#define BLIS_ZPACKM_12XK_3M_KERNEL BLIS_ZPACKM_12XK_3M_KERNEL_REF
#endif

// packm_14xk_3m kernels

#ifndef BLIS_CPACKM_14XK_3M_KERNEL
#define BLIS_CPACKM_14XK_3M_KERNEL BLIS_CPACKM_14XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_14XK_3M_KERNEL
#define BLIS_ZPACKM_14XK_3M_KERNEL BLIS_ZPACKM_14XK_3M_KERNEL_REF
#endif

// packm_16xk_3m kernels

#ifndef BLIS_CPACKM_16XK_3M_KERNEL
#define BLIS_CPACKM_16XK_3M_KERNEL BLIS_CPACKM_16XK_3M_KERNEL_REF
#endif

#ifndef BLIS_ZPACKM_16XK_3M_KERNEL
#define BLIS_ZPACKM_16XK_3M_KERNEL BLIS_ZPACKM_16XK_3M_KERNEL_REF
#endif



// -- Define default 3m-specific blocksize macros ------------------------------

// Define complex 3m register blocksizes in terms of blocksizes used for
// real kernels.

// 3m register blocksizes
#define BLIS_DEFAULT_3M_MR_C     BLIS_DEFAULT_MR_S
#define BLIS_DEFAULT_3M_KR_C     BLIS_DEFAULT_KR_S
#define BLIS_DEFAULT_3M_NR_C     BLIS_DEFAULT_NR_S

#define BLIS_DEFAULT_3M_MR_Z     BLIS_DEFAULT_MR_D
#define BLIS_DEFAULT_3M_KR_Z     BLIS_DEFAULT_KR_D
#define BLIS_DEFAULT_3M_NR_Z     BLIS_DEFAULT_NR_D

// 3m register blocksize extensions
#define BLIS_EXTEND_3M_MR_C      BLIS_EXTEND_MR_S
#define BLIS_EXTEND_3M_KR_C      0
#define BLIS_EXTEND_3M_NR_C      BLIS_EXTEND_NR_S

#define BLIS_EXTEND_3M_MR_Z      BLIS_EXTEND_MR_D
#define BLIS_EXTEND_3M_KR_Z      0
#define BLIS_EXTEND_3M_NR_Z      BLIS_EXTEND_NR_D

// Define complex 3m cache blocksizes in terms of blocksizes used for
// real operations (if they have not yet already been defined).

// 3m cache blocksizes
#ifndef BLIS_DEFAULT_3M_MC_C
#define BLIS_DEFAULT_3M_MC_C     BLIS_DEFAULT_MC_S
#endif
#ifndef BLIS_DEFAULT_3M_KC_C
#define BLIS_DEFAULT_3M_KC_C   ((BLIS_DEFAULT_KC_S)/2)
#endif
#ifndef BLIS_DEFAULT_3M_NC_C
#define BLIS_DEFAULT_3M_NC_C     BLIS_DEFAULT_NC_S
#endif

#ifndef BLIS_DEFAULT_3M_MC_Z
#define BLIS_DEFAULT_3M_MC_Z     BLIS_DEFAULT_MC_D
#endif
#ifndef BLIS_DEFAULT_3M_KC_Z
#define BLIS_DEFAULT_3M_KC_Z   ((BLIS_DEFAULT_KC_D)/2)
#endif
#ifndef BLIS_DEFAULT_3M_NC_Z
#define BLIS_DEFAULT_3M_NC_Z     BLIS_DEFAULT_NC_D
#endif

// 3m cache blocksize extensions
#ifndef BLIS_EXTEND_3M_MC_C
#define BLIS_EXTEND_3M_MC_C      BLIS_EXTEND_MC_S
#endif
#ifndef BLIS_EXTEND_3M_KC_C
#define BLIS_EXTEND_3M_KC_C    ((BLIS_EXTEND_KC_S)/2)
#endif
#ifndef BLIS_EXTEND_3M_NC_C
#define BLIS_EXTEND_3M_NC_C      BLIS_EXTEND_NC_S
#endif

#ifndef BLIS_EXTEND_3M_MC_Z
#define BLIS_EXTEND_3M_MC_Z      BLIS_EXTEND_MC_D
#endif
#ifndef BLIS_EXTEND_3M_KC_Z
#define BLIS_EXTEND_3M_KC_Z    ((BLIS_EXTEND_KC_D)/2)
#endif
#ifndef BLIS_EXTEND_3M_NC_Z
#define BLIS_EXTEND_3M_NC_Z      BLIS_EXTEND_NC_D
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
      ( BLIS_DEFAULT_3M_MC_C % BLIS_DEFAULT_3M_MR_C != 0 ) || \
      ( BLIS_DEFAULT_3M_MC_Z % BLIS_DEFAULT_3M_MR_Z != 0 )    \
    )
  #error "MC (3m) must be multiple of MR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_3M_MC_C % BLIS_DEFAULT_3M_NR_C != 0 ) || \
      ( BLIS_DEFAULT_3M_MC_Z % BLIS_DEFAULT_3M_NR_Z != 0 )    \
    )
  #error "MC (3m) must be multiple of NR for all datatypes."
#endif

//
// NC must be a whole multiple of NR and MR.
//
#if ( \
      ( BLIS_DEFAULT_3M_NC_C % BLIS_DEFAULT_3M_NR_C != 0 ) || \
      ( BLIS_DEFAULT_3M_NC_Z % BLIS_DEFAULT_3M_NR_Z != 0 )    \
    )
  #error "NC (3m) must be multiple of NR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_3M_NC_C % BLIS_DEFAULT_3M_MR_C != 0 ) || \
      ( BLIS_DEFAULT_3M_NC_Z % BLIS_DEFAULT_3M_MR_Z != 0 )    \
    )
  #error "NC (3m) must be multiple of MR for all datatypes."
#endif

//
// KC must be a whole multiple of KR, MR, and NR.
//
#if ( \
      ( BLIS_DEFAULT_3M_KC_C % BLIS_DEFAULT_3M_KR_C != 0 ) || \
      ( BLIS_DEFAULT_3M_KC_Z % BLIS_DEFAULT_3M_KR_Z != 0 )    \
    )
  #error "KC (3m) must be multiple of KR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_3M_KC_C % BLIS_DEFAULT_3M_MR_C != 0 ) || \
      ( BLIS_DEFAULT_3M_KC_Z % BLIS_DEFAULT_3M_MR_Z != 0 )    \
    )
  #error "KC (3m) must be multiple of MR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_3M_KC_C % BLIS_DEFAULT_3M_NR_C != 0 ) || \
      ( BLIS_DEFAULT_3M_KC_Z % BLIS_DEFAULT_3M_NR_Z != 0 )    \
    )
  #error "KC (3m) must be multiple of NR for all datatypes."
#endif



// -- Compute extended blocksizes ----------------------------------------------

//
// Compute maximum cache blocksizes.
//

#define BLIS_MAXIMUM_3M_MC_C  ( BLIS_DEFAULT_3M_MC_C + BLIS_EXTEND_3M_MC_C )
#define BLIS_MAXIMUM_3M_KC_C  ( BLIS_DEFAULT_3M_KC_C + BLIS_EXTEND_3M_KC_C )
#define BLIS_MAXIMUM_3M_NC_C  ( BLIS_DEFAULT_3M_NC_C + BLIS_EXTEND_3M_NC_C )

#define BLIS_MAXIMUM_3M_MC_Z  ( BLIS_DEFAULT_3M_MC_Z + BLIS_EXTEND_3M_MC_Z )
#define BLIS_MAXIMUM_3M_KC_Z  ( BLIS_DEFAULT_3M_KC_Z + BLIS_EXTEND_3M_KC_Z )
#define BLIS_MAXIMUM_3M_NC_Z  ( BLIS_DEFAULT_3M_NC_Z + BLIS_EXTEND_3M_NC_Z )

//
// Compute leading dimension blocksizes used when packing micro-panels.
//

#define BLIS_PACKDIM_3M_MR_C  ( BLIS_DEFAULT_3M_MR_C + BLIS_EXTEND_3M_MR_C )
#define BLIS_PACKDIM_3M_KR_C  ( BLIS_DEFAULT_3M_KR_C + BLIS_EXTEND_3M_KR_C )
#define BLIS_PACKDIM_3M_NR_C  ( BLIS_DEFAULT_3M_NR_C + BLIS_EXTEND_3M_NR_C )

#define BLIS_PACKDIM_3M_MR_Z  ( BLIS_DEFAULT_3M_MR_Z + BLIS_EXTEND_3M_MR_Z )
#define BLIS_PACKDIM_3M_KR_Z  ( BLIS_DEFAULT_3M_KR_Z + BLIS_EXTEND_3M_KR_Z )
#define BLIS_PACKDIM_3M_NR_Z  ( BLIS_DEFAULT_3M_NR_Z + BLIS_EXTEND_3M_NR_Z )



#endif 
