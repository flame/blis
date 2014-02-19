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

#ifndef BLIS_KERNEL_4M_MACRO_DEFS_H
#define BLIS_KERNEL_4M_MACRO_DEFS_H


// -- Define datatype-agnostic base 4m kernel names ----------------------------

// If any datatype-agnostic base name for a given micro-kernel does not
// exist, default to the reference kernel base name.

#ifndef GEMM4M_UKERNEL
#define GEMM4M_UKERNEL        gemm4m_ref_mxn
#endif

#ifndef GEMMTRSM4M_L_UKERNEL
#define GEMMTRSM4M_L_UKERNEL  gemmtrsm4m_l_ref_mxn
#endif

#ifndef GEMMTRSM4M_U_UKERNEL
#define GEMMTRSM4M_U_UKERNEL  gemmtrsm4m_u_ref_mxn
#endif

#ifndef TRSM4M_L_UKERNEL
#define TRSM4M_L_UKERNEL      trsm4m_l_ref_mxn
#endif

#ifndef TRSM4M_U_UKERNEL
#define TRSM4M_U_UKERNEL      trsm4m_u_ref_mxn
#endif


// -- Construct 4m kernel function names ---------------------------------------

// For each datatype-specific micro-kernel base name, construct the full
// function name of the corresponding micro-kernel.

#define BLIS_CGEMM4M_UKERNEL       PASTEMAC(c,GEMM4M_UKERNEL)
#define BLIS_ZGEMM4M_UKERNEL       PASTEMAC(z,GEMM4M_UKERNEL)

#define BLIS_CGEMMTRSM4M_L_UKERNEL PASTEMAC(c,GEMMTRSM4M_L_UKERNEL)
#define BLIS_ZGEMMTRSM4M_L_UKERNEL PASTEMAC(z,GEMMTRSM4M_L_UKERNEL)

#define BLIS_CGEMMTRSM4M_U_UKERNEL PASTEMAC(c,GEMMTRSM4M_U_UKERNEL)
#define BLIS_ZGEMMTRSM4M_U_UKERNEL PASTEMAC(z,GEMMTRSM4M_U_UKERNEL)

#define BLIS_CTRSM4M_L_UKERNEL     PASTEMAC(c,TRSM4M_L_UKERNEL)
#define BLIS_ZTRSM4M_L_UKERNEL     PASTEMAC(z,TRSM4M_L_UKERNEL)

#define BLIS_CTRSM4M_U_UKERNEL     PASTEMAC(c,TRSM4M_U_UKERNEL)
#define BLIS_ZTRSM4M_U_UKERNEL     PASTEMAC(z,TRSM4M_U_UKERNEL)


// -- Define 4m-specific blocksize macros --------------------------------------

// Define complex 4m register blocksizes in terms of blocksizes used for
// real kernels.

// 4m register blocksizes
#define BLIS_DEFAULT_4M_MR_C     BLIS_DEFAULT_MR_S
#define BLIS_DEFAULT_4M_KR_C     BLIS_DEFAULT_KR_S
#define BLIS_DEFAULT_4M_NR_C     BLIS_DEFAULT_NR_S

#define BLIS_DEFAULT_4M_MR_Z     BLIS_DEFAULT_MR_D
#define BLIS_DEFAULT_4M_KR_Z     BLIS_DEFAULT_KR_D
#define BLIS_DEFAULT_4M_NR_Z     BLIS_DEFAULT_NR_D

// 4m register blocksize extensions
#define BLIS_EXTEND_4M_MR_C      BLIS_EXTEND_MR_S
#define BLIS_EXTEND_4M_KR_C      BLIS_EXTEND_KR_S
#define BLIS_EXTEND_4M_NR_C      BLIS_EXTEND_NR_S

#define BLIS_EXTEND_4M_MR_Z      BLIS_EXTEND_MR_D
#define BLIS_EXTEND_4M_KR_Z      BLIS_EXTEND_KR_D
#define BLIS_EXTEND_4M_NR_Z      BLIS_EXTEND_NR_D

// Define complex 4m cache blocksizes in terms of blocksizes used for
// real operations (if they have not yet already been defined).

// 4m cache blocksizes
#ifndef BLIS_DEFAULT_4M_MC_C
#define BLIS_DEFAULT_4M_MC_C     ((BLIS_DEFAULT_MC_S)/1)
#endif
#ifndef BLIS_DEFAULT_4M_KC_C
#define BLIS_DEFAULT_4M_KC_C     ((BLIS_DEFAULT_KC_S)/2)
#endif
#ifndef BLIS_DEFAULT_4M_NC_C
#define BLIS_DEFAULT_4M_NC_C     ((BLIS_DEFAULT_NC_S)/1)
#endif

#ifndef BLIS_DEFAULT_4M_MC_Z
#define BLIS_DEFAULT_4M_MC_Z     ((BLIS_DEFAULT_MC_D)/1)
#endif
#ifndef BLIS_DEFAULT_4M_KC_Z
#define BLIS_DEFAULT_4M_KC_Z     ((BLIS_DEFAULT_KC_D)/2)
#endif
#ifndef BLIS_DEFAULT_4M_NC_Z
#define BLIS_DEFAULT_4M_NC_Z     ((BLIS_DEFAULT_NC_D)/1)
#endif

// 4m cache blocksize extensions
#ifndef BLIS_EXTEND_4M_MC_C
#define BLIS_EXTEND_4M_MC_C      0
#endif
#ifndef BLIS_EXTEND_4M_KC_C
#define BLIS_EXTEND_4M_KC_C      0
#endif
#ifndef BLIS_EXTEND_4M_NC_C
#define BLIS_EXTEND_4M_NC_C      0
#endif

#ifndef BLIS_EXTEND_4M_MC_Z
#define BLIS_EXTEND_4M_MC_Z      0
#endif
#ifndef BLIS_EXTEND_4M_KC_Z
#define BLIS_EXTEND_4M_KC_Z      0
#endif
#ifndef BLIS_EXTEND_4M_NC_Z
#define BLIS_EXTEND_4M_NC_Z      0
#endif


// -- Kernel macro checks ------------------------------------------------------

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
      ( BLIS_DEFAULT_4M_MC_C % BLIS_DEFAULT_4M_MR_C != 0 ) || \
      ( BLIS_DEFAULT_4M_MC_Z % BLIS_DEFAULT_4M_MR_Z != 0 )    \
    )
  #error "MC (4m) must be multiple of MR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_4M_MC_C % BLIS_DEFAULT_4M_NR_C != 0 ) || \
      ( BLIS_DEFAULT_4M_MC_Z % BLIS_DEFAULT_4M_NR_Z != 0 )    \
    )
  #error "MC (4m) must be multiple of NR for all datatypes."
#endif

//
// NC must be a whole multiple of NR and MR.
//
#if ( \
      ( BLIS_DEFAULT_4M_NC_C % BLIS_DEFAULT_4M_NR_C != 0 ) || \
      ( BLIS_DEFAULT_4M_NC_Z % BLIS_DEFAULT_4M_NR_Z != 0 )    \
    )
  #error "NC (4m) must be multiple of NR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_4M_NC_C % BLIS_DEFAULT_4M_MR_C != 0 ) || \
      ( BLIS_DEFAULT_4M_NC_Z % BLIS_DEFAULT_4M_MR_Z != 0 )    \
    )
  #error "NC (4m) must be multiple of MR for all datatypes."
#endif

//
// KC must be a whole multiple of KR, MR, and NR.
//
#if ( \
      ( BLIS_DEFAULT_4M_KC_C % BLIS_DEFAULT_4M_KR_C != 0 ) || \
      ( BLIS_DEFAULT_4M_KC_Z % BLIS_DEFAULT_4M_KR_Z != 0 )    \
    )
  #error "KC (4m) must be multiple of KR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_4M_KC_C % BLIS_DEFAULT_4M_MR_C != 0 ) || \
      ( BLIS_DEFAULT_4M_KC_Z % BLIS_DEFAULT_4M_MR_Z != 0 )    \
    )
  #error "KC (4m) must be multiple of MR for all datatypes."
#endif

#if ( \
      ( BLIS_DEFAULT_4M_KC_C % BLIS_DEFAULT_4M_NR_C != 0 ) || \
      ( BLIS_DEFAULT_4M_KC_Z % BLIS_DEFAULT_4M_NR_Z != 0 )    \
    )
  #error "KC (4m) must be multiple of NR for all datatypes."
#endif


// -- Compute maximum cache blocksizes -----------------------------------------

#define BLIS_MAXIMUM_4M_MC_C  ( BLIS_DEFAULT_4M_MC_C + BLIS_EXTEND_4M_MC_C )
#define BLIS_MAXIMUM_4M_KC_C  ( BLIS_DEFAULT_4M_KC_C + BLIS_EXTEND_4M_KC_C )
#define BLIS_MAXIMUM_4M_NC_C  ( BLIS_DEFAULT_4M_NC_C + BLIS_EXTEND_4M_NC_C )

#define BLIS_MAXIMUM_4M_MC_Z  ( BLIS_DEFAULT_4M_MC_Z + BLIS_EXTEND_4M_MC_Z )
#define BLIS_MAXIMUM_4M_KC_Z  ( BLIS_DEFAULT_4M_KC_Z + BLIS_EXTEND_4M_KC_Z )
#define BLIS_MAXIMUM_4M_NC_Z  ( BLIS_DEFAULT_4M_NC_Z + BLIS_EXTEND_4M_NC_Z )

// -- Compute leading dim blocksizes used for packing --------------------------

#define BLIS_PACKDIM_4M_MR_C  ( BLIS_DEFAULT_4M_MR_C + BLIS_EXTEND_4M_MR_C )
#define BLIS_PACKDIM_4M_KR_C  ( BLIS_DEFAULT_4M_KR_C + BLIS_EXTEND_4M_KR_C )
#define BLIS_PACKDIM_4M_NR_C  ( BLIS_DEFAULT_4M_NR_C + BLIS_EXTEND_4M_NR_C )

#define BLIS_PACKDIM_4M_MR_Z  ( BLIS_DEFAULT_4M_MR_Z + BLIS_EXTEND_4M_MR_Z )
#define BLIS_PACKDIM_4M_KR_Z  ( BLIS_DEFAULT_4M_KR_Z + BLIS_EXTEND_4M_KR_Z )
#define BLIS_PACKDIM_4M_NR_Z  ( BLIS_DEFAULT_4M_NR_Z + BLIS_EXTEND_4M_NR_Z )



#endif 
