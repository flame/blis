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

#ifndef BLIS_KERNEL_H
#define BLIS_KERNEL_H


// -- LEVEL-3 MICRO-KERNEL CONSTANTS -------------------------------------------

// -- Default cache blocksizes --

//
// Constraints:
//
// (1) MC must be a multiple of:
//     (a) MR (for zero-padding purposes)
//     (b) NR (for zero-padding purposes when MR and NR are "swapped")
// (2) NC must be a multiple of
//     (a) NR (for zero-padding purposes)
//     (b) MR (for zero-padding purposes when MR and NR are "swapped")
// (3) KC must be a multiple of
//     (a) MR and
//     (b) NR (for triangular operations such as trmm and trsm).
//

#define BLIS_DEFAULT_MC_S              64
#define BLIS_DEFAULT_KC_S              128
#define BLIS_DEFAULT_NC_S              4096

#define BLIS_DEFAULT_MC_D              96
#define BLIS_DEFAULT_KC_D              256
#define BLIS_DEFAULT_NC_D              4096

#define BLIS_DEFAULT_MC_C              64
#define BLIS_DEFAULT_KC_C              128
#define BLIS_DEFAULT_NC_C              4096

#define BLIS_DEFAULT_MC_Z              64
#define BLIS_DEFAULT_KC_Z              128
#define BLIS_DEFAULT_NC_Z              4096

// -- Cache blocksize extensions (for optimizing edge cases) --

// NOTE: These cache blocksize "extensions" have the same constraints as
// the corresponding default blocksizes above. When these values are
// non-zero, blocksizes used at edge cases are extended (enlarged) if
// such an extension would encompass the remaining portion of the
// matrix dimension.

#define BLIS_EXTEND_MC_S               0 //(BLIS_DEFAULT_MC_S/4)
#define BLIS_EXTEND_KC_S               0 //(BLIS_DEFAULT_KC_S/4)
#define BLIS_EXTEND_NC_S               0 //(BLIS_DEFAULT_NC_S/4)

#define BLIS_EXTEND_MC_D               0 //(BLIS_DEFAULT_MC_D/4)
#define BLIS_EXTEND_KC_D               0 //(BLIS_DEFAULT_KC_D/4)
#define BLIS_EXTEND_NC_D               0 //(BLIS_DEFAULT_NC_D/4)

#define BLIS_EXTEND_MC_C               0 //(BLIS_DEFAULT_MC_C/4)
#define BLIS_EXTEND_KC_C               0 //(BLIS_DEFAULT_KC_C/4)
#define BLIS_EXTEND_NC_C               0 //(BLIS_DEFAULT_NC_C/4)

#define BLIS_EXTEND_MC_Z               0 //(BLIS_DEFAULT_MC_Z/4)
#define BLIS_EXTEND_KC_Z               0 //(BLIS_DEFAULT_KC_Z/4)
#define BLIS_EXTEND_NC_Z               0 //(BLIS_DEFAULT_NC_Z/4)

// -- Default register blocksizes for micro-kernel --

// NOTE: When using the reference configuration, these register blocksizes
// in the m and n dimensions should all be equal to the size expected by
// the reference micro-kernel(s).

#define BLIS_DEFAULT_MR_S              8
#define BLIS_DEFAULT_NR_S              4

#define BLIS_DEFAULT_MR_D              8
#define BLIS_DEFAULT_NR_D              4

#define BLIS_DEFAULT_MR_C              8
#define BLIS_DEFAULT_NR_C              4

#define BLIS_DEFAULT_MR_Z              8
#define BLIS_DEFAULT_NR_Z              4

// NOTE: If the micro-kernel, which is typically unrolled to a factor
// of f, handles leftover edge cases (ie: when k % f > 0) then these
// register blocksizes in the k dimension can be defined to 1.

#define BLIS_DEFAULT_KR_S              1
#define BLIS_DEFAULT_KR_D              1
#define BLIS_DEFAULT_KR_C              1
#define BLIS_DEFAULT_KR_Z              1

// -- Register blocksize extensions (for packed micro-panels) --

// NOTE: These register blocksize "extensions" determine whether the
// leading dimensions used within the packed micro-panels are equal to
// or greater than their corresponding register blocksizes above.

#define BLIS_EXTEND_MR_S               0
#define BLIS_EXTEND_NR_S               0

#define BLIS_EXTEND_MR_D               0
#define BLIS_EXTEND_NR_D               0

#define BLIS_EXTEND_MR_C               0
#define BLIS_EXTEND_NR_C               0

#define BLIS_EXTEND_MR_Z               0
#define BLIS_EXTEND_NR_Z               0

// Register blocksize extensions in the k dimension are not used.

#define BLIS_EXTEND_KR_S               0
#define BLIS_EXTEND_KR_D               0
#define BLIS_EXTEND_KR_C               0
#define BLIS_EXTEND_KR_Z               0

// -- Default incremental packing blocksizes (n dimension) --

// NOTE: These incremental packing blocksizes (for the n dimension) are only
// used by certain blocked variants. But when the *are* used, they MUST be
// be an integer multiple of NR!

#define BLIS_DEFAULT_NI_FAC            16
#define BLIS_DEFAULT_NI_S              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_S)
#define BLIS_DEFAULT_NI_D              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_D)
#define BLIS_DEFAULT_NI_C              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_C)
#define BLIS_DEFAULT_NI_Z              (BLIS_DEFAULT_NI_FAC * BLIS_DEFAULT_NR_Z)



// -- LEVEL-2 KERNEL CONSTANTS -------------------------------------------------

// NOTE: These values determine high-level cache blocking for level-2
// operations ONLY. So, if gemv is performed with a 2000x2000 matrix A and
// MC = NC = 1000, then a total of four unblocked (or unblocked fused)
// gemv subproblems are called. The blocked algorithms are only useful in
// that they provide the opportunity for packing vectors. (Matrices can also
// be packed here, but this tends to be much too expensive in practice to
// actually employ.)

#define BLIS_DEFAULT_L2_MC_S           1000
#define BLIS_DEFAULT_L2_NC_S           1000

#define BLIS_DEFAULT_L2_MC_D           1000
#define BLIS_DEFAULT_L2_NC_D           1000

#define BLIS_DEFAULT_L2_MC_C           1000
#define BLIS_DEFAULT_L2_NC_C           1000

#define BLIS_DEFAULT_L2_MC_Z           1000
#define BLIS_DEFAULT_L2_NC_Z           1000



// -- LEVEL-1F KERNEL CONSTANTS ------------------------------------------------

// -- Default fusing factors for level-1f operations --

// NOTE: Default fusing factors are not used by the reference implementations
// of level-1f operations. They are here only for use when these operations
// are optimized.

#define BLIS_DEFAULT_FUSE_FAC_S        8
#define BLIS_DEFAULT_FUSE_FAC_D        4
#define BLIS_DEFAULT_FUSE_FAC_C        4
#define BLIS_DEFAULT_FUSE_FAC_Z        2

#define BLIS_AXPYF_FUSE_FAC_S          BLIS_DEFAULT_FUSE_FAC_S
#define BLIS_AXPYF_FUSE_FAC_D          BLIS_DEFAULT_FUSE_FAC_D
#define BLIS_AXPYF_FUSE_FAC_C          BLIS_DEFAULT_FUSE_FAC_C
#define BLIS_AXPYF_FUSE_FAC_Z          BLIS_DEFAULT_FUSE_FAC_Z

#define BLIS_DOTXF_FUSE_FAC_S          BLIS_DEFAULT_FUSE_FAC_S
#define BLIS_DOTXF_FUSE_FAC_D          BLIS_DEFAULT_FUSE_FAC_D
#define BLIS_DOTXF_FUSE_FAC_C          BLIS_DEFAULT_FUSE_FAC_C
#define BLIS_DOTXF_FUSE_FAC_Z          BLIS_DEFAULT_FUSE_FAC_Z

#define BLIS_DOTXAXPYF_FUSE_FAC_S      BLIS_DEFAULT_FUSE_FAC_S
#define BLIS_DOTXAXPYF_FUSE_FAC_D      BLIS_DEFAULT_FUSE_FAC_D
#define BLIS_DOTXAXPYF_FUSE_FAC_C      BLIS_DEFAULT_FUSE_FAC_C
#define BLIS_DOTXAXPYF_FUSE_FAC_Z      BLIS_DEFAULT_FUSE_FAC_Z



// -- LEVEL-1V KERNEL CONSTANTS ------------------------------------------------

// -- Default register blocksizes for vectors --

// NOTE: Register blocksizes for vectors are used when packing
// non-contiguous vectors. Similar to that of KR, they can
// typically be set to 1.

#define BLIS_DEFAULT_VR_S              1
#define BLIS_DEFAULT_VR_D              1
#define BLIS_DEFAULT_VR_C              1
#define BLIS_DEFAULT_VR_Z              1



// -- LEVEL-3 KERNEL DEFINITIONS -----------------------------------------------

#include "bli_gemm_opt_8x4_ref_u4_nodupl_avx1.h"

// -- gemm --

#define GEMM_UKERNEL         gemm_opt_8x4_ref_u4_nodupl_avx1

// -- trsm-related --

#define GEMMTRSM_L_UKERNEL   gemmtrsm_l_ref_mxn
#define GEMMTRSM_U_UKERNEL   gemmtrsm_u_ref_mxn

#define TRSM_L_UKERNEL       trsm_l_ref_mxn
#define TRSM_U_UKERNEL       trsm_u_ref_mxn



// -- LEVEL-1M KERNEL DEFINITIONS ----------------------------------------------

// -- packm --

#define PACKM_2XK_KERNEL     packm_ref_2xk
#define PACKM_4XK_KERNEL     packm_ref_4xk
#define PACKM_6XK_KERNEL     packm_ref_6xk
#define PACKM_8XK_KERNEL     packm_ref_8xk
#define PACKM_10XK_KERNEL    packm_ref_10xk
#define PACKM_12XK_KERNEL    packm_ref_12xk
#define PACKM_14XK_KERNEL    packm_ref_14xk
#define PACKM_16XK_KERNEL    packm_ref_16xk

// -- unpackm --

#define UNPACKM_2XK_KERNEL   unpackm_ref_2xk
#define UNPACKM_4XK_KERNEL   unpackm_ref_4xk
#define UNPACKM_6XK_KERNEL   unpackm_ref_6xk
#define UNPACKM_8XK_KERNEL   unpackm_ref_8xk
#define UNPACKM_10XK_KERNEL  unpackm_ref_10xk
#define UNPACKM_12XK_KERNEL  unpackm_ref_12xk
#define UNPACKM_14XK_KERNEL  unpackm_ref_14xk
#define UNPACKM_16XK_KERNEL  unpackm_ref_16xk



// -- LEVEL-1F KERNEL DEFINITIONS ----------------------------------------------

// -- axpy2v --

#define AXPY2V_KERNEL        axpy2v_unb_var1

// -- dotaxpyv --

#define DOTAXPYV_KERNEL      dotaxpyv_unb_var1

// -- axpyf --

#define AXPYF_KERNEL         axpyf_unb_var1

// -- dotxf --

#define DOTXF_KERNEL         dotxf_unb_var1

// -- dotxaxpyf --

#define DOTXAXPYF_KERNEL     dotxaxpyf_unb_var1



// -- LEVEL-1V KERNEL DEFINITIONS ----------------------------------------------

// -- addv --

#define ADDV_KERNEL          addv_unb_var1

// -- axpyv --

#define AXPYV_KERNEL         axpyv_unb_var1

// -- copyv --

#define COPYV_KERNEL         copyv_unb_var1

// -- dotv --

#define DOTV_KERNEL          dotv_unb_var1

// -- dotxv --

#define DOTXV_KERNEL         dotxv_unb_var1

// -- invertv --

#define INVERTV_KERNEL       invertv_unb_var1

// -- scal2v --

#define SCAL2V_KERNEL        scal2v_unb_var1

// -- scalv --

#define SCALV_KERNEL         scalv_unb_var1

// -- setv --

#define SETV_KERNEL          setv_unb_var1

// -- subv --

#define SUBV_KERNEL          subv_unb_var1

// -- swapv --

#define SWAPV_KERNEL         swapv_unb_var1



#endif

