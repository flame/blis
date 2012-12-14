/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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


// -- LEVEL-3 KERNEL DEFINITIONS -----------------------------------------------

// -- dupl --

#define DUPL_KERNEL          dupl_unb_var1

// -- gemm --

#define GEMM_UKERNEL         gemm_ref_4x4

// -- trsm-related --

#define GEMMTRSM_L_UKERNEL   gemmtrsm_l_ref_4x4
#define GEMMTRSM_U_UKERNEL   gemmtrsm_u_ref_4x4

#define TRSM_L_UKERNEL       trsm_l_ref_4x4
#define TRSM_U_UKERNEL       trsm_u_ref_4x4



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



// -- LEVEL-1 KERNEL DEFINITIONS -----------------------------------------------

// -- axpyv --

#define AXPYV_KERNEL         axpyv_unb_var1

// -- copynzv --

#define COPYNZV_KERNEL       copynzv_unb_var1

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


