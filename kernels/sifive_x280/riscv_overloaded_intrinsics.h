/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, SiFive, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

// 6. Configuration-Setting and Utility Functions
#define RVV_TYPE_F_(PRECISION, LMUL) vfloat##PRECISION##LMUL##_t
#define RVV_TYPE_F(PRECISION, LMUL) RVV_TYPE_F_(PRECISION, LMUL)
#define VSETVL_(PRECISION, LMUL) __riscv_vsetvl_e##PRECISION##LMUL
#define VSETVL(PRECISION, LMUL) VSETVL_(PRECISION, LMUL)

// 7. Vector Loads and Stores
// Loads
#define VLE_V_F_(PRECISION, LMUL)   __riscv_vle##PRECISION##_v_f##PRECISION##LMUL
#define VLE_V_F(PRECISION, LMUL)   VLE_V_F_(PRECISION, LMUL)
#define VLSE_V_F_(PRECISION, LMUL) __riscv_vlse##PRECISION##_v_f##PRECISION##LMUL
#define VLSE_V_F(PRECISION, LMUL) VLSE_V_F_(PRECISION, LMUL)
#define VLSEG2_V_F_(PRECISION, LMUL)   __riscv_vlseg2e##PRECISION##_v_f##PRECISION##LMUL
#define VLSEG2_V_F(PRECISION, LMUL)   VLSEG2_V_F_(PRECISION, LMUL)
#define VLSSEG2_V_F_(PRECISION, LMUL)   __riscv_vlsseg2e##PRECISION##_v_f##PRECISION##LMUL
#define VLSSEG2_V_F(PRECISION, LMUL)   VLSSEG2_V_F_(PRECISION, LMUL)
// Stores
#define VSE_V_F_(PRECISION, LMUL)   __riscv_vse##PRECISION##_v_f##PRECISION##LMUL
#define VSE_V_F(PRECISION, LMUL) VSE_V_F_(PRECISION, LMUL)
#define VSSE_V_F_(PRECISION, LMUL) __riscv_vsse##PRECISION##_v_f##PRECISION##LMUL
#define VSSE_V_F(PRECISION, LMUL) VSSE_V_F_(PRECISION, LMUL)
#define VSSEG2_V_F_(PRECISION, LMUL)   __riscv_vsseg2e##PRECISION##_v_f##PRECISION##LMUL
#define VSSEG2_V_F(PRECISION, LMUL) VSSEG2_V_F_(PRECISION, LMUL)
#define VSSSEG2_V_F_(PRECISION, LMUL) __riscv_vssseg2e##PRECISION##_v_f##PRECISION##LMUL
#define VSSSEG2_V_F(PRECISION, LMUL) VSSSEG2_V_F_(PRECISION, LMUL)

// 13. Vector Floating-Point Operations
#define VFADD_VV_(PRECISION, LMUL) __riscv_vfadd_vv_f##PRECISION##LMUL
#define VFADD_VV(PRECISION, LMUL) VFADD_VV_(PRECISION, LMUL)
#define VFSUB_VV_(PRECISION, LMUL) __riscv_vfsub_vv_f##PRECISION##LMUL
#define VFSUB_VV(PRECISION, LMUL) VFSUB_VV_(PRECISION, LMUL)
#define VFMUL_VF_(PRECISION, LMUL) __riscv_vfmul_vf_f##PRECISION##LMUL
#define VFMUL_VF(PRECISION, LMUL) VFMUL_VF_(PRECISION, LMUL)
#define VFMUL_VV_(PRECISION, LMUL) __riscv_vfmul_vv_f##PRECISION##LMUL
#define VFMUL_VV(PRECISION, LMUL) VFMUL_VV_(PRECISION, LMUL)
#define VFMUL_VF_(PRECISION, LMUL) __riscv_vfmul_vf_f##PRECISION##LMUL
#define VFMUL_VF(PRECISION, LMUL) VFMUL_VF_(PRECISION, LMUL)
#define VFMACC_VF_(PRECISION, LMUL) __riscv_vfmacc_vf_f##PRECISION##LMUL
#define VFMACC_VF(PRECISION, LMUL) VFMACC_VF_(PRECISION, LMUL)
#define VFMACC_VV_(PRECISION, LMUL) __riscv_vfmacc_vv_f##PRECISION##LMUL
#define VFMACC_VV(PRECISION, LMUL) VFMACC_VV_(PRECISION, LMUL)
#define VFMACC_VV_TU_(PRECISION, LMUL) __riscv_vfmacc_vv_f##PRECISION##LMUL##_tu
#define VFMACC_VV_TU(PRECISION, LMUL) VFMACC_VV_TU_(PRECISION, LMUL)
#define VFMSAC_VF_(PRECISION, LMUL) __riscv_vfmsac_vf_f##PRECISION##LMUL
#define VFMSAC_VF(PRECISION, LMUL) VFMSAC_VF_(PRECISION, LMUL)
#define VFNMSAC_VF_(PRECISION, LMUL) __riscv_vfnmsac_vf_f##PRECISION##LMUL
#define VFNMSAC_VF(PRECISION, LMUL) VFNMSAC_VF_(PRECISION, LMUL)
#define VFNMSAC_VV_TU_(PRECISION, LMUL) __riscv_vfnmsac_vv_f##PRECISION##LMUL##_tu
#define VFNMSAC_VV_TU(PRECISION, LMUL) VFNMSAC_VV_TU_(PRECISION, LMUL)
#define VFMADD_VF_(PRECISION, LMUL) __riscv_vfmadd_vf_f##PRECISION##LMUL
#define VFMADD_VF(PRECISION, LMUL)  VFMADD_VF_(PRECISION, LMUL)
#define VFMSUB_VF_(PRECISION, LMUL) __riscv_vfmsub_vf_f##PRECISION##LMUL
#define VFMSUB_VF(PRECISION, LMUL) VFMSUB_VF_(PRECISION, LMUL)
#define VFNEG_VF_(PRECISION, LMUL) __riscv_vfneg_v_f##PRECISION##LMUL
#define VFNEG_VF(PRECISION, LMUL)  VFNEG_VF_(PRECISION, LMUL)
#define VFMV_V_V_(PRECISION, LMUL) VREINTERPRET_V_I_F(PRECISION, LMUL)(  __riscv_vmv_v_v_i##PRECISION##LMUL( VREINTERPRET_V_F_I(PRECISION, LMUL) CURRY_1ARG
#define VFMV_V_V(PRECISION, LMUL) VFMV_V_V_(PRECISION, LMUL)

// 14. Vector Reduction Operations
#define VF_REDUSUM_VS_(PRECISION, LMUL) __riscv_vfredusum_vs_f##PRECISION##LMUL##_f##PRECISION##m1
#define VF_REDUSUM_VS(PRECISION, LMUL) VF_REDUSUM_VS_(PRECISION, LMUL)

// 16. Vector Permutation Operations
#define VFMV_S_F_(PRECISION, LMUL) __riscv_vfmv_s_f_f##PRECISION##LMUL
#define VFMV_S_F(PRECISION, LMUL) VFMV_S_F_(PRECISION, LMUL)
#define VFMV_F_S_(PRECISION) __riscv_vfmv_f_s_f##PRECISION##m1_f##PRECISION
#define VFMV_F_S(PRECISION) VFMV_F_S_(PRECISION)

// Miscellaneous Vector Function
#define VREINTERPRET_V_I_F_(PRECISION, LMUL) __riscv_vreinterpret_v_i##PRECISION##LMUL##_f##PRECISION##LMUL
#define VREINTERPRET_V_I_F(PRECISION, LMUL) VREINTERPRET_V_I_F_(PRECISION, LMUL)
#define VREINTERPRET_V_F_I_(PRECISION, LMUL) __riscv_vreinterpret_v_f##PRECISION##LMUL##_i##PRECISION##LMUL
#define VREINTERPRET_V_F_I(PRECISION, LMUL) VREINTERPRET_V_F_I_(PRECISION, LMUL)


// Non-vector functions
#define CURRY_1ARG(arg1, ...) (arg1), __VA_ARGS__))
