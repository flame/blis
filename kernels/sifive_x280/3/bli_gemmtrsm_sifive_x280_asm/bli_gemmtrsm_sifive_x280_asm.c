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

// clang-format off
#include "blis.h"
#include "../../riscv_cmul_macros_asm.h"
#include "../../bli_kernels_sifive_x280.h"
#include <stdint.h>
#include <riscv_vector.h>

#define GEMMTRSM_L(PRECISION_CHAR, T) void bli_##PRECISION_CHAR##gemmtrsm_l_sifive_x280_asm(\
          dim_t               m,      \
          dim_t               n,      \
          dim_t               k,      \
    const T*         restrict alpha_, \
    const T*         restrict a10_,   \
    const T*         restrict a11_,   \
    const T*         restrict b01_,   \
          T*         restrict b11_,   \
          T*         restrict c11_,   \
          inc_t               rsc,    \
          inc_t               csc,    \
    const auxinfo_t* restrict data,   \
    const cntx_t*    restrict cntx    \
    )

#define GEMMTRSM_U(PRECISION_CHAR, T) void bli_##PRECISION_CHAR##gemmtrsm_u_sifive_x280_asm(\
          dim_t               m,      \
          dim_t               n,      \
          dim_t               k,      \
    const T*         restrict alpha_, \
    const T*         restrict a12_,   \
    const T*         restrict a11_,   \
    const T*         restrict b21_,   \
          T*         restrict b11_,   \
          T*         restrict c11_,   \
          inc_t               rsc,    \
          inc_t               csc,    \
    const auxinfo_t* restrict data,   \
    const cntx_t*    restrict cntx    \
    )

#define GEMMTRSM(macro, ...)  macro(__VA_ARGS__)

// Single precision real
#define DATATYPE float
#define PRECISION_CHAR s
#define PACKMR 8
#define PACKNR 64
#define VLE "vle32.v"
#define VSE "vse32.v"
#define VSSE "vsse32.v"
#define FLT_LOAD "flw"
#define FLT_SIZE sizeof(float)
#define LOG_FLT_SIZE 2


#include "./bli_gemmtrsm_l_sifive_x280_asm_real.c"
#include "./bli_gemmtrsm_u_sifive_x280_asm_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PACKMR
#undef PACKNR
#undef VLE
#undef VSE
#undef VSSE
#undef FLT_LOAD
#undef FLT_SIZE
#undef LOG_FLT_SIZE

// Double precision real
#define DATATYPE double
#define PRECISION_CHAR d
#define PACKMR 8
#define PACKNR 32
#define VLE "vle64.v"
#define VSE "vse64.v"
#define VSSE "vsse64.v"
#define FLT_LOAD "fld"
#define FLT_SIZE sizeof(double)
#define LOG_FLT_SIZE 3

#include "./bli_gemmtrsm_l_sifive_x280_asm_real.c"
#include "./bli_gemmtrsm_u_sifive_x280_asm_real.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PACKMR
#undef PACKNR
#undef VLE
#undef VSE
#undef VSSE
#undef FLT_LOAD
#undef FLT_SIZE
#undef LOG_FLT_SIZE

// Single precision complex
#define DATATYPE scomplex
#define PRECISION_CHAR c
#define PACKMR 8
#define PACKNR 32
#define VLSEG2 "vlseg2e32.v "
#define VSSEG2 "vsseg2e32.v "
#define VSSSEG2 "vssseg2e32.v "
#define FLT_LOAD "flw "
#define FLT_SIZE sizeof(float)

#include "./bli_gemmtrsm_l_sifive_x280_asm_complex.c"
#include "./bli_gemmtrsm_u_sifive_x280_asm_complex.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PACKMR
#undef PACKNR
#undef VLSEG2
#undef VSSEG2
#undef VSSSEG2
#undef FLT_LOAD
#undef FLT_SIZE

// Double precision complex
#define DATATYPE dcomplex
#define PRECISION_CHAR z
#define PACKMR 8
#define PACKNR 16
#define VLSEG2 "vlseg2e64.v "
#define VSSEG2 "vsseg2e64.v "
#define VSSSEG2 "vssseg2e64.v "
#define FLT_LOAD "fld "
#define FLT_SIZE sizeof(double)

#include "./bli_gemmtrsm_l_sifive_x280_asm_complex.c"
#include "./bli_gemmtrsm_u_sifive_x280_asm_complex.c"

#undef DATATYPE
#undef PRECISION_CHAR
#undef PACKMR
#undef PACKNR
#undef VLSEG
#undef VSSEG
#undef VSSSEG
#undef FLT_LOAD
#undef FLT_SIZE



#undef GEMMTRSM
#undef GEMMTRSM_L
#undef GEMMTRSM_U


