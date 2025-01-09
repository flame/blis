/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

static const int64_t mask_3[4] = {-1, -1, -1, 0};
static const int64_t mask_1[4] = {-1, 0, 0, 0};

static void bli_dgemmsup_rv_haswell_asm_6x7m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

static void bli_dgemmsup_rv_haswell_asm_6x5m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

static void bli_dgemmsup_rv_haswell_asm_6x3m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

static void bli_dgemmsup_rv_haswell_asm_6x1m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     );

#define C_TRANSPOSE_6x7_TILE(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Broadcasting Beta into ymm15 vector register*/\
	vbroadcastsd(mem(rbx), ymm15)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3, R4 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm15, ymm(R1))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm(R2))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm(R3))\
	vfmadd231pd(mem(rcx, rax, 1), ymm15, ymm(R4))\
	/*Storing it back to C matrix.*/ \
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	/*Moving to operate on last 2 rows of 6 rows.*/ \
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2, 3 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm15, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm15, xmm2)\
	vfmadd231pd(mem(rdx, rax, 1), xmm15, xmm3)\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpckhpd(ymm(R8), ymm(R7), ymm1)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vunpckhpd(ymm(R10), ymm(R9), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)\
\
	vfmadd231pd(mem(rcx        ), ymm15, ymm5)\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm7)\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm9)\
\
	vmovupd(ymm5, mem(rcx        ))\
	vmovupd(ymm7, mem(rcx, rsi, 1))\
	vmovupd(ymm9, mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
	vunpckhpd(ymm(R12), ymm(R11), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm4)\
\
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm15, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm15, xmm2)\
\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x7_TILE_BZ(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Storing transposed 4x4 tile back to C matrix*/\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Storing transposed 2x4 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpckhpd(ymm(R8), ymm(R7), ymm1)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vunpckhpd(ymm(R10), ymm(R9), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)\
\
	/*Storing transposed 4x3 tile back to C matrix*/\
	vmovupd(ymm5, mem(rcx        ))\
	vmovupd(ymm7, mem(rcx, rsi, 1))\
	vmovupd(ymm9, mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
	vunpckhpd(ymm(R12), ymm(R11), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm4)\
\
	/*Storing transposed 2x3 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x5_TILE(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Broadcasting Beta into ymm15 vector register*/\
	vbroadcastsd(mem(rbx), ymm15)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3, R4 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm15, ymm(R1))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm15, ymm(R2))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm15, ymm(R3))\
	vfmadd231pd(mem(rcx, rax, 1), ymm15, ymm(R4))\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2, 3 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm15, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm15, xmm2)\
	vfmadd231pd(mem(rdx, rax, 1), xmm15, xmm3)\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x1 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
\
	vfmadd231pd(mem(rcx        ), ymm15, ymm5)\
	vmovupd(ymm5, mem(rcx        ))\
\
	/*Transposing 2x1 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
	vfmadd231pd(mem(rdx        ), xmm15, xmm0)\
\
	vmovupd(xmm0, mem(rdx        ))

#define C_TRANSPOSE_6x5_TILE_BZ(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12) \
	/*Transposing 4x4 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm15)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm15, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
	vperm2f128(imm(0x31), ymm15, ymm1, ymm(R4))\
\
	/*Storing transposed 4x4 tile back to C matrix*/\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
	vmovupd(ymm(R4), mem(rcx, rax, 1))\
\
	lea(mem(rcx, rsi, 4), rcx)\
\
	/*Transposing 2x4 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm3)\
\
	/*Storing transposed 4x2 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))\
	vmovupd(xmm3, mem(rdx, rax, 1))\
\
	lea(mem(rdx, rsi, 4), rdx)\
\
	/*Transposing 4x1 tile*/ \
	vunpcklpd(ymm(R8), ymm(R7), ymm0)\
	vunpcklpd(ymm(R10), ymm(R9), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)\
\
	/*Storing transposed 4x1 tile back to C matrix*/\
	vmovupd(ymm5, mem(rcx        ))\
\
	/*Transposing 2x1 tile*/ \
	vunpcklpd(ymm(R12), ymm(R11), ymm0)\
\
	/*Storing transposed 2x1 tile back to C matrix*/\
	vmovupd(xmm0, mem(rdx        ))

#define C_TRANSPOSE_6x3_TILE(R1, R2, R3, R4, R5, R6) \
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
\
	vbroadcastsd(mem(rbx), ymm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm3, ymm(R1))\
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm(R2))\
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm(R3))\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm3, xmm0)\
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)\
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x3_TILE_BZ(R1, R2, R3, R4, R5, R6) \
	/*Transposing 4x3 tile*/ \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpckhpd(ymm(R2), ymm(R1), ymm1)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vunpckhpd(ymm(R4), ymm(R3), ymm3)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
	vinsertf128(imm(0x1), xmm3, ymm1, ymm(R2))\
	vperm2f128(imm(0x31), ymm2, ymm0, ymm(R3))\
\
	vmovupd(ymm(R1), mem(rcx        ))\
	vmovupd(ymm(R2), mem(rcx, rsi, 1))\
	vmovupd(ymm(R3), mem(rcx, rsi, 2))\
\
	/*Transposing 2x3 tile*/ \
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
	vunpckhpd(ymm(R6), ymm(R5), ymm1)\
	vextractf128(imm(0x1), ymm0, xmm2)\
	vextractf128(imm(0x1), ymm1, xmm4)\
\
	vmovupd(xmm0, mem(rdx        ))\
	vmovupd(xmm1, mem(rdx, rsi, 1))\
	vmovupd(xmm2, mem(rdx, rsi, 2))

#define C_TRANSPOSE_6x1_TILE(R1, R2, R3, R4, R5, R6) \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
\
	vbroadcastsd(mem(rbx), ymm3)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*R1, R2, R3 holds final result*/ \
	vfmadd231pd(mem(rcx        ), ymm3, ymm(R1))\
	vmovupd(ymm(R1), mem(rcx        ))\
\
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
\
	/*Scaling C matrix by Beta and adding it to fma result.*/ \
	/*0, 1, 2 holds final result*/ \
	vfmadd231pd(mem(rdx        ), xmm3, xmm0)\
	vmovupd(xmm0, mem(rdx        ))\

#define C_TRANSPOSE_6x1_TILE_BZ(R1, R2, R3, R4, R5, R6) \
	vunpcklpd(ymm(R2), ymm(R1), ymm0)\
	vunpcklpd(ymm(R4), ymm(R3), ymm2)\
	vinsertf128(imm(0x1), xmm2, ymm0, ymm(R1))\
\
	vmovupd(ymm(R1), mem(rcx        ))\
\
	vunpcklpd(ymm(R6), ymm(R5), ymm0)\
\
	vmovupd(xmm0, mem(rdx        ))\
/*
   rrr:
	 --------        ------        --------
	 --------        ------        --------
	 --------   +=   ------ ...    --------
	 --------        ------        --------
	 --------        ------            :
	 --------        ------            :

   rcr:
	 --------        | | | |       --------
	 --------        | | | |       --------
	 --------   +=   | | | | ...   --------
	 --------        | | | |       --------
	 --------        | | | |           :
	 --------        | | | |           :

   Assumptions:
   - B is row-stored;
   - A is row- or column-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.

   NOTE: These kernels explicitly support column-oriented IO, implemented
   via an in-register transpose. And thus they also support the crr and
   ccr cases, though only crr is ever utilized (because ccr is handled by
   transposing the operation and executing rcr, which does not incur the
   cost of the in-register transpose).

   crr:
	 | | | | | | | |       ------        --------
	 | | | | | | | |       ------        --------
	 | | | | | | | |  +=   ------ ...    --------
	 | | | | | | | |       ------        --------
	 | | | | | | | |       ------            :
	 | | | | | | | |       ------            :
*/

// Prototype reference microkernels.
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )


void bli_dgemmsup_rv_haswell_asm_6x8m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t n_left = n0 % 8;

	// First check whether this is a edge case in the n dimension. If so,
	// dispatch other 6x?m kernels, as needed.
	if ( n_left )
	{
		double* restrict cij = c;
		double* restrict bj  = b;
		double* restrict ai  = a;

		switch(n_left)
		{
			case 7:
			{
				bli_dgemmsup_rv_haswell_asm_6x7m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 6:
			{
				bli_dgemmsup_rv_haswell_asm_6x6m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 5:
			{
				bli_dgemmsup_rv_haswell_asm_6x5m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 4:
			{
				bli_dgemmsup_rv_haswell_asm_6x4m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 3:
			{
				bli_dgemmsup_rv_haswell_asm_6x3m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 2:
			{
				bli_dgemmsup_rv_haswell_asm_6x2m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			case 1:
			{
				bli_dgemmsup_rv_haswell_asm_6x1m
				(
					conja, conjb, m0, n_left, k0,
					alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
					beta, cij, rs_c0, cs_c0, data, cntx
				);
				break;
			}
			default:
			{
				break;
			}
		}
		return;
	}

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X8I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))


	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rbx, 0*32))

	vfmadd231pd(mem(rbx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rbx, 1*32))
	add(rdi, rbx)


	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rbx, 0*32))

	vfmadd231pd(mem(rbx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rbx, 1*32))


	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rdx, 0*32))

	vfmadd231pd(mem(rdx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rdx, 1*32))
	add(rdi, rdx)


	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rdx, 0*32))

	vfmadd231pd(mem(rdx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(ymm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X8I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 8;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x8,
		  bli_dgemmsup_rv_haswell_asm_2x8,
		  bli_dgemmsup_rv_haswell_asm_3x8,
		  bli_dgemmsup_rv_haswell_asm_4x8,
		  bli_dgemmsup_rv_haswell_asm_5x8
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*
24x24 block

                         1 1 1 1 1 1  1 1 1 1 2 2 2 2
     0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5  6 7 8 9 0 1 2 3
     |- - - - - - - -|- - - - - - - -| - - - - - - - -|
0    |               |               |                |
1    | m_off_24 = 0  |               |                |
2    | n_off_24 = 0  |               |                |
3    |    m_idx = 0  |               |                |
4    |    n_idx = 0  |               |                |
5    |- - - - - - - -|- - - - - - - -|- - - - - - - - |
6    |               |               |                |
7    | m_off_24 = 6  | m_off_24 = 6  |                |
8    | n_off_24 = 0  | n_off_24 = 8  |                |
9    |    m_idx = 1  |    m_idx = 1  |                |
10   |    n_idx = 0  |    n_idx = 1  |                |
11   |- - - - - - - -|- - - - - - - -|- - - - - - - - |
12   |               |               |                |
13   |               | m_off_24 = 12 | m_off_24 = 12  |
14   |               | n_off_24 = 8  | n_off_24 = 16  |
15   |               |    m_idx = 2  |    m_idx = 2   |
16   |               |    n_idx = 1  |    n_idx = 2   |
17   |- - - - - - - -|- - - - - - - -|- - - - - - - - |
18   |               |               |                |
19   |               |               | m_off_24 = 18  |
20   |               |               | n_off_24 = 16  |
21   |               |               |    m_idx = 3   |
22   |               |               |    n_idx = 2   |
23   |- - - - - - - -|- - - - - - - -|- - - - - - - - |
*/

#define PREFETCH_C() \
\
	cmp(imm(8), rdi) \
	jz(.DCOLPFETCH) \
	label(.DROWPFETCH) \
 \
	lea(mem(r12, rdi, 2), rdx) \
	lea(mem(rdx, rdi, 1), rdx) \
	prefetch(0, mem(r12,         7*8)) \
	prefetch(0, mem(r12, rdi, 1, 7*8)) \
	prefetch(0, mem(r12, rdi, 2, 7*8)) \
	prefetch(0, mem(rdx,         7*8)) \
	prefetch(0, mem(rdx, rdi, 1, 7*8)) \
	prefetch(0, mem(rdx, rdi, 2, 7*8)) \
 \
	jmp(.DPOSTPFETCH) \
	label(.DCOLPFETCH) \
 \
	mov(var(cs_c), rsi) \
	lea(mem(, rsi, 8), rsi) \
	lea(mem(r12, rsi, 2), rdx) \
	lea(mem(rdx, rsi, 1), rdx) \
	prefetch(0, mem(r12,         5*8)) \
	prefetch(0, mem(r12, rsi, 1, 5*8)) \
	prefetch(0, mem(r12, rsi, 2, 5*8)) \
	prefetch(0, mem(rdx,         5*8)) \
	prefetch(0, mem(rdx, rsi, 1, 5*8)) \
	prefetch(0, mem(rdx, rsi, 2, 5*8)) \
	lea(mem(rdx, rsi, 2), rdx) \
	prefetch(0, mem(rdx, rsi, 1, 5*8)) \
	prefetch(0, mem(rdx, rsi, 2, 5*8)) \

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 0 and n_offset is 0(0x0)
(0x0)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		0    x - - - - - - -
|		1    x x - - - - - -
m		2    x x x - - - - -
off		3    x x x x - - - -
24		4    x x x x x - - -
|		5    x x x x x x - -
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_0x0_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a8  = bli_auxinfo_ps_a( data ) * sizeof( double );


	// -------------------------------------------------------------------------

	begin_asm()

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	//for triangular kernels we can skip 1st loop around micro kernel
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)                      // reset rax to current upanel of a.

	PREFETCH_C()
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;

	label(.DPOSTPFETCH)                // done prefetching c


	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.

	// skip computation of ymm5, ymm7, ymm9, ymm11 and compute only half of ymm4, ymm6, ymm13, ymm15
	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(xmm1, xmm2, xmm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(xmm1, xmm3, xmm15)


	// ---------------------------------- iteration 1

	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(xmm1, xmm2, xmm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(xmm1, xmm3, xmm15)


	// ---------------------------------- iteration 2

	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(xmm1, xmm2, xmm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(xmm1, xmm3, xmm15)


	// ---------------------------------- iteration 3

	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(xmm1, xmm2, xmm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(xmm1, xmm3, xmm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.


	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(xmm1, xmm2, xmm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(xmm1, xmm3, xmm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm4)
	vmovlpd(xmm4, mem(rcx, 0*32))		// write back only lower half of xmm (8 bytes)

	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx, 0*32))		// write only lower half of ymm6 to c

	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(xmm8, mem(rcx, 0*32))		// write lower half of ymm (16 bytes)
	vextractf128(imm(1), ymm8, xmm1)	// move upper half of ymm to xmm
	vmovlpd(xmm1, mem(rcx, 2*8))		// write only lower half of xmm (8 bytes) to rcx + 16

	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm13)
	vmovlpd(xmm13, mem(rcx, 1*32))		// write back only xmm13[0]
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm15)
	vmovupd(xmm15, mem(rcx, 1*32))		// write xmm to c (16 bytes)
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vextractf128(imm(1), ymm6, xmm1)                // move upper half of ymm to xmm1 (ymm6[2], ymm6[3])
	vmovhpd(xmm6, mem(rcx, rsi, 1, 1*8))            // write upper half of xmm6(ymm6[1]) to c + rsi + 8
	vmovupd(xmm1, mem(rcx, rsi, 1, 2*8))            // write xmm1 (ymm6[2], ymm6[3]) to c + rsi + 16
	vextractf128(imm(1), ymm8, xmm1)                // move upper half of ymm8 to xmm1
	vmovupd(xmm1, mem(rcx, rsi, 2, 2*8))            // write upper half of ymm8 to c + rsi*2 + 16
	vextractf128(imm(1), ymm10, xmm1)               // move uppper half of ymm10 to xmm1
	vmovhpd(xmm1, mem(rcx, rax, 1, 3*8))            // move ymm8[3] to c + rsi*3 + 3*8

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx             ))        // move the first half of ymm13 to c
	vmovhpd(xmm1, mem(rdx, rsi, 1, 1*8))        // move the last 8 bits of ymm13


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovlpd(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm8, mem(rcx, 0*32))
	vextractf128(imm(1), ymm8, xmm1)
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovlpd(xmm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(xmm15, mem(rcx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vmovupd(ymm4, mem(rcx        ))
	vextractf128(imm(1), ymm6, xmm1)        // move upper half of ymm to xmm1 (ymm6[2], ymm6[3])
	vmovhpd(xmm6, mem(rcx, rsi, 1, 1*8))    // write upper half of xmm6(ymm6[1]) to c + rsi + 8
	vmovupd(xmm1, mem(rcx, rsi, 1, 2*8))    // write xmm1 (ymm6[2], ymm6[3]) to c + rsi + 16
	vextractf128(imm(1), ymm8, xmm1)        // move upper half of ymm8 to xmm1
	vmovupd(xmm1, mem(rcx, rsi, 2, 2*8))    // write upper half of ymm8 to c + rsi*2 + 16
	vextractf128(imm(1), ymm10, xmm1)       // move uppper half of ymm10 to xmm1
	vmovhpd(xmm1, mem(rcx, rax, 1, 3*8))    // move ymm8[3] to c + rsi*3 + 3*8

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vmovupd(xmm0, mem(rdx             ))        // move the first half of ymm13 to c
	vmovhpd(xmm1, mem(rdx, rsi, 1, 1*8))        // move the last 8 bits of ymm13


	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6",
	  "ymm8", "ymm10", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 8(6x8)
(6x8)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		6    - - - - - - - -
|		7    - - - - - - - -
m		8    x - - - - - - -
off		9    x x - - - - - -
24		10   x x x - - - - -
|		11   x x x x - - - -
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_6x8_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	// -------------------------------------------------------------------------

	begin_asm()
	mov(var(a), r14)
	mov(var(c), r12)

	mov(var(rs_a), r8)
	mov(var(cs_a), r9)
	mov(var(rs_b), r10)
	mov(var(rs_c), rdi)
	lea(mem(, r8, 8), r8)
	lea(mem(, r9, 8), r9)
	lea(mem(, r10, 8), r10)
	lea(mem(, rdi, 8), rdi)

	lea(mem(r8, r8, 2), r13)         	//3*r8
	lea(mem(r8, r8, 4), r15)          	//5*r8

	vxorpd(ymm8,  ymm8,  ymm8)
	vmovapd( ymm8, ymm10)
	vmovapd( ymm8, ymm12)
	vmovapd( ymm8, ymm14)
	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	cmp(imm(8), rdi)
	jz(.DCOLPFETCH)

	label(.DROWPFETCH)
	lea(mem(r12, rdi, 2), rdx)
	lea(mem(rdx, rdi, 1), rdx)
	prefetch(0, mem(r12, rdi, 2, 1*8))
	prefetch(0, mem(rdx,         2*8))
	prefetch(0, mem(rdx, rdi, 1, 3*8))
	prefetch(0, mem(rdx, rdi, 2, 4*8))
	jmp(.DPOSTPFETCH)

	label(.DCOLPFETCH)
	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(r12, rsi, 2), rdx)
	lea(mem(rdx, rsi, 1), rdx)
	prefetch(0, mem(r12,         5*8))
	prefetch(0, mem(r12, rsi, 1, 5*8))
	prefetch(0, mem(r12, rsi, 2, 5*8))
	prefetch(0, mem(rdx, 5*8))

	label(.DPOSTPFETCH)
	mov(var(k_iter), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	// computer xmm8, xmm10, ymm12, ymm14 only
	label(.DLOOPKITER)
	//0
	vmovupd(mem(rbx,  0*32), ymm0)
	vbroadcastsd(mem(rax, r8,  2), ymm1)
	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vbroadcastsd(mem(rax, r8,  4), ymm3)
	vbroadcastsd(mem(rax, r15, 1), ymm4)
	vfmadd231pd(xmm0, xmm1, xmm8)
	vfmadd231pd(xmm0, xmm2, xmm10)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm0, ymm4, ymm14)
	add(r10, rbx)                      // b += rs_b;
	add(r9, rax)                       // a += cs_a;
	//1
	vmovupd(mem(rbx,  0*32), ymm0)
	vbroadcastsd(mem(rax, r8,  2), ymm1)
	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vbroadcastsd(mem(rax, r8,  4), ymm3)
	vbroadcastsd(mem(rax, r15, 1), ymm4)
	vfmadd231pd(xmm0, xmm1, xmm8)
	vfmadd231pd(xmm0, xmm2, xmm10)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm0, ymm4, ymm14)
	add(r10, rbx)                      // b += rs_b;
	add(r9, rax)                       // a += cs_a;
	//2
	vmovupd(mem(rbx,  0*32), ymm0)
	vbroadcastsd(mem(rax, r8,  2), ymm1)
	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vbroadcastsd(mem(rax, r8,  4), ymm3)
	vbroadcastsd(mem(rax, r15, 1), ymm4)
	vfmadd231pd(xmm0, xmm1, xmm8)
	vfmadd231pd(xmm0, xmm2, xmm10)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm0, ymm4, ymm14)
	add(r10, rbx)                      // b += rs_b;
	add(r9, rax)                       // a += cs_a;
	//3
	vmovupd(mem(rbx,  0*32), ymm0)
	vbroadcastsd(mem(rax, r8,  2), ymm1)
	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vbroadcastsd(mem(rax, r8,  4), ymm3)
	vbroadcastsd(mem(rax, r15, 1), ymm4)
	vfmadd231pd(xmm0, xmm1, xmm8)
	vfmadd231pd(xmm0, xmm2, xmm10)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm0, ymm4, ymm14)
	add(r10, rbx)                      // b += rs_b;
	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.


	label(.DCONSIDKLEFT)
	mov(var(k_left), rsi)
	test(rsi, rsi)
	je(.DPOSTACCUM)

	label(.DLOOPKLEFT)
	vmovupd(mem(rbx,  0*32), ymm0)
	vbroadcastsd(mem(rax, r8,  2), ymm1)
	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vbroadcastsd(mem(rax, r8,  4), ymm3)
	vbroadcastsd(mem(rax, r15, 1), ymm4)
	vfmadd231pd(xmm0, xmm1, xmm8)
	vfmadd231pd(xmm0, xmm2, xmm10)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm0, ymm4, ymm14)
	add(r10, rbx)                      // b += rs_b;
	add(r9, rax)                       // a += cs_a;

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)

	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;


	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)
	lea(mem(rcx , rdi, 2), rcx)
	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm8)
	vmovlpd(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm10)
	vmovupd(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(xmm12, mem(rcx, 0*32))
	vextractf128(imm(1), ymm12, xmm1)
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx, 0*32))

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)
	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)

	vextractf128(imm(1), ymm4, xmm1)
	vmovupd(xmm1, mem(rcx,  2*8   ))     // write upper half of ymm4 to c
	vextractf128(imm(1), ymm6, xmm1)
	vmovhpd(xmm1, mem(rcx, rsi, 1, 3*8)) // write last element of ymm6

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovhpd(xmm4, mem(rdx, rax, 1, 1*8))    // write only last 8 bytes of second half of ymm14

	lea(mem(rdx, rsi, 4), rdx)

	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)
	lea(mem(rcx , rdi, 2), rcx)

	vmovlpd(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm12, mem(rcx, 0*32))
	vextractf128(imm(1), ymm12, xmm1)
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vbroadcastsd(mem(rbx), ymm3)

	vextractf128(imm(1), ymm4, xmm1)
	vmovupd(xmm1, mem(rcx,  2*8   ))        // write upper half of ymm4 to c
	vextractf128(imm(1), ymm6, xmm1)
	vmovhpd(xmm1, mem(rcx, rsi, 1, 3*8))    // write last element of ymm6

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovhpd(xmm4, mem(rdx, rax, 1, 1*8))    // write only last 8 bytes of second half of ymm14


	label(.DDONE)
	vzeroupper()


    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6",
	  "ymm8", "ymm10", "ymm12", "ymm14",
	  "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 16(12x16)
(12x16)_L


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		12   - - - - - - - -
|		13   - - - - - - - -
m		14   - - - - - - - -
off		15   - - - - - - - -
24		16   x - - - - - - -
|		17   x x - - - - - -
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_12x16_L
	(
		conj_t              conja,
		conj_t              conjb,
		dim_t               m0,
		dim_t               n0,
		dim_t               k0,
		double*     restrict alpha,
		double*     restrict a, inc_t rs_a0, inc_t cs_a0,
		double*     restrict b, inc_t rs_b0, inc_t cs_b0,
		double*     restrict beta,
		double*     restrict c, inc_t rs_c0, inc_t cs_c0,
		auxinfo_t* restrict data,
		cntx_t*    restrict cntx
	)
{
		uint64_t k_iter = k0 / 4;
		uint64_t k_left = k0 % 4;
		uint64_t rs_a   = rs_a0;
		uint64_t cs_a   = cs_a0;
		uint64_t rs_b   = rs_b0;
		uint64_t cs_b   = cs_b0;
		uint64_t rs_c   = rs_c0;
		uint64_t cs_c   = cs_c0;
		begin_asm()
		mov(var(a), r14)
		mov(var(b), rbx)
		mov(var(c), r12)
		mov(r14, rax)

		mov(var(rs_a), r8)
		mov(var(cs_a), r9)
		lea(mem(, r8, 8), r8)
		lea(mem(, r9, 8), r9)

		mov(var(rs_b), r10)
		lea(mem(, r10, 8), r10)

		mov(var(rs_c), rdi)
		lea(mem(, rdi, 8), rdi)

		lea(mem(r8, r8, 4), r15)

		vxorpd(ymm12, ymm12, ymm12)
		vmovapd(ymm12, ymm14)

		cmp(imm(8), rdi)
		jz(.DCOLPFETCH)

		label(.DROWPFETCH)
		lea(mem(r12, rdi, 2), rdx)
		lea(mem(rdx, rdi, 1), rdx)
		prefetch(0, mem(rdx, rdi, 1, 1*8))
		prefetch(0, mem(rdx, rdi, 2, 2*8))
		jmp(.DPOSTPFETCH)

		label(.DCOLPFETCH)
		mov(var(cs_c), rsi)
		lea(mem(, rsi, 8), rsi)
		prefetch(0, mem(r12, 5*8))
		prefetch(0, mem(r12, rsi, 1, 5*8))

		label(.DPOSTPFETCH)
		lea(mem(rax, r8, 4), rax)
		mov(var(k_iter), rsi)
		test(rsi, rsi)
		je(.DCONSILEFT)

		//compute xmm12 and xmm 14
		label(.DMAIN)
		//0
		vmovupd(mem(rbx,  0*32), xmm0)
		vbroadcastsd(mem(rax, r8,  4), ymm3)
		vbroadcastsd(mem(rax, r15, 1), ymm4)
		vfmadd231pd(xmm0, xmm3, xmm12)
		vfmadd231pd(xmm0, xmm4, xmm14)
		add(r10, rbx)
		add(r9, rax)
		//1
		vmovupd(mem(rbx,  0*32), xmm0)
		vbroadcastsd(mem(rax, r8,  4), ymm3)
		vbroadcastsd(mem(rax, r15, 1), ymm4)
		vfmadd231pd(xmm0, xmm3, xmm12)
		vfmadd231pd(xmm0, xmm4, xmm14)
		add(r10, rbx)
		add(r9, rax)
		//2
		vmovupd(mem(rbx,  0*32), xmm0)
		vbroadcastsd(mem(rax, r8,  4), ymm3)
		vbroadcastsd(mem(rax, r15, 1), ymm4)
		vfmadd231pd(xmm0, xmm3, xmm12)
		vfmadd231pd(xmm0, xmm4, xmm14)
		add(r10, rbx)
		add(r9, rax)
		//3
		vmovupd(mem(rbx,  0*32), xmm0)
		vbroadcastsd(mem(rax, r8,  4), ymm3)
		vbroadcastsd(mem(rax, r15, 1), ymm4)
		vfmadd231pd(xmm0, xmm3, xmm12)
		vfmadd231pd(xmm0, xmm4, xmm14)
		add(r10, rbx)
		add(r9, rax)

		dec(rsi)
		jne(.DMAIN)

		label(.DCONSILEFT)
		mov(var(k_left), rsi)
		test(rsi, rsi)
		je(.DPOSTACC)

		label(.DLEFT)
		vmovupd(mem(rbx,  0*32), xmm0)
		vbroadcastsd(mem(rax, r8,  4), ymm3)
		vbroadcastsd(mem(rax, r15, 1), ymm4)
		vfmadd231pd(xmm0, xmm3, xmm12)
		vfmadd231pd(xmm0, xmm4, xmm14)
		add(r10, rbx)
		add(r9, rax)
		dec(rsi)
		jne(.DLEFT)

		label(.DPOSTACC)
		mov(r12, rcx)
		mov(var(alpha), rax)
		mov(var(beta), rbx)
		vbroadcastsd(mem(rax), ymm0)
		vbroadcastsd(mem(rbx), ymm3)
		vmulpd(ymm0, ymm12, ymm12)
		vmulpd(ymm0, ymm14, ymm14)

		mov(var(cs_c), rsi)
		lea(mem(, rsi, 8), rsi)
		vxorpd(ymm0, ymm0, ymm0)
		vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
		je(.DBETAZERO)

		cmp(imm(8), rdi)                   //rs_c == 0?
		je(.DCOLSTOR)

		label(.DROWSTOR)
		lea(mem(rcx, rdi, 4), rcx)         //rcx += 4 * rdi
		vfmadd231pd(mem(rcx, 0*32), xmm3, xmm12)
		vmovlpd(xmm12, mem(rcx))
		add(rdi, rcx)
		vfmadd231pd(mem(rcx, 0*32), xmm3, xmm14)
		vmovlpd(xmm14, mem(rcx))
		vmovhpd(xmm14, mem(rcx, rsi, 1))
		jmp(.DDONE)

		label(.DCOLSTOR)

		lea(mem(rcx, rdi, 4), rdx)
		vunpcklpd(ymm14, ymm12, ymm0)
		vunpckhpd(ymm14, ymm12, ymm1)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm12)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm14)

		vfmadd231pd(mem(rdx), xmm3, xmm12)
		vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm14)
		vmovupd(xmm12, mem(rdx        ))
		vmovhpd(xmm14, mem(rdx, rsi, 1, 1*8))
		jmp(.DDONE)

		label(.DBETAZERO)
		cmp(imm(8), rdi)	              //rs_c == 0?
		je(.DCOLSTORBZ)

		label(.DROWSTORBZ)
		lea(mem(rcx, rdi, 4), rcx)       //rcx += 4 * rdi
		vmovlpd(xmm12, mem(rcx))
		add(rdi, rcx)
		vmovlpd(xmm14, mem(rcx))
		vmovhpd(xmm14, mem(rcx, rsi, 1))
		jmp(.DDONE)

		label(.DCOLSTORBZ)

		lea(mem(rcx, rdi, 4), rdx)
		vunpcklpd(ymm14, ymm12, ymm0)
		vunpckhpd(ymm14, ymm12, ymm1)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm12)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm14)

		vmovupd(xmm12, mem(rdx        ))
		vmovhpd(xmm14, mem(rdx, rsi, 1, 1*8))
		jmp(.DDONE)

		label(.DDONE)
		vzeroupper()

		end_asm(
			: // output operands (none)
			: // input operands
			[k_iter] "m" (k_iter),
			[k_left] "m" (k_left),
			[a]      "m" (a),
			[rs_a]   "m" (rs_a),
			[cs_a]   "m" (cs_a),
			[b]      "m" (b),
			[rs_b]   "m" (rs_b),
			[cs_b]   "m" (cs_b),
			[alpha]  "m" (alpha),
      		[beta]   "m" (beta),
			[c]      "m" (c),
			[rs_c]   "m" (rs_c),
			[cs_c]   "m" (cs_c)/*,
			[a_next] "m" (a_next),
			[b_next] "m" (b_next)*/
			: // register clobber list
			"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
			"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
			"xmm0", "xmm1", "xmm2", "xmm3",
			"xmm4", "xmm5", "xmm6", "xmm7",
			"xmm8", "xmm9", "xmm10", "xmm11",
			"xmm12", "xmm13", "xmm14", "xmm15",
			"ymm0", "ymm1", "ymm3", "ymm4", "ymm12", "ymm14",
	 		"memory"
			)
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 12, n_offset is 16(12x16) and m_offset is 18, n_offset is 16 (18x16)
(16x12)+(16x18)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		12   - - - - - - - -
|		13   - - - - - - - -
m		14   - - - - - - - -
off		15   - - - - - - - -
24		16   x - - - - - - -
|		17   x x - - - - - -
↓
↑		18   x x x - - - - -
|		19   x x x x - - - -
m		20   x x x x x - - -
off		21   x x x x x x - -
24		22   x x x x x x x -
|		23   x x x x x x x x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_16x12_combined_L
	(
		conj_t              conja,
		conj_t              conjb,
		dim_t               m0,
		dim_t               n0,
		dim_t               k0,
		double*     restrict alpha,
		double*     restrict a, inc_t rs_a0, inc_t cs_a0,
		double*     restrict b, inc_t rs_b0, inc_t cs_b0,
		double*     restrict beta,
		double*     restrict c, inc_t rs_c0, inc_t cs_c0,
		auxinfo_t* restrict data,
		cntx_t*    restrict cntx
	)
	{
		uint64_t k_iter = k0 / 4;
		uint64_t k_left = k0 % 4;
		uint64_t rs_a   = rs_a0;
		uint64_t cs_a   = cs_a0;
		uint64_t rs_b   = rs_b0;
		uint64_t cs_b   = cs_b0;
		uint64_t rs_c   = rs_c0;
		uint64_t cs_c   = cs_c0;
		uint64_t ps_a8 =  bli_auxinfo_ps_a( data ) * sizeof( double );
		double* a_next = ( (double*)a ) + rs_a * 6;
		begin_asm()
		mov(var(a), r14)
		mov(var(b), rbx)
		mov(var(c), r12)
		mov(var(a_next), r11)
		mov(r14, rax)

		mov(var(rs_a), r8)
		mov(var(cs_a), r9)
		lea(mem(, r8, 8), r8)
		lea(mem(, r9, 8), r9)

		lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
		lea(mem(r8, r8, 4), r15) 			// 5
		mov(var(rs_b), r10)
		lea(mem(, r10, 8), r10)

		mov(var(rs_c), rdi)
		lea(mem(, rdi, 8), rdi)

		lea(mem(r8, r8, 4), r15)

		vxorpd(ymm4, ymm4, ymm4)
		vmovapd( ymm4, ymm5)
		vmovapd( ymm4, ymm6)
		vmovapd( ymm4, ymm7)
		vmovapd( ymm4, ymm8)
		vmovapd( ymm4, ymm9)
		vmovapd( ymm4, ymm10)
		vmovapd( ymm4, ymm11)
		vmovapd( ymm4, ymm12)
		vmovapd( ymm4, ymm13)
		vmovapd( ymm4, ymm14)
		vmovapd( ymm4, ymm15)

		cmp(imm(8), rdi)
		jz(.DCOLPFETCH)

		label(.DROWPFETCH)
		lea(mem(r12, rdi, 2), rdx)
		lea(mem(rdx, rdi, 1), rdx)

		prefetch(0, mem(rdx, rdi, 1, 1*8))		// c + 4 * rs_c
		prefetch(0, mem(rdx, rdi, 2, 2*8))
		lea(mem(rdx, rdi, 2), rdx)
		lea(mem(rdx, rdi, 1), rdx)				// c + 6 *rsc
		prefetch(0, mem(rdx,         7*8))
		prefetch(0, mem(rdx, rdi, 1, 7*8))
		prefetch(0, mem(rdx, rdi, 2, 7*8))
		lea(mem(rdx, rdi, 2), rdx)
		lea(mem(rdx, rdi, 1), rdx)
		prefetch(0, mem(rdx,         7*8))
		prefetch(0, mem(rdx, rdi, 1, 7*8))
		prefetch(0, mem(rdx, rdi, 2, 7*8))

		jmp(.DPOSTPFETCH)

		label(.DCOLPFETCH)
		mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
		lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
		lea(mem(r12, rsi, 2), rdx)         //
		lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
		prefetch(0, mem(r12,         11*8)) // prefetch c + 0*cs_c
		prefetch(0, mem(r12, rsi, 1, 11*8)) // prefetch c + 1*cs_c
		prefetch(0, mem(r12, rsi, 2, 11*8)) // prefetch c + 2*cs_c
		prefetch(0, mem(rdx,         11*8)) // prefetch c + 3*cs_c
		prefetch(0, mem(rdx, rsi, 1, 11*8)) // prefetch c + 4*cs_c
		prefetch(0, mem(rdx, rsi, 2, 11*8)) // prefetch c + 5*cs_c
		lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
		prefetch(0, mem(rdx, rsi, 1, 11*8)) // prefetch c + 6*cs_c
		prefetch(0, mem(rdx, rsi, 2, 11*8)) // prefetch c + 7*cs_c

		label(.DPOSTPFETCH)
		mov(var(ps_a8), rdx)
		lea(mem(rax, rdx, 1), rdx)	//rdx = a + ps_a8		//for prefetch
		mov(var(ps_a8), rcx)
		lea(mem(r11, rcx, 1), rcx)	//rdx = a + ps_a8		//for prefetch
		mov(var(k_iter), rsi)
		test(rsi, rsi)
		je(.DCONSILEFT)

		// ymm5 and ymm7 contains the data for 16x12 block, other registers contains data for 16x18 block
		label(.DMAIN)
		//0
		prefetch(0, mem(rdx, 5*8))
		prefetch(0, mem(rcx, 5*8))
		vmovupd(mem(rbx,  0*32), ymm0)
		vmovupd(mem(rbx,  1*32), ymm1)

		vbroadcastsd(mem(rax, r8,  4), ymm2)
		vbroadcastsd(mem(rax, r15, 1), ymm3)
		vfmadd231pd(xmm0, xmm2, xmm5)
		vfmadd231pd(xmm0, xmm3, xmm7)
		vbroadcastsd(mem(r11        ), ymm2)
		vbroadcastsd(mem(r11, r8,  1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm4)
		vfmadd231pd(ymm0, ymm3, ymm6)

		vbroadcastsd(mem(r11, r8,  2), ymm2)
		vbroadcastsd(mem(r11, r13, 1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm8)
		vfmadd231pd(ymm1, ymm2, ymm9)
		vfmadd231pd(ymm0, ymm3, ymm10)
		vfmadd231pd(ymm1, ymm3, ymm11)

		vbroadcastsd(mem(r11, r8,  4), ymm2)
		vbroadcastsd(mem(r11, r15, 1), ymm3)
		add(r9, r11)                       // a += cs_a;
		vfmadd231pd(ymm0, ymm2, ymm12)
		vfmadd231pd(ymm1, ymm2, ymm13)
		vfmadd231pd(ymm0, ymm3, ymm14)
		vfmadd231pd(ymm1, ymm3, ymm15)

		add(r10, rbx)
		add(r9, rax)
		//1
		prefetch(0, mem(rdx, r9, 1, 5*8))
		prefetch(0, mem(rcx, r9, 1, 5*8))
		vmovupd(mem(rbx,  0*32), ymm0)
		vmovupd(mem(rbx,  1*32), ymm1)

		vbroadcastsd(mem(rax, r8,  4), ymm2)
		vbroadcastsd(mem(rax, r15, 1), ymm3)
		vfmadd231pd(xmm0, xmm2, xmm5)
		vfmadd231pd(xmm0, xmm3, xmm7)
		vbroadcastsd(mem(r11        ), ymm2)
		vbroadcastsd(mem(r11, r8,  1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm4)
		vfmadd231pd(ymm0, ymm3, ymm6)

		vbroadcastsd(mem(r11, r8,  2), ymm2)
		vbroadcastsd(mem(r11, r13, 1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm8)
		vfmadd231pd(ymm1, ymm2, ymm9)
		vfmadd231pd(ymm0, ymm3, ymm10)
		vfmadd231pd(ymm1, ymm3, ymm11)

		vbroadcastsd(mem(r11, r8,  4), ymm2)
		vbroadcastsd(mem(r11, r15, 1), ymm3)
		add(r9, r11)                       // a += cs_a;
		vfmadd231pd(ymm0, ymm2, ymm12)
		vfmadd231pd(ymm1, ymm2, ymm13)
		vfmadd231pd(ymm0, ymm3, ymm14)
		vfmadd231pd(ymm1, ymm3, ymm15)

		add(r10, rbx)
		add(r9, rax)
		//2
		prefetch(0, mem(rdx, r9, 2, 5*8))
		prefetch(0, mem(rcx, r9, 2, 5*8))
		vmovupd(mem(rbx,  0*32), ymm0)
		vmovupd(mem(rbx,  1*32), ymm1)

		vbroadcastsd(mem(rax, r8,  4), ymm2)
		vbroadcastsd(mem(rax, r15, 1), ymm3)
		vfmadd231pd(xmm0, xmm2, xmm5)
		vfmadd231pd(xmm0, xmm3, xmm7)
		vbroadcastsd(mem(r11        ), ymm2)
		vbroadcastsd(mem(r11, r8,  1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm4)
		vfmadd231pd(ymm0, ymm3, ymm6)

		vbroadcastsd(mem(r11, r8,  2), ymm2)
		vbroadcastsd(mem(r11, r13, 1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm8)
		vfmadd231pd(ymm1, ymm2, ymm9)
		vfmadd231pd(ymm0, ymm3, ymm10)
		vfmadd231pd(ymm1, ymm3, ymm11)

		vbroadcastsd(mem(r11, r8,  4), ymm2)
		vbroadcastsd(mem(r11, r15, 1), ymm3)
		add(r9, r11)                       // a += cs_a;
		vfmadd231pd(ymm0, ymm2, ymm12)
		vfmadd231pd(ymm1, ymm2, ymm13)
		vfmadd231pd(ymm0, ymm3, ymm14)
		vfmadd231pd(ymm1, ymm3, ymm15)

		add(r10, rbx)
		add(r9, rax)
		//3
		lea(mem(rdx, r9,  2), rdx)
		lea(mem(rcx, r9,  2), rcx)
		prefetch(0, mem(rdx, r9, 1, 5*8))
		prefetch(0, mem(rcx, r9, 1, 5*8))
		lea(mem(rdx, r9,  2), rdx)
		lea(mem(rcx, r9,  2), rcx)

		vmovupd(mem(rbx,  0*32), ymm0)
		vmovupd(mem(rbx,  1*32), ymm1)

		vbroadcastsd(mem(rax, r8,  4), ymm2)
		vbroadcastsd(mem(rax, r15, 1), ymm3)
		vfmadd231pd(xmm0, xmm2, xmm5)
		vfmadd231pd(xmm0, xmm3, xmm7)
		vbroadcastsd(mem(r11        ), ymm2)
		vbroadcastsd(mem(r11, r8,  1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm4)
		vfmadd231pd(ymm0, ymm3, ymm6)

		vbroadcastsd(mem(r11, r8,  2), ymm2)
		vbroadcastsd(mem(r11, r13, 1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm8)
		vfmadd231pd(ymm1, ymm2, ymm9)
		vfmadd231pd(ymm0, ymm3, ymm10)
		vfmadd231pd(ymm1, ymm3, ymm11)

		vbroadcastsd(mem(r11, r8,  4), ymm2)
		vbroadcastsd(mem(r11, r15, 1), ymm3)
		add(r9, r11)                       // a += cs_a;
		vfmadd231pd(ymm0, ymm2, ymm12)
		vfmadd231pd(ymm1, ymm2, ymm13)
		vfmadd231pd(ymm0, ymm3, ymm14)
		vfmadd231pd(ymm1, ymm3, ymm15)

		add(r10, rbx)
		add(r9, rax)

		dec(rsi)
		jne(.DMAIN)

		label(.DCONSILEFT)
		mov(var(k_left), rsi)
		test(rsi, rsi)
		je(.DPOSTACC)

		label(.DLEFT)
		prefetch(0, mem(rdx, 5*8))
		prefetch(0, mem(rcx, 5*8))
		add(r9, rcx)
		add(r9, rdx)
		vmovupd(mem(rbx,  0*32), ymm0)
		vmovupd(mem(rbx,  1*32), ymm1)

		vbroadcastsd(mem(rax, r8,  4), ymm2)
		vbroadcastsd(mem(rax, r15, 1), ymm3)
		vfmadd231pd(xmm0, xmm2, xmm5)
		vfmadd231pd(xmm0, xmm3, xmm7)
		vbroadcastsd(mem(r11        ), ymm2)
		vbroadcastsd(mem(r11, r8,  1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm4)
		vfmadd231pd(ymm0, ymm3, ymm6)

		vbroadcastsd(mem(r11, r8,  2), ymm2)
		vbroadcastsd(mem(r11, r13, 1), ymm3)
		vfmadd231pd(ymm0, ymm2, ymm8)
		vfmadd231pd(ymm1, ymm2, ymm9)
		vfmadd231pd(ymm0, ymm3, ymm10)
		vfmadd231pd(ymm1, ymm3, ymm11)

		vbroadcastsd(mem(r11, r8,  4), ymm2)
		vbroadcastsd(mem(r11, r15, 1), ymm3)
		add(r9, r11)                       // a += cs_a;
		vfmadd231pd(ymm0, ymm2, ymm12)
		vfmadd231pd(ymm1, ymm2, ymm13)
		vfmadd231pd(ymm0, ymm3, ymm14)
		vfmadd231pd(ymm1, ymm3, ymm15)

		add(r10, rbx)
		add(r9, rax)

		dec(rsi)
		jne(.DLEFT)

		label(.DPOSTACC)
		mov(r12, rcx)
		mov(var(alpha), rax)
		mov(var(beta), rbx)
		vbroadcastsd(mem(rax), ymm0)
		vbroadcastsd(mem(rbx), ymm3)
		vmulpd(ymm0, ymm5, ymm5)
		vmulpd(ymm0, ymm7, ymm7)
		vmulpd(ymm0, ymm4, ymm4)
		vmulpd(ymm0, ymm6, ymm6)
		vmulpd(ymm0, ymm8, ymm8)
		vmulpd(ymm0, ymm9, ymm9)
		vmulpd(ymm0, ymm10, ymm10)
		vmulpd(ymm0, ymm11, ymm11)
		vmulpd(ymm0, ymm12, ymm12)
		vmulpd(ymm0, ymm13, ymm13)
		vmulpd(ymm0, ymm14, ymm14)
		vmulpd(ymm0, ymm15, ymm15)

		mov(var(cs_c), rsi)
		lea(mem(, rsi, 8), rsi)
		vxorpd(ymm0, ymm0, ymm0)
		lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
		vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
		je(.DBETAZERO)

		cmp(imm(8), rdi)	//rs_c == 8?
		je(.DCOLSTOR)

		label(.DROWSTOR)
		lea(mem(rcx, rdi, 4), rcx)        //rcx += 4 * rdi
		vfmadd231pd(mem(rcx, 0*32), xmm3, xmm5)
		vmovlpd(xmm5, mem(rcx))
		add(rdi, rcx)
		vfmadd231pd(mem(rcx, 0*32), xmm3, xmm7)
		vmovlpd(xmm7, mem(rcx))
		vmovhpd(xmm7, mem(rcx, rsi, 1))

		//for lower 6x8
		lea(mem(rcx, rdi, 1), rcx)        //rcx += 1 * rdi
		vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
		vmovupd(xmm4, mem(rcx, 0*32))
		vextractf128(imm(1), ymm4, xmm1)
		vmovlpd(xmm1, mem(rcx, 2*8))
		add(rdi, rcx)

		vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
		vmovupd(ymm6, mem(rcx, 0*32))
		add(rdi, rcx)

		vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
		vmovupd(ymm8, mem(rcx, 0*32))

		vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
		vmovlpd(xmm9, mem(rcx, 1*32))
		add(rdi, rcx)


		vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
		vmovupd(ymm10, mem(rcx, 0*32))

		vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
		vmovupd(xmm11, mem(rcx, 1*32))
		add(rdi, rcx)


		vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
		vmovupd(ymm12, mem(rcx, 0*32))

		vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
		vmovupd(xmm13, mem(rcx, 1*32))
		vextractf128(imm(1), ymm13, xmm1)
		vmovlpd(xmm1, mem(rcx, 1*32+2*8))
		add(rdi, rcx)


		vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
		vmovupd(ymm14, mem(rcx, 0*32))

		vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
		vmovupd(ymm15, mem(rcx, 1*32))

		jmp(.DDONE)

		label(.DCOLSTOR)
		vbroadcastsd(mem(rbx), ymm3)

		lea(mem(rcx, rdi, 4), rdx)        //rdx = rcx + 4* rs_c
		vunpcklpd(ymm7, ymm5, ymm0)
		vunpckhpd(ymm7, ymm5, ymm1)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

		vfmadd231pd(mem(rdx), xmm3, xmm5)
		vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm7)
		vmovupd(xmm5, mem(rdx        ))
		vmovhpd(xmm7, mem(rdx, rsi, 1, 1*8))

		lea(mem(rcx, rdi, 4), rcx)
		lea(mem(rcx, rdi, 2), rcx)
		lea(mem(rcx, rdi, 4), rdx)

		vunpcklpd(ymm6, ymm4, ymm0)
		vunpckhpd(ymm6, ymm4, ymm1)
		vunpcklpd(ymm10, ymm8, ymm2)
		vunpckhpd(ymm10, ymm8, ymm3)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
		vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
		vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

		vbroadcastsd(mem(rbx), ymm3)

		vfmadd231pd(mem(rcx        ), ymm3, ymm4)
		vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
		vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
		vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
		vmovupd(ymm4, mem(rcx        ))
		vmovupd(ymm6, mem(rcx, rsi, 1))
		vmovupd(ymm8, mem(rcx, rsi, 2))
		vextractf128(imm(1), ymm10, xmm1)
		vmovhpd(xmm10, mem(rcx, rax, 1, 1*8))
		vmovupd(xmm1,  mem(rcx, rax, 1, 2*8))


		lea(mem(rcx, rsi, 4), rcx)

		vunpcklpd(ymm14, ymm12, ymm0)
		vunpckhpd(ymm14, ymm12, ymm1)
		vextractf128(imm(0x1), ymm0, xmm2)
		vextractf128(imm(0x1), ymm1, xmm4)

		vfmadd231pd(mem(rdx        ), xmm3, xmm0)
		vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
		vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
		vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
		vmovupd(xmm0, mem(rdx        ))
		vmovupd(xmm1, mem(rdx, rsi, 1))
		vmovupd(xmm2, mem(rdx, rsi, 2))
		vmovupd(xmm4, mem(rdx, rax, 1))

		lea(mem(rdx, rsi, 4), rdx)

										// begin I/O on columns 4-7
		vunpcklpd(ymm7, ymm5, ymm0)
		vunpckhpd(ymm7, ymm5, ymm1)
		vunpcklpd(ymm11, ymm9, ymm2)
		vunpckhpd(ymm11, ymm9, ymm3)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

		vbroadcastsd(mem(rbx), ymm3)

		vfmadd231pd(mem(rcx        ), ymm3, ymm5)
		vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
		vextractf128(imm(1), ymm5, xmm1)
		vmovupd(xmm1, mem(rcx, 2*8   ))
		vextractf128(imm(1), ymm7, xmm1)
		vmovhpd(xmm1, mem(rcx, rsi, 1, 3*8))

		vunpcklpd(ymm15, ymm13, ymm0)
		vunpckhpd(ymm15, ymm13, ymm1)
		vextractf128(imm(0x1), ymm0, xmm2)
		vextractf128(imm(0x1), ymm1, xmm4)

		vfmadd231pd(mem(rdx        ), xmm3, xmm0)
		vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
		vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
		vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
		vmovupd(xmm0, mem(rdx        ))
		vmovupd(xmm1, mem(rdx, rsi, 1))
		vmovupd(xmm2, mem(rdx, rsi, 2))
		vmovhpd(xmm4, mem(rdx, rax, 1, 1*8))
		jmp(.DDONE)

		label(.DBETAZERO)
		cmp(imm(8), rdi)
		je(.DCOLSTORBZ)

		label(.DROWSTORBZ)
		lea(mem(rcx, rdi, 4), rcx)        //rcx += 4 * rdi
		vmovlpd(xmm5, mem(rcx))
		add(rdi, rcx)
		vmovlpd(xmm7, mem(rcx))
		vmovhpd(xmm7, mem(rcx, rsi, 1))

		//For lower 6x8 block
		lea(mem(rcx, rdi, 1), rcx)        //rcx += 1 * rdi
		vmovupd(xmm4, mem(rcx, 0*32))
		vextractf128(imm(1), ymm4, xmm1)
		vmovlpd(xmm1, mem(rcx, 2*8))
		add(rdi, rcx)

		vmovupd(ymm6, mem(rcx, 0*32))
		add(rdi, rcx)

		vmovupd(ymm8, mem(rcx, 0*32))

		vmovlpd(xmm9, mem(rcx, 1*32))
		add(rdi, rcx)

		vmovupd(ymm10, mem(rcx, 0*32))

		vmovupd(xmm11, mem(rcx, 1*32))
		add(rdi, rcx)


		vmovupd(ymm12, mem(rcx, 0*32))

		vmovupd(xmm13, mem(rcx, 1*32))
		vextractf128(imm(1), ymm13, xmm1)
		vmovlpd(xmm1, mem(rcx, 1*32+2*8))
		add(rdi, rcx)

		vmovupd(ymm14, mem(rcx, 0*32))

		vmovupd(ymm15, mem(rcx, 1*32))

		jmp(.DDONE)

		label(.DCOLSTORBZ)
		vbroadcastsd(mem(rbx), ymm3)

		lea(mem(rcx, rdi, 4), rdx)        //rdx = rcx + 4* rs_c
		vunpcklpd(ymm7, ymm5, ymm0)
		vunpckhpd(ymm7, ymm5, ymm1)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

		vmovupd(xmm5, mem(rdx        ))
		vmovhpd(xmm7, mem(rdx, rsi, 1, 1*8))

		lea(mem(rcx, rdi, 4), rcx)
		lea(mem(rcx, rdi, 2), rcx)
		lea(mem(rcx, rdi, 4), rdx)

		vunpcklpd(ymm6, ymm4, ymm0)
		vunpckhpd(ymm6, ymm4, ymm1)
		vunpcklpd(ymm10, ymm8, ymm2)
		vunpckhpd(ymm10, ymm8, ymm3)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
		vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
		vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

		vbroadcastsd(mem(rbx), ymm3)

		vmovupd(ymm4, mem(rcx        ))
		vmovupd(ymm6, mem(rcx, rsi, 1))
		vmovupd(ymm8, mem(rcx, rsi, 2))
		vextractf128(imm(1), ymm10, xmm1)
		vmovhpd(xmm10, mem(rcx, rax, 1, 1*8))
		vmovupd(xmm1,  mem(rcx, rax, 1, 2*8))


		lea(mem(rcx, rsi, 4), rcx)

		vunpcklpd(ymm14, ymm12, ymm0)
		vunpckhpd(ymm14, ymm12, ymm1)
		vextractf128(imm(0x1), ymm0, xmm2)
		vextractf128(imm(0x1), ymm1, xmm4)

		vmovupd(xmm0, mem(rdx        ))
		vmovupd(xmm1, mem(rdx, rsi, 1))
		vmovupd(xmm2, mem(rdx, rsi, 2))
		vmovupd(xmm4, mem(rdx, rax, 1))

		lea(mem(rdx, rsi, 4), rdx)

		// begin I/O on columns 4-7
		vunpcklpd(ymm7, ymm5, ymm0)
		vunpckhpd(ymm7, ymm5, ymm1)
		vunpcklpd(ymm11, ymm9, ymm2)
		vunpckhpd(ymm11, ymm9, ymm3)
		vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
		vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

		vbroadcastsd(mem(rbx), ymm3)

		vextractf128(imm(1), ymm5, xmm1)
		vmovupd(xmm1, mem(rcx, 2*8   ))
		vextractf128(imm(1), ymm7, xmm1)
		vmovhpd(xmm1, mem(rcx, rsi, 1, 3*8))

		vunpcklpd(ymm15, ymm13, ymm0)
		vunpckhpd(ymm15, ymm13, ymm1)
		vextractf128(imm(0x1), ymm0, xmm2)
		vextractf128(imm(0x1), ymm1, xmm4)

		vmovupd(xmm0, mem(rdx        ))
		vmovupd(xmm1, mem(rdx, rsi, 1))
		vmovupd(xmm2, mem(rdx, rsi, 2))
		vmovhpd(xmm4, mem(rdx, rax, 1, 1*8))
		jmp(.DDONE)

		label(.DDONE)
		vzeroupper()

		end_asm(
			: // output operands (none)
			: // input operands
			[k_iter] "m" (k_iter),
			[k_left] "m" (k_left),
			[a]      "m" (a),
			[a_next] "m" (a_next),
			[rs_a]   "m" (rs_a),
			[cs_a]   "m" (cs_a),
      		[ps_a8]  "m" (ps_a8),
			[b]      "m" (b),
			[rs_b]   "m" (rs_b),
			[cs_b]   "m" (cs_b),
			[alpha]  "m" (alpha),
      		[beta]   "m" (beta),
			[c]      "m" (c),
			[rs_c]   "m" (rs_c),
			[cs_c]   "m" (cs_c)/*,
			[a_next] "m" (a_next),
			[b_next] "m" (b_next)*/
			: // register clobber list
			"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
			"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
			"xmm0", "xmm1", "xmm2", "xmm3",
			"xmm4", "xmm5", "xmm6", "xmm7",
			"xmm8", "xmm9", "xmm10", "xmm11",
			"xmm12", "xmm13", "xmm14", "xmm15",
			"ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
			"ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
			"ymm12", "ymm13", "ymm14", "ymm15",
			"memory"
			)
	}
/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 0(6x0)
(6x0)_L


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		6    x x x x x x x -
|		7    x x x x x x x x
m		8    x x x x x x x x
off		9    x x x x x x x x
24		10   x x x x x x x x
|		11   x x x x x x x x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_6x0_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a8  = bli_auxinfo_ps_a( data ) * sizeof( double );


	// -------------------------------------------------------------------------

	begin_asm()


	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)                      // reset rax to current upanel of a.



	PREFETCH_C()

	label(.DPOSTPFETCH)                // done prefetching c


	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;



	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

	prefetch(0, mem(rdx, 5*8))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 1

	prefetch(0, mem(rdx, r9, 1, 5*8))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 2

	prefetch(0, mem(rdx, r9, 2, 5*8))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 3

	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(xmm5, mem(rcx, 1*32))
	vextractf128(imm(1), ymm5, xmm1)
	vmovlpd(xmm1, mem(rcx, 1*32+2*8))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rcx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovhpd(xmm11, mem(rcx, rax, 1, 1*8))
	vextractf128(imm(1), ymm11, xmm1)
	vmovupd(xmm1, mem(rcx, rax, 1, 2*8))


	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	vextractf128(imm(1), ymm5, xmm1)
	vmovlpd(xmm1, mem(rcx, 1*32+2*8))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(ymm15, mem(rcx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovhpd(xmm11, mem(rcx, rax, 1, 1*8))
	vextractf128(imm(1), ymm11, xmm1)
	vmovupd(xmm1, mem(rcx, rax, 1, 2*8))


	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)


	// Handle edge cases in the m dimension, if they exist.
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 8(12x8)
(12x8)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		12   x x x x x - - -
|		13   x x x x x x - -
m		14   x x x x x x x -
off		15   x x x x x x x x
24		16   x x x x x x x x
|		17   x x x x x x x x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_12x8_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a8  = bli_auxinfo_ps_a( data ) * sizeof( double );


	// -------------------------------------------------------------------------

	begin_asm()

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)                      // reset rax to current upanel of a.



	PREFETCH_C()

	label(.DPOSTPFETCH)                // done prefetching c


	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;



	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 1

	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 2

	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 3

	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovlpd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(xmm9, mem(rcx, 1*32))
	vextractf128(imm(1), ymm9, xmm1)
	vmovlpd(xmm1, mem(rcx, 1*32+2*8))

	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rcx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vextractf128(imm(1), ymm7, xmm1)
	vmovhpd(xmm7, mem(rcx, rsi, 1, 1*8))
	vmovupd(xmm1, mem(rcx, rsi, 1, 2*8))
	vextractf128(imm(1), ymm9, xmm1)
	vmovupd(xmm1, mem(rcx, rsi, 2, 2*8))
	vextractf128(imm(1), ymm11, xmm1)
	vmovhpd(xmm1, mem(rcx, rax, 1, 3*8))


	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))



	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovlpd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(xmm9, mem(rcx, 1*32))
	vextractf128(imm(1), ymm9, xmm1)
	vmovlpd(xmm1, mem(rcx, 1*32+2*8))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(ymm15, mem(rcx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vextractf128(imm(1), ymm7, xmm1)
	vmovhpd(xmm7, mem(rcx, rsi, 1, 1*8))
	vmovupd(xmm1, mem(rcx, rsi, 1, 2*8))
	vextractf128(imm(1), ymm9, xmm1)
	vmovupd(xmm1, mem(rcx, rsi, 2, 2*8))
	vextractf128(imm(1), ymm11, xmm1)
	vmovhpd(xmm1, mem(rcx, rax, 1, 3*8))


	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)


	// Handle edge cases in the m dimension, if they exist.
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}
/*

Following kernel computes the 6x8 block for the Lower vairant(L) of gemmt where
m_offset in 24x24 block is 18 and n_offset is 16(18x16)
(18x16)_L


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		18   x x x - - - - -
|		19   x x x x - - - -
m		20   x x x x x - - -
off		21   x x x x x x - -
24		22   x x x x x x x -
|		23   x x x x x x x x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_18x16_L
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a8  = bli_auxinfo_ps_a( data ) * sizeof( double );


	// -------------------------------------------------------------------------

	begin_asm()


	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)                      // reset rax to current upanel of a.



	PREFETCH_C()

	label(.DPOSTPFETCH)                // done prefetching c


	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.



	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

	prefetch(0, mem(rdx, 5*8))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 1

	prefetch(0, mem(rdx, r9, 1, 5*8))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 2

	prefetch(0, mem(rdx, r9, 2, 5*8))
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	// ---------------------------------- iteration 3

	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(xmm4, mem(rcx, 0*32))
	vextractf128(imm(1), ymm4, xmm1)
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovlpd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(xmm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(xmm13, mem(rcx, 1*32))
	vextractf128(imm(1), ymm13, xmm1)
	vmovlpd(xmm1, mem(rcx, 1*32+2*8))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rcx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vextractf128(imm(1), ymm10, xmm1)
	vmovhpd(xmm10, mem(rcx, rax, 1, 1*8))
	vmovupd(xmm1,  mem(rcx, rax, 1, 2*8))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vextractf128(imm(1), ymm5, xmm1)
	vmovupd(xmm1, mem(rcx, 2*8   ))
	vextractf128(imm(1), ymm7, xmm1)
	vmovhpd(xmm1, mem(rcx, rsi, 1, 3*8))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovhpd(xmm4, mem(rdx, rax, 1, 1*8))



	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(xmm4, mem(rcx, 0*32))
	vextractf128(imm(1), ymm4, xmm1)
	vmovlpd(xmm1, mem(rcx, 2*8))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovlpd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(xmm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(xmm13, mem(rcx, 1*32))
	vextractf128(imm(1), ymm13, xmm1)
	vmovlpd(xmm1, mem(rcx, 1*32+2*8))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(ymm15, mem(rcx, 1*32))
	//add(rdi, rcx)

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vextractf128(imm(1), ymm10, xmm1)
	vmovhpd(xmm10, mem(rcx, rax, 1, 1*8))
	vmovupd(xmm1,  mem(rcx, rax, 1, 2*8))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vextractf128(imm(1), ymm5, xmm1)
	vmovupd(xmm1, mem(rcx, 2*8   ))
	vextractf128(imm(1), ymm7, xmm1)
	vmovhpd(xmm1, mem(rcx, rsi, 1, 3*8))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovhpd(xmm4, mem(rdx, rax, 1, 1*8))

	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)


	// Handle edge cases in the m dimension, if they exist.
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}
/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 0 and n_offset is 0(0x0)
(0x0)_U

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		0    x x x x x x x x
|		1    - x x x x x x x
m		2    - - x x x x x x
off		3    - - - x x x x x
24		4    - - - - x x x x
|		5    - - - - - x x x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_0x0_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	uint64_t ps_a8   = bli_auxinfo_ps_a( data ) * sizeof( double );

	begin_asm()


	mov(var(a), r14)
	mov(var(rs_a), r8)
	mov(var(cs_a), r9)
	lea(mem(, r8, 8), r8)
	lea(mem(, r9, 8), r9)

	lea(mem(r8, r8, 2), r13)
	lea(mem(r8, r8, 4), r15)

	mov(var(rs_b), r10)
	lea(mem(, r10, 8), r10)

	mov(var(c), r12)
	mov(var(rs_c), rdi)
	lea(mem(, rdi, 8), rdi)

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)
	mov(r14, rax)



	PREFETCH_C()

	label(.DPOSTPFETCH)

	mov(var(ps_a8), rdx)
	lea(mem(rax, rdx, 1), rdx)
	lea(mem(r9, r9, 2), rcx)

	mov(var(k_iter), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	//ymm12, ymm14 can be skipped
	label(.DLOOPKITER)                 // MAIN LOOP
	//0
	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//1
	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)
	//2

	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)
	//3
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKITER)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)
	test(rsi, rsi)
	je(.DPOSTACCUM)

	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKLEFT)

	label(.DPOSTACCUM)



	mov(r12, rcx)
	mov(var(alpha), rax)
	mov(var(beta), rbx)
	vbroadcastsd(mem(rax), ymm0)
	vbroadcastsd(mem(rbx), ymm3)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(rcx, rdi, 4), rdx)         // c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // 3*cs_c;


	vxorpd(ymm0, ymm0, ymm0)
	vucomisd(xmm0, xmm3)
	je(.DBETAZERO)

	cmp(imm(8), rdi)
	jz(.DCOLSTORED)

	label(.DROWSTORED)

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovhpd(xmm6, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm6, xmm6)
	vmovupd(xmm6, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovupd(xmm8, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vextractf128(imm(0x1), ymm10, xmm10)
	vmovhpd(xmm10, mem(rcx, 0*32+3*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovhpd(xmm15, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm15, xmm15)
	vmovupd(xmm15, mem(rcx, 1*32+2*8))


	jmp(.DDONE)


	label(.DCOLSTORED)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovlpd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovlpd(xmm8, mem(rcx, rsi, 2, 1*16))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovlpd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)				   // if beta zero


	cmp(imm(8), rdi)
	jz(.DCOLSTORBZ)

	label(.DROWSTORBZ)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovhpd(xmm6, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm6, xmm6)
	vmovupd(xmm6, mem(rcx, 0*32+2*8))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm8, xmm8)
	vmovupd(xmm8, mem(rcx, 0*32+2*8))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm10, xmm10)
	vmovhpd(xmm10, mem(rcx, 0*32+3*8))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovhpd(xmm15, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm15, xmm15)
	vmovupd(xmm15, mem(rcx, 1*32+2*8))

	jmp(.DDONE)

	label(.DCOLSTORBZ)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovlpd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovlpd(xmm8, mem(rcx, rsi, 2, 1*16))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovlpd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))


	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 8(6x8)
(6x8)_U

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		6    x x x x x x x x
|		7    x x x x x x x x
m		8    x x x x x x x x
off		9    - x x x x x x x
24		10   - - x x x x x x
|		11   - - - x x x x x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_6x8_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	uint64_t ps_a8   = bli_auxinfo_ps_a( data ) * sizeof( double );

	begin_asm()


	mov(var(a), r14)
	mov(var(rs_a), r8)
	mov(var(cs_a), r9)
	lea(mem(, r8, 8), r8)
	lea(mem(, r9, 8), r9)

	lea(mem(r8, r8, 2), r13)
	lea(mem(r8, r8, 4), r15)

	mov(var(rs_b), r10)
	lea(mem(, r10, 8), r10)

	mov(var(c), r12)
	mov(var(rs_c), rdi)
	lea(mem(, rdi, 8), rdi)

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)
	mov(r14, rax)


	PREFETCH_C()

	label(.DPOSTPFETCH)

	mov(var(ps_a8), rdx)
	lea(mem(rax, rdx, 1), rdx)
	lea(mem(r9, r9, 2), rcx)

	mov(var(k_iter), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	label(.DLOOPKITER)                 // MAIN LOOP
	//0
	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//1
	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//2

	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//3
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKITER)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)
	test(rsi, rsi)
	je(.DPOSTACCUM)

	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKLEFT)

	label(.DPOSTACCUM)



	mov(r12, rcx)
	mov(var(alpha), rax)
	mov(var(beta), rbx)
	vbroadcastsd(mem(rax), ymm0)
	vbroadcastsd(mem(rbx), ymm3)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(rcx, rdi, 4), rdx)         // c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // 3*cs_c;


	vxorpd(ymm0, ymm0, ymm0)
	vucomisd(xmm0, xmm3)
	je(.DBETAZERO)

	cmp(imm(8), rdi)
	jz(.DCOLSTORED)

	label(.DROWSTORED)

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovhpd(xmm10, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm10, xmm10)
	vmovupd(xmm10, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vextractf128(imm(0x1), ymm12, xmm12)
	vmovupd(xmm12, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vextractf128(imm(0x1), ymm14, xmm14)
	vmovhpd(xmm14, mem(rcx, 0*32+3*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rcx, 1*32))

	jmp(.DDONE)


	label(.DCOLSTORED)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(xmm4, mem(rcx        ))
	vextractf128(imm(0x1), ymm4, xmm4)
	vmovlpd(xmm4, mem(rcx, 2*8   ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovlpd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)				   // if beta zero


	cmp(imm(8), rdi)
	jz(.DCOLSTORBZ)

	label(.DROWSTORBZ)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovhpd(xmm10, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm10, xmm10)
	vmovupd(xmm10, mem(rcx, 0*32+2*8))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vextractf128(imm(0x1), ymm12, xmm12)
	vmovupd(xmm12, mem(rcx, 0*32+2*8))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm14, xmm14)
	vmovhpd(xmm14, mem(rcx, 0*32+3*8))
	vmovupd(ymm15, mem(rcx, 1*32))

	jmp(.DDONE)

	label(.DCOLSTORBZ)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(xmm4, mem(rcx        ))
	vextractf128(imm(0x1), ymm4, xmm4)
	vmovlpd(xmm4, mem(rcx, 2*8   ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovlpd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))


	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 16(12x16)
(12x16)_U


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		12   x x x x x x x x
|		13   x x x x x x x x
m		14   x x x x x x x x
off		15   x x x x x x x x
24		16   x x x x x x x x
|		17   - x x x x x x x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_12x16_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	uint64_t ps_a8   = bli_auxinfo_ps_a( data ) * sizeof( double );

	begin_asm()


	mov(var(a), r14)
	mov(var(rs_a), r8)
	mov(var(cs_a), r9)
	lea(mem(, r8, 8), r8)
	lea(mem(, r9, 8), r9)

	lea(mem(r8, r8, 2), r13)
	lea(mem(r8, r8, 4), r15)

	mov(var(rs_b), r10)
	lea(mem(, r10, 8), r10)

	mov(var(c), r12)
	mov(var(rs_c), rdi)
	lea(mem(, rdi, 8), rdi)

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)
	mov(r14, rax)


	PREFETCH_C()

	label(.DPOSTPFETCH)

	mov(var(ps_a8), rdx)
	lea(mem(rax, rdx, 1), rdx)
	lea(mem(r9, r9, 2), rcx)

	mov(var(k_iter), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	label(.DLOOPKITER)                 // MAIN LOOP
	//0
	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	//1
	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//2

	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	//3
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKITER)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)
	test(rsi, rsi)
	je(.DPOSTACCUM)

	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKLEFT)

	label(.DPOSTACCUM)



	mov(r12, rcx)
	mov(var(alpha), rax)
	mov(var(beta), rbx)
	vbroadcastsd(mem(rax), ymm0)
	vbroadcastsd(mem(rbx), ymm3)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(rcx, rdi, 4), rdx)         // c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // 3*cs_c;


	vxorpd(ymm0, ymm0, ymm0)
	vucomisd(xmm0, xmm3)
	je(.DBETAZERO)

	cmp(imm(8), rdi)
	jz(.DCOLSTORED)

	label(.DROWSTORED)

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm14)
	vmovhpd(xmm14, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm14, xmm14)
	vmovupd(xmm14, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovupd(ymm15, mem(rcx, 1*32))

	jmp(.DDONE)


	label(.DCOLSTORED)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovlpd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)
	jz(.DCOLSTORBZ)

	label(.DROWSTORBZ)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovhpd(xmm14, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm14, xmm14)
	vmovupd(xmm14, mem(rcx, 0*32+2*8))
	vmovupd(ymm15, mem(rcx, 1*32))

	jmp(.DDONE)

	label(.DCOLSTORBZ)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovlpd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))


	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 6 and n_offset is 0(6x0)
(6x0)_U


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		6    - - - - - - x x
|		7    - - - - - - - x
m		8    - - - - - - - -
off		9    - - - - - - - -
24		10   - - - - - - - -
|		11   - - - - - - - -
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_6x0_U
	(
		conj_t              conja,
		conj_t              conjb,
		dim_t               m0,
		dim_t               n0,
		dim_t               k0,
		double*     restrict alpha,
		double*     restrict a, inc_t rs_a0, inc_t cs_a0,
		double*     restrict b, inc_t rs_b0, inc_t cs_b0,
		double*     restrict beta,
		double*     restrict c, inc_t rs_c0, inc_t cs_c0,
		auxinfo_t* restrict data,
		cntx_t*    restrict cntx
	)
{
		uint64_t k_iter = k0 / 4;
		uint64_t k_left = k0 % 4;
		uint64_t rs_a   = rs_a0;
		uint64_t cs_a   = cs_a0;
		uint64_t rs_b   = rs_b0;
		uint64_t cs_b   = cs_b0;
		uint64_t rs_c   = rs_c0;
		uint64_t cs_c   = cs_c0;
		uint64_t ps_a8 = bli_auxinfo_ps_a( data ) * sizeof( double );

		begin_asm()
		mov(var(a), r14)
		mov(var(b), rbx)
		mov(var(c), r12)
		mov(r14, rax)

		mov(var(rs_a), r8)
		mov(var(cs_a), r9)
		lea(mem(, r8, 8), r8)
		lea(mem(, r9, 8), r9)

		mov(var(rs_b), r10)
		lea(mem(, r10, 8), r10)

		mov(var(rs_c), rdi)
		lea(mem(, rdi, 8), rdi)

		lea(mem(r8, r8, 4), r15)

		vxorpd(ymm5, ymm5, ymm5)
		vxorpd(ymm7, ymm7, ymm7)

		cmp(imm(8), rdi)
		jz(.DCOLPFETCH)

		label(.DROWPFETCH)
		lea(mem(r12, rdi, 2), rdx)
		lea(mem(rdx, rdi, 1), rdx)
		prefetch(0, mem(rdx, rdi, 1, 1*8))
		prefetch(0, mem(rdx, rdi, 2, 2*8))
		jmp(.DPOSTPFETCH)

		label(.DCOLPFETCH)
		mov(var(cs_c), rsi)
		lea(mem(, rsi, 8), rsi)
		prefetch(0, mem(r12, 5*8))
		prefetch(0, mem(r12, rsi, 1, 5*8))

		label(.DPOSTPFETCH)
		mov(var(k_iter), rsi)
		test(rsi, rsi)
		lea(mem(rbx, 1*16), rbx)
		je(.DCONSILEFT)

		//compute xmm5 and xmm7 only
		label(.DMAIN)
		//0
		vmovupd(mem(rbx,  1*32), xmm1)
		vbroadcastsd(mem(rax        ), ymm2)
		vbroadcastsd(mem(rax, r8,  1), ymm3)
		vfmadd231pd(xmm1, xmm2, xmm5)
		vfmadd231pd(xmm1, xmm3, xmm7)
		add(r10, rbx)
		add(r9, rax)
		//1
		vmovupd(mem(rbx,  1*32), xmm1)
		vbroadcastsd(mem(rax        ), ymm2)
		vbroadcastsd(mem(rax, r8,  1), ymm3)
		vfmadd231pd(xmm1, xmm2, xmm5)
		vfmadd231pd(xmm1, xmm3, xmm7)
		add(r10, rbx)
		add(r9, rax)
		//2
		vmovupd(mem(rbx,  1*32), xmm1)
		vbroadcastsd(mem(rax        ), ymm2)
		vbroadcastsd(mem(rax, r8,  1), ymm3)
		vfmadd231pd(xmm1, xmm2, xmm5)
		vfmadd231pd(xmm1, xmm3, xmm7)
		add(r10, rbx)
		add(r9, rax)
		//3
		vmovupd(mem(rbx,  1*32), xmm1)
		vbroadcastsd(mem(rax        ), ymm2)
		vbroadcastsd(mem(rax, r8,  1), ymm3)
		vfmadd231pd(xmm1, xmm2, xmm5)
		vfmadd231pd(xmm1, xmm3, xmm7)
		add(r10, rbx)
		add(r9, rax)

		dec(rsi)
		jne(.DMAIN)

		label(.DCONSILEFT)
		mov(var(k_left), rsi)
		test(rsi, rsi)
		je(.DPOSTACC)

		label(.DLEFT)
		vmovupd(mem(rbx,  1*32), xmm1)
		vbroadcastsd(mem(rax        ), ymm2)
		vbroadcastsd(mem(rax, r8,  1), ymm3)
		vfmadd231pd(xmm1, xmm2, xmm5)
		vfmadd231pd(xmm1, xmm3, xmm7)
		add(r10, rbx)
		add(r9, rax)
		dec(rsi)
		jne(.DLEFT)

		label(.DPOSTACC)
		mov(r12, rcx)
		mov(var(alpha), rax)
		mov(var(beta), rbx)
		vbroadcastsd(mem(rax), ymm0)
		vbroadcastsd(mem(rbx), ymm3)
		lea(mem(rsi, rsi, 2), rax)
		vmulpd(ymm0, ymm5, ymm5)
		vmulpd(ymm0, ymm7, ymm7)

		mov(var(cs_c), rsi)
		lea(mem(, rsi, 8), rsi)
		vxorpd(ymm0, ymm0, ymm0)
		vucomisd(xmm0, xmm3)
		je(.DBETAZERO)

		cmp(imm(8), rdi)
		je(.DCOLSTOR)

		label(.DROWSTOR)
		lea(mem(rcx, 1*32), rcx)
		lea(mem(rcx, 1*16), rcx)

		vfmadd231pd(mem(rcx, 0*32), xmm3, xmm5)
		vmovlpd(xmm5, mem(rcx))
		vmovhpd(xmm5, mem(rcx, rsi, 1))
		add(rdi, rcx)
		vfmadd231pd(mem(rcx, 0*32), xmm3, xmm7)
		vmovhpd(xmm7, mem(rcx, rsi, 1))

		jmp(.DDONE)

		label(.DCOLSTOR)

		vbroadcastsd(mem(rbx), ymm3)
		lea(mem(rcx, rsi, 4), rcx)
		lea(mem(rcx, rsi, 2), rcx)
		vunpcklpd(xmm7, xmm5, xmm0)
		vunpckhpd(xmm7, xmm5, xmm1)
		vfmadd231pd(mem(rcx        ), xmm3, xmm0)
		vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)
		vmovlpd(xmm0, mem(rcx        ))
		vmovupd(xmm1, mem(rcx, rsi, 1))
		jmp(.DDONE)

		label(.DBETAZERO)
		cmp(imm(8), rdi)
		je(.DCOLSTORBZ)

		label(.DROWSTORBZ)
		lea(mem(rcx, 1*32), rcx)
		lea(mem(rcx, 1*16), rcx)

		vmovlpd(xmm5, mem(rcx))
		vmovhpd(xmm5, mem(rcx, rsi, 1))
		add(rdi, rcx)
		vmovhpd(xmm7, mem(rcx, rsi, 1))

		jmp(.DDONE)

		label(.DCOLSTORBZ)

		lea(mem(rcx, rsi, 4), rcx)
		lea(mem(rcx, rsi, 2), rcx)
		vunpcklpd(xmm7, xmm5, xmm0)
		vunpckhpd(xmm7, xmm5, xmm1)
		vmovlpd(xmm0, mem(rcx        ))
		vmovupd(xmm1, mem(rcx, rsi, 1))
		jmp(.DDONE)


		label(.DDONE)
		vzeroupper()

		end_asm(
			: // output operands (none)
			: // input operands
			[k_iter] "m" (k_iter),
			[k_left] "m" (k_left),
			[a]      "m" (a),
			[rs_a]   "m" (rs_a),
			[cs_a]   "m" (cs_a),
      		[ps_a8]  "m" (ps_a8),
			[b]      "m" (b),
			[rs_b]   "m" (rs_b),
			[cs_b]   "m" (cs_b),
			[alpha]  "m" (alpha),
      		[beta]   "m" (beta),
			[c]      "m" (c),
			[rs_c]   "m" (rs_c),
			[cs_c]   "m" (cs_c)/*,
			[a_next] "m" (a_next),
			[b_next] "m" (b_next)*/
			: // register clobber list
			"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
			"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
			"xmm0", "xmm1", "xmm2", "xmm3",
			"xmm4", "xmm5", "xmm6", "xmm7",
			"xmm8", "xmm9", "xmm10", "xmm11",
			"xmm12", "xmm13", "xmm14", "xmm15",
			"ymm0", "ymm2", "ymm3", "ymm5", "ymm7",
			"memory"
			)
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 12 and n_offset is 8(12x8)
(12x8)_U

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		   8 9 10 11 12 13 14 15

↑		12   - - - - x x x x
|		13   - - - - - x x x
m		14   - - - - - - x x
off		15   - - - - - - - x
24		16   - - - - - - - -
|		17   - - - - - - - -
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_12x8_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	uint64_t ps_a8   = bli_auxinfo_ps_a( data ) * sizeof( double );

	begin_asm()


	mov(var(a), r14)
	mov(var(rs_a), r8)
	mov(var(cs_a), r9)
	lea(mem(, r8, 8), r8)
	lea(mem(, r9, 8), r9)

	lea(mem(r8, r8, 2), r13)
	lea(mem(r8, r8, 4), r15)

	mov(var(rs_b), r10)
	lea(mem(, r10, 8), r10)

	mov(var(c), r12)
	mov(var(rs_c), rdi)
	lea(mem(, rdi, 8), rdi)

	vxorpd(ymm5,  ymm5,  ymm5)
	vmovapd( ymm5, ymm7)
	vmovapd( ymm5, ymm9)
	vmovapd( ymm5, ymm11)

	mov(var(b), rbx)
	mov(r14, rax)

	cmp(imm(8), rdi)
	jz(.DCOLPFETCH)
	label(.DROWPFETCH)

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)
	label(.DCOLPFETCH)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(r12, rsi, 2), rdx)
	lea(mem(rdx, rsi, 1), rdx)
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)

	mov(var(ps_a8), rdx)
	lea(mem(rax, rdx, 1), rdx)
	lea(mem(r9, r9, 2), rcx)

	mov(var(k_iter), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	//compute ymm5, 7, 9, 11 only
	label(.DLOOPKITER)                 // MAIN LOOP
	//0
	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)
	//1
	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)
	//2

	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)
	//3
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)

	dec(rsi)
	jne(.DLOOPKITER)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)
	test(rsi, rsi)
	je(.DPOSTACCUM)

	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)

	dec(rsi)
	jne(.DLOOPKLEFT)

	label(.DPOSTACCUM)



	mov(r12, rcx)
	mov(var(alpha), rax)
	mov(var(beta), rbx)
	vbroadcastsd(mem(rax), ymm0)
	vbroadcastsd(mem(rbx), ymm3)

	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm11, ymm11)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(rcx, rdi, 4), rdx)         // c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // 3*cs_c;


	vxorpd(ymm0, ymm0, ymm0)
	vucomisd(xmm0, xmm3)
	je(.DBETAZERO)

	cmp(imm(8), rdi)
	jz(.DCOLSTORED)

	label(.DROWSTORED)
	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovhpd(xmm7, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm7, xmm7)
	vmovupd(xmm7, mem(rcx, 1*32+2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vextractf128(imm(0x1), ymm9, xmm9)
	vmovupd(xmm9, mem(rcx, 1*32+2*8))
	add(rdi, rcx)

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vextractf128(imm(0x1), ymm11, xmm11)
	vmovhpd(xmm11, mem(rcx, 1*32+3*8))


	jmp(.DDONE)


	label(.DCOLSTORED)

	lea(mem(rdx, rsi, 4), rdx)
	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovlpd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	vmovupd(xmm9, mem(rcx, rsi, 2))
	vextractf128(imm(0x1), ymm9, xmm9)
	vmovlpd(xmm9, mem(rcx, rsi, 2, 1*16))
	vmovupd(ymm11, mem(rcx, rax, 1))

	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)
	jz(.DCOLSTORBZ)

	label(.DROWSTORBZ)

	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovhpd(xmm7, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm7, xmm7)
	vmovupd(xmm7, mem(rcx, 1*32+2*8))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm9, xmm9)
	vmovupd(xmm9, mem(rcx, 1*32+2*8))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm11, xmm11)
	vmovhpd(xmm11, mem(rcx, 1*32+3*8))

	jmp(.DDONE)

	label(.DCOLSTORBZ)

	lea(mem(rdx, rsi, 4), rdx)
	lea(mem(rcx, rsi, 4), rcx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovlpd(xmm5, mem(rcx        ))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	vmovupd(xmm9, mem(rcx, rsi, 2))
	vextractf128(imm(0x1), ymm9, xmm9)
	vmovlpd(xmm9, mem(rcx, rsi, 2, 2*8))
	vmovupd(ymm11, mem(rcx, rax, 1))

	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm5",
	  "ymm7", "ymm9", "ymm11",
	  "memory"
	)
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 18 and n_offset is 16(18x16)
(18x16)_U


the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		  16 17 18 19 20 21 22 23

↑		18   - - x x x x x x
|		19   - - - x x x x x
m		20   - - - - x x x x
off		21   - - - - - x x x
24		22   - - - - - - x x
|		23   - - - - - - - x
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_18x16_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	uint64_t ps_a8   = bli_auxinfo_ps_a( data ) * sizeof( double );

	begin_asm()


	mov(var(a), r14)
	mov(var(rs_a), r8)
	mov(var(cs_a), r9)
	lea(mem(, r8, 8), r8)
	lea(mem(, r9, 8), r9)

	lea(mem(r8, r8, 2), r13)
	lea(mem(r8, r8, 4), r15)

	mov(var(rs_b), r10)
	lea(mem(, r10, 8), r10)

	mov(var(c), r12)
	mov(var(rs_c), rdi)
	lea(mem(, rdi, 8), rdi)

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)
	mov(r14, rax)



	PREFETCH_C()

	label(.DPOSTPFETCH)

	mov(var(ps_a8), rdx)
	lea(mem(rax, rdx, 1), rdx)
	lea(mem(r9, r9, 2), rcx)

	mov(var(k_iter), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	//skip ymm8, 10, 12, 14
	label(.DLOOPKITER)                 // MAIN LOOP
	//0
	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//1
	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//2
	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	//3
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKITER)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)
	test(rsi, rsi)
	je(.DPOSTACCUM)

	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	dec(rsi)
	jne(.DLOOPKLEFT)

	label(.DPOSTACCUM)



	mov(r12, rcx)
	mov(var(alpha), rax)
	mov(var(beta), rbx)
	vbroadcastsd(mem(rax), ymm0)
	vbroadcastsd(mem(rbx), ymm3)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm15, ymm15)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(rcx, rdi, 4), rdx)         // c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // 3*cs_c;


	vxorpd(ymm0, ymm0, ymm0)
	vucomisd(xmm0, xmm3)
	je(.DBETAZERO)

	cmp(imm(8), rdi)
	jz(.DCOLSTORED)

	label(.DROWSTORED)

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vextractf128(imm(0x1), ymm4, xmm4)
	vmovupd(xmm4, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vextractf128(imm(0x1), ymm6, xmm6)
	vmovhpd(xmm6, mem(rcx, 0*32+3*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovhpd(xmm11, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm11, xmm11)
	vmovupd(xmm11, mem(rcx, 1*32+2*8))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vextractf128(imm(0x1), ymm13, xmm13)
	vmovupd(xmm13, mem(rcx, 1*32+2*8))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vextractf128(imm(0x1), ymm15, xmm15)
	vmovhpd(xmm15, mem(rcx, 1*32+3*8))
	//add(rdi, rcx)


	jmp(.DDONE)


	label(.DCOLSTORED)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovlpd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(xmm5, mem(rcx        ))
	vextractf128(imm(0x1), ymm5, xmm5)
	vmovlpd(xmm5, mem(rcx, 2*8   ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovlpd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)
	jz(.DCOLSTORBZ)

	label(.DROWSTORBZ)

	vextractf128(imm(0x1), ymm4, xmm4)
	vmovupd(xmm4, mem(rcx, 0*32+2*8))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm6, xmm6)
	vmovhpd(xmm6, mem(rcx, 0*32+3*8))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovhpd(xmm11, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm11, xmm11)
	vmovupd(xmm11, mem(rcx, 1*32+2*8))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm13, xmm13)
	vmovupd(xmm13, mem(rcx, 1*32+2*8))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm15, xmm15)
	vmovhpd(xmm15, mem(rcx, 1*32+3*8))

	jmp(.DDONE)

	label(.DCOLSTORBZ)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovlpd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(xmm5, mem(rcx        ))
	vextractf128(imm(0x1), ymm5, xmm5)
	vmovlpd(xmm5, mem(rcx, 2*8   ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovlpd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))


	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm13", "ymm15",
	  "memory"
	)
}

/*

Following kernel computes the 6x8 block for the Upper vairant(U) of gemmt where
m_offset in 24x24 block is 0, n_offset is 0(0x0) and m_offset is 6, n_offset is 0 (6x0)
(0x0)+(6x0)_L

the region marked with 'x' is computed by following kernel
the region marked with '-' is not computed

	     	<-- n_off_24 -- >
 		     0 1 2 3 4 5 6 7

↑		0    x x x x x x x x
|		1    - x x x x x x x
m		2    - - x x x x x x
off		3    - - - x x x x x
24		4    - - - - x x x x
|		5    - - - - - x x x
↓
↑		6    - - - - - - x x
|		7    - - - - - - - x
m		8    - - - - - - - -
off		9    - - - - - - - -
24		10   - - - - - - - -
|		11   - - - - - - - -
↓


*/
void bli_dgemmsup_rv_haswell_asm_6x8m_0x0_combined_U
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*     restrict alpha,
       double*     restrict a, inc_t rs_a0, inc_t cs_a0,
       double*     restrict b, inc_t rs_b0, inc_t cs_b0,
       double*     restrict beta,
       double*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	uint64_t ps_a8   = bli_auxinfo_ps_a( data ) * sizeof( double );

	begin_asm()


	mov(var(a), r14)
	mov(var(rs_a), r8)
	mov(var(cs_a), r9)
	lea(mem(, r8, 8), r8)
	lea(mem(, r9, 8), r9)

	lea(mem(r8, r8, 2), r13)
	lea(mem(r8, r8, 4), r15)

	mov(var(rs_b), r10)
	lea(mem(, r10, 8), r10)

	mov(var(c), r12)
	mov(var(rs_c), rdi)
	lea(mem(, rdi, 8), rdi)

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)

	mov(var(b), rbx)
	mov(r14, rax)



	cmp(imm(8), rdi)
	jz(.DCOLPFETCH)
	label(.DROWPFETCH)

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         7*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)
	label(.DCOLPFETCH)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(r12, rsi, 2), rdx)
	lea(mem(rdx, rsi, 1), rdx)
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)

	mov(var(ps_a8), rdx)
	lea(mem(rax, rdx, 1), rdx)
	lea(mem(r9, r9, 2), rcx)

	mov(var(k_iter), rsi)
	test(rsi, rsi)
	je(.DCONSIDKLEFT)

	//ymm12 and ymm14 are used for 0x6 block
	label(.DLOOPKITER)                 // MAIN LOOP
	//0
	prefetch(0, mem(rdx, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovupd(mem(rbx,  1*64), ymm0)
	add(r10, rbx)                      // b += rs_b;
	lea(mem(rax, r13, 2), r11)
	vbroadcastsd(mem(r11       ), ymm2)
	vbroadcastsd(mem(r11, r8, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm14)

	add(r9, rax)                       // a += cs_a;



	//1
	prefetch(0, mem(rdx, r9, 1, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovupd(mem(rbx,  1*64), ymm0)
	add(r10, rbx)                      // b += rs_b;
	lea(mem(rax, r13, 2), r11)
	vbroadcastsd(mem(r11       ), ymm2)
	vbroadcastsd(mem(r11, r8, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm14)

	add(r9, rax)                       // a += cs_a;


	//2

	prefetch(0, mem(rdx, r9, 2, 5*8))

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovupd(mem(rbx,  1*64), ymm0)
	add(r10, rbx)                      // b += rs_b;
	lea(mem(rax, r13, 2), r11)
	vbroadcastsd(mem(r11       ), ymm2)
	vbroadcastsd(mem(r11, r8, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm14)
	add(r9, rax)                       // a += cs_a;


	//3
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovupd(mem(rbx,  1*64), ymm0)
	add(r10, rbx)                      // b += rs_b;
	lea(mem(rax, r13, 2), r11)
	vbroadcastsd(mem(r11       ), ymm2)
	vbroadcastsd(mem(r11, r8, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm14)
	add(r9, rax)                       // a += cs_a;


	dec(rsi)
	jne(.DLOOPKITER)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)
	test(rsi, rsi)
	je(.DPOSTACCUM)

	label(.DLOOPKLEFT)                 // EDGE LOOP

	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovupd(mem(rbx,  1*64), ymm0)
	add(r10, rbx)                      // b += rs_b;
	lea(mem(rax, r13, 2), r11)
	vbroadcastsd(mem(r11       ), ymm2)
	vbroadcastsd(mem(r11, r8, 1), ymm3)
	vfmadd231pd(ymm1, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm14)
	add(r9, rax)                       // a += cs_a;


	dec(rsi)
	jne(.DLOOPKLEFT)

	label(.DPOSTACCUM)



	mov(r12, rcx)
	mov(var(alpha), rax)
	mov(var(beta), rbx)
	vbroadcastsd(mem(rax), ymm0)
	vbroadcastsd(mem(rbx), ymm3)

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)

	mov(var(cs_c), rsi)
	lea(mem(, rsi, 8), rsi)
	lea(mem(rcx, rdi, 4), rdx)         // c +  4*rs_c;
	lea(mem(rsi, rsi, 2), rax)         // 3*cs_c;


	vxorpd(ymm0, ymm0, ymm0)
	vucomisd(xmm0, xmm3)
	je(.DBETAZERO)

	cmp(imm(8), rdi)
	jz(.DCOLSTORED)

	label(.DROWSTORED)

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovhpd(xmm6, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm6, xmm6)
	vmovupd(xmm6, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm8)
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovupd(xmm8, mem(rcx, 0*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm10)
	vextractf128(imm(0x1), ymm10, xmm10)
	vmovhpd(xmm10, mem(rcx, 0*32+3*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, rdi, 2, 1*32), ymm3, ymm12)
	vextractf128(imm(0x1), ymm12, xmm12)
	vmovupd(xmm12, mem(rcx, rdi, 2, 1*32+2*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, rdi, 2, 1*32), ymm3, ymm14)
	vextractf128(imm(0x1), ymm14, xmm14)
	vmovhpd(xmm14, mem(rcx, rdi, 2, 1*32+3*8))

	vfmadd231pd(mem(rcx, 1*32), ymm3, ymm15)
	vmovhpd(xmm15, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm15, xmm15)
	vmovupd(xmm15, mem(rcx, 1*32+2*8))
	//add(rdi, rcx)


	jmp(.DDONE)


	label(.DCOLSTORED)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovlpd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovlpd(xmm8, mem(rcx, rsi, 2, 1*16))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)
	lea(mem(rcx, 6*8), r11)
	lea(mem(r11, rsi, 2), r11)
	vfmadd231pd(mem(r11 ), xmm3, xmm2)
	vfmadd231pd(mem(r11, rsi, 1), xmm3, xmm4)
	vmovlpd(xmm2, mem(r11))
	vmovupd(xmm4, mem(r11, rsi, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovlpd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	jmp(.DDONE)                        // jump to end.

	label(.DBETAZERO)


	cmp(imm(8), rdi)
	jz(.DCOLSTORBZ)

	label(.DROWSTORBZ)

	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(ymm5, mem(rcx, 1*32))
	add(rdi, rcx)

	vmovhpd(xmm6, mem(rcx, 0*32+1*8))
	vextractf128(imm(0x1), ymm6, xmm6)
	vmovupd(xmm6, mem(rcx, 0*32+2*8))
	vmovupd(ymm7, mem(rcx, 1*32))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm8, xmm8)
	vmovupd(xmm8, mem(rcx, 0*32+2*8))
	vmovupd(ymm9, mem(rcx, 1*32))
	add(rdi, rcx)

	vextractf128(imm(0x1), ymm10, xmm10)
	vmovhpd(xmm10, mem(rcx, 0*32+3*8))
	vmovupd(ymm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vextractf128(imm(0x1), ymm12, xmm12)
	vmovupd(xmm12, mem(rcx, rdi, 2, 1*32+2*8))
	vmovupd(ymm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vextractf128(imm(0x1), ymm14, xmm14)
	vmovhpd(xmm14, mem(rcx, rdi, 2, 1*32+3*8))
	vmovhpd(xmm15, mem(rcx, 1*32+1*8))
	vextractf128(imm(0x1), ymm15, xmm15)
	vmovupd(xmm15, mem(rcx, 1*32+2*8))

	jmp(.DDONE)

	label(.DCOLSTORBZ)

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovlpd(xmm4, mem(rcx        ))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vextractf128(imm(0x1), ymm8, xmm8)
	vmovlpd(xmm8, mem(rcx, rsi, 2, 1*16))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)
	lea(mem(rcx, rdi, 4), r11)
	lea(mem(r11, rdi, 2), r11)
	lea(mem(r11, rsi, 2), r11)
	vmovlpd(xmm2, mem(r11))
	vmovupd(xmm4, mem(r11, rsi, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-7
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovlpd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))


	label(.DDONE)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)
}

static void bli_dgemmsup_rv_haswell_asm_6x7m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  -1 |  -1 | 0   |  ----> Mask vector( mask_3 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//
// Since we have 7 elements to load, kernel will use one normal load
// that loads 4 elements into vector register and for remainder 3 elements,
// kernel is using mask_3 which is set to -1, -1, -1, 0 so that the
// 3 elements will be loaded and 4th element will be set to 0 in destination vector.
//
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_3;

	if ( m_iter == 0 ) goto consider_edge_cases_7;

	// -------------------------------------------------------------------------
	begin_asm()

	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X7I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm3)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	lea(mem(rdx, rsi, 2), rcx)         // rcx = c + 5*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c
	prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 6*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.

	label(.DLOOPKITER)                 // MAIN LOOP
	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements based on mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements based on mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements based on mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 3 elements based on mask_3 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha

	vmulpd(ymm0, ymm5, ymm5)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)           // scale by alpha
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	                                   // now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm4)
	//Loads 4 element
	vmovupd(ymm3, mem(rcx, 0*32))
	//Loads 3 elements based on mask_3 mask vector
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm5)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm6)

	vmovupd(ymm5, mem(rcx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	//-----------------------2

	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm7)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm8)

	vmovupd(ymm7, mem(rbx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rbx, 1*32))

	add(rdi, rbx)
	//-----------------------3

	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm9)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm10)

	vmovupd(ymm9, mem(rbx, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(rbx, 1*32))

	//-----------------------4

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm11)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm12)

	vmovupd(ymm11, mem(rdx, 0*32))
	vmaskmovpd(ymm12, ymm15, mem(rdx, 1*32))

	add(rdi, rdx)
	//-----------------------5

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm13)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm14)

	vmovupd(ymm13, mem(rdx, 0*32))
	vmaskmovpd(ymm14, ymm15, mem(rdx, 1*32))

	//-----------------------6

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)
	C_TRANSPOSE_6x7_TILE(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)

	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1

	vmovupd(ymm5, mem(rcx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------2

	vmovupd(ymm7, mem(rcx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------3

	vmovupd(ymm9, mem(rcx, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------4

	vmovupd(ymm11, mem(rcx, 0*32))
	vmaskmovpd(ymm12, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------5

	vmovupd(ymm13, mem(rcx, 0*32))
	vmaskmovpd(ymm14, ymm15, mem(rcx, 1*32))

	//-----------------------6



	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	C_TRANSPOSE_6x7_TILE_BZ(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.RESETPARAM)
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask
	jmp(.DDONE)

	label(.DDONE)
	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X7I)                    // iterate again if ii != 0.


	label(.DRETURN)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[m_iter]   "m" (m_iter),
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[ps_a8]    "m" (ps_a8),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[mask_vec] "m" (mask_vec),
	[rs_c]     "m" (rs_c),
	[n0]       "m" (n0),
	[cs_c]     "m" (cs_c)
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm5", "ymm6", "ymm7", "ymm8", "ymm9",
	  "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
	  "ymm15",
	  "memory"
	)

	consider_edge_cases_7:
	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		dgemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x7,
			bli_dgemmsup_rv_haswell_asm_2x7,
			bli_dgemmsup_rv_haswell_asm_3x7,
			bli_dgemmsup_rv_haswell_asm_4x7,
			bli_dgemmsup_rv_haswell_asm_5x7
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

static void bli_dgemmsup_rv_haswell_asm_6x5m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5);

// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  0  |  0  | 0   |  ----> Mask vector( mask_1 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  0  |  0  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//
// Since we have 5 elements to load, kernel will use one normal load
// that loads 4 elements into vector register and for remainder 1 element,
// kernel is using mask_1 which is set to -1, 0, 0, 0 static that the
// 1 element will be loaded and other 3 elements will be set to 0 in destination vector.
//
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_1;

	if ( m_iter == 0 ) goto consider_edge_cases_5;

	// -------------------------------------------------------------------------
	begin_asm()

	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X5I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm3)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.

	label(.DLOOPKITER)                 // MAIN LOOP
	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.

	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif
	//Loads 4 element
	vmovupd(mem(rbx,  0*32), ymm0)
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm3)
	vfmadd231pd(ymm1, ymm2, ymm4)

	vbroadcastsd(mem(rax, r8,  1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm1, ymm2, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm7)
	vfmadd231pd(ymm1, ymm2, ymm8)

	vbroadcastsd(mem(rax, r13, 1), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm1, ymm2, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vfmadd231pd(ymm0, ymm2, ymm11)
	vfmadd231pd(ymm1, ymm2, ymm12)

	vbroadcastsd(mem(rax, r15, 1), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm1, ymm2, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm1)       // load beta and duplicate

	vmulpd(ymm0, ymm3, ymm3)           // scale by alpha
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha

	vmulpd(ymm0, ymm5, ymm5)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)

	vmulpd(ymm0, ymm7, ymm7)           // scale by alpha
	vmulpd(ymm0, ymm8, ymm8)

	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)

	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)

	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	                                   // now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm1)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm3)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm4)
	//Loads 4 element
	vmovupd(ymm3, mem(rcx, 0*32))
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1

	vfmadd231pd(mem(rcx, 0*32), ymm1, ymm5)
	vmaskmovpd(mem(rcx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm6)

	vmovupd(ymm5, mem(rcx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	//-----------------------2

	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm7)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm8)

	vmovupd(ymm7, mem(rbx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rbx, 1*32))

	add(rdi, rbx)
	//-----------------------3

	vfmadd231pd(mem(rbx, 0*32), ymm1, ymm9)
	vmaskmovpd(mem(rbx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm10)

	vmovupd(ymm9, mem(rbx, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(rbx, 1*32))

	//-----------------------4

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm11)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm12)

	vmovupd(ymm11, mem(rdx, 0*32))
	vmaskmovpd(ymm12, ymm15, mem(rdx, 1*32))

	add(rdi, rdx)
	//-----------------------5

	vfmadd231pd(mem(rdx, 0*32), ymm1, ymm13)
	vmaskmovpd(mem(rdx, 1*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm1, ymm14)

	vmovupd(ymm13, mem(rdx, 0*32))
	vmaskmovpd(ymm14, ymm15, mem(rdx, 1*32))

	//-----------------------6

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)

	C_TRANSPOSE_6x5_TILE(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)

	vmovupd(ymm3, mem(rcx, 0*32))
	vmaskmovpd(ymm4, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------1

	vmovupd(ymm5, mem(rcx, 0*32))
	vmaskmovpd(ymm6, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------2

	vmovupd(ymm7, mem(rcx, 0*32))
	vmaskmovpd(ymm8, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------3

	vmovupd(ymm9, mem(rcx, 0*32))
	vmaskmovpd(ymm10, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------4

	vmovupd(ymm11, mem(rcx, 0*32))
	vmaskmovpd(ymm12, ymm15, mem(rcx, 1*32))

	add(rdi, rcx)
	//-----------------------5

	vmovupd(ymm13, mem(rcx, 0*32))
	vmaskmovpd(ymm14, ymm15, mem(rcx, 1*32))

	//-----------------------6



	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	C_TRANSPOSE_6x5_TILE_BZ(3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14)
	jmp(.RESETPARAM)

	label(.RESETPARAM)
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)           //load mask
	jmp(.DDONE)

	label(.DDONE)
	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X5I)                    // iterate again if ii != 0.


	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter]   "m" (m_iter),
      [k_iter]   "m" (k_iter),
      [k_left]   "m" (k_left),
      [a]        "m" (a),
      [rs_a]     "m" (rs_a),
      [cs_a]     "m" (cs_a),
      [ps_a8]    "m" (ps_a8),
      [b]        "m" (b),
      [rs_b]     "m" (rs_b),
      [cs_b]     "m" (cs_b),
      [alpha]    "m" (alpha),
      [beta]     "m" (beta),
      [c]        "m" (c),
      [mask_vec] "m" (mask_vec),
      [rs_c]     "m" (rs_c),
      [n0]       "m" (n0),
      [cs_c]     "m" (cs_c)/*,
      [a_next]   "m" (a_next),
      [b_next]   "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases_5:
	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		dgemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x5,
			bli_dgemmsup_rv_haswell_asm_2x5,
			bli_dgemmsup_rv_haswell_asm_3x5,
			bli_dgemmsup_rv_haswell_asm_4x5,
			bli_dgemmsup_rv_haswell_asm_5x5
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5);
}

static void bli_dgemmsup_rv_haswell_asm_6x3m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  -1 |  -1 | 0   |  ----> Mask vector( mask_3 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//

// kernel is using mask_3 which is set to -1, -1, -1, 0 so that the
// 3 elements will be loaded and 4th element will be set to 0 in destination vector.
//
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_3;

	if ( m_iter == 0 ) goto consider_edge_cases_nleft_3;

	begin_asm()
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b

	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X3I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm14)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         2*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 2*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 2*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         2*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 2*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 2*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	// use rcx, rdx for prefetching lines
	// from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
									// contains the k_left loop.

	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                           // else, we prepare to enter k_left loop.

	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif
	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	// now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  4*rs_c;

	//Loads 3 elements as per mask_3 mask vector
	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))

	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vmaskmovpd(ymm8, ymm15, mem(rbx, 0*32))
	add(rdi, rbx)

	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vmaskmovpd(ymm10, ymm15, mem(rbx, 0*32))

	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vmaskmovpd(ymm12, ymm15, mem(rdx, 0*32))
	add(rdi, rdx)

	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vmaskmovpd(ymm14, ymm15, mem(rdx, 0*32))

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)

	C_TRANSPOSE_6x3_TILE(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)

	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm8, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm10, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm12, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm14, ymm15, mem(rcx, 0*32))


	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORBZ)

	C_TRANSPOSE_6x3_TILE_BZ(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DDONE)
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X3I)                    // iterate again if ii != 0.

	label(.DRETURN)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[m_iter]   "m" (m_iter),
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[ps_a8]    "m" (ps_a8),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[rs_c]     "m" (rs_c),
	[n0]       "m" (n0),
	[mask_vec] "m" (mask_vec),
	[cs_c]     "m" (cs_c)/*,
	[a_next]   "m" (a_next),
	[b_next]   "m" (b_next)*/
	: // register clobber list
	"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	"xmm0", "xmm1", "xmm2", "xmm3",
	"xmm4", "xmm5", "xmm6", "xmm7",
	"xmm8", "xmm9", "xmm10", "xmm11",
	"xmm12", "xmm13", "xmm14", "xmm15",
	"ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	"ymm6", "ymm8", "ymm10", "ymm12", "ymm14",
	"ymm15", "memory"
	)

	consider_edge_cases_nleft_3:
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		dgemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x3,
			bli_dgemmsup_rv_haswell_asm_2x3,
			bli_dgemmsup_rv_haswell_asm_3x3,
			bli_dgemmsup_rv_haswell_asm_4x3,
			bli_dgemmsup_rv_haswell_asm_5x3
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
			conja, conjb, m_left, nr_cur, k0,
			alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}


static void bli_dgemmsup_rv_haswell_asm_6x1m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
// Sets up the mask for loading relevant remainder elements in load direction
// int64_t array of size 4 represents the mask for 4 elements of AVX2 vector register.
//
// Low end           High end
//  ________________________
// |     |     |     |     |
// | 1   |  2  |  3  | 4   |  ----> Source vector
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | -1  |  0   |  0  | 0   |  ----> Mask vector( mask_1 )
// |_____|_____|_____|_____|
//
//  ________________________
// |     |     |     |     |
// | 1   |  0  |  0  | 0   |  ----> Destination vector
// |_____|_____|_____|_____|
//

// kernel is using mask_1 which is set to -1, 0, 0, 0 so that the
// 1 element will be loaded and 4th element will be set to 0 in destination vector.
//
	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	int64_t const *mask_vec = mask_1;

	if ( m_iter == 0 ) goto consider_edge_cases_nleft_1;

	begin_asm()
	mov(var(mask_vec), rdx)
	vmovdqu(mem(rdx), ymm15)       //load mask

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(rs_b), r10)                // load rs_b

	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X1I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]

	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm1)

	mov(var(b), rbx)                   // load address of b.
	mov(r14, rax)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         2*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 2*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 2*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         2*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 2*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 2*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	// use rcx, rdx for prefetching lines
	// from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif

	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
									// contains the k_left loop.

	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif
	//Loads 1 elements as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm1)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm5, ymm4, ymm4)
	vaddpd(ymm7, ymm6, ymm6)
	vaddpd(ymm9, ymm8, ymm8)
	vaddpd(ymm11, ymm10, ymm10)
	vaddpd(ymm13, ymm12, ymm12)
	vaddpd(ymm1, ymm14, ymm14)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
                                           // else, we prepare to enter k_left loop.

	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif
	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.

	label(.DPOSTACCUM)

	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)


	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;

	// now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case

	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	//Loads 1 element as per mask_1 mask vector
	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm0)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(mem(rcx, 0*32), ymm15, ymm1)
	vfmadd231pd(ymm1, ymm3, ymm6)
	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))

	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm2)
	vfmadd231pd(ymm2, ymm3, ymm8)
	vmaskmovpd(ymm8, ymm15, mem(rbx, 0*32))
	add(rdi, rbx)

	vmaskmovpd(mem(rbx, 0*32), ymm15, ymm4)
	vfmadd231pd(ymm4, ymm3, ymm10)
	vmaskmovpd(ymm10, ymm15, mem(rbx, 0*32))

	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm5)
	vfmadd231pd(ymm5, ymm3, ymm12)
	vmaskmovpd(ymm12, ymm15, mem(rdx, 0*32))
	add(rdi, rdx)

	vmaskmovpd(mem(rdx, 0*32), ymm15, ymm6)
	vfmadd231pd(ymm6, ymm3, ymm14)
	vmaskmovpd(ymm14, ymm15, mem(rdx, 0*32))

	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORED)

	C_TRANSPOSE_6x1_TILE(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DBETAZERO)

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case

	label(.DROWSTORBZ)

	vmaskmovpd(ymm4, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm6, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm8, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm10, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm12, ymm15, mem(rcx, 0*32))
	add(rdi, rcx)

	vmaskmovpd(ymm14, ymm15, mem(rcx, 0*32))


	jmp(.DDONE)                        // jump to end.

	label(.DCOLSTORBZ)

	C_TRANSPOSE_6x1_TILE_BZ(4, 6, 8, 10, 12, 14)
	jmp(.DDONE)

	label(.DDONE)
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X1I)                    // iterate again if ii != 0.

	label(.DRETURN)
	vzeroupper()

	end_asm(
	: // output operands (none)
	: // input operands
	[m_iter]   "m" (m_iter),
	[k_iter]   "m" (k_iter),
	[k_left]   "m" (k_left),
	[a]        "m" (a),
	[rs_a]     "m" (rs_a),
	[cs_a]     "m" (cs_a),
	[ps_a8]    "m" (ps_a8),
	[b]        "m" (b),
	[rs_b]     "m" (rs_b),
	[cs_b]     "m" (cs_b),
	[alpha]    "m" (alpha),
	[beta]     "m" (beta),
	[c]        "m" (c),
	[rs_c]     "m" (rs_c),
	[n0]       "m" (n0),
	[mask_vec] "m" (mask_vec),
	[cs_c]     "m" (cs_c)/*,
	[a_next]   "m" (a_next),
	[b_next]   "m" (b_next)*/
	: // register clobber list
	"rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	"r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	"xmm0", "xmm1", "xmm2", "xmm3",
	"xmm4", "xmm5", "xmm6", "xmm7",
	"xmm8", "xmm9", "xmm10", "xmm11",
	"xmm12", "xmm13", "xmm14", "xmm15",
	"ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	"ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	"ymm12", "ymm13", "ymm14", "ymm15", "memory"
	)

	consider_edge_cases_nleft_1:
	if ( m_left )
	{
		const dim_t      nr_cur = n0;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

		dgemmsup_ker_ft ker_fps[6] =
		{
			NULL,
			bli_dgemmsup_rv_haswell_asm_1x1,
			bli_dgemmsup_rv_haswell_asm_2x1,
			bli_dgemmsup_rv_haswell_asm_3x1,
			bli_dgemmsup_rv_haswell_asm_4x1,
			bli_dgemmsup_rv_haswell_asm_5x1
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
			conja, conjb, m_left, nr_cur, k0,
			alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
}

void bli_dgemmsup_rv_haswell_asm_6x6m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X8I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm1,  ymm1,  ymm1)         // zero ymm1 since we only use the lower
	vxorpd(ymm4,  ymm4,  ymm4)         // half (xmm1), and nans/infs may slow us
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))

	vfmadd231pd(mem(rcx, 1*32), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, 1*32))


	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rbx, 0*32))

	vfmadd231pd(mem(rbx, 1*32), xmm3, xmm9)
	vmovupd(xmm9, mem(rbx, 1*32))
	add(rdi, rbx)


	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rbx, 0*32))

	vfmadd231pd(mem(rbx, 1*32), xmm3, xmm11)
	vmovupd(xmm11, mem(rbx, 1*32))


	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rdx, 0*32))

	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm13)
	vmovupd(xmm13, mem(rdx, 1*32))
	add(rdi, rdx)


	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rdx, 0*32))

	vfmadd231pd(mem(rdx, 1*32), xmm3, xmm15)
	vmovupd(xmm15, mem(rdx, 1*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	vmovupd(xmm5, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	vmovupd(xmm7, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	vmovupd(xmm9, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	vmovupd(xmm11, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	vmovupd(xmm13, mem(rcx, 1*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	vmovupd(xmm15, mem(rcx, 1*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)

	                                   // begin I/O on columns 4-5
	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)

	vmovupd(ymm5, mem(rcx        ))
	vmovupd(ymm7, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X8I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
	  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
	  "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 6;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x6;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x6;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x6;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x6,
		  bli_dgemmsup_rv_haswell_asm_2x6,
		  bli_dgemmsup_rv_haswell_asm_3x6,
		  bli_dgemmsup_rv_haswell_asm_4x6,
		  bli_dgemmsup_rv_haswell_asm_5x6
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}


void bli_dgemmsup_rv_haswell_asm_6x4m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         3*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx,         5*8)) // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm5, ymm4, ymm4)
	vaddpd(ymm7, ymm6, ymm6)
	vaddpd(ymm9, ymm8, ymm8)
	vaddpd(ymm11, ymm10, ymm10)
	vaddpd(ymm13, ymm12, ymm12)
	vaddpd(ymm15, ymm14, ymm14)



	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx, 0*32))


	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm8)
	vmovupd(ymm8, mem(rbx, 0*32))
	add(rdi, rbx)


	vfmadd231pd(mem(rbx, 0*32), ymm3, ymm10)
	vmovupd(ymm10, mem(rbx, 0*32))


	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm12)
	vmovupd(ymm12, mem(rdx, 0*32))
	add(rdi, rdx)


	vfmadd231pd(mem(rdx, 0*32), ymm3, ymm14)
	vmovupd(ymm14, mem(rdx, 0*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(ymm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm12, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(ymm14, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X4I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14",  "ymm15", "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 4;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x4;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x4;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x4;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x4,
		  bli_dgemmsup_rv_haswell_asm_2x4,
		  bli_dgemmsup_rv_haswell_asm_3x4,
		  bli_dgemmsup_rv_haswell_asm_4x4,
		  bli_dgemmsup_rv_haswell_asm_5x4
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_6x2m
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.

	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X2I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(xmm4,  xmm4,  xmm4)
	vmovapd( ymm4, ymm5)
	vmovapd( ymm4, ymm6)
	vmovapd( ymm4, ymm7)
	vmovapd( ymm4, ymm8)
	vmovapd( ymm4, ymm9)
	vmovapd( ymm4, ymm10)
	vmovapd( ymm4, ymm11)
	vmovapd( ymm4, ymm12)
	vmovapd( ymm4, ymm13)
	vmovapd( ymm4, ymm14)
	vmovapd( ymm4, ymm15)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12,         1*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 1*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 1*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx,         1*8)) // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 1*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 1*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	prefetch(0, mem(r12,         5*8)) // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	mov(var(ps_a8), rdx)               // load ps_a8
	lea(mem(rax, rdx, 1), rdx)         // rdx = a + ps_a8
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
	                                   // use rcx, rdx for prefetching lines
	                                   // from next upanel of a.
#else
	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;
#endif




	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.


	label(.DLOOPKITER)                 // MAIN LOOP


	// ---------------------------------- iteration 0

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)


	// ---------------------------------- iteration 2

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	// ---------------------------------- iteration 3

#if 0
	prefetch(0, mem(rdx, 5*8))
#else
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm5)
	vfmadd231pd(xmm0, xmm3, xmm7)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm9)
	vfmadd231pd(xmm0, xmm3, xmm11)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm13)
	vfmadd231pd(xmm0, xmm3, xmm15)



	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.

	vaddpd(ymm5, ymm4, ymm4)
	vaddpd(ymm7, ymm6, ymm6)
	vaddpd(ymm9, ymm8, ymm8)
	vaddpd(ymm11, ymm10, ymm10)
	vaddpd(ymm13, ymm12, ymm12)
	vaddpd(ymm15, ymm14, ymm14)

	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT)                 // EDGE LOOP

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)
#endif

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)


	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.



	label(.DPOSTACCUM)



	mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)






	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	//lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;



	                                   // now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)

	lea(mem(rcx, rdi, 2), rbx)         // load address of c +  2*rs_c;

	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx, 0*32), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx, 0*32))


	vfmadd231pd(mem(rbx, 0*32), xmm3, xmm8)
	vmovupd(xmm8, mem(rbx, 0*32))
	add(rdi, rbx)


	vfmadd231pd(mem(rbx, 0*32), xmm3, xmm10)
	vmovupd(xmm10, mem(rbx, 0*32))


	vfmadd231pd(mem(rdx, 0*32), xmm3, xmm12)
	vmovupd(xmm12, mem(rdx, 0*32))
	add(rdi, rdx)


	vfmadd231pd(mem(rdx, 0*32), xmm3, xmm14)
	vmovupd(xmm14, mem(rdx, 0*32))


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)

	                                   // begin I/O on columns 0-3
	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx        ), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vfmadd231pd(mem(rdx        ), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.




	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case



	label(.DROWSTORBZ)


	vmovupd(xmm4, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm6, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm8, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm10, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm12, mem(rcx, 0*32))
	add(rdi, rcx)


	vmovupd(xmm14, mem(rcx, 0*32))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)

	                                   // begin I/O on columns 0-3
	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vmovupd(ymm4, mem(rcx        ))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vmovupd(xmm0, mem(rdx        ))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)




	label(.DDONE)




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X2I)                    // iterate again if ii != 0.




	label(.DRETURN)
	vzeroupper()

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4",
	  "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11",
	  "ymm12", "ymm13", "ymm14", "ymm15", "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 2;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x2;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x2;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x2;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x2,
		  bli_dgemmsup_rv_haswell_asm_2x2,
		  bli_dgemmsup_rv_haswell_asm_3x2,
		  bli_dgemmsup_rv_haswell_asm_4x2,
		  bli_dgemmsup_rv_haswell_asm_5x2
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
	}
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}
