/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

#define INIT_REG \
    vxorps( zmm0,zmm0,zmm0 ) \
    vxorps( zmm1,zmm1,zmm1 ) \
    vxorps( zmm2,zmm2,zmm2 ) \
    vxorps( zmm3,zmm3,zmm3 ) \
    vxorps( zmm4,zmm4,zmm4 ) \
    vxorps( zmm5,zmm5,zmm5 ) \
    vxorps( zmm6,zmm6,zmm6 ) \
    vxorps( zmm7,zmm7,zmm7 ) \
    vxorps( zmm8,zmm8,zmm8 ) \
    vxorps( zmm9,zmm9,zmm9 ) \
    vxorps( zmm10,zmm10,zmm10 ) \
    vxorps( zmm11,zmm11,zmm11 ) \
    vxorps( zmm12,zmm12,zmm12 ) \
    vxorps( zmm13,zmm13,zmm13 ) \
    vxorps( zmm14,zmm14,zmm14 ) \
    vxorps( zmm15,zmm15,zmm15 ) \
    vxorps( zmm16,zmm16,zmm16 ) \
    vxorps( zmm17,zmm17,zmm17 ) \
    vxorps( zmm18,zmm18,zmm18 ) \
    vxorps( zmm19,zmm19,zmm19 ) \
    vxorps( zmm20,zmm20,zmm20 ) \
    vxorps( zmm21,zmm21,zmm21 ) \
    vxorps( zmm22,zmm22,zmm22 ) \
    vxorps( zmm23,zmm23,zmm23 ) \
    vxorps( zmm24,zmm24,zmm24 ) \
    vxorps( zmm25,zmm25,zmm25 ) \
    vxorps( zmm26,zmm26,zmm26 ) \
    vxorps( zmm27,zmm27,zmm27 ) \
    vxorps( zmm28,zmm28,zmm28 ) \
    vxorps( zmm29,zmm29,zmm29 ) \
    vxorps( zmm30,zmm30,zmm30 ) \
    vxorps( zmm31,zmm31,zmm31 )

#define VFMA6( R0, R1, R2, R3, R4, R5 ) \
    vfmadd231ps( zmm0, zmm6, zmm(R0) ) \
    vfmadd231ps( zmm1, zmm6, zmm(R1) ) \
    vfmadd231ps( zmm2, zmm6, zmm(R2) ) \
    vfmadd231ps( zmm3, zmm6, zmm(R3) ) \
    vfmadd231ps( zmm4, zmm6, zmm(R4) ) \
    vfmadd231ps( zmm5, zmm6, zmm(R5) )

#define VFMA5( R0, R1, R2, R3, R4 ) \
    vfmadd231ps( zmm0, zmm6, zmm(R0) ) \
    vfmadd231ps( zmm1, zmm6, zmm(R1) ) \
    vfmadd231ps( zmm2, zmm6, zmm(R2) ) \
    vfmadd231ps( zmm3, zmm6, zmm(R3) ) \
    vfmadd231ps( zmm4, zmm6, zmm(R4) )

#define VFMA4( R0, R1, R2, R3 ) \
    vfmadd231ps( zmm0, zmm6, zmm(R0) ) \
    vfmadd231ps( zmm1, zmm6, zmm(R1) ) \
    vfmadd231ps( zmm2, zmm6, zmm(R2) ) \
    vfmadd231ps( zmm3, zmm6, zmm(R3) )

#define VFMA3( R0, R1, R2 ) \
    vfmadd231ps( zmm0, zmm6, zmm(R0) ) \
    vfmadd231ps( zmm1, zmm6, zmm(R1) ) \
    vfmadd231ps( zmm2, zmm6, zmm(R2) )

#define VFMA2( R0, R1 ) \
    vfmadd231ps( zmm0, zmm6, zmm(R0) ) \
    vfmadd231ps( zmm1, zmm6, zmm(R1) )

#define VFMA1( R0 ) \
    vfmadd231ps( zmm0, zmm6, zmm(R0) )

#define ZMM_TO_YMM( R0, R1, R2, R3, R4, R5, R6, R7 ) \
    VEXTRACTF32X8( imm(0x01), zmm(R0), ymm0 ) \
    VADDPS( ymm0, ymm(R0), ymm(R4) ) \
    VEXTRACTF32X8( imm(0x01), zmm(R1), ymm1 ) \
    VADDPS( ymm1, ymm(R1), ymm(R5) ) \
    VEXTRACTF32X8( imm(0x01), zmm(R2), ymm2 ) \
    VADDPS( ymm2, ymm(R2), ymm(R6) ) \
    VEXTRACTF32X8( imm(0x01), zmm(R3), ymm3 ) \
    VADDPS( ymm3, ymm(R3), ymm(R7) )

#define ACCUM_YMM( R0, R1, R2, R3, R4 ) \
    vhaddps( ymm(R1), ymm(R0), ymm0 ) \
    vextractf128( imm(0x01), ymm0, xmm1 ) \
    vaddps( xmm0, xmm1, xmm0 ) \
    vhaddps( ymm(R3), ymm(R2), ymm2 ) \
    vextractf128( imm(0x01), ymm2, xmm1 ) \
    vaddps( xmm2, xmm1, xmm2 ) \
    vhaddps( xmm2, xmm0, xmm(R4) )

#define ALPHA_SCALE \
    mov( var(alpha), rax ) \
    vbroadcastss( (rax), xmm0 ) \
    vmulps( xmm0, xmm4, xmm4 ) \
    vmulps( xmm0, xmm5, xmm5 ) \
    vmulps( xmm0, xmm6, xmm6 )

#define ALPHA_SCALE2 \
    mov( var(alpha), rax ) \
    vbroadcastss( (rax), xmm0 ) \
    vmulps( xmm0, xmm4, xmm4 ) \
    vmulps( xmm0, xmm5, xmm5 )

#define ALPHA_SCALE1 \
    mov( var(alpha), rax ) \
    vbroadcastss( (rax), xmm0 ) \
    vmulps( xmm0, xmm4, xmm4 )

#define C_STOR \
    mov( var( rs_c ), rdi ) \
	lea( mem( , rdi, 4 ), rdi ) \
    mov( var(beta), rax ) \
    vbroadcastss( (rax), xmm0 ) \
    vfmadd231ps( (rcx), xmm0, xmm4 ) \
    vmovups( xmm4, (rcx) ) \
    add( rdi, rcx ) \
    vbroadcastss( (rax), xmm0 ) \
    vfmadd231ps( (rcx), xmm0, xmm5 ) \
    vmovups( xmm5, (rcx) ) \
    add( rdi, rcx ) \
    vbroadcastss( (rax), xmm0 ) \
    vfmadd231ps( (rcx), xmm0, xmm6 ) \
    vmovups( xmm6, (rcx) ) \
    add( rdi, rcx )

#define C_STOR2 \
    mov( var( rs_c ), rdi ) \
	lea( mem( , rdi, 4 ), rdi ) \
    mov( var(beta), rax ) \
    vbroadcastss( (rax), xmm0 ) \
    vfmadd231ps( (rcx), xmm0, xmm4 ) \
    vmovups( xmm4, (rcx) ) \
    add( rdi, rcx ) \
    vbroadcastss( (rax), xmm0 ) \
    vfmadd231ps( (rcx), xmm0, xmm5 ) \
    vmovups( xmm5, (rcx) ) \
    add( rdi, rcx )

#define C_STOR1 \
    mov( var( rs_c ), rdi ) \
	lea( mem( , rdi, 4 ), rdi ) \
    mov( var(beta), rax ) \
    vbroadcastss( (rax), xmm0 ) \
    vfmadd231ps( (rcx), xmm0, xmm4 ) \
    vmovups( xmm4, (rcx) ) \
    add( rdi, rcx )

#define C_STOR_BZ \
    mov( var( rs_c ), rdi ) \
	lea( mem( , rdi, 4 ), rdi ) \
    vmovups( xmm4, mem( rcx ) ) \
    add( rdi, rcx ) \
    vmovups( xmm5, mem( rcx ) ) \
    add( rdi, rcx ) \
    vmovups( xmm6, mem( rcx ) ) \
    add( rdi, rcx )

#define C_STOR_BZ2 \
    mov( var( rs_c ), rdi ) \
	lea( mem( , rdi, 4 ), rdi ) \
    vmovups( xmm4, mem( rcx ) ) \
    add( rdi, rcx ) \
    vmovups( xmm5, mem( rcx ) ) \
    add( rdi, rcx )

#define C_STOR_BZ1 \
    mov( var( rs_c ), rdi ) \
	lea( mem( , rdi, 4 ), rdi ) \
    vmovups( xmm4, mem( rcx ) ) \
    add( rdi, rcx )
