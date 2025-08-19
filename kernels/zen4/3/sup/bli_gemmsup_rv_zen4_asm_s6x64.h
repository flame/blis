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

/**
 * VFMA4 - performs 4 VFMAs for k-loop
 * zmm0-3 - contains 4 rows of B
 * R0 - register containing A broadcast
 * R1-4 - registers to store intermediate result
 */
#define VFMA4( R0, R1, R2, R3, R4) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) ) \
    vfmadd231ps( zmm1,zmm(R0),zmm(R2) ) \
    vfmadd231ps( zmm2,zmm(R0),zmm(R3) ) \
    vfmadd231ps( zmm3,zmm(R0),zmm(R4) )

#define VFMA3( R0, R1, R2, R3) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) ) \
    vfmadd231ps( zmm1,zmm(R0),zmm(R2) ) \
    vfmadd231ps( zmm2,zmm(R0),zmm(R3) )

#define VFMA2( R0, R1, R2 ) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) ) \
    vfmadd231ps( zmm1,zmm(R0),zmm(R2) )

#define VFMA1( R0, R1 ) \
    vfmadd231ps( zmm0,zmm(R0),zmm(R1) )

/**
 * ALPHA_SCALE4 - scales 4 ZMM registers by alpha
 * R0 - register having alpha
 * R1-4 - registers to be scaled
 */
#define ALPHA_SCALE4( R0, R1, R2, R3, R4 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) ) \
    vmulps( zmm(R0), zmm(R2), zmm(R2) ) \
    vmulps( zmm(R0), zmm(R3), zmm(R3) ) \
    vmulps( zmm(R0), zmm(R4), zmm(R4) )

#define ALPHA_SCALE3( R0, R1, R2, R3 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) ) \
    vmulps( zmm(R0), zmm(R2), zmm(R2) ) \
    vmulps( zmm(R0), zmm(R3), zmm(R3) )

#define ALPHA_SCALE2( R0, R1, R2 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) ) \
    vmulps( zmm(R0), zmm(R2), zmm(R2) )

#define ALPHA_SCALE1( R0, R1 ) \
    vmulps( zmm(R0), zmm(R1), zmm(R1) )

/**
 * UPDATE_C4 - loads 4 C rows, performs 4 VFMAs (scaling by beta), stores to buffer & increments C ptr
 * R0 -> register having beta
 * R1-4 -> registers having intermediate results ( alpha * A * B )
 */
#define UPDATE_C4( R0, R1, R2, R3, R4 ) \
    vmovups( (rcx), zmm1 ) \
    vfmadd231ps( zmm(R0),zmm1,zmm(R1) ) \
    vmovups( zmm(R1),(rcx) ) \
    vmovups( 0x40(rcx),zmm1 ) \
    vfmadd231ps( zmm(R0),zmm1,zmm(R2) ) \
    vmovups( zmm(R2),0x40(rcx) ) \
    vmovups( 0x80(rcx),zmm1 ) \
    vfmadd231ps( zmm(R0),zmm1,zmm(R3) ) \
    vmovups( zmm(R3),0x80(rcx) ) \
    vmovups( 0xc0(rcx),zmm1 ) \
    vfmadd231ps( zmm(R0),zmm1,zmm(R4) ) \
    vmovups( zmm(R4),0xc0(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C3( R0, R1, R2, R3 ) \
    vmovups( (rcx), zmm1 ) \
    vfmadd231ps( zmm(R0),zmm1,zmm(R1) ) \
    vmovups( zmm(R1),(rcx) ) \
    vmovups( 0x40(rcx),zmm1 ) \
    vfmadd231ps( zmm(R0),zmm1,zmm(R2) ) \
    vmovups( zmm(R2),0x40(rcx) ) \
    vmovups( 0x80(rcx),zmm1 ) \
    vfmadd231ps( zmm(R0),zmm1,zmm(R3) ) \
    vmovups( zmm(R3),0x80(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C2( R0, R1, R2 ) \
    vmovups( (rcx), zmm1 ) \
    vfmadd231ps( zmm(R0), zmm1, zmm(R1) ) \
    vmovups( zmm(R1), (rcx) ) \
    vmovups( 0x40(rcx), zmm1 ) \
    vfmadd231ps( zmm(R0), zmm1, zmm(R2) ) \
    vmovups( zmm(R2), 0x40(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C1( R0, R1 ) \
    vmovups( (rcx), zmm1 ) \
    vfmadd231ps( zmm(R0), zmm1, zmm(R1) ) \
    vmovups( zmm(R1), (rcx) ) \
    add( rdi, rcx )

/**
 * UPDATE_C4_BZ - stores result to buffer & increments C ptr
 * R0-3 -> registers having intermediate results ( alpha * A * B )
 */
#define UPDATE_C4_BZ( R0, R1, R2, R3 ) \
    vmovups( zmm(R0),(rcx) ) \
    vmovups( zmm(R1),0x40(rcx) ) \
    vmovups( zmm(R2),0x80(rcx) ) \
    vmovups( zmm(R3),0xc0(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C3_BZ( R0, R1, R2 ) \
    vmovups( zmm(R0),(rcx) ) \
    vmovups( zmm(R1),0x40(rcx) ) \
    vmovups( zmm(R2),0x80(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C2_BZ( R0, R1 ) \
    vmovups( zmm(R0),(rcx) ) \
    vmovups( zmm(R1),0x40(rcx) ) \
    add( rdi, rcx )

#define UPDATE_C1_BZ( R0 ) \
    vmovups( zmm(R0),(rcx) ) \
    add( rdi, rcx )

#define TRANSPOSE_4X16( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L( 0X44, R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L( 0XEE, R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H( 0X44, R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H( 0XEE, R0, R1, R2, R3 )

#define TRANSPOSE_4X16L( IMM, R0, R1, R2, R3 ) \
    vunpcklps( ZMM(R1), ZMM(R0), zmm6 ) \
    vunpcklps( ZMM(R3), ZMM(R2), zmm7 ) \
    VSHUFPS( imm(IMM), zmm7, zmm6, zmm5 ) \
    VINSERTF32X4( imm(0x0), mem(rcx), zmm0, zmm0 ) \
    VINSERTF32X4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    VINSERTF32X4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    VINSERTF32X4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    VFMADD231PS( zmm0, zmm4, zmm5 ) \
    VEXTRACTF32X4( imm(0x00), zmm5, mem(rcx) ) \
    VEXTRACTF32X4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    VEXTRACTF32X4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    VEXTRACTF32X4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )

#define TRANSPOSE_4X16H( IMM, R0, R1, R2, R3 ) \
    vunpckhps( ZMM(R1), ZMM(R0), zmm6 ) \
    vunpckhps( ZMM(R3), ZMM(R2), zmm7 ) \
    VSHUFPS( imm(IMM), zmm7, zmm6, zmm5 ) \
    VINSERTF32X4( imm(0x0), mem(rcx), zmm0, zmm0 ) \
    VINSERTF32X4( imm(0x1), mem(rcx, rdi, 4), zmm0, zmm0 ) \
    VINSERTF32X4( imm(0x2), mem(rcx, rdi, 8), zmm0, zmm0 ) \
    VINSERTF32X4( imm(0x3), mem(rcx, r12, 4), zmm0, zmm0 ) \
    VFMADD231PS( zmm0, zmm4, zmm5 ) \
    VEXTRACTF32X4( imm(0x00), zmm5, mem(rcx) ) \
    VEXTRACTF32X4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    VEXTRACTF32X4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    VEXTRACTF32X4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )

#define TRANSPOSE_4X16_BZ( R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ( 0X44, R0, R1, R2, R3 ) \
    TRANSPOSE_4X16L_BZ( 0XEE, R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ( 0X44, R0, R1, R2, R3 ) \
    TRANSPOSE_4X16H_BZ( 0XEE, R0, R1, R2, R3 )

#define TRANSPOSE_4X16L_BZ( IMM, R0, R1, R2, R3 ) \
    vunpcklps( ZMM(R1), ZMM(R0), zmm6 ) \
    vunpcklps( ZMM(R3), ZMM(R2), zmm7 ) \
    VSHUFPS( imm(IMM), zmm7, zmm6, zmm5 ) \
    VEXTRACTF32X4( imm(0x00), zmm5, mem(rcx) ) \
    VEXTRACTF32X4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    VEXTRACTF32X4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    VEXTRACTF32X4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )

#define TRANSPOSE_4X16H_BZ( IMM, R0, R1, R2, R3 ) \
    vunpckhps( ZMM(R1), ZMM(R0), zmm6 ) \
    vunpckhps( ZMM(R3), ZMM(R2), zmm7 ) \
    VSHUFPS( imm(IMM), zmm7, zmm6, zmm5 ) \
    VEXTRACTF32X4( imm(0x00), zmm5, mem(rcx) ) \
    VEXTRACTF32X4( imm(0x01), zmm5, mem(rcx, rdi, 4) ) \
    VEXTRACTF32X4( imm(0x02), zmm5, mem(rcx, rdi, 8) ) \
    VEXTRACTF32X4( imm(0x03), zmm5, mem(rcx, r12, 4) ) \
    add( rdi, rcx )

#define TRANSPOSE_2X16( R0, R1 ) \
    MOV( rcx, r12 ) \
    TRANSPOSE_2X16L( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    MOV( rcx, r12 ) \
    TRANSPOSE_2X16H( R0, R1 )

#define TRANSPOSE_2X16L( R0, R1 ) \
    VUNPCKLPS( zmm(R1), zmm(R0), zmm5 ) \
    FETCH_C_2X16 \
    MOV( r12, rcx ) \
    VFMADD231PS( zmm0, zmm4, zmm5 ) \
    UPDATE_C_2X16( 5 )

#define TRANSPOSE_2X16H( R0, R1 ) \
    VUNPCKHPS( zmm(R1), zmm(R0), zmm5 ) \
    FETCH_C_2X16 \
    MOV( r12, rcx ) \
    VFMADD231PS( zmm0, zmm4, zmm5 ) \
    UPDATE_C_2X16( 5 )

#define FETCH_C_2X16 \
    VMOVLPD( mem(rcx), xmm0, xmm0 ) \
    VMOVHPD( mem(rcx, rdi, 1), xmm0, xmm0 ) \
    LEA( mem(rcx, rdi, 4), rcx ) \
    VMOVLPD( mem(rcx), xmm1, xmm1 ) \
    VMOVHPD( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    VINSERTF32X4( imm(0x1), xmm1, zmm0, zmm0 ) \
    LEA( mem(rcx, rdi, 4), rcx ) \
    VMOVLPD( mem(rcx), xmm1, xmm1 ) \
    VMOVHPD( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    VINSERTF32X4( imm(0x2), xmm1, zmm0, zmm0 ) \
    LEA( mem(rcx, rdi, 4), rcx ) \
    VMOVLPD( mem(rcx), xmm1, xmm1 ) \
    VMOVHPD( mem(rcx, rdi, 1), xmm1, xmm1 ) \
    VINSERTF32X4( imm(0x3), xmm1, zmm0, zmm0 )

#define TRANSPOSE_2X16_BZ( R0, R1 ) \
    mov( rcx, r12 ) \
    TRANSPOSE_2X16L_BZ( R0, R1 ) \
    lea( mem(r12, rdi, 2), rcx ) \
    TRANSPOSE_2X16H_BZ( R0, R1 )

#define TRANSPOSE_2X16L_BZ( R0, R1 ) \
    vunpcklps( zmm(R1), zmm(R0), zmm5 ) \
    UPDATE_C_2X16( 5 ) \

#define TRANSPOSE_2X16H_BZ( R0, R1 ) \
    vunpckhps( zmm(R1), zmm(R0), zmm5 ) \
    UPDATE_C_2X16( 5 )

#define UPDATE_C_2X16( R0 ) \
    VEXTRACTF32X4( imm(0x0), zmm(R0), xmm0 ) \
    vmovlpd( xmm0, mem(rcx) ) \
    vmovhpd( xmm0, mem(rcx, rdi, 1) ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    VEXTRACTF32X4( imm(0x1), zmm(R0), xmm1 ) \
    vmovlpd( xmm1, mem(rcx) ) \
    vmovhpd( xmm1, mem(rcx, rdi, 1) ) \
    VEXTRACTF32X4( imm(0x2), zmm(R0), xmm2 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm2, mem(rcx) ) \
    vmovhpd( xmm2, mem(rcx, rdi, 1) ) \
    VEXTRACTF32X4( imm(0x3), zmm(R0), xmm3 ) \
    lea( mem(rcx, rdi, 4), rcx ) \
    vmovlpd( xmm3, mem(rcx) ) \
    vmovhpd( xmm3, mem(rcx, rdi, 1) )

#define UPDATE_C_1X16_BZ(R0) \
    UPDATE_C_1X16_BZ_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x02, R0 ) \
    UPDATE_C_1X16_BZ_UTIL( 0x03, R0 )

#define UPDATE_C_1X16_BZ_UTIL( IMM, R0 ) \
    VEXTRACTF32X4( imm(IMM), zmm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm1 ) \
    vshufps( imm(0x02), xmm0, xmm0, xmm2 ) \
    vshufps( imm(0x03), xmm0, xmm0, xmm3 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm1, (rcx, rdi, 1) ) \
    vmovss( xmm2, (rcx, rdi, 2) ) \
    vmovss( xmm3, (rcx, r12, 1) ) \
    lea( (rcx, rdi, 4), rcx )

#define UPDATE_C_1X16( R0 ) \
    UPDATE_C_1X16_UTIL( 0x00, R0 ) \
    UPDATE_C_1X16_UTIL( 0x01, R0 ) \
    UPDATE_C_1X16_UTIL( 0x02, R0 ) \
    UPDATE_C_1X16_UTIL( 0x03, R0 )

#define UPDATE_C_1X16_UTIL( IMM, R0 ) \
    VEXTRACTF32X4( imm(IMM), zmm(R0), xmm0 ) \
    vshufps( imm(0x01), xmm0, xmm0, xmm6 ) \
    vshufps( imm(0x02), xmm0, xmm0, xmm7 ) \
    vshufps( imm(0x03), xmm0, xmm0, xmm12 ) \
    vmovss( (rcx), xmm1 ) \
    vmovss( (rcx, rdi, 1), xmm2 ) \
    vmovss( (rcx, rdi, 2), xmm3 ) \
    vmovss( (rcx, r12, 1), xmm5 ) \
    vfmadd231ps( xmm1, xmm4, xmm0 ) \
    vfmadd231ps( xmm2, xmm4, xmm6 ) \
    vfmadd231ps( xmm3, xmm4, xmm7 ) \
    vfmadd231ps( xmm5, xmm4, xmm12 ) \
    vmovss( xmm0, (rcx) ) \
    vmovss( xmm6, (rcx, rdi, 1) ) \
    vmovss( xmm7, (rcx, rdi, 2) ) \
    vmovss( xmm12, (rcx, r12, 1) ) \
    lea( (rcx, rdi, 4), rcx )
