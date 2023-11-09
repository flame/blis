/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#define A_L1_PREFETCH_DIST 12 // in units of k iterations
#define B_L1_PREFETCH_DIST 12 // e.g. 4 k iterations ~= 56 cycles
#define TAIL_NITER 5 // in units of 4x unrolled k iterations
                     // e.g. 5 -> 4*5 k iterations ~= 280 cycles

#define PREFETCH_A_L1(n, k) \
    PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*16*8 + (2*n+k)*64))
#define PREFETCH_B_L1(n, k) \
    PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*14*8 + (2*n+k)*56))

#define LOOP_ALIGN ALIGN32

#define SUBITER(n) \
\
        PREFETCH_A_L1(n, 0) \
        \
        VBROADCASTSD(MEM(RBX, (14*n + 0)*8), ZMM(2)) \
        VBROADCASTSD(MEM(RBX, (14*n + 1)*8), ZMM(3)) \
        VFMADD231PD(ZMM(0), ZMM(2), ZMM(4)) \
        VFMADD231PD(ZMM(1), ZMM(2), ZMM(5)) \
        VFMADD231PD(ZMM(0), ZMM(3), ZMM(6)) \
        VFMADD231PD(ZMM(1), ZMM(3), ZMM(7)) \
        \
        VBROADCASTSD(MEM(RBX, (14*n + 2)*8), ZMM(2)) \
        VBROADCASTSD(MEM(RBX, (14*n + 3)*8), ZMM(3)) \
        VFMADD231PD(ZMM(0), ZMM(2), ZMM(8) ) \
        VFMADD231PD(ZMM(1), ZMM(2), ZMM(9) ) \
        VFMADD231PD(ZMM(0), ZMM(3), ZMM(10)) \
        VFMADD231PD(ZMM(1), ZMM(3), ZMM(11)) \
        \
        PREFETCH_B_L1(n, 0) \
        \
        VBROADCASTSD(MEM(RBX, (14*n + 4)*8), ZMM(2)) \
        VBROADCASTSD(MEM(RBX, (14*n + 5)*8), ZMM(3)) \
        VFMADD231PD(ZMM(0), ZMM(2), ZMM(12)) \
        VFMADD231PD(ZMM(1), ZMM(2), ZMM(13)) \
        VFMADD231PD(ZMM(0), ZMM(3), ZMM(14)) \
        VFMADD231PD(ZMM(1), ZMM(3), ZMM(15)) \
        \
        VBROADCASTSD(MEM(RBX, (14*n + 6)*8), ZMM(2)) \
        VBROADCASTSD(MEM(RBX, (14*n + 7)*8), ZMM(3)) \
        VFMADD231PD(ZMM(0), ZMM(2), ZMM(16)) \
        VFMADD231PD(ZMM(1), ZMM(2), ZMM(17)) \
        VFMADD231PD(ZMM(0), ZMM(3), ZMM(18)) \
        VFMADD231PD(ZMM(1), ZMM(3), ZMM(19)) \
        \
        PREFETCH_A_L1(n, 1) \
        \
        VBROADCASTSD(MEM(RBX, (14*n + 8)*8), ZMM(2)) \
        VBROADCASTSD(MEM(RBX, (14*n + 9)*8), ZMM(3)) \
        VFMADD231PD(ZMM(0), ZMM(2), ZMM(20)) \
        VFMADD231PD(ZMM(1), ZMM(2), ZMM(21)) \
        VFMADD231PD(ZMM(0), ZMM(3), ZMM(22)) \
        VFMADD231PD(ZMM(1), ZMM(3), ZMM(23)) \
        \
        VBROADCASTSD(MEM(RBX, (14*n + 10)*8), ZMM(2)) \
        VBROADCASTSD(MEM(RBX, (14*n + 11)*8), ZMM(3)) \
        VFMADD231PD(ZMM(0), ZMM(2), ZMM(24)) \
        VFMADD231PD(ZMM(1), ZMM(2), ZMM(25)) \
        VFMADD231PD(ZMM(0), ZMM(3), ZMM(26)) \
        VFMADD231PD(ZMM(1), ZMM(3), ZMM(27)) \
        \
        PREFETCH_B_L1(n, 1) \
        \
        VBROADCASTSD(MEM(RBX, (14*n + 12)*8), ZMM(2)) \
        VBROADCASTSD(MEM(RBX, (14*n + 13)*8), ZMM(3)) \
        VFMADD231PD(ZMM(0), ZMM(2), ZMM(28)) \
        VFMADD231PD(ZMM(1), ZMM(2), ZMM(29)) \
        VFMADD231PD(ZMM(0), ZMM(3), ZMM(30)) \
        VFMADD231PD(ZMM(1), ZMM(3), ZMM(31)) \
        \
        VMOVAPD(MEM(RAX,((n*2)+2)*8*8), ZMM(0)) \
        VMOVAPD(MEM(RAX,((n*2)+3)*8*8), ZMM(1))

#define UPDATE_C_COL_SCATTERED(R1,R2) \
\
        KXNORW(K(0), K(0), K(1)) \
        KXNORW(K(0), K(0), K(2)) \
        VSCATTERQPD(ZMM(R1), MEM(RCX,ZMM(2),1) MASK_K(1)) \
        VSCATTERQPD(ZMM(R2), MEM(R14,ZMM(2),1) MASK_K(2)) \
        ADD(RDI, RCX) \
        ADD(RDI, R14) \

/* scatter only first 6 elements of r1 and r2 */
#define UPDATE_C_COL_SCATTERED_2x6(R1,R2) \
\
        KXNORW(K(0), K(0), K(1)) \
        KXNORW(K(0), K(0), K(2)) \
        MOVQ(IMM(0b00111111), RAX) \
        KMOVQ(RAX, K(2)) \
        KMOVQ(RAX, K(1)) \
        VSCATTERQPD(ZMM(R1), MEM(RCX,ZMM(2),1) MASK_K(1)) \
        VSCATTERQPD(ZMM(R2), MEM(R14,ZMM(2),1) MASK_K(2)) \
        ADD(RDI, RCX) \
        ADD(RDI, R14) \

/*
Transpose 8 zmm registers and store the output in the given 8 registers
    Note: Requires offsetPointer for scatter instruction
          and 512 bytes of free memory (rcx) for transpose.
    Input :
            R1 = [ 0,  1,  2,  3,  4,  5,  6,  7]
            R2 = [ 8,  9, 10, 11, 12, 13, 14, 15]
            R3 = [16, 17, 18, 19, 20, 21, 22, 23]
            R4 = [24, 25, 26, 27, 28, 29, 30, 31]
            R5 = [32, 33, 34, 35, 36, 37, 38, 39]
            R6 = [40, 41, 42, 43, 44, 45, 46, 47]
            R7 = [48, 49, 50, 51, 52, 53, 54, 55]
            R18= [56, 57, 58, 59, 60, 61, 62, 63]
    Output :
            R1 = [0,  8, 16, 24, 32, 40, 48, 56]
            R2 = [1,  9, 17, 25, 33, 41, 49, 57]
            R3 = [2, 10, 18, 26, 34, 42, 50, 58]
            R4 = [3, 11, 19, 27, 35, 43, 51, 59]
            R5 = [4, 12, 20, 28, 36, 44, 52, 60]
            R6 = [5, 13, 21, 29, 37, 45, 53, 61]
            R7 = [6, 14, 22, 30, 38, 46, 54, 62]
            R18= [7, 15, 23, 31, 39, 47, 55, 63]
*/
#define TRANSPOSE_REGISTERS_8x8(R1, R2, R3, R4, R5, R6, R7, R18) \
\
        MOV(R8, RCX) \
        MOV(VAR(cs_c), RSI) \
        MOV(R9, RDI) \
        LEA(MEM(RCX, RSI, 8), RDX) \
        MOV(VAR(offsetPtr), R13) \
        MOV(RDI, R12) \
        CMP(RSI, R12) \
        CMOVL(RSI, R12) \
        VPBROADCASTQ(R12, ZMM(0)) \
        VPMULLQ(MEM(R13   ), ZMM(0), ZMM(2)) \
        \
        KXNORW(K(0), K(0), K(1)) \
        KXNORW(K(0), K(0), K(2)) \
        KXNORW(K(0), K(0), K(3)) \
        KXNORW(K(0), K(0), K(4)) \
        VSCATTERQPD(ZMM(R1), MEM(RCX,ZMM(2),1) MASK_K(1)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(1)) \
        VSCATTERQPD(ZMM(R2), MEM(RCX,ZMM(2),1) MASK_K(2)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(2)) \
        VSCATTERQPD(ZMM(R3), MEM(RCX,ZMM(2),1) MASK_K(3)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(3)) \
        VSCATTERQPD(ZMM(R4), MEM(RCX,ZMM(2),1) MASK_K(4)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(4)) \
        VSCATTERQPD(ZMM(R5), MEM(RCX,ZMM(2),1) MASK_K(1)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(1)) \
        VSCATTERQPD(ZMM(R6), MEM(RCX,ZMM(2),1) MASK_K(2)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(2)) \
        VSCATTERQPD(ZMM(R7), MEM(RCX,ZMM(2),1) MASK_K(3)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(3)) \
        VSCATTERQPD(ZMM(R18), MEM(RCX,ZMM(2),1) MASK_K(4)) \
        \
        MOV(R8, RCX) \
        \
        VMOVUPD(MEM(RCX), ZMM(R1))\
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R2)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R3)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R4)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R5)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R6)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R7)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R18)) \

/*
Transpose six zmm registers and store the output in the given 8 registers
    Note: Require offsetPointer for scatter instruction
          and 512 bytes of free memory (rcx) for transpose.
    Input :
            R1 = [ 0,  1,  2,  3,  4,  5,  6,  7]
            R2 = [ 8,  9, 10, 11, 12, 13, 14, 15]
            R3 = [16, 17, 18, 19, 20, 21, 22, 23]
            R4 = [24, 25, 26, 27, 28, 29, 30, 31]
            R5 = [32, 33, 34, 35, 36, 37, 38, 39]
            R6 = [40, 41, 42, 43, 44, 45, 46, 47]
    Output :
            R1 = [0,  8, 16, 24, 32, 40, -, -]
            R2 = [1,  9, 17, 25, 33, 41, -, -]
            R3 = [2, 10, 18, 26, 34, 42, -, -]
            R4 = [3, 11, 19, 27, 35, 43, -, -]
            R5 = [4, 12, 20, 28, 36, 44, -, -]
            R6 = [5, 13, 21, 29, 37, 45, -, -]
            R7 = [6, 14, 22, 30, 38, 46, -, -]
           R18 = [7, 15, 23, 31, 39, 47, -, -]
*/
#define TRANSPOSE_REGISTERS_6x8(R1, R2, R3, R4, R5, R6, R7, R18) \
\
        MOV(R8, RCX) \
        MOV(VAR(cs_c), RSI) \
        MOV(R9, RDI) \
        LEA(MEM(RCX, RSI, 8), RDX) \
        MOV(VAR(offsetPtr), R13) \
        MOV(RDI, R12) \
        CMP(RSI, R12) \
        CMOVL(RSI, R12) \
        VPBROADCASTQ(R12, ZMM(0)) \
        VPMULLQ(MEM(R13   ), ZMM(0), ZMM(2)) \
        LEA(MEM(RCX, R12, 4), RCX) \
        LEA(MEM(RCX, R12, 1), RCX) \
        \
        KXNORW(K(0), K(0), K(1)) \
        KXNORW(K(0), K(0), K(2)) \
        KXNORW(K(0), K(0), K(3)) \
        KXNORW(K(0), K(0), K(4)) \
        VSCATTERQPD(ZMM(R1), MEM(RCX,ZMM(2),1) MASK_K(1)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(1)) \
        VSCATTERQPD(ZMM(R2), MEM(RCX,ZMM(2),1) MASK_K(2)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(2)) \
        VSCATTERQPD(ZMM(R3), MEM(RCX,ZMM(2),1) MASK_K(3)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(3)) \
        VSCATTERQPD(ZMM(R4), MEM(RCX,ZMM(2),1) MASK_K(4)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(4)) \
        VSCATTERQPD(ZMM(R5), MEM(RCX,ZMM(2),1) MASK_K(1)) \
        ADD(IMM(1*8), RCX) \
        KXNORW(K(0), K(0), K(1)) \
        VSCATTERQPD(ZMM(R6), MEM(RCX,ZMM(2),1) MASK_K(2)) \
        \
        MOV(R8, RCX) \
        LEA(MEM(RCX, R12, 4), RCX) \
        LEA(MEM(RCX, R12, 1), RCX) \
        \
        VMOVUPD(MEM(RCX), ZMM(R1))\
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R2)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R3)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R4)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R5)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R6)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R7)) \
        ADD(R12, RCX) \
        VMOVUPD(MEM(RCX), ZMM(R18)) \

// Offsets for scatter/gather instructions
static int64_t offsets[16] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15};


void bli_dgemmtrsm_l_zen_asm_16x14
(
    dim_t               k_,
    double*    restrict alpha,
    double*    restrict a10,
    double*    restrict a11,
    double*    restrict b01,
    double*    restrict b11,
    double*    restrict c11, inc_t rs_c_, inc_t cs_c_,
    auxinfo_t* restrict data,
    cntx_t*    restrict cntx
)
{
        AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_9);
        const int64_t k = k_;
        uint64_t rs_c   = rs_c_ * 8;
        const int64_t* offsetPtr = &offsets[0];
        uint64_t cs_c   = cs_c_ * 8;

        BEGIN_ASM()

        //clear out registers
        VXORPD(ZMM(4), ZMM(4), ZMM(4))
        VMOVAPD(ZMM(4), ZMM(5) )
        VMOVAPD(ZMM(4), ZMM(6) )
        VMOVAPD(ZMM(4), ZMM(7) )
        VMOVAPD(ZMM(4), ZMM(8) )
        VMOVAPD(ZMM(4), ZMM(9) )
        VMOVAPD(ZMM(4), ZMM(10))
        VMOVAPD(ZMM(4), ZMM(11))
        VMOVAPD(ZMM(4), ZMM(12))
        VMOVAPD(ZMM(4), ZMM(13))
        VMOVAPD(ZMM(4), ZMM(14))
        VMOVAPD(ZMM(4), ZMM(15))
        VMOVAPD(ZMM(4), ZMM(16))
        VMOVAPD(ZMM(4), ZMM(17))
        VMOVAPD(ZMM(4), ZMM(18))
        VMOVAPD(ZMM(4), ZMM(19))
        VMOVAPD(ZMM(4), ZMM(20))
        VMOVAPD(ZMM(4), ZMM(21))
        VMOVAPD(ZMM(4), ZMM(22))
        VMOVAPD(ZMM(4), ZMM(23))
        VMOVAPD(ZMM(4), ZMM(24))
        VMOVAPD(ZMM(4), ZMM(25))
        VMOVAPD(ZMM(4), ZMM(26))
        VMOVAPD(ZMM(4), ZMM(27))
        VMOVAPD(ZMM(4), ZMM(28))
        VMOVAPD(ZMM(4), ZMM(29))
        VMOVAPD(ZMM(4), ZMM(30))
        VMOVAPD(ZMM(4), ZMM(31))

        MOV(VAR(k), RSI)

        MOV(VAR(a10), RAX)                               // load address of a
        MOV(VAR(b01), RBX)                               // load address of b
        MOV(VAR(c11), R8)                                // load address of c

        LEA(MEM(RSI,RSI,2), RDX)
        LEA(MEM(,RDX,4), RDX)
        LEA(MEM(RDX,RSI,4), RDX)                         // 16 * K
        LEA(MEM(RAX,RDX,8,-128), RDX)                    // a_next
        LEA(MEM(R8,63), R12)                             // c for prefetching

        MOV(IMM(14), RDI)
        LEA(MEM(, RDI, 8), RDI)

        MOV(VAR(rs_c), R9)
        MOV(VAR(cs_c), R13)

        MOV(IMM(0), R11)
        MOV(R13, R15)

        CMP(IMM(8), R13)
        JNE(.DBEFORELOOP)
                MOV(IMM(2), R11)
                MOV(R9, R15)

        LABEL(.DBEFORELOOP)

        VMOVAPD(MEM(RAX,  0*8*8), ZMM(0))
        VMOVAPD(MEM(RAX,  1*8*8), ZMM(1))                // preload a

        MOV(RSI, R10)
	AND(IMM(3), R10)                                // R10 = K % 4
        SAR(IMM(2), RSI)                                // RSI = K / 4

        /*
        MAIN LOOP
                Note: This loop runs (K/4 - 14 - TAIL_NITER) times
        */
        SUB(R11, RSI)
        SUB(IMM(14+TAIL_NITER), RSI)
        JLE(K_LE_80)

                LOOP_ALIGN
                LABEL(LOOP1)
                        SUBITER(0)
                        PREFETCH(1, MEM(RDX))
                        SUBITER(1)
                        SUB(IMM(1), RSI)
                        SUBITER(2)
                        PREFETCH(1, MEM(RDX,64))
                        SUBITER(3)

                        LEA(MEM(RAX,4*16*8), RAX)
                        LEA(MEM(RBX,4*14*8), RBX)
                        LEA(MEM(RDX,16*8), RDX)

                JNZ(LOOP1)

        LABEL(K_LE_80)

        /*
        C prefetch Loop
                Note: This loop runs 14 times,
                These 14 iterations are done separately so that c11 can be prefetched here.
        */
        ADD(R11, RSI)
        ADD(IMM(14), RSI)
        JLE(K_LE_24)

                LOOP_ALIGN
                LABEL(LOOP2)
                        PREFETCH(0, MEM(R12))
                        SUBITER(0)
                        PREFETCH(1, MEM(RDX))
                        SUBITER(1)
                        PREFETCH(0, MEM(R12,64))
                        SUB(IMM(1), RSI)
                        SUBITER(2)
                        PREFETCH(1, MEM(RDX,64))
                        SUBITER(3)

                        LEA(MEM(RAX,4*16*8), RAX)
                        LEA(MEM(RBX,4*14*8), RBX)
                        LEA(MEM(RDX,16*8), RDX)
                        LEA(MEM(R12,R15,1), R12)

                JNZ(LOOP2)

        LABEL(K_LE_24)

        /*
        TAIL_NITER Loop
                Note: This loop runs TAIL_NITER times,
                This loop is used to provide some distance between c11 prefetch and usage of c11.
        */
        ADD(IMM(0+TAIL_NITER), RSI)
        JLE(TAIL)

                LOOP_ALIGN
                LABEL(LOOP3)

                        SUBITER(0)
                        PREFETCH(1, MEM(RDX))
                        SUBITER(1)
                        SUB(IMM(1), RSI)
                        SUBITER(2)
                        PREFETCH(1, MEM(RDX,64))
                        SUBITER(3)

                        LEA(MEM(RAX,4*16*8), RAX)
                        LEA(MEM(RBX,4*14*8), RBX)
                        LEA(MEM(RDX,16*8), RDX)

                JNZ(LOOP3)

        /*
        K Left Loop
                This loop runs K % 4 times.
        */
        LABEL(TAIL)
        MOV(R10, RSI)
        TEST(RSI, RSI)
        JE(.DPOSTACCUM)
                LOOP_ALIGN
                LABEL(TAIL_LOOP)

                        SUB(IMM(1), RSI)
                        SUBITER(0)

                        LEA(MEM(RAX,16*8), RAX)
                        LEA(MEM(RBX,14*8), RBX)

                JNZ(TAIL_LOOP)

        LABEL(.DPOSTACCUM)
	        /* GEMM output before transpose                                 GEMM output after transpose
                                                                        __________________________________
                ___________________________                             |______zmm4______|______zmm20___x x|
               | | | | | | | | | | | | | | |                            |______zmm6______|______zmm22___x x|
               |z|z|z|z|z|z|z|z|z|z|z|z|z|z|                            |______zmm8______|______zmm24___x x|
               |m|m|m|m|m|m|m|m|m|m|m|m|m|m|                            |______zmm10_____|______zmm26___x x|
               |m|m|m|m|m|m|m|m|m|m|m|m|m|m|                            |______zmm12_____|______zmm28___x x|
               |4|6|8|1|1|1|1|1|2|2|2|2|2|3|                            |______zmm14_____|______zmm30___x x|
               | | | |0|2|4|6|8|0|2|4|6|8|0|                            |______zmm16_____|_____c11______x x|
               | | | | | | | | | | | | | | |                            |______zmm18_____|_____c11+cs___x x|
               ____________________________                             |______zmm5______|______zmm21___x x|
               | | | | | | | | | | | | | | |                            |______zmm7______|______zmm23___x x|
               |z|z|z|z|z|z|z|z|z|z|z|z|z|z|                            |______zmm9______|______zmm25___x x|
               |m|m|m|m|m|m|m|m|m|m|m|m|m|m|                            |______zmm11_____|______zmm27___x x|
               |m|m|m|m|m|m|m|m|m|m|m|m|m|m|                            |______zmm13_____|______zmm29___x x|
               |5|7|9|1|1|1|1|1|2|2|2|2|2|3|                            |______zmm15_____|______zmm31___x x|
               | | | |1|3|5|7|9|1|3|5|7|9|1|                            |______zmm17_____|____c11+cs*2__x x|
               | | | | | | | | | | | | | | |                            |______zmm19_____|____c11+cs*4__x x|
               _____________________________
                */
	        TRANSPOSE_REGISTERS_8x8(4, 6, 8, 10, 12, 14, 16, 18)         // transpose the output of GEMM
                TRANSPOSE_REGISTERS_8x8(5, 7, 9, 11, 13, 15, 17, 19)
                TRANSPOSE_REGISTERS_6x8(20, 22, 24, 26, 28, 30, 0, 1)
                VMOVUPD(ZMM(0), MEM(R8     ))
                VMOVUPD(ZMM(1), MEM(R8, R12, 1))                             // zmm0 and zmm1 are needed for other computations,
                                                                             // therefore store zmm0, zmm1 's data in rcx
                TRANSPOSE_REGISTERS_6x8(21, 23, 25, 27, 29, 31, 0, 1)
                VMOVUPD(ZMM(0), MEM(R8, R12, 2))
                VMOVUPD(ZMM(1), MEM(R8, R12, 4))                             // zmm0 and zmm1 are needed for other computations,
                                                                             // therefore store zmm0, zmm1 's data in rcx
                MOV(IMM(14), RDI)
                LEA(MEM(, RDI, 8), RDI)

                MOV(VAR(alpha), RBX)
                VBROADCASTSD(MEM(RBX), ZMM(3))

                MOV(IMM(1), RSI)
                LEA(MEM(, RSI, 8), RSI)

                MOV(VAR(b11), RCX)
                LEA(MEM(RCX, RSI, 8), RDX)

                MOV(RCX, R11)
                MOV(RDX, R14)

                // Scale by Alpha
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(4))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(6))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(8))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(10))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(12))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(14))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(16))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(18))
                ADD(RDI, RCX)

                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(5))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(7))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(9))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(11))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(13))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(15))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(17))
                ADD(RDI, RCX)
                VFMSUB231PD(MEM(RCX), ZMM(3), ZMM(19))



                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(20))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(22))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(24))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(26))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(28))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(30))
                ADD(RDI, RDX)
                VMOVUPD(MEM(R8     ), ZMM(0))
                VMOVUPD(MEM(R8, R12, 1), ZMM(1))
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(0))
                VMOVUPD(ZMM(0), MEM(R8        ))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(1))
                VMOVUPD(ZMM(1), MEM(R8, R12, 1))
                ADD(RDI, RDX)

                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(21))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(23))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(25))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(27))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(29))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(31))
                ADD(RDI, RDX)
                VMOVUPD(MEM(R8, R12, 2), ZMM(0))
                VMOVUPD(MEM(R8, R12, 4), ZMM(1))
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(0))
                VMOVUPD(ZMM(0), MEM(R8, R12, 2))
                ADD(RDI, RDX)
                VFMSUB231PD(MEM(RDX), ZMM(3), ZMM(1))
                VMOVUPD(ZMM(1), MEM(R8, R12, 4))

                /*
                TRSM region
                        Each row requires 1 iteration, therefore 16 iterations are present
                */
                MOV(VAR(a11), RAX)
                MOV(R11, RCX)
                MOV(R14, RDX)


                //iteration 0 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (0+0*16)*8), ZMM(0))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(4) , ZMM(4) )
                        VMULPD(ZMM(0), ZMM(20), ZMM(20))
                #else
                        VDIVPD(ZMM(0), ZMM(4) , ZMM(4) )
                        VDIVPD(ZMM(0), ZMM(20), ZMM(20))
                #endif

                VMOVUPD(ZMM(4), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(20), YMM(0))
                VMOVUPD(YMM(20), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))                   // move only first six values to rcx
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 1 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (1+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (1+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VSUBPD(ZMM(2), ZMM(6) , ZMM(6) )
                VSUBPD(ZMM(3), ZMM(22), ZMM(22))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(6) , ZMM(6) )
                        VMULPD(ZMM(1), ZMM(22), ZMM(22))
                #else
                        VDIVPD(ZMM(1), ZMM(6) , ZMM(6))
                        VDIVPD(ZMM(1), ZMM(22), ZMM(22))
                #endif

                VMOVUPD(ZMM(6), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(22), YMM(0))
                VMOVUPD(YMM(22), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 2 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (2+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (2+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))


                VBROADCASTSD(MEM(RAX, (2+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VSUBPD(ZMM(2), ZMM(8) , ZMM(8) )
                VSUBPD(ZMM(3), ZMM(24), ZMM(24))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(8) , ZMM(8) )
                        VMULPD(ZMM(0), ZMM(24), ZMM(24))
                #else
                        VDIVPD(ZMM(0), ZMM(8) , ZMM(8) )
                        VDIVPD(ZMM(0), ZMM(24), ZMM(24))
                #endif

                VMOVUPD(ZMM(8), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(24), YMM(0))
                VMOVUPD(YMM(24), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                 ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 3 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (3+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (3+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (3+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (3+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VSUBPD(ZMM(2), ZMM(10), ZMM(10))
                VSUBPD(ZMM(3), ZMM(26), ZMM(26))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(10), ZMM(10))
                        VMULPD(ZMM(1), ZMM(26), ZMM(26))
                #else
                        VDIVPD(ZMM(1), ZMM(10), ZMM(10))
                        VDIVPD(ZMM(1), ZMM(26), ZMM(26))
                #endif

                VMOVUPD(ZMM(10), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(26), YMM(0))
                VMOVUPD(YMM(26), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 4 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (4+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (4+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (4+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (4+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (4+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VSUBPD(ZMM(2), ZMM(12), ZMM(12))
                VSUBPD(ZMM(3), ZMM(28), ZMM(28))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(12), ZMM(12))
                        VMULPD(ZMM(0), ZMM(28), ZMM(28))
                #else
                        VDIVPD(ZMM(0), ZMM(12), ZMM(12))
                        VDIVPD(ZMM(0), ZMM(28), ZMM(28))
                #endif

                VMOVUPD(ZMM(12), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(28), YMM(0))
                VMOVUPD(YMM(28), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 5 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (5+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (5+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (5+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (5+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (5+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (5+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VSUBPD(ZMM(2), ZMM(14), ZMM(14))
                VSUBPD(ZMM(3), ZMM(30), ZMM(30))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(14), ZMM(14))
                        VMULPD(ZMM(1), ZMM(30), ZMM(30))
                #else
                        VDIVPD(ZMM(1), ZMM(14), ZMM(14))
                        VDIVPD(ZMM(1), ZMM(30), ZMM(30))
                #endif

                VMOVUPD(ZMM(14), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(30), YMM(0))
                VMOVUPD(YMM(30), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 6 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (6+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (6+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (6+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (6+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (6+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (6+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (6+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8), ZMM(1))
                VSUBPD(ZMM(2), ZMM(16), ZMM(16))
                VSUBPD(ZMM(3), ZMM(1) , ZMM(1) )

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(16), ZMM(16))
                        VMULPD(ZMM(0), ZMM(1) , ZMM(1) )
                #else
                        VDIVPD(ZMM(0), ZMM(16), ZMM(16))
                        VDIVPD(ZMM(0), ZMM(1) , ZMM(1) )
                #endif

                VMOVUPD(ZMM(1), MEM(R8        ))

                VMOVUPD(ZMM(16), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(1), YMM(0))
                VMOVUPD(YMM(1), MEM(RDX    ))
                VMOVUPD(XMM(0), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 7 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (7+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (7+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (7+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (7+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (7+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (7+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (7+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (7+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VSUBPD(ZMM(2), ZMM(18), ZMM(18))
                VSUBPD(ZMM(3), ZMM(0) , ZMM(0) )

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(18), ZMM(18))
                        VMULPD(ZMM(1), ZMM(0) , ZMM(0) )
                #else
                        VDIVPD(ZMM(1), ZMM(18), ZMM(18))
                        VDIVPD(ZMM(1), ZMM(0) , ZMM(0) )
                #endif

                VMOVUPD(ZMM(0), MEM(R8, R12, 1))

                VMOVUPD(ZMM(18), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(0), YMM(1))
                VMOVUPD(YMM(0), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 8 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (8+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (8+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (8+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (8+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (8+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (8+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (8+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (8+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (8+8*16)*8), ZMM(0))

                VSUBPD(ZMM(2), ZMM(5) , ZMM(5) )
                VSUBPD(ZMM(3), ZMM(21), ZMM(21))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(5), ZMM(5))
                        VMULPD(ZMM(0), ZMM(21), ZMM(21))
                #else
                        VDIVPD(ZMM(0), ZMM(5), ZMM(5))
                        VDIVPD(ZMM(0), ZMM(21), ZMM(21))
                #endif

                VMOVUPD(ZMM(5), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(21), YMM(1))
                VMOVUPD(YMM(21), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 9 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (9+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (9+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (9+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (9+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (9+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (9+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (9+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (9+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (9+8*16)*8), ZMM(0))

                VBROADCASTSD(MEM(RAX, (9+9*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(5), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(21), ZMM(3))

                VSUBPD(ZMM(2), ZMM(7), ZMM(7))
                VSUBPD(ZMM(3), ZMM(23), ZMM(23))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(7), ZMM(7))
                        VMULPD(ZMM(1), ZMM(23), ZMM(23))
                #else
                        VDIVPD(ZMM(1), ZMM(7), ZMM(7))
                        VDIVPD(ZMM(1), ZMM(23), ZMM(23))
                #endif

                VMOVUPD(ZMM(7), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(23), YMM(1))
                VMOVUPD(YMM(23), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 10 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (10+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (10+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (10+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (10+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (10+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (10+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (10+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (10+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (10+8*16)*8), ZMM(0))

                VBROADCASTSD(MEM(RAX, (10+9*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(5), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(21), ZMM(3))

                VBROADCASTSD(MEM(RAX, (10+10*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(7), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(23), ZMM(3))

                VSUBPD(ZMM(2), ZMM(9), ZMM(9))
                VSUBPD(ZMM(3), ZMM(25), ZMM(25))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(9), ZMM(9))
                        VMULPD(ZMM(0), ZMM(25), ZMM(25))
                #else
                        VDIVPD(ZMM(0), ZMM(9), ZMM(9))
                        VDIVPD(ZMM(0), ZMM(25), ZMM(25))
                #endif

                VMOVUPD(ZMM(9), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(25), YMM(1))
                VMOVUPD(YMM(25), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 11 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (11+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (11+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (11+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (11+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (11+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (11+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (11+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (11+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (11+8*16)*8), ZMM(0))

                VBROADCASTSD(MEM(RAX, (11+9*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(5), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(21), ZMM(3))

                VBROADCASTSD(MEM(RAX, (11+10*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(7), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(23), ZMM(3))

                VBROADCASTSD(MEM(RAX, (11+11*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(9), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(25), ZMM(3))

                VSUBPD(ZMM(2), ZMM(11), ZMM(11))
                VSUBPD(ZMM(3), ZMM(27), ZMM(27))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(11), ZMM(11))
                        VMULPD(ZMM(1), ZMM(27), ZMM(27))
                #else
                        VDIVPD(ZMM(1), ZMM(11), ZMM(11))
                        VDIVPD(ZMM(1), ZMM(27), ZMM(27))
                #endif

                VMOVUPD(ZMM(11), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(27), YMM(1))
                VMOVUPD(YMM(27), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 12 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (12+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (12+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (12+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (12+8*16)*8), ZMM(0))

                VBROADCASTSD(MEM(RAX, (12+9*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(5), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(21), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+10*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(7), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(23), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+11*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(9), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(25), ZMM(3))

                VBROADCASTSD(MEM(RAX, (12+12*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(11), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(27), ZMM(3))

                VSUBPD(ZMM(2), ZMM(13), ZMM(13))
                VSUBPD(ZMM(3), ZMM(29), ZMM(29))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(13), ZMM(13))
                        VMULPD(ZMM(0), ZMM(29), ZMM(29))
                #else
                        VDIVPD(ZMM(0), ZMM(13), ZMM(13))
                        VDIVPD(ZMM(0), ZMM(29), ZMM(29))
                #endif

                VMOVUPD(ZMM(13), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(29), YMM(1))
                VMOVUPD(YMM(29), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 13 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (13+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (13+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (13+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (13+8*16)*8), ZMM(0))

                VBROADCASTSD(MEM(RAX, (13+9*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(5), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(21), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+10*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(7), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(23), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+11*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(9), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(25), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+12*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(11), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(27), ZMM(3))

                VBROADCASTSD(MEM(RAX, (13+13*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(13), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(29), ZMM(3))

                VSUBPD(ZMM(2), ZMM(15), ZMM(15))
                VSUBPD(ZMM(3), ZMM(31), ZMM(31))

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(15), ZMM(15))
                        VMULPD(ZMM(1), ZMM(31), ZMM(31))
                #else
                        VDIVPD(ZMM(1), ZMM(15), ZMM(15))
                        VDIVPD(ZMM(1), ZMM(31), ZMM(31))
                #endif

                VMOVUPD(ZMM(15), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(31), YMM(1))
                VMOVUPD(YMM(31), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 14 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (14+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (14+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (14+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (14+8*16)*8), ZMM(0))

                VBROADCASTSD(MEM(RAX, (14+9*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(5), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(21), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+10*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(7), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(23), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+11*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(9), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(25), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+12*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(11), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(27), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+13*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(13), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(29), ZMM(3))

                VBROADCASTSD(MEM(RAX, (14+14*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(15), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(31), ZMM(3))

                VMOVUPD(MEM(R8, R12, 2), ZMM(1))
                VSUBPD(ZMM(2), ZMM(17), ZMM(17))
                VSUBPD(ZMM(3), ZMM(1) , ZMM(1) )

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(0), ZMM(17), ZMM(17))
                        VMULPD(ZMM(0), ZMM(1) , ZMM(1) )
                #else
                        VDIVPD(ZMM(0), ZMM(17), ZMM(17))
                        VDIVPD(ZMM(0), ZMM(1) , ZMM(1) )
                #endif

                VMOVUPD(ZMM(1), MEM(R8, R12, 2))

                VMOVUPD(ZMM(17), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(1), YMM(0))
                VMOVUPD(YMM(1), MEM(RDX    ))
                VMOVUPD(XMM(0), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)

                //iteration 15 --------------------------------------------
                VBROADCASTSD(MEM(RAX, (15+0*16)*8), ZMM(0))
                VBROADCASTSD(MEM(RAX, (15+1*16)*8), ZMM(1))

                VMULPD(ZMM(0), ZMM(4) , ZMM(2))
                VMULPD(ZMM(0), ZMM(20), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+2*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(6) , ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(22), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+3*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(8) , ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(24), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+4*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(10), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(26), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+5*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(12), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(28), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+6*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(14), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(30), ZMM(3))

                VMOVUPD(MEM(R8     ), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(16), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (15+7*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 1), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(18), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(0) , ZMM(3))
                VBROADCASTSD(MEM(RAX, (15+8*16)*8), ZMM(0))

                VBROADCASTSD(MEM(RAX, (15+9*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(5), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(21), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+10*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(7), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(23), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+11*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(9), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(25), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+12*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(11), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(27), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+13*16)*8), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(13), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(29), ZMM(3))

                VBROADCASTSD(MEM(RAX, (15+14*16)*8), ZMM(0))
                VFMADD231PD(ZMM(1), ZMM(15), ZMM(2))
                VFMADD231PD(ZMM(1), ZMM(31), ZMM(3))

                VMOVUPD(MEM(R8, R12, 2), ZMM(1))
                VFMADD231PD(ZMM(0), ZMM(17), ZMM(2))
                VFMADD231PD(ZMM(0), ZMM(1 ), ZMM(3))
                VBROADCASTSD(MEM(RAX, (15+15*16)*8), ZMM(1))

                VMOVUPD(MEM(R8, R12, 4), ZMM(0))
                VSUBPD(ZMM(2), ZMM(19), ZMM(19))
                VSUBPD(ZMM(3), ZMM(0) , ZMM(0) )

                #ifdef BLIS_ENABLE_TRSM_PREINVERSION
                        VMULPD(ZMM(1), ZMM(19), ZMM(19))
                        VMULPD(ZMM(1), ZMM(0) , ZMM(0) )
                #else
                        VDIVPD(ZMM(1), ZMM(19), ZMM(19))
                        VDIVPD(ZMM(1), ZMM(0) , ZMM(0) )
                #endif

                VMOVUPD(ZMM(0), MEM(R8, R12, 4))

                VMOVUPD(ZMM(19), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(0), YMM(1))
                VMOVUPD(YMM(0), MEM(RDX    ))
                VMOVUPD(XMM(1), MEM(RDX,4*8))
                ADD(RDI, RCX)
                ADD(RDI, RDX)


        /*
        Storage Region (Post TRSM)
        */
        MOV(R8, RCX)
        MOV(R9, RDI)
        MOV(VAR(cs_c), RSI)

        LEA(MEM(RCX, RSI, 8), RDX)
        LEA(MEM(RCX, RDI, 8), R14)

        LEA(MEM(RSI, RSI, 2), R12)
        LEA(MEM(RSI, RSI, 4), R13)
        LEA(MEM(R13, RSI, 2), R15)

        CMP(IMM(8), RSI)
        JZ(.DROWSTORED)

        CMP(IMM(8), RDI)
        JZ(.DCOLSTORED)

        LABEL(.DROWSTORED)

                VMOVUPD(MEM(R8  ), ZMM(1))
                VMOVUPD(ZMM(4), MEM(RCX))
                VMOVUPD(MEM(R8, RDI, 1), ZMM(4))

                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(20), YMM(0))
                VMOVUPD(YMM(20), MEM(RDX    ))

                VMOVUPD(MEM(R8, RDI, 2), ZMM(20))

                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(6), MEM(RCX))

                VMOVUPD(MEM(R8, RDI, 4), ZMM(6))

                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(22), YMM(0))
                VMOVUPD(YMM(22), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(8), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(24), YMM(0))
                VMOVUPD(YMM(24), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(10), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(26), YMM(0))
                VMOVUPD(YMM(26), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(12), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(28), YMM(0))
                VMOVUPD(YMM(28), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(14), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(30), YMM(0))
                VMOVUPD(YMM(30), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(16), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(1), YMM(0))
                VMOVUPD(YMM(1), MEM(RDX    ))
                VMOVUPD(XMM(0), MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(18), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(4), YMM(0))
                VMOVUPD(YMM(4), MEM(RDX    ))
                VMOVUPD(XMM(0), MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(5), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(21), YMM(0))
                VMOVUPD(YMM(21), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(7), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(23), YMM(0))
                VMOVUPD(YMM(23), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(9), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(25), YMM(0))
                VMOVUPD(YMM(25), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(11), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(27), YMM(0))
                VMOVUPD(YMM(27), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(13), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(29), YMM(0))
                VMOVUPD(YMM(29), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(15), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(31), YMM(0))
                VMOVUPD(YMM(31), MEM(RDX    ))
                VMOVUPD(XMM(0) , MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(17), MEM(RCX))
                ADD(RDI, RCX)
                VEXTRACTF64X4(IMM(1), ZMM(20), YMM(0))
                VMOVUPD(YMM(20), MEM(RDX    ))
                VMOVUPD(XMM(0), MEM(RDX,4*8))
                ADD(RDI, RDX)

                VMOVUPD(ZMM(19), MEM(RCX))
                VEXTRACTF64X4(IMM(1), ZMM(6), YMM(0))
                VMOVUPD(YMM(6), MEM(RDX    ))
                VMOVUPD(XMM(0), MEM(RDX,4*8))


        JMP(.DDONE)
        LABEL(.DCOLSTORED)


                MOV(VAR(offsetPtr), R12)
                LEA(MEM(RCX, RSI, 8), RDX)
                VPBROADCASTQ(RSI, ZMM(0))
                VPMULLQ(MEM(R12), ZMM(0), ZMM(2))
                VPMULLQ(MEM(R12,64), ZMM(0), ZMM(3))

                VMOVUPD(MEM(RCX        ), ZMM(0))
                VMOVUPD(MEM(RCX, RSI, 1), ZMM(1))

                MOV(RDX, RCX)
                LEA(MEM(RCX, RDI, 8), R14)
                UPDATE_C_COL_SCATTERED_2x6(20,21)
                VMOVUPD(MEM(R8, RSI, 2), ZMM(20))
                VMOVUPD(MEM(R8, RSI, 4), ZMM(21))
                UPDATE_C_COL_SCATTERED_2x6(22,23)
                UPDATE_C_COL_SCATTERED_2x6(24,25)
                UPDATE_C_COL_SCATTERED_2x6(26,27)
                UPDATE_C_COL_SCATTERED_2x6(28,29)
                UPDATE_C_COL_SCATTERED_2x6(30,31)
                UPDATE_C_COL_SCATTERED_2x6(0 ,20 )
                UPDATE_C_COL_SCATTERED_2x6(1 ,21 )

                MOV(R8, RCX)
                LEA(MEM(RCX, RDI, 8), R14)
                UPDATE_C_COL_SCATTERED( 4, 5)
                UPDATE_C_COL_SCATTERED( 6, 7)
                UPDATE_C_COL_SCATTERED( 8, 9)
                UPDATE_C_COL_SCATTERED(10,11)
                UPDATE_C_COL_SCATTERED(12,13)
                UPDATE_C_COL_SCATTERED(14,15)
                UPDATE_C_COL_SCATTERED(16,17)
                UPDATE_C_COL_SCATTERED(18,19)


        LABEL(.DDONE)

        VZEROUPPER()

        end_asm(
        : // output operands (none)
        : // input operands
      [a10]    "m" (a10),    // 1
      [k]      "m" (k),      // 2
      [b01]    "m" (b01),    // 3
      [a11]    "m" (a11),    // 6
      [b11]    "m" (b11),    // 7
      [c11]    "m" (c11),    // 8
      [rs_c]   "m" (rs_c),   // 9
      [cs_c]   "m" (cs_c),   // 10,
      [alpha]  "m" (alpha),
      [offsetPtr] "m" (offsetPtr)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "xmm0", "xmm1",
          "ymm0", "ymm1", "ymm4", "ymm6", "ymm20", "ymm21", "ymm22", "ymm23",
          "ymm24", "ymm25", "ymm26", "ymm27", "ymm28", "ymm29", "ymm30", "ymm31",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31",
          "k0", "k1", "k2", "k3", "k4", "memory"
        )

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_9);
}
