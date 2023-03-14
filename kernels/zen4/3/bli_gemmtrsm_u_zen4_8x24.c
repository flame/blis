/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc.All rights reserved.

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
   AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "bli_x86_asm_macros.h"
// BLIS_ASM_SYNTAX_INTEL syntax is followed

#define TAIL_NITER 5

#define LOOP_ALIGN ALIGN32

// Update C when C is general stored
#define UPDATE_C_SCATTERED(R1,R2,R3) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R2)) \
    VSCATTERQPD(MEM(RCX,ZMM(4),1) MASK_K(3), ZMM(R3))  \
    LEA(RCX, MEM(RCX,R12,1))

// 8x8 in register transpose, used for column stored C
#define TRANSPOSE_8X8(R0, R1, R2, R3, R4, R5, R6, R7) \
\
    VUNPCKLPD(ZMM(6), ZMM(R0), ZMM(R1)) \
    VUNPCKLPD(ZMM(7), ZMM(R2), ZMM(R3)) \
    VUNPCKLPD(ZMM(2), ZMM(R4), ZMM(R5)) \
    VUNPCKLPD(ZMM(3), ZMM(R6), ZMM(R7)) \
    VMOVUPD(ZMM(0), ZMM(R0)) \
    VMOVUPD(ZMM(1), ZMM(R4)) \
    /*Stage2*/ \
    VSHUFF64X2(ZMM(4), ZMM(6), ZMM(7), IMM(0x88)) \
    VSHUFF64X2(ZMM(5), ZMM(2), ZMM(3), IMM(0x88)) \
    /*Stage3  1,5*/ \
    VSHUFF64X2(ZMM(R0), ZMM(4), ZMM(5), IMM(0x88)) \
    VSHUFF64X2(ZMM(R4), ZMM(4), ZMM(5), IMM(0xDD)) \
    /*Stage2*/ \
    VSHUFF64X2(ZMM(4), ZMM(6), ZMM(7), IMM(0xDD)) \
    VSHUFF64X2(ZMM(5), ZMM(2), ZMM(3), IMM(0xDD)) \
    /*Stage3  3,7*/ \
    VUNPCKHPD(ZMM(6), ZMM(0 ), ZMM(R1)) \
    VUNPCKHPD(ZMM(7), ZMM(R2), ZMM(R3)) \
    VUNPCKHPD(ZMM(2), ZMM(1 ), ZMM(R5)) \
    VUNPCKHPD(ZMM(3), ZMM(R6), ZMM(R7)) \
    VSHUFF64X2(ZMM(R2), ZMM(4), ZMM(5), IMM(0x88)) \
    VSHUFF64X2(ZMM(R6), ZMM(4), ZMM(5), IMM(0xDD)) \
    \
    /*Stage2*/ \
    VSHUFF64X2(ZMM(4), ZMM(6), ZMM(7), IMM(0x88)) \
    VSHUFF64X2(ZMM(5), ZMM(2), ZMM(3), IMM(0x88)) \
    /*Stage3  2,6*/ \
    VSHUFF64X2(ZMM(R1), ZMM(4), ZMM(5), IMM(0x88)) \
    VSHUFF64X2(ZMM(R5), ZMM(4), ZMM(5), IMM(0xDD)) \
    /*Stage2*/ \
    VSHUFF64X2(ZMM(4), ZMM(6), ZMM(7), IMM(0xDD)) \
    VSHUFF64X2(ZMM(5), ZMM(2), ZMM(3), IMM(0xDD)) \
    /*Stage3  4,8*/ \
    VSHUFF64X2(ZMM(R3), ZMM(4), ZMM(5), IMM(0x88)) \
    VSHUFF64X2(ZMM(R7), ZMM(4), ZMM(5), IMM(0xDD)) \

// Update C when C is column stored
#define UPDATE_C_COL_STORE(R0, R1, R2, R3, R4, R5, R6, R7) \
    \
    VMOVUPD(MEM(RCX), ZMM(R0)) \
    VMOVUPD(MEM(RCX, R10, 1), ZMM(R1)) \
    VMOVUPD(MEM(RCX, R10, 2), ZMM(R2)) \
    VMOVUPD(MEM(RCX, R11, 1), ZMM(R3)) \
    VMOVUPD(MEM(RCX, R10, 4), ZMM(R4)) \
    VMOVUPD(MEM(RCX, R12, 1), ZMM(R5)) \
    VMOVUPD(MEM(RCX, R11, 2), ZMM(R6)) \
    VMOVUPD(MEM(RCX, R13, 1), ZMM(R7)) \
    LEA(RCX, MEM(RCX,R10,8))

#define SUBITER(n) \
\
    VBROADCASTSD(ZMM(6), MEM(RAX,(8*n+ 0)*8)) \
    VBROADCASTSD(ZMM(7), MEM(RAX,(8*n+ 1)*8)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(6)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(6)) \
    VFMADD231PD(ZMM(10), ZMM(2), ZMM(6)) \
    \
    VBROADCASTSD(ZMM(6), MEM(RAX,(8*n+ 2)*8)) \
    VFMADD231PD(ZMM(11), ZMM(0), ZMM(7)) \
    VFMADD231PD(ZMM(12), ZMM(1), ZMM(7)) \
    VFMADD231PD(ZMM(13), ZMM(2), ZMM(7)) \
    \
    VBROADCASTSD(ZMM(7), MEM(RAX,(8*n+ 3)*8)) \
    VFMADD231PD(ZMM(14), ZMM(0), ZMM(6)) \
    VFMADD231PD(ZMM(15), ZMM(1), ZMM(6)) \
    VFMADD231PD(ZMM(16), ZMM(2), ZMM(6)) \
    \
    VBROADCASTSD(ZMM(6), MEM(RAX,(8*n+ 4)*8)) \
    VFMADD231PD(ZMM(17), ZMM(0), ZMM(7)) \
    VFMADD231PD(ZMM(18), ZMM(1), ZMM(7)) \
    VFMADD231PD(ZMM(19), ZMM(2), ZMM(7)) \
    \
    VBROADCASTSD(ZMM(7), MEM(RAX,(8*n+ 5)*8)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(6)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(6)) \
    VFMADD231PD(ZMM(22), ZMM(2), ZMM(6)) \
    \
    VBROADCASTSD(ZMM(6), MEM(RAX,(8*n+ 6)*8)) \
    VFMADD231PD(ZMM(23), ZMM(0), ZMM(7)) \
    VFMADD231PD(ZMM(24), ZMM(1), ZMM(7)) \
    VFMADD231PD(ZMM(25), ZMM(2), ZMM(7)) \
    \
    VBROADCASTSD(ZMM(7), MEM(RAX,(8*n+ 7)*8)) \
    VFMADD231PD(ZMM(26), ZMM(0), ZMM(6)) \
    VFMADD231PD(ZMM(27), ZMM(1), ZMM(6)) \
    VFMADD231PD(ZMM(28), ZMM(2), ZMM(6)) \
    \
    VFMADD231PD(ZMM(29), ZMM(0), ZMM(7)) \
    VFMADD231PD(ZMM(30), ZMM(1), ZMM(7)) \
    VFMADD231PD(ZMM(31), ZMM(2), ZMM(7)) \
    \
    VMOVAPD(ZMM(0), MEM(RBX,(24*n+0)*8)) \
    VMOVAPD(ZMM(1), MEM(RBX,(24*n+8)*8)) \
    VMOVAPD(ZMM(2), MEM(RBX,(24*n+16)*8)) \

//This is an array used for the scatter/gather instructions.
static int64_t offsets[24] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};

/*
 * number of accumulation registers = 24/8 * 8 = 24     zmm8 to zmm31
 * number of registers used for load B = 24/8 = 3       zmm0 to zmm2
 * number of regusters used for broadcast A = 2         zmm6 and zmm7
 */
void bli_dgemmtrsm_u_zen4_asm_8x24
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
    const int64_t* offsetPtr = &offsets[0];
    const int64_t k = k_;
    const int64_t rs_c = rs_c_*8; //convert strides to bytes
    const int64_t cs_c = cs_c_*8; //convert strides to bytes

    BEGIN_ASM()

    VXORPD(YMM(8) , YMM(8), YMM(8))
    VXORPD(YMM(9) , YMM(9), YMM(9))
    VXORPD(YMM(10), YMM(10), YMM(10)) //clear out registers
    VXORPD(YMM(11), YMM(11), YMM(11)) //clear out registers
    VMOVAPD(YMM(12), YMM(8))
    VMOVAPD(YMM(13), YMM(8))
    VMOVAPD(YMM(14), YMM(8))
    VMOVAPD(YMM(15), YMM(8))
    VXORPD(YMM(16), YMM(16), YMM(16)) //clear out registers
    VXORPD(YMM(17), YMM(17), YMM(17)) //clear out registers
    VMOVAPD(YMM(18), YMM(8))
    VMOVAPD(YMM(19), YMM(8))
    VMOVAPD(YMM(20), YMM(8))
    VMOVAPD(YMM(21), YMM(8))
    VXORPD(YMM(22), YMM(22), YMM(22)) //clear out registers
    VXORPD(YMM(23), YMM(23), YMM(23)) //clear out registers
    VMOVAPD(YMM(24), YMM(8))
    VMOVAPD(YMM(25), YMM(8))
    VMOVAPD(YMM(26), YMM(8))
    VMOVAPD(YMM(27), YMM(8))
    VXORPD(YMM(28), YMM(28), YMM(28)) //clear out registers
    VXORPD(YMM(29), YMM(29), YMM(29)) //clear out registers
    VMOVAPD(YMM(30), YMM(8))
    VMOVAPD(YMM(31), YMM(8))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a10)) //load address of a
    MOV(RBX, VAR(b01)) //load address of b
    MOV(RCX, VAR(b11)) //load address of c
    MOV(R15, VAR(c11))

    LEA(R9, MEM(R15,63)) // c for prefetching

    VMOVAPD(ZMM(0), MEM(RBX, 0*8)) //pre-load b
    VMOVAPD(ZMM(1), MEM(RBX, 8*8)) //pre-load b
    VMOVAPD(ZMM(2), MEM(RBX,16*8)) //pre-load b

    LEA(RBX, MEM(RBX,24*8)) //adjust b for pre-load

    MOV(R12, VAR(rs_c))
    MOV(R10, VAR(cs_c))

    MOV(R11, IMM(8))  // prefetch loop count
                      // r11 = NR for row store
                      // r11 = MR for col store
    MOV(R8, R12)      // prefetch loop increment
                      // r8 = cs_c for row store
                      // r8 = rs_c for col store
    MOV(R13, IMM(64)) // r13 = 0  for row store
                      // r13 = 64 for col store
    CMP(R10, IMM(8))  // jmp if c row stor
    JZ(POST_STRIDE)
        MOV(R8 , R10) // r8 = cs_c  -  prefetch loop increment
        MOV(R11, IMM(24)) // r11 = 24  -  prefetch loop count
        MOV(R13, IMM(0)) // r13 = 0

    LABEL(POST_STRIDE)

    MOV(RDI, RSI) // RDI = k
    AND(RSI, IMM(3)) // RSI = k & 3, RSI = k % 4
    SAR(RDI, IMM(2)) // RSI = k >> 2, RSI = k / 4

    SUB(RDI, R11) // subtract prefetch loop count
    SUB(RDI, IMM(0+TAIL_NITER)) // '0+' needed for preprocessor
    JLE(K_LE_80)

        LOOP_ALIGN
        LABEL(LOOP1)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*8*8))
            LEA(RBX, MEM(RBX,4*24*8))

        JNZ(LOOP1)

    LABEL(K_LE_80)

    ADD(RDI, R11) // add prefetch loop count
    JLE(K_LE_26)

        LOOP_ALIGN
        LABEL(LOOP2)

            PREFETCH(0, MEM(R9))
            SUBITER(0)
            PREFETCH(0, MEM(R9,R13, 1)) // prefetch R9+64 if col store,
                                        // prefetch R9+0 if row store
            SUBITER(1)
            SUB(RDI, IMM(1))
            PREFETCH(0, MEM(R9,R13, 2)) // prefetch R9+128 if col store,
                                        // prefetch R9+0 if row store
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*8*8))
            LEA(RBX, MEM(RBX,4*24*8))
            LEA(R9, MEM(R9,R8,1))       // r9 += rs_c if col store,
                                        // r9 += cs_c if row store

        JNZ(LOOP2)

    LABEL(K_LE_26)

    ADD(RDI, IMM(0+TAIL_NITER))
    JLE(TAIL)

        LOOP_ALIGN
        LABEL(LOOP3)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*8*8))
            LEA(RBX, MEM(RBX,4*24*8))

        JNZ(LOOP3)

    LABEL(TAIL)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

        LOOP_ALIGN
        LABEL(TAIL_LOOP)

            SUB(RSI, IMM(1))
            SUBITER(0)

            LEA(RAX, MEM(RAX,8*8))
            LEA(RBX, MEM(RBX,24*8))

        JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RBX, VAR(alpha))
    VBROADCASTSD(ZMM(3), MEM(RBX))
    MOV(RSI, IMM(1*8))

    LEA(RAX, MEM(RCX, RSI, 8))
    LEA(RDX, MEM(RAX, RSI, 8))

    MOV(R13, RCX)
    MOV(R14, RAX)
    MOV(R15, RDX)

// #region - TRSM
    MOV(RDI, IMM(24 * 8))

    VFMSUB231PD(ZMM(8 ), ZMM(3), MEM(RCX))
    ADD(RCX, RDI)
    VFMSUB231PD(ZMM(9 ), ZMM(3), MEM(RAX))
    ADD(RAX, RDI)
    VFMSUB231PD(ZMM(10), ZMM(3), MEM(RDX))
    ADD(RDX, RDI)

    VFMSUB231PD(ZMM(11), ZMM(3), MEM(RCX))
    ADD(RCX, RDI)
    VFMSUB231PD(ZMM(12), ZMM(3), MEM(RAX))
    ADD(RAX, RDI)
    VFMSUB231PD(ZMM(13), ZMM(3), MEM(RDX))
    ADD(RDX, RDI)

    VFMSUB231PD(ZMM(14), ZMM(3), MEM(RCX))
    ADD(RCX, RDI)
    VFMSUB231PD(ZMM(15), ZMM(3), MEM(RAX))
    ADD(RAX, RDI)
    VFMSUB231PD(ZMM(16), ZMM(3), MEM(RDX))
    ADD(RDX, RDI)

    VFMSUB231PD(ZMM(17), ZMM(3), MEM(RCX))
    ADD(RCX, RDI)
    VFMSUB231PD(ZMM(18), ZMM(3), MEM(RAX))
    ADD(RAX, RDI)
    VFMSUB231PD(ZMM(19), ZMM(3), MEM(RDX))
    ADD(RDX, RDI)

    VFMSUB231PD(ZMM(20), ZMM(3), MEM(RCX))
    ADD(RCX, RDI)
    VFMSUB231PD(ZMM(21), ZMM(3), MEM(RAX))
    ADD(RAX, RDI)
    VFMSUB231PD(ZMM(22), ZMM(3), MEM(RDX))
    ADD(RDX, RDI)

    VFMSUB231PD(ZMM(23), ZMM(3), MEM(RCX))
    ADD(RCX, RDI)
    VFMSUB231PD(ZMM(24), ZMM(3), MEM(RAX))
    ADD(RAX, RDI)
    VFMSUB231PD(ZMM(25), ZMM(3), MEM(RDX))
    ADD(RDX, RDI)

    VFMSUB231PD(ZMM(26), ZMM(3), MEM(RCX))
    ADD(RCX, RDI)
    VFMSUB231PD(ZMM(27), ZMM(3), MEM(RAX))
    ADD(RAX, RDI)
    VFMSUB231PD(ZMM(28), ZMM(3), MEM(RDX))
    ADD(RDX, RDI)

    VFMSUB231PD(ZMM(29), ZMM(3), MEM(RCX))
    VFMSUB231PD(ZMM(30), ZMM(3), MEM(RAX))
    VFMSUB231PD(ZMM(31), ZMM(3), MEM(RDX))

    LEA(RAX, MEM(RDI, RDI, 2))
    LEA(RAX, MEM(RAX, RDI, 4))
    ADD(R13, RAX)
    ADD(R14, RAX)
    ADD(R15, RAX)

    MOV(RAX, VAR(a11))
    //iteration 0 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (7+7*8)*8))
    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM(29), ZMM(29), ZMM(0))
        VMULPD(ZMM(30), ZMM(30), ZMM(0))
        VMULPD(ZMM(31), ZMM(31), ZMM(0))
    #else
        VDIVPD(ZMM(29), ZMM(29), ZMM(0))
        VDIVPD(ZMM(30), ZMM(30), ZMM(0))
        VDIVPD(ZMM(31), ZMM(31), ZMM(0))
    #endif
    VMOVUPD(MEM(R13), ZMM(29))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM(30))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(31))
    SUB(R15, RDI)

    //iteration 1 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (6+7*8)*8))
    VBROADCASTSD(ZMM(1), MEM(RAX, (6+6*8)*8))

    VMULPD(ZMM(2), ZMM(29), ZMM(0))
    VMULPD(ZMM(3), ZMM(30), ZMM(0))
    VMULPD(ZMM(4), ZMM(31), ZMM(0))

    VSUBPD(ZMM(26), ZMM(26), ZMM(2))
    VSUBPD(ZMM(27), ZMM(27), ZMM(3))
    VSUBPD(ZMM(28), ZMM(28), ZMM(4))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM(26), ZMM(26), ZMM(1))
        VMULPD(ZMM(27), ZMM(27), ZMM(1))
        VMULPD(ZMM(28), ZMM(28), ZMM(1))
    #else
        VDIVPD(ZMM(26), ZMM(26), ZMM(1))
        VDIVPD(ZMM(27), ZMM(27), ZMM(1))
        VDIVPD(ZMM(28), ZMM(28), ZMM(1))
    #endif
    VMOVUPD(MEM(R13), ZMM(26))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM(27))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(28))
    SUB(R15, RDI)

    //iteration 2 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (5+7*8)*8))
    VBROADCASTSD(ZMM(1), MEM(RAX, (5+6*8)*8))

    VMULPD(ZMM(2), ZMM(29), ZMM(0))
    VMULPD(ZMM(3), ZMM(30), ZMM(0))
    VMULPD(ZMM(4), ZMM(31), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (5+5*8)*8))
    VFMADD231PD(ZMM(2), ZMM(26), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(27), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(28), ZMM(1))

    VSUBPD(ZMM(23), ZMM(23), ZMM(2))
    VSUBPD(ZMM(24), ZMM(24), ZMM(3))
    VSUBPD(ZMM(25), ZMM(25), ZMM(4))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM(23), ZMM(23), ZMM(0))
        VMULPD(ZMM(24), ZMM(24), ZMM(0))
        VMULPD(ZMM(25), ZMM(25), ZMM(0))
    #else
        VDIVPD(ZMM(23), ZMM(23), ZMM(0))
        VDIVPD(ZMM(24), ZMM(24), ZMM(0))
        VDIVPD(ZMM(25), ZMM(25), ZMM(0))
    #endif
    VMOVUPD(MEM(R13), ZMM(23))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM(24))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(25))
    SUB(R15, RDI)

    //iteration 3 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (4+7*8)*8))
    VBROADCASTSD(ZMM(1), MEM(RAX, (4+6*8)*8))

    VMULPD(ZMM(2), ZMM(29), ZMM(0))
    VMULPD(ZMM(3), ZMM(30), ZMM(0))
    VMULPD(ZMM(4), ZMM(31), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (4+5*8)*8))
    VFMADD231PD(ZMM(2), ZMM(26), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(27), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(28), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (4+4*8)*8))
    VFMADD231PD(ZMM(2), ZMM(23), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(24), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(25), ZMM(0))

    VSUBPD(ZMM(20), ZMM(20), ZMM(2))
    VSUBPD(ZMM(21), ZMM(21), ZMM(3))
    VSUBPD(ZMM(22), ZMM(22), ZMM(4))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM(20), ZMM(20), ZMM(1))
        VMULPD(ZMM(21), ZMM(21), ZMM(1))
        VMULPD(ZMM(22), ZMM(22), ZMM(1))
    #else
        VDIVPD(ZMM(20), ZMM(20), ZMM(1))
        VDIVPD(ZMM(21), ZMM(21), ZMM(1))
        VDIVPD(ZMM(22), ZMM(22), ZMM(1))
    #endif
    VMOVUPD(MEM(R13), ZMM(20))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM(21))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(22))
    SUB(R15, RDI)

    //iteration 4 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (3+7*8)*8))
    VBROADCASTSD(ZMM(1), MEM(RAX, (3+6*8)*8))

    VMULPD(ZMM(2), ZMM(29), ZMM(0))
    VMULPD(ZMM(3), ZMM(30), ZMM(0))
    VMULPD(ZMM(4), ZMM(31), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (3+5*8)*8))
    VFMADD231PD(ZMM(2), ZMM(26), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(27), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(28), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (3+4*8)*8))
    VFMADD231PD(ZMM(2), ZMM(23), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(24), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(25), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (3+3*8)*8))
    VFMADD231PD(ZMM(2), ZMM(20), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(21), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(22), ZMM(1))

    VSUBPD(ZMM(17), ZMM(17), ZMM(2))
    VSUBPD(ZMM(18), ZMM(18), ZMM(3))
    VSUBPD(ZMM(19), ZMM(19), ZMM(4))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM(17), ZMM(17), ZMM(0))
        VMULPD(ZMM(18), ZMM(18), ZMM(0))
        VMULPD(ZMM(19), ZMM(19), ZMM(0))
    #else
        VDIVPD(ZMM(17), ZMM(17), ZMM(0))
        VDIVPD(ZMM(18), ZMM(18), ZMM(0))
        VDIVPD(ZMM(19), ZMM(19), ZMM(0))
    #endif
    VMOVUPD(MEM(R13), ZMM(17))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM(18))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(19))
    SUB(R15, RDI)

    //iteration 5 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (2+7*8)*8))
    VBROADCASTSD(ZMM(1), MEM(RAX, (2+6*8)*8))

    VMULPD(ZMM(2), ZMM(29), ZMM(0))
    VMULPD(ZMM(3), ZMM(30), ZMM(0))
    VMULPD(ZMM(4), ZMM(31), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (2+5*8)*8))
    VFMADD231PD(ZMM(2), ZMM(26), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(27), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(28), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (2+4*8)*8))
    VFMADD231PD(ZMM(2), ZMM(23), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(24), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(25), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (2+3*8)*8))
    VFMADD231PD(ZMM(2), ZMM(20), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(21), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(22), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (2+2*8)*8))
    VFMADD231PD(ZMM(2), ZMM(17), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(18), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(19), ZMM(0))

    VSUBPD(ZMM(14), ZMM(14), ZMM(2))
    VSUBPD(ZMM(15), ZMM(15), ZMM(3))
    VSUBPD(ZMM(16), ZMM(16), ZMM(4))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM(14), ZMM(14), ZMM(1))
        VMULPD(ZMM(15), ZMM(15), ZMM(1))
        VMULPD(ZMM(16), ZMM(16), ZMM(1))
    #else
        VDIVPD(ZMM(14), ZMM(14), ZMM(1))
        VDIVPD(ZMM(15), ZMM(15), ZMM(1))
        VDIVPD(ZMM(16), ZMM(16), ZMM(1))
    #endif
    VMOVUPD(MEM(R13), ZMM(14))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM(15))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(16))
    SUB(R15, RDI)

    //iteration 6 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (1+7*8)*8))
    VBROADCASTSD(ZMM(1), MEM(RAX, (1+6*8)*8))

    VMULPD(ZMM(2), ZMM(29), ZMM(0))
    VMULPD(ZMM(3), ZMM(30), ZMM(0))
    VMULPD(ZMM(4), ZMM(31), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (1+5*8)*8))
    VFMADD231PD(ZMM(2), ZMM(26), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(27), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(28), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (1+4*8)*8))
    VFMADD231PD(ZMM(2), ZMM(23), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(24), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(25), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (1+3*8)*8))
    VFMADD231PD(ZMM(2), ZMM(20), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(21), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(22), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (1+2*8)*8))
    VFMADD231PD(ZMM(2), ZMM(17), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(18), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(19), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (1+1*8)*8))
    VFMADD231PD(ZMM(2), ZMM(14), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(15), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(16), ZMM(1))

    VSUBPD(ZMM(11), ZMM(11), ZMM(2))
    VSUBPD(ZMM(12), ZMM(12), ZMM(3))
    VSUBPD(ZMM(13), ZMM(13), ZMM(4))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM(11), ZMM(11), ZMM(0))
        VMULPD(ZMM(12), ZMM(12), ZMM(0))
        VMULPD(ZMM(13), ZMM(13), ZMM(0))
    #else
        VDIVPD(ZMM(11), ZMM(11), ZMM(0))
        VDIVPD(ZMM(12), ZMM(12), ZMM(0))
        VDIVPD(ZMM(13), ZMM(13), ZMM(0))
    #endif
    VMOVUPD(MEM(R13), ZMM(11))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM(12))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(13))
    SUB(R15, RDI)

    //iteration 7 --------------------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (0+7*8)*8))
    VBROADCASTSD(ZMM(1), MEM(RAX, (0+6*8)*8))

    VMULPD(ZMM(2), ZMM(29), ZMM(0))
    VMULPD(ZMM(3), ZMM(30), ZMM(0))
    VMULPD(ZMM(4), ZMM(31), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (0+5*8)*8))
    VFMADD231PD(ZMM(2), ZMM(26), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(27), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(28), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (0+4*8)*8))
    VFMADD231PD(ZMM(2), ZMM(23), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(24), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(25), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (0+3*8)*8))
    VFMADD231PD(ZMM(2), ZMM(20), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(21), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(22), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (0+2*8)*8))
    VFMADD231PD(ZMM(2), ZMM(17), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(18), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(19), ZMM(0))

    VBROADCASTSD(ZMM(0), MEM(RAX, (0+1*8)*8))
    VFMADD231PD(ZMM(2), ZMM(14), ZMM(1))
    VFMADD231PD(ZMM(3), ZMM(15), ZMM(1))
    VFMADD231PD(ZMM(4), ZMM(16), ZMM(1))

    VBROADCASTSD(ZMM(1), MEM(RAX, (0+0*8)*8))
    VFMADD231PD(ZMM(2), ZMM(11), ZMM(0))
    VFMADD231PD(ZMM(3), ZMM(12), ZMM(0))
    VFMADD231PD(ZMM(4), ZMM(13), ZMM(0))

    VSUBPD(ZMM( 8), ZMM( 8), ZMM(2))
    VSUBPD(ZMM( 9), ZMM( 9), ZMM(3))
    VSUBPD(ZMM(10), ZMM(10), ZMM(4))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VMULPD(ZMM( 8), ZMM( 8), ZMM(1))
        VMULPD(ZMM( 9), ZMM( 9), ZMM(1))
        VMULPD(ZMM(10), ZMM(10), ZMM(1))
    #else
        VDIVPD(ZMM( 8), ZMM( 8), ZMM(1))
        VDIVPD(ZMM( 9), ZMM( 9), ZMM(1))
        VDIVPD(ZMM(10), ZMM(10), ZMM(1))
    #endif
    VMOVUPD(MEM(R13), ZMM( 8))
    SUB(R13, RDI)
    VMOVUPD(MEM(R14), ZMM( 9))
    SUB(R14, RDI)
    VMOVUPD(MEM(R15), ZMM(10))
    SUB(R15, RDI)

// #endregion - trsm

    MOV(RAX, R12)
    MOV(RBX, R10)
    MOV(RCX, VAR(c11))

    CMP(RAX, IMM(8))
    JE(COLUPDATE)

    CMP(RBX, IMM(8))
    JE(ROWUPDATE)

    LABEL(SCATTERUPDATE)
        MOV(RDI, VAR(offsetPtr))
        VPBROADCASTQ(ZMM(0), R10)
        VPMULLQ(ZMM(2), ZMM(0), MEM(RDI))
        VPMULLQ(ZMM(3), ZMM(0), MEM(RDI, 8*8))
        VPMULLQ(ZMM(4), ZMM(0), MEM(RDI,16*8))
        UPDATE_C_SCATTERED( 8,  9, 10)
        UPDATE_C_SCATTERED(11, 12, 13)
        UPDATE_C_SCATTERED(14, 15, 16)
        UPDATE_C_SCATTERED(17, 18, 19)
        UPDATE_C_SCATTERED(20, 21, 22)
        UPDATE_C_SCATTERED(23, 24, 25)
        UPDATE_C_SCATTERED(26, 27, 28)
        UPDATE_C_SCATTERED(29, 30, 31)
        JMP(END)
    LABEL(ROWUPDATE)
        LEA(R11, MEM(RAX, RAX, 2)) //R11 = rs_c * 3, R11 = rs_c + rs_c * 2
        LEA(R12, MEM(RAX, RAX, 4)) //R12 = rs_c * 5, R12 = rs_c + rs_c * 4
        LEA(R13, MEM(RAX, R11, 2)) //R13 = rs_c * 7, R13 = rs_c + R11 * 2

        // ROW0
        VMOVUPD(MEM(RCX     ), ZMM( 8))
        VMOVUPD(MEM(RCX, 64 ), ZMM( 9))
        VMOVUPD(MEM(RCX, 128), ZMM(10))

        // ROW1
        VMOVUPD(MEM(RCX, RAX, 1,    ), ZMM(11))
        VMOVUPD(MEM(RCX, RAX, 1, 64 ), ZMM(12))
        VMOVUPD(MEM(RCX, RAX, 1, 128), ZMM(13))

        // ROW2
        VMOVUPD(MEM(RCX, RAX, 2,    ), ZMM(14))
        VMOVUPD(MEM(RCX, RAX, 2, 64 ), ZMM(15))
        VMOVUPD(MEM(RCX, RAX, 2, 128), ZMM(16))

        // ROW3
        VMOVUPD(MEM(RCX, R11, 1,    ), ZMM(17))
        VMOVUPD(MEM(RCX, R11, 1, 64 ), ZMM(18))
        VMOVUPD(MEM(RCX, R11, 1, 128), ZMM(19))

        // ROW4
        VMOVUPD(MEM(RCX, RAX, 4,    ), ZMM(20))
        VMOVUPD(MEM(RCX, RAX, 4, 64 ), ZMM(21))
        VMOVUPD(MEM(RCX, RAX, 4, 128), ZMM(22))

        // ROW5
        VMOVUPD(MEM(RCX, R12, 1,    ), ZMM(23))
        VMOVUPD(MEM(RCX, R12, 1, 64 ), ZMM(24))
        VMOVUPD(MEM(RCX, R12, 1, 128), ZMM(25))

        // ROW6
        VMOVUPD(MEM(RCX, R11, 2,    ), ZMM(26))
        VMOVUPD(MEM(RCX, R11, 2, 64 ), ZMM(27))
        VMOVUPD(MEM(RCX, R11, 2, 128), ZMM(28))

        // ROW7
        VMOVUPD(MEM(RCX, R13, 1,    ), ZMM(29))
        VMOVUPD(MEM(RCX, R13, 1, 64 ), ZMM(30))
        VMOVUPD(MEM(RCX, R13, 1, 128), ZMM(31))
        JMP(END)

    LABEL(COLUPDATE)
        LEA(R11, MEM(R10, R10, 2)) //R11 = cs_c * 3
        LEA(R12, MEM(R10, R10, 4)) //R12 = cs_c * 5
        LEA(R13, MEM(R10, R11, 2)) //R13 = cs_c * 7
        TRANSPOSE_8X8( 8, 11, 14, 17, 20, 23, 26, 29)
        TRANSPOSE_8X8( 9, 12, 15, 18, 21, 24, 27, 30)
        TRANSPOSE_8X8(10, 13, 16, 19, 22, 25, 28, 31)
        UPDATE_C_COL_STORE( 8, 11, 14, 17, 20, 23, 26, 29)
        UPDATE_C_COL_STORE( 9, 12, 15, 18, 21, 24, 27, 30)
        UPDATE_C_COL_STORE(10, 13, 16, 19, 22, 25, 28, 31)

    LABEL(END)

    VZEROUPPER()

    END_ASM(
        : // output operands (none)
        : // input operands
        [a10]       "m" (a10),
        [k]         "m" (k),
        [b01]       "m" (b01),
        [a11]       "m" (a11),
        [b11]       "m" (b11),
        [c11]       "m" (c11),
        [rs_c]      "m" (rs_c),
        [cs_c]      "m" (cs_c),
        [alpha]     "m" (alpha),
        [offsetPtr] "m" (offsetPtr)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
    )
}
