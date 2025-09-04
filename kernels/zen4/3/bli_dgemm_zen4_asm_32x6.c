/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "bli_x86_asm_macros.h"

#define TAIL_NITER 5 // in units of 4x unrolled k iterations
                     // e.g. 5 -> 4*5 k iterations ~= 280 cycles


/*
 * A Registers:  ZMM0, ZMM1, ZMM2, ZMM3
 * B Registers:  ZMM4, xMM5, xMM6, xMM7
 * C Registers:  ZMM[8-31]
 */

#define LOOP_ALIGN ALIGN32

#define UPDATE_C(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX)) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,64)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,128)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,192)) \
    VMOVUPD(MEM(RCX), ZMM(R1)) \
    VMOVUPD(MEM(RCX,64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,128), ZMM(R3)) \
    VMOVUPD(MEM(RCX,192), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RBX,1))

#define UPDATE_C_BZ(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX), ZMM(R1)) \
    VMOVUPD(MEM(RCX,64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,128), ZMM(R3)) \
    VMOVUPD(MEM(RCX,192), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RBX,1))


#define UPDATE_C_COL_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VGATHERQPD(ZMM(0) MASK_K(1), MEM(RCX,ZMM(4),1)) \
    VFMADD231PD(ZMM(R1), ZMM(0), ZMM(1)) \
    VGATHERQPD(ZMM(0) MASK_K(2), MEM(RCX,ZMM(5),1)) \
    VFMADD231PD(ZMM(R2), ZMM(0), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(4),1) MASK_K(3), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(5),1) MASK_K(4), ZMM(R2)) \
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VGATHERQPD(ZMM(0) MASK_K(1), MEM(RCX,ZMM(6),1)) \
    VFMADD231PD(ZMM(R3), ZMM(0), ZMM(1)) \
    VGATHERQPD(ZMM(0) MASK_K(2), MEM(RCX,ZMM(7),1)) \
    VFMADD231PD(ZMM(R4), ZMM(0), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(6),1) MASK_K(3), ZMM(R3))  \
    VSCATTERQPD(MEM(RCX,ZMM(7),1) MASK_K(4), ZMM(R4))  \
    LEA(RCX, MEM(RCX,RBX,1))

#define UPDATE_C_BZ_COL_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(4),1) MASK_K(1), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(5),1) MASK_K(2), ZMM(R2)) \
    VSCATTERQPD(MEM(RCX,ZMM(6),1) MASK_K(3), ZMM(R3)) \
    VSCATTERQPD(MEM(RCX,ZMM(7),1) MASK_K(4), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RBX,1))


#if 0
#define SUBITER(n) \
\
    VMOVUPD(XMM( 5), MEM(RBX,(6*n+ 0)*8)) \
    \
    VBROADCASTSD(ZMM( 4), XMM( 5)) \
    VPERMILPD(XMM( 5), XMM( 5), IMM(3)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(10), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(11), ZMM(3), ZMM(4)) \
    \
    VBROADCASTSD(ZMM( 4), XMM( 5)) \
    VMOVUPD(XMM( 6), MEM(RBX,(6*n+ 2)*8)) \
    VFMADD231PD(ZMM(12), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(13), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(14), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(15), ZMM(3), ZMM(4)) \
    \
    VBROADCASTSD(ZMM( 4), XMM( 6)) \
    VPERMILPD(XMM( 6), XMM( 6), IMM(3)) \
    VFMADD231PD(ZMM(16), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(17), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(18), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(19), ZMM(3), ZMM(4)) \
    \
    VBROADCASTSD(ZMM( 4), XMM( 6)) \
    VMOVUPD(XMM( 7), MEM(RBX,(6*n+ 4)*8)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(22), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(23), ZMM(3), ZMM(4)) \
    \
    VBROADCASTSD(ZMM( 4), XMM( 7)) \
    VPERMILPD(XMM( 7), XMM( 7), IMM(3)) \
    VFMADD231PD(ZMM(24), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(25), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(26), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(27), ZMM(3), ZMM(4)) \
    \
    VBROADCASTSD(ZMM( 4), XMM( 7)) \
    VFMADD231PD(ZMM(28), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(29), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(30), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(31), ZMM(3), ZMM(4)) \
    \
    VMOVAPD(ZMM(0), MEM(RAX,(32*n+ 0)*8)) \
    VMOVAPD(ZMM(1), MEM(RAX,(32*n+ 8)*8)) \
    VMOVAPD(ZMM(2), MEM(RAX,(32*n+16)*8)) \
    VMOVAPD(ZMM(3), MEM(RAX,(32*n+24)*8))


#else

#define SUBITER(n) \
\
    VBROADCASTSD(ZMM( 4), MEM(RBX,(6*n+ 0)*8)) \
    VBROADCASTSD(ZMM( 5), MEM(RBX,(6*n+ 1)*8)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(10), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(11), ZMM(3), ZMM(4)) \
    \
    VBROADCASTSD(ZMM( 4), MEM(RBX,(6*n+ 2)*8)) \
    VFMADD231PD(ZMM(12), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(13), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(14), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(15), ZMM(3), ZMM(5)) \
    \
    VBROADCASTSD(ZMM( 5), MEM(RBX,(6*n+ 3)*8)) \
    VFMADD231PD(ZMM(16), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(17), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(18), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(19), ZMM(3), ZMM(4)) \
    \
    VBROADCASTSD(ZMM( 4), MEM(RBX,(6*n+ 4)*8)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(22), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(23), ZMM(3), ZMM(5)) \
    \
    VBROADCASTSD(ZMM( 5), MEM(RBX,(6*n+ 5)*8)) \
    VFMADD231PD(ZMM(24), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(25), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(26), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(27), ZMM(3), ZMM(4)) \
    \
    VFMADD231PD(ZMM(28), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(29), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(30), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(31), ZMM(3), ZMM(5)) \
    \
    VMOVAPD(ZMM(0), MEM(RAX,(32*n+ 0)*8)) \
    VMOVAPD(ZMM(1), MEM(RAX,(32*n+ 8)*8)) \
    VMOVAPD(ZMM(2), MEM(RAX,(32*n+16)*8)) \
    VMOVAPD(ZMM(3), MEM(RAX,(32*n+24)*8))
#endif

//This is an array used for the scatter/gather instructions.
static int64_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

void bli_dgemm_zen4_asm_32x6(
                              dim_t            k_,
                              double* restrict alpha,
                              double* restrict a,
                              double* restrict b,
                              double* restrict beta,
                              double* restrict c, inc_t rs_c_, inc_t cs_c_,
                              auxinfo_t*       data,
                              cntx_t* restrict cntx
                            )
{
    (void)data;
    (void)cntx;

    const int64_t* offsetPtr = &offsets[0];
    const int64_t k = k_;
    const int64_t rs_c = rs_c_*8; // stride in bytes
    const int64_t cs_c = cs_c_*8; // stride in bytes

    BEGIN_ASM()

    VXORPD(YMM( 4), YMM( 4), YMM( 4)) //clear out registers
    VXORPD(YMM( 5), YMM( 5), YMM( 5)) //clear out registers
    VMOVAPD(YMM(6) , YMM(4))
    VMOVAPD(YMM(7) , YMM(4))
    VMOVAPD(YMM(8) , YMM(4))
    VMOVAPD(YMM(9) , YMM(4))
    VXORPD(YMM(10), YMM(10), YMM(10)) //clear out registers
    VXORPD(YMM(11), YMM(11), YMM(11)) //clear out registers
    VMOVAPD(YMM(12), YMM(4))
    VMOVAPD(YMM(13), YMM(4))
    VMOVAPD(YMM(14), YMM(4))
    VMOVAPD(YMM(15), YMM(4))
    VXORPD(YMM(16), YMM(16), YMM(16)) //clear out registers
    VXORPD(YMM(17), YMM(17), YMM(17)) //clear out registers
    VMOVAPD(YMM(18), YMM(4))
    VMOVAPD(YMM(19), YMM(4))
    VMOVAPD(YMM(20), YMM(4))
    VMOVAPD(YMM(21), YMM(4))
    VXORPD(YMM(22), YMM(22), YMM(22)) //clear out registers
    VXORPD(YMM(23), YMM(23), YMM(23)) //clear out registers
    VMOVAPD(YMM(24), YMM(4))
    VMOVAPD(YMM(25), YMM(4))
    VMOVAPD(YMM(26), YMM(4))
    VMOVAPD(YMM(27), YMM(4))
    VXORPD(YMM(28), YMM(28), YMM(28)) //clear out registers
    VXORPD(YMM(29), YMM(29), YMM(29)) //clear out registers
    VMOVAPD(YMM(30), YMM(4))
    VMOVAPD(YMM(31), YMM(4))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c

    //LEA(R9, MEM(RCX,63)) // c for prefetching
    MOV(R9, RCX)

    VMOVAPD(ZMM(0), MEM(RAX, 0*8)) //pre-load a
    VMOVAPD(ZMM(1), MEM(RAX, 8*8)) //pre-load a
    VMOVAPD(ZMM(2), MEM(RAX,16*8)) //pre-load a
    VMOVAPD(ZMM(3), MEM(RAX,24*8)) //pre-load a
    LEA(RAX, MEM(RAX,32*8)) //adjust a for pre-load

    MOV(R12, VAR(rs_c))
    MOV(R10, VAR(cs_c))

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    SUB(RDI, IMM(6+TAIL_NITER))
    JLE(K_LE_80)

        LOOP_ALIGN
        LABEL(LOOP1)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*32*8))
            LEA(RBX, MEM(RBX,4*6*8))


        JNZ(LOOP1)

    LABEL(K_LE_80)

    ADD(RDI, IMM(6))
    JLE(K_LE_24)

        LOOP_ALIGN
        LABEL(LOOP2)

            PREFETCH(0, MEM(R9))
            SUBITER(0)
            PREFETCH(0, MEM(R9,64))
            SUBITER(1)
            PREFETCH(0, MEM(R9,128))
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(0, MEM(R9,192))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*32*8))
            LEA(RBX, MEM(RBX,4*6*8))
            LEA(R9, MEM(R9,R10,1))

        JNZ(LOOP2)

    LABEL(K_LE_24)

    ADD(RDI, IMM(0+TAIL_NITER))
    JLE(TAIL)

        LOOP_ALIGN
        LABEL(LOOP3)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*32*8))
            LEA(RBX, MEM(RBX,4*6*8))

        JNZ(LOOP3)

    LABEL(TAIL)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

        LOOP_ALIGN
        LABEL(TAIL_LOOP)

            SUB(RSI, IMM(1))
            SUBITER(0)
            LEA(RAX, MEM(RAX,32*8))
            LEA(RBX, MEM(RBX,6*8))

        JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    VXORPD(YMM(2), YMM(2), YMM(2))

    MOV(RAX, R12)
    MOV(RBX, R10)

    // Check if C is column stride.
    CMP(RAX, IMM(8))
    JNE(SCATTEREDUPDATE)

        VCOMISD(XMM(1), XMM(2))
        JE(COLSTORBZ)

            UPDATE_C( 8, 9,10,11)
            UPDATE_C(12,13,14,15)
            UPDATE_C(16,17,18,19)
            UPDATE_C(20,21,22,23)
            UPDATE_C(24,25,26,27)
            UPDATE_C(28,29,30,31)

        JMP(END)
        LABEL(COLSTORBZ)

            UPDATE_C_BZ( 8, 9,10,11)
            UPDATE_C_BZ(12,13,14,15)
            UPDATE_C_BZ(16,17,18,19)
            UPDATE_C_BZ(20,21,22,23)
            UPDATE_C_BZ(24,25,26,27)
            UPDATE_C_BZ(28,29,30,31)

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        VMULPD(ZMM( 8), ZMM( 8), ZMM(0))
        VMULPD(ZMM( 9), ZMM( 9), ZMM(0))
        VMULPD(ZMM(10), ZMM(10), ZMM(0))
        VMULPD(ZMM(11), ZMM(11), ZMM(0))
        VMULPD(ZMM(12), ZMM(12), ZMM(0))
        VMULPD(ZMM(13), ZMM(13), ZMM(0))
        VMULPD(ZMM(14), ZMM(14), ZMM(0))
        VMULPD(ZMM(15), ZMM(15), ZMM(0))
        VMULPD(ZMM(16), ZMM(16), ZMM(0))
        VMULPD(ZMM(17), ZMM(17), ZMM(0))
        VMULPD(ZMM(18), ZMM(18), ZMM(0))
        VMULPD(ZMM(19), ZMM(19), ZMM(0))
        VMULPD(ZMM(20), ZMM(20), ZMM(0))
        VMULPD(ZMM(21), ZMM(21), ZMM(0))
        VMULPD(ZMM(22), ZMM(22), ZMM(0))
        VMULPD(ZMM(23), ZMM(23), ZMM(0))
        VMULPD(ZMM(24), ZMM(24), ZMM(0))
        VMULPD(ZMM(25), ZMM(25), ZMM(0))
        VMULPD(ZMM(26), ZMM(26), ZMM(0))
        VMULPD(ZMM(27), ZMM(27), ZMM(0))
        VMULPD(ZMM(28), ZMM(28), ZMM(0))
        VMULPD(ZMM(29), ZMM(29), ZMM(0))
        VMULPD(ZMM(30), ZMM(30), ZMM(0))
        VMULPD(ZMM(31), ZMM(31), ZMM(0))

        VCOMISD(XMM(1), XMM(2))

        MOV(RDI, VAR(offsetPtr))
        VPBROADCASTQ(ZMM(0), RAX)
        VPMULLQ(ZMM(4), ZMM(0), MEM(RDI))
        VPMULLQ(ZMM(5), ZMM(0), MEM(RDI,64))
        VPMULLQ(ZMM(6), ZMM(0), MEM(RDI,128))
        VPMULLQ(ZMM(7), ZMM(0), MEM(RDI,192))

        JE(SCATTERBZ)

            UPDATE_C_COL_SCATTERED( 8, 9,10,11)
            UPDATE_C_COL_SCATTERED(12,13,14,15)
            UPDATE_C_COL_SCATTERED(16,17,18,19)
            UPDATE_C_COL_SCATTERED(20,21,22,23)
            UPDATE_C_COL_SCATTERED(24,25,26,27)
            UPDATE_C_COL_SCATTERED(28,29,30,31)

        JMP(END)
        LABEL(SCATTERBZ)

            UPDATE_C_BZ_COL_SCATTERED( 8, 9,10,11)
            UPDATE_C_BZ_COL_SCATTERED(12,13,14,15)
            UPDATE_C_BZ_COL_SCATTERED(16,17,18,19)
            UPDATE_C_BZ_COL_SCATTERED(20,21,22,23)
            UPDATE_C_BZ_COL_SCATTERED(24,25,26,27)
            UPDATE_C_BZ_COL_SCATTERED(28,29,30,31)

    LABEL(END)

    VZEROUPPER()

    END_ASM
    (
        : // output operands
        : // input operands
          [k]         "m" (k),
          [a]         "m" (a),
          [b]         "m" (b),
          [alpha]     "m" (alpha),
          [beta]      "m" (beta),
          [c]         "m" (c),
          [rs_c]      "m" (rs_c),
          [cs_c]      "m" (cs_c),
          [offsetPtr] "m" (offsetPtr)
        : // register clobber list
          "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "k0", "k1", "k2", "k3", "k4", "xmm1",
          "xmm2", "ymm2", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9",
          "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "ymm16",
          "ymm17", "ymm18", "ymm19", "ymm20", "ymm21", "ymm22", "ymm23",
          "ymm24", "ymm25", "ymm26", "ymm27", "ymm28", "ymm29", "ymm30",
          "ymm31", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
    )
}
