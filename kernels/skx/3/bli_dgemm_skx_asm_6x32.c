/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived from this software without specific prior written permission.

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

#define A_L1_PREFETCH_DIST 4 // in units of k iterations
#define B_L1_PREFETCH_DIST 4 // e.g. 4 k iterations ~= 48 cycles
#define TAIL_NITER 6 // in units of 4x unrolled k iterations
                     // e.g. 6 -> 4*6 k iterations ~= 288 cycles

#define PREFETCH_A_L1(n, k) \
    PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*6*8 + (n+k)*48))
#define PREFETCH_B_L1(n, k) \
    PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*32*8 + (4*n+k)*64))

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
    LEA(RCX, MEM(RCX,RAX,1))

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
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_ROW_SCATTERED(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),1)) \
    VGATHERQPD(ZMM(7) MASK_K(2), MEM(RCX,ZMM(3),1)) \
    VFMADD231PD(ZMM(R1), ZMM(6), ZMM(1)) \
    VFMADD231PD(ZMM(R2), ZMM(7), ZMM(1)) \
    VGATHERQPD(ZMM(6) MASK_K(3), MEM(RCX,ZMM(4),1)) \
    VGATHERQPD(ZMM(7) MASK_K(4), MEM(RCX,ZMM(5),1)) \
    VFMADD231PD(ZMM(R3), ZMM(6), ZMM(1)) \
    VFMADD231PD(ZMM(R4), ZMM(7), ZMM(1)) \
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R2)) \
    VSCATTERQPD(MEM(RCX,ZMM(4),1) MASK_K(3), ZMM(R3)) \
    VSCATTERQPD(MEM(RCX,ZMM(5),1) MASK_K(4), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_BZ_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    KXNORW(K(3), K(0), K(0)) \
    KXNORW(K(4), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),1) MASK_K(1), ZMM(R1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),1) MASK_K(2), ZMM(R2)) \
    VSCATTERQPD(MEM(RCX,ZMM(4),1) MASK_K(3), ZMM(R3)) \
    VSCATTERQPD(MEM(RCX,ZMM(5),1) MASK_K(4), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,1))

#define SUBITER(n) \
\
    PREFETCH_A_L1(n, 0) \
    PREFETCH_B_L1(n, 0) \
    \
    VBROADCASTSD(ZMM(4), MEM(RAX,(6*n+0)*8)) \
    VBROADCASTSD(ZMM(5), MEM(RAX,(6*n+1)*8)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(10), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(11), ZMM(3), ZMM(4)) \
    VFMADD231PD(ZMM(12), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(13), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(14), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(15), ZMM(3), ZMM(5)) \
    \
    PREFETCH_B_L1(n, 1) \
    \
    VBROADCASTSD(ZMM(4), MEM(RAX,(6*n+2)*8)) \
    VBROADCASTSD(ZMM(5), MEM(RAX,(6*n+3)*8)) \
    VFMADD231PD(ZMM(16), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(17), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(18), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(19), ZMM(3), ZMM(4)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(22), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(23), ZMM(3), ZMM(5)) \
    \
    PREFETCH_B_L1(n, 2) \
    PREFETCH_B_L1(n, 3) \
    \
    VBROADCASTSD(ZMM(4), MEM(RAX,(6*n+4)*8)) \
    VBROADCASTSD(ZMM(5), MEM(RAX,(6*n+5)*8)) \
    VFMADD231PD(ZMM(24), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(25), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(26), ZMM(2), ZMM(4)) \
    VFMADD231PD(ZMM(27), ZMM(3), ZMM(4)) \
    VFMADD231PD(ZMM(28), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(29), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(30), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(31), ZMM(3), ZMM(5)) \
    \
    VMOVAPD(ZMM(0), MEM(RBX,(32*n+ 0)*8)) \
    VMOVAPD(ZMM(1), MEM(RBX,(32*n+ 8)*8)) \
    VMOVAPD(ZMM(2), MEM(RBX,(32*n+16)*8)) \
    VMOVAPD(ZMM(3), MEM(RBX,(32*n+24)*8))

//This is an array used for the scatter/gather instructions.
static int64_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

void bli_dgemm_skx_asm_6x32(
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
    const int64_t rs_c = rs_c_*8;
    const int64_t cs_c = cs_c_*8;

    BEGIN_ASM()

    VXORPD(YMM( 8), YMM( 8), YMM( 8)) //clear out registers
    VXORPD(YMM( 9), YMM( 9), YMM( 9))
    VXORPD(YMM(10), YMM(10), YMM(10))
    VXORPD(YMM(11), YMM(11), YMM(11))
    VXORPD(YMM(12), YMM(12), YMM(12))
    VXORPD(YMM(13), YMM(13), YMM(13))
    VXORPD(YMM(14), YMM(14), YMM(14))
    VXORPD(YMM(15), YMM(15), YMM(15))
    VXORPD(YMM(16), YMM(16), YMM(16))
    VXORPD(YMM(17), YMM(17), YMM(17))
    VXORPD(YMM(18), YMM(18), YMM(18))
    VXORPD(YMM(19), YMM(19), YMM(19))
    VXORPD(YMM(20), YMM(20), YMM(20))
    VXORPD(YMM(21), YMM(21), YMM(21))
    VXORPD(YMM(22), YMM(22), YMM(22))
    VXORPD(YMM(23), YMM(23), YMM(23))
    VXORPD(YMM(24), YMM(24), YMM(24))
    VXORPD(YMM(25), YMM(25), YMM(25))
    VXORPD(YMM(26), YMM(26), YMM(26))
    VXORPD(YMM(27), YMM(27), YMM(27))
    VXORPD(YMM(28), YMM(28), YMM(28))
    VXORPD(YMM(29), YMM(29), YMM(29))
    VXORPD(YMM(30), YMM(30), YMM(30))
    VXORPD(YMM(31), YMM(31), YMM(31))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c

    RORX(RDX, RSI, IMM(64-8)) // size of b micropanel
    LEA(RDX, MEM(RBX,RDX,1,-128)) // b_next

    VMOVAPD(ZMM(0), MEM(RBX, 0*8)) //pre-load b
    VMOVAPD(ZMM(1), MEM(RBX, 8*8)) //pre-load b
    VMOVAPD(ZMM(2), MEM(RBX,16*8)) //pre-load b
    VMOVAPD(ZMM(3), MEM(RBX,24*8)) //pre-load b
    LEA(RBX, MEM(RBX,32*8)) //adjust b for pre-load

    MOV(R12, VAR(rs_c))
    MOV(R10, VAR(cs_c))

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    SUB(RDI, IMM(12+TAIL_NITER))
    JLE(K_LE_72)

        LOOP_ALIGN
        LABEL(LOOP1)

            PREFETCH(1, MEM(RDX))
            SUBITER(0)
            PREFETCH(1, MEM(RDX,64))
            SUBITER(1)
            PREFETCH(1, MEM(RDX,128))
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(1, MEM(RDX,192))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*6*8))
            LEA(RBX, MEM(RBX,4*32*8))
            LEA(RDX, MEM(RDX,32*8))

        JNZ(LOOP1)

    LABEL(K_LE_72)

    LEA(R9, MEM(RCX,63)) // c for prefetching (first half)

    ADD(RDI, IMM(6))
    JLE(K_LE_48)

        LOOP_ALIGN
        LABEL(LOOP2)

            PREFETCH(0, MEM(R9))
            PREFETCH(1, MEM(RDX))
            SUBITER(0)
            PREFETCH(1, MEM(RDX,64))
            SUBITER(1)
            PREFETCH(0, MEM(R9,64))
            PREFETCH(1, MEM(RDX,128))
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(1, MEM(RDX,192))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*6*8))
            LEA(RBX, MEM(RBX,4*32*8))
            LEA(RDX, MEM(RDX,32*8))
            LEA(R9, MEM(R9,R12,1))

        JNZ(LOOP2)

    LABEL(K_LE_48)

    LEA(R9, MEM(RCX,191)) // c for prefetching (second half)

    ADD(RDI, IMM(6))
    JLE(K_LE_24)

        LOOP_ALIGN
        LABEL(LOOP3)

            PREFETCH(0, MEM(R9))
            PREFETCH(1, MEM(RDX))
            SUBITER(0)
            PREFETCH(1, MEM(RDX,64))
            SUBITER(1)
            PREFETCH(0, MEM(R9,64))
            PREFETCH(1, MEM(RDX,128))
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(1, MEM(RDX,192))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*6*8))
            LEA(RBX, MEM(RBX,4*32*8))
            LEA(RDX, MEM(RDX,32*8))
            LEA(R9, MEM(R9,R12,1))

        JNZ(LOOP3)

    LABEL(K_LE_24)

    ADD(RDI, IMM(0+TAIL_NITER))
    JLE(TAIL)

        LOOP_ALIGN
        LABEL(LOOP4)

            PREFETCH(1, MEM(RDX))
            SUBITER(0)
            PREFETCH(1, MEM(RDX,64))
            SUBITER(1)
            PREFETCH(1, MEM(RDX,128))
            SUB(RDI, IMM(1))
            SUBITER(2)
            PREFETCH(1, MEM(RDX,192))
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*6*8))
            LEA(RBX, MEM(RBX,4*32*8))
            LEA(RDX, MEM(RDX,32*8))

        JNZ(LOOP4)

    LABEL(TAIL)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

        LOOP_ALIGN
        LABEL(TAIL_LOOP)

            SUB(RSI, IMM(1))
            SUBITER(0)

            LEA(RAX, MEM(RAX,6*8))
            LEA(RBX, MEM(RBX,32*8))

        JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    VXORPD(YMM(2), YMM(2), YMM(2))

    MOV(RAX, R12)
    MOV(RBX, R10)

    // Check if C is row stride.
    CMP(RBX, IMM(8))
    JNE(SCATTEREDUPDATE)

        VCOMISD(XMM(1), XMM(2))
        JE(ROWSTORBZ)

            UPDATE_C( 8, 9,10,11)
            UPDATE_C(12,13,14,15)
            UPDATE_C(16,17,18,19)
            UPDATE_C(20,21,22,23)
            UPDATE_C(24,25,26,27)
            UPDATE_C(28,29,30,31)

        JMP(END)
        LABEL(ROWSTORBZ)

            UPDATE_C_BZ( 8, 9,10,11)
            UPDATE_C_BZ(12,13,14,15)
            UPDATE_C_BZ(16,17,18,19)
            UPDATE_C_BZ(20,21,22,23)
            UPDATE_C_BZ(24,25,26,27)
            UPDATE_C_BZ(28,29,30,31)

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        VCOMISD(XMM(1), XMM(2))

        MOV(RDI, VAR(offsetPtr))
        VPBROADCASTQ(ZMM(6), RBX)
        VPMULLQ(ZMM(2), ZMM(6), MEM(RDI))
        VPMULLQ(ZMM(3), ZMM(6), MEM(RDI,64))
        VPMULLQ(ZMM(4), ZMM(6), MEM(RDI,128))
        VPMULLQ(ZMM(5), ZMM(6), MEM(RDI,192))

        JE(SCATTERBZ)

            UPDATE_C_ROW_SCATTERED( 8, 9,10,11)
            UPDATE_C_ROW_SCATTERED(12,13,14,15)
            UPDATE_C_ROW_SCATTERED(16,17,18,19)
            UPDATE_C_ROW_SCATTERED(20,21,22,23)
            UPDATE_C_ROW_SCATTERED(24,25,26,27)
            UPDATE_C_ROW_SCATTERED(28,29,30,31)

        JMP(END)
        LABEL(SCATTERBZ)

            UPDATE_C_BZ_ROW_SCATTERED( 8, 9,10,11)
            UPDATE_C_BZ_ROW_SCATTERED(12,13,14,15)
            UPDATE_C_BZ_ROW_SCATTERED(16,17,18,19)
            UPDATE_C_BZ_ROW_SCATTERED(20,21,22,23)
            UPDATE_C_BZ_ROW_SCATTERED(24,25,26,27)
            UPDATE_C_BZ_ROW_SCATTERED(28,29,30,31)

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
          "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
          "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31", "memory"
    )
}
