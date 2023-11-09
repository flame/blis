/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#define A_L1_PREFETCH_DIST 6
#define B_L1_PREFETCH_DIST 6
#define TAIL_NITER 7
// #define PREFETCH_A
#define PREFETCH_B
// #define PREFETCH_A_NEXT
#define PREFETCH_B_NEXT
#define PREFETCH_C              // perfetch c in middle loop over 4 iterations of k


#ifdef PREFETCH_A
    #define PREFETCH_A_L1(n) \
        PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*4*16 + 4*n*16))
#else
    #define PREFETCH_A_L1(n)
#endif

#ifdef PREFETCH_B
    #define PREFETCH_B_L1(n, k) \
        PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*12*16 + (12*n+(4*k))*16))
#else
    #define PREFETCH_B_L1(n, k)
#endif


/*
 * A Registers:  ZMM3, ZMM4, ZMM29, ZMM30
 * B Registers:  ZMM0, ZMM1, ZMM2
 * C Registers:  ZMM[8-28]
 */

#define LOOP_ALIGN ALIGN32

#define SUBITER(n) \
\
    PREFETCH_A_L1(n)\
    VBROADCASTSD(ZMM(3), MEM(RAX, (8*n+2)*8)) \
    VFMADD231PD(ZMM(5) , ZMM(0), ZMM(29)) \
    VFMADD231PD(ZMM(6) , ZMM(1), ZMM(29)) \
    VFMADD231PD(ZMM(7) , ZMM(2), ZMM(29)) \
    VBROADCASTSD(ZMM(4), MEM(RAX, (8*n+3)*8)) \
    VFMADD231PD(ZMM(8) , ZMM(0), ZMM(30)) \
    VFMADD231PD(ZMM(9) , ZMM(1), ZMM(30)) \
    VFMADD231PD(ZMM(10), ZMM(2), ZMM(30)) \
    \
    PREFETCH_B_L1(n, 0)\
    VBROADCASTSD(ZMM(29), MEM(RAX, (8*n+4)*8)) \
    VFMADD231PD(ZMM(11), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(12), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(13), ZMM(2), ZMM(3)) \
    VBROADCASTSD(ZMM(30), MEM(RAX, (8*n+5)*8)) \
    VFMADD231PD(ZMM(14), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(15), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(16), ZMM(2), ZMM(4)) \
    \
    PREFETCH_B_L1(n, 1)\
    VBROADCASTSD(ZMM(3), MEM(RAX, (8*n+6)*8)) \
    VFMADD231PD(ZMM(17), ZMM(0), ZMM(29)) \
    VFMADD231PD(ZMM(18), ZMM(1), ZMM(29)) \
    VFMADD231PD(ZMM(19), ZMM(2), ZMM(29)) \
    VBROADCASTSD(ZMM(4), MEM(RAX, (8*n+7)*8)) \
    VFMADD231PD(ZMM(20), ZMM(0), ZMM(30)) \
    VFMADD231PD(ZMM(21), ZMM(1), ZMM(30)) \
    VFMADD231PD(ZMM(22), ZMM(2), ZMM(30)) \
    \
    PREFETCH_B_L1(n, 2)\
    VBROADCASTSD(ZMM(29), MEM(RAX, (8*n+8)*8)) \
    VFMADD231PD(ZMM(23), ZMM(0), ZMM(3)) \
    VFMADD231PD(ZMM(24), ZMM(1), ZMM(3)) \
    VFMADD231PD(ZMM(25), ZMM(2), ZMM(3)) \
    VBROADCASTSD(ZMM(30), MEM(RAX, (8*n+9)*8)) \
    VFMADD231PD(ZMM(26), ZMM(0), ZMM(4)) \
    VFMADD231PD(ZMM(27), ZMM(1), ZMM(4)) \
    VFMADD231PD(ZMM(28), ZMM(2), ZMM(4)) \
    \
    VMOVAPD(ZMM(0), MEM(RBX, (12*n+0)*16)) \
    VMOVAPD(ZMM(1), MEM(RBX, (12*n+4)*16)) \
    VMOVAPD(ZMM(2), MEM(RBX, (12*n+8)*16))

#define SCALE_REG(a, b, c) \
    VPERMILPD(ZMM(3), a, IMM(0x55)) \
    VMULPD(a, a, b) \
    VMULPD(ZMM(3), ZMM(3), c) \
    VFMADDSUB132PD(a, ZMM(3), ZMM(31)) \

#define STORE_C_ROW(R1, R2, R3) \
    VMOVUPD(ZMM(0), MEM(RCX)) \
    SCALE_REG(ZMM(0), ZMM(1), ZMM(2)) \
    VADDPD(ZMM(0), ZMM(0), ZMM(R1)) \
    VMOVUPD(MEM(RCX), ZMM(0)) \
    \
    VMOVUPD(ZMM(0), MEM(RCX, R10, 4)) \
    SCALE_REG(ZMM(0), ZMM(1), ZMM(2)) \
    VADDPD(ZMM(0), ZMM(0), ZMM(R2)) \
    VMOVUPD(MEM(RCX, R10, 4), ZMM(0)) \
    \
    VMOVUPD(ZMM(0), MEM(RCX, R10, 8)) \
    SCALE_REG(ZMM(0), ZMM(1), ZMM(2)) \
    VADDPD(ZMM(0), ZMM(0), ZMM(R3)) \
    VMOVUPD(MEM(RCX, R10, 8), ZMM(0)) \

#define LOAD_ROW_GEN() \
    VMOVUPD(XMM(0), MEM(RDX))         \
    VMOVUPD(XMM(27), MEM(RDX, R10, 1)) \
    VMOVUPD(XMM(28), MEM(RDX, R10, 2)) \
    VMOVUPD(XMM(29), MEM(RDX, R11, 1)) \
    VINSERTF64X2(ZMM(0), ZMM(0), XMM(27), IMM(0x1)) \
    VINSERTF64X2(ZMM(0), ZMM(0), XMM(28), IMM(0x2)) \
    VINSERTF64X2(ZMM(0), ZMM(0), XMM(29), IMM(0x3)) \

#define STORE_ROW_GEN() \
    VEXTRACTF64X2(XMM(27), ZMM(0), IMM(0x1)) \
    VEXTRACTF64X2(XMM(28), ZMM(0), IMM(0x2)) \
    VEXTRACTF64X2(XMM(29), ZMM(0), IMM(0x3)) \
    VMOVUPD(MEM(RDX)        , XMM(0)) \
    VMOVUPD(MEM(RDX, R10, 1), XMM(27)) \
    VMOVUPD(MEM(RDX, R10, 2), XMM(28)) \
    VMOVUPD(MEM(RDX, R11, 1), XMM(29)) \

#define STORE_C_COL_GEN(R1, R2, R3) \
    MOV(RDX, RCX) \
    LEA(RCX, MEM(RCX, R12, 1)) \
    LOAD_ROW_GEN() \
    SCALE_REG(ZMM(0), ZMM(1), ZMM(2)) \
    VADDPD(ZMM(0), ZMM(0), ZMM(R1)) \
    STORE_ROW_GEN() \
    LEA(RDX, MEM(RDX, R10, 4)) \
    \
    LOAD_ROW_GEN() \
    SCALE_REG(ZMM(0), ZMM(1), ZMM(2)) \
    VADDPD(ZMM(0), ZMM(0), ZMM(R2)) \
    STORE_ROW_GEN() \
    LEA(RDX, MEM(RDX, R10, 4)) \
    \
    LOAD_ROW_GEN() \
    SCALE_REG(ZMM(0), ZMM(1), ZMM(2)) \
    VADDPD(ZMM(0), ZMM(0), ZMM(R3)) \
    STORE_ROW_GEN() \

/**********************************************************/
/* Kernel : bli_zgemm_zen4_asm_4x12                       */
/* It performs  C = C * beta + alpha * A * B              */
/* It is row preferred kernel, A and B are packed         */
/* C could be Row/Col/Gen Stored Matrix                   */
/* Registers are allocated as below                       */
/* Broadcast A :  ZMM(3, 4, 29, 30)                       */
/* load B :  ZMM(0, 1, 2)                                 */
/* Accumulation of B(real,imag)*Areal :                   */
/*       ZMM(5-7 , 11-13, 17-19, 23-25)                   */
/* Accumulation of B(real,imag)*Aimag :                   */
/*       ZMM(8-10, 14-16, 20-22, 26-28)                   */
/* Computation of A(real,imag)*B(real,imag):              */
/*       ZMM(5-7 , 11-13, 17-19, 23-25)                   */
/**********************************************************/
void bli_zgemm_zen4_asm_4x12(
                              dim_t            k_,
                              dcomplex* restrict alpha,
                              dcomplex* restrict a,
                              dcomplex* restrict b,
                              dcomplex* restrict beta,
                              dcomplex* restrict c, inc_t rs_c_, inc_t cs_c_,
                              auxinfo_t*       data,
                              cntx_t* restrict cntx
                            )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
    const int64_t k = k_;
    /*rowstride * size of one dcomplex element*/
    const int64_t rs_c = rs_c_*16;
    /*colstride * size of one dcomplex element*/
    const int64_t cs_c = cs_c_*16;


    char beta_mul_type  = BLIS_MUL_DEFAULT;
    if(beta->imag == 0.0 && beta->real == 0.0 )
    {
        beta_mul_type = BLIS_MUL_ZERO;
    }
    double one = 1; // used for FMADDSUB instruction
    double *one_addr = &one;

    BEGIN_ASM()

    VXORPD(XMM(5) , XMM(5) , XMM(5) )
    VXORPD(XMM(6) , XMM(6) , XMM(6) )
    VXORPD(XMM(7) , XMM(7) , XMM(7) )
    VXORPD(XMM(8) , XMM(8) , XMM(8) )
    VXORPD(XMM(9) , XMM(9) , XMM(9) )
    VXORPD(XMM(10), XMM(10), XMM(10))
    VXORPD(XMM(11), XMM(11), XMM(11))
    VXORPD(XMM(12), XMM(12), XMM(12))
    VXORPD(XMM(13), XMM(13), XMM(13))
    VXORPD(XMM(14), XMM(14), XMM(14))
    VXORPD(XMM(15), XMM(15), XMM(15))
    VXORPD(XMM(16), XMM(16), XMM(16))
    VXORPD(XMM(17), XMM(17), XMM(17))
    VXORPD(XMM(18), XMM(18), XMM(18))
    VXORPD(XMM(19), XMM(19), XMM(19))
    VXORPD(XMM(20), XMM(20), XMM(20))
    VXORPD(XMM(21), XMM(21), XMM(21))
    VXORPD(XMM(22), XMM(22), XMM(22))
    VXORPD(XMM(23), XMM(23), XMM(23))
    VXORPD(XMM(24), XMM(24), XMM(24))
    VXORPD(XMM(25), XMM(25), XMM(25))
    VXORPD(XMM(26), XMM(26), XMM(26))
    VXORPD(XMM(27), XMM(27), XMM(27))
    VXORPD(XMM(28), XMM(28), XMM(28))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c

    #ifdef PREFETCH_C
        LEA(R9, MEM(RCX, 63)) // c for prefetch, first cache line
        LEA(R8, MEM(R9, 128)) // c for prefetch, second cache line
    #endif


    VMOVAPD(ZMM(0), MEM(RBX, 0*16)) //pre-load b
    VMOVAPD(ZMM(1), MEM(RBX, 4*16)) //pre-load b
    VMOVAPD(ZMM(2), MEM(RBX, 8*16)) //pre-load b
    VBROADCASTSD(ZMM(29), MEM(RAX, 0))
    VBROADCASTSD(ZMM(30), MEM(RAX, 8))
    LEA(RBX, MEM(RBX, 12*16)) //adjust b for pre-load

    MOV(R12, VAR(rs_c))
    MOV(R10, VAR(cs_c))

    #if defined PREFETCH_A_NEXT || defined PREFETCH_B_NEXT
        MOV(RDI, RSI)
        IMUL(RDI, IMM(16*4)) // rdi = k * 16*4
    #endif

    #ifdef PREFETCH_A_NEXT
        LEA(R14, MEM(RAX, RDI, 1)) // r14(a_next) = A + (k*16*4)
    #endif

    #ifdef PREFETCH_B_NEXT
        IMUL(RDI, IMM(3)) // rdi = k * 16*12
        LEA(R15, MEM(RBX, RDI, 1)) // r15(b_next) = B + (k*16*12)
    #endif

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))
    /************************************************************/
    /* Operation:                                               */
    /* SUBITER = (Ar, Ai)*(Br, Bi) = Ar*(Br, Bi) , Ai*(Br, Bi)  */
    /* C_PREFETCH loop count:                                   */
    /*          LOOP1:   k/4 - TAIL_NITER - 4                   */
    /*          LOOP2:   4                                      */
    /*          LOOP4:   TAIL_NITER                             */
    /*          TAIL_LOOP: k%4                                  */
    /*                                                          */
    /* No prefetch loop count:                                  */
    /*          LOOP1:   k/4                                    */
    /*          TAIL_LOOP: k%4                                  */
    /************************************************************/
    #ifdef PREFETCH_C
            /* prefetch c over 4 iterations of k*/
            SUB(RDI, IMM(4+TAIL_NITER))
    #endif
    JLE(K_PREFETCH_C)

        LOOP_ALIGN
        LABEL(LOOP1)
            #ifdef PREFETCH_A_NEXT
                PREFETCH(1, MEM(R14))
            #endif
            SUBITER(0)
            #ifdef PREFETCH_B_NEXT
                PREFETCH(1, MEM(R15))
            #endif
            SUBITER(1)
            #ifdef PREFETCH_A_NEXT
                PREFETCH(2, MEM(R14, 64))
            #endif
            SUB(RDI, IMM(1))
            SUBITER(2)
            #ifdef PREFETCH_B_NEXT
                PREFETCH(2, MEM(R15, 64))
            #endif
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*4*16))
            LEA(RBX, MEM(RBX,4*12*16))
            #ifdef PREFETCH_A_NEXT
                LEA(R14, MEM(R14,128))
            #endif
            #ifdef PREFETCH_B_NEXT
                LEA(R15, MEM(R15,64))
            #endif

        JNZ(LOOP1)

    LABEL(K_PREFETCH_C)

#ifdef PREFETCH_C
    ADD(RDI, IMM(4))
    JLE(K_TAIL_NITER)

        LOOP_ALIGN
        LABEL(LOOP2)
            SUBITER(0)
            PREFETCH(0, MEM(R9))
            SUBITER(1)
            PREFETCH(0, MEM(R9, 64))
            SUB(RDI, IMM(1))
            PREFETCH(0, MEM(R9,128))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*4*16))
            LEA(RBX, MEM(RBX,4*12*16))
            LEA(R9, MEM(R9,R12,1))
        JNZ(LOOP2)

    LABEL(K_TAIL_NITER)

    ADD(RDI, IMM(0+TAIL_NITER))
    JLE(TAIL)

        LOOP_ALIGN
        LABEL(LOOP4)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*4*16))
            LEA(RBX, MEM(RBX,4*12*16))

        JNZ(LOOP4)

#endif //PREFETCH_C

    LABEL(TAIL)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

        LOOP_ALIGN
        LABEL(TAIL_LOOP)

            SUB(RSI, IMM(1))
            SUBITER(0)
            LEA(RAX, MEM(RAX,4*16))
            LEA(RBX, MEM(RBX,12*16))

        JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    VPERMILPD(ZMM8 , ZMM8 , IMM(0x55))
    VPERMILPD(ZMM9 , ZMM9 , IMM(0x55))
    VPERMILPD(ZMM10, ZMM10, IMM(0x55))
    VPERMILPD(ZMM14, ZMM14, IMM(0x55))
    VPERMILPD(ZMM15, ZMM15, IMM(0x55))
    VPERMILPD(ZMM16, ZMM16, IMM(0x55))
    VPERMILPD(ZMM20, ZMM20, IMM(0x55))
    VPERMILPD(ZMM21, ZMM21, IMM(0x55))
    VPERMILPD(ZMM22, ZMM22, IMM(0x55))
    VPERMILPD(ZMM26, ZMM26, IMM(0x55))
    VPERMILPD(ZMM27, ZMM27, IMM(0x55))
    VPERMILPD(ZMM28, ZMM28, IMM(0x55))

    MOV(R8, VAR(one_addr))
    VBROADCASTSD(ZMM(31), MEM(R8))
    VFMADDSUB132PD(ZMM(5) , ZMM(8) , ZMM(31))
    VFMADDSUB132PD(ZMM(6) , ZMM(9) , ZMM(31))
    VFMADDSUB132PD(ZMM(7) , ZMM(10), ZMM(31))

    VFMADDSUB132PD(ZMM(11), ZMM(14), ZMM(31))
    VFMADDSUB132PD(ZMM(12), ZMM(15), ZMM(31))
    VFMADDSUB132PD(ZMM(13), ZMM(16), ZMM(31))

    VFMADDSUB132PD(ZMM(17), ZMM(20), ZMM(31))
    VFMADDSUB132PD(ZMM(18), ZMM(21), ZMM(31))
    VFMADDSUB132PD(ZMM(19), ZMM(22), ZMM(31))

    VFMADDSUB132PD(ZMM(23), ZMM(26), ZMM(31))
    VFMADDSUB132PD(ZMM(24), ZMM(27), ZMM(31))
    VFMADDSUB132PD(ZMM(25), ZMM(28), ZMM(31))

    MOV(RAX, VAR(alpha))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RAX, 8))

    SCALE_REG(ZMM(5) , ZMM(0), ZMM(1))
    SCALE_REG(ZMM(6) , ZMM(0), ZMM(1))
    SCALE_REG(ZMM(7) , ZMM(0), ZMM(1))

    SCALE_REG(ZMM(11), ZMM(0), ZMM(1))
    SCALE_REG(ZMM(12), ZMM(0), ZMM(1))
    SCALE_REG(ZMM(13), ZMM(0), ZMM(1))

    SCALE_REG(ZMM(17), ZMM(0), ZMM(1))
    SCALE_REG(ZMM(18), ZMM(0), ZMM(1))
    SCALE_REG(ZMM(19), ZMM(0), ZMM(1))

    SCALE_REG(ZMM(23), ZMM(0), ZMM(1))
    SCALE_REG(ZMM(24), ZMM(0), ZMM(1))
    SCALE_REG(ZMM(25), ZMM(0), ZMM(1))

    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(1), MEM(RBX))
    VBROADCASTSD(ZMM(2), MEM(RBX, 8))


    MOV(AL, VAR(beta_mul_type))
    CMP(AL, IMM(0))
    JE(.ZBETAZERO)

    CMP(R10, IMM(16)) //CS == 1 IMPLIES ROW STORED
    JNZ(.ZCOLSTORED)

    LABEL(.ZROWSTORED)
        STORE_C_ROW(5 , 6 , 7 )      ADD(RCX, R12)
        STORE_C_ROW(11, 12, 13)      ADD(RCX, R12)
        STORE_C_ROW(17, 18, 19)      ADD(RCX, R12)
        STORE_C_ROW(23, 24, 25)
        JMP(.ZDONE)

    LABEL(.ZCOLSTORED)
        LEA(R11, MEM(R10, R10, 2))
        STORE_C_COL_GEN(5, 6, 7)
        STORE_C_COL_GEN(11, 12, 13)
        STORE_C_COL_GEN(17, 18, 19)
        STORE_C_COL_GEN(23, 24, 25)
        JMP(.ZDONE)

    LABEL(.ZBETAZERO)
        CMP(R10, IMM(16))
        JZ(.ZROWSTORBZ)

    LABEL(.ZCOLSTORBZ)
        LEA(R11, MEM(R10, R10, 2))
        MOV(RDX, RCX)
        ADD(RCX, R12)
        VMOVUPD(ZMM(0), ZMM(5))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(6))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(7))         STORE_ROW_GEN()

        MOV(RDX, RCX)
        LEA(RCX, MEM(RCX, R12, 1))
        VMOVUPD(ZMM(0), ZMM(11))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(12))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(13))         STORE_ROW_GEN()

        MOV(RDX, RCX)
        LEA(RCX, MEM(RCX, R12, 1))
        VMOVUPD(ZMM(0), ZMM(17))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(18))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(19))         STORE_ROW_GEN()

        MOV(RDX, RCX)
        VMOVUPD(ZMM(0), ZMM(23))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(24))         STORE_ROW_GEN()
        LEA(RDX, MEM(RDX, R10, 4))
        VMOVUPD(ZMM(0), ZMM(25))         STORE_ROW_GEN()

        JMP(.ZDONE)


    LABEL(.ZROWSTORBZ)
        VMOVUPD(MEM(RCX        ), ZMM(5))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(6))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(7))
        LEA(RCX, MEM(RCX, R12, 1))

        VMOVUPD(MEM(RCX        ), ZMM(11))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(12))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(13))
        LEA(RCX, MEM(RCX, R12, 1))

        VMOVUPD(MEM(RCX        ), ZMM(17))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(18))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(19))
        LEA(RCX, MEM(RCX, R12, 1))

        VMOVUPD(MEM(RCX        ), ZMM(23))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(24))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(25))

    LABEL(.ZDONE)

    VZEROUPPER()

    END_ASM
    (
    : // output operands (none)
    : // input operands
      [beta_mul_type] "m" (beta_mul_type),
      [k]             "m" (k),
      [a]             "m" (a),
      [b]             "m" (b),
      [alpha]         "m" (alpha),
      [beta]          "m" (beta),
      [c]             "m" (c),
      [rs_c]          "m" (rs_c),
      [cs_c]          "m" (cs_c),
      [one_addr]      "m" (one_addr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6",
      "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20",
      "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
      "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",
      "xmm14", "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20",
      "xmm21", "xmm22", "xmm23", "xmm24", "xmm25", "xmm26",
      "xmm27", "xmm28", "xmm29", "xmm30", "xmm31",
      "memory"
    )
}
