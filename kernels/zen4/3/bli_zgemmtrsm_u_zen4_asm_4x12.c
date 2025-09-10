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

#define SCALE_REG(a, b, c, out) \
    VPERMILPD(ZMM(3), a, IMM(0x55)) \
    VMULPD(out, a, b) \
    VMULPD(ZMM(3), ZMM(3), c) \
    VFMADDSUB132PD(out, ZMM(3), ZMM(31)) \

#define DIVIDE_COMPLEX(R1, c, d, csq_dsq) \
    VPERMILPD(ZMM(3), R1, IMM(0x55)) \
    VMULPD(R1, R1, c) \
    VMULPD(ZMM(3), ZMM(3), d) \
    VMULPD(ZMM(3), ZMM(3), ZMM(2)) \
    VFMADDSUB132PD(R1, ZMM(3), ZMM(31)) \
    VDIVPD(R1, R1, csq_dsq) \

#define STORE_REG_GEN(reg) \
    VEXTRACTF64X2(XMM(27), ZMM(reg), IMM(0x1)) \
    VEXTRACTF64X2(XMM(28), ZMM(reg), IMM(0x2)) \
    VEXTRACTF64X2(XMM(29), ZMM(reg), IMM(0x3)) \
    VMOVUPD(MEM(RDX)        , XMM(reg)) \
    VMOVUPD(MEM(RDX, R10, 1), XMM(27)) \
    VMOVUPD(MEM(RDX, R10, 2), XMM(28)) \
    VMOVUPD(MEM(RDX, R11, 1), XMM(29)) \


/**********************************************************/
/* Kernel : bli_zgemmtrsm_l_zen4_asm_4x12                 */
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
void bli_zgemmtrsm_u_zen4_asm_4x12(
                                dim_t               k_,
                                dcomplex*    restrict alpha,
                                dcomplex*    restrict a10,
                                dcomplex*    restrict a11,
                                dcomplex*    restrict b01,
                                dcomplex*    restrict b11,
                                dcomplex*    restrict c11, inc_t rs_c_, inc_t cs_c_,
                                auxinfo_t*   restrict data,
                                cntx_t*      restrict cntx
                            )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
    const int64_t k = k_;
    /*rowstride * size of one dcomplex element*/
    const int64_t rs_c = rs_c_*16;
    /*colstride * size of one dcomplex element*/
    const int64_t cs_c = cs_c_*16;
    double one = 1; // used for FMADDSUB instruction
    double neg_one = -1; // used for complex division
    double *one_addr = &one;
    double *neg_one_addr = &neg_one;

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
    MOV(RAX, VAR(a10)) //load address of a
    MOV(RBX, VAR(b01)) //load address of b
    MOV(RCX, VAR(b11)) //load address of c
    MOV(R9, VAR(c11)) //load address of c
    MOV(R11, VAR(neg_one_addr))

    #ifdef PREFETCH_C
        LEA(R9, MEM(R9, 63)) // c for prefetch, first cache line
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

    /******************************************************/
    /* Permute imag component register. Shuffle even      */
    /* and odd components                                 */
    /* SRC: ZMM8 =(Ai0*Br0, Ai0*Bi0, Ai0*Br1, Ai0*Bi1, ..)*/
    /* DST: ZMM8 =(Ai0*Bi0, Ai0*Br0, Ai0*Bi1, Ai0*Br1, ..)*/
    /******************************************************/
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

    /*******************************************************/
    /* SRC: ZMM5 = (Ar0*Br0, Ar0*Bi0, Ar0*Br1, Ar0*Bi1, ..)*/
    /* SRC: ZMM8 = (Ai0*Bi0, Ai0*Br0, Ai0*Bi1, Ai0*Br1, ..)*/
    /* DST: ZMM8 =(Ar0*Br0-Ai0*Bi0, Ai0*Br0+Ar0*Bi0,       */
    /*             Ar0*Br1-Ai0*Bi1, Ai0*Br1+Ar0*Bi1, ..)   */
    /*******************************************************/
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
    MOV(RDX, RCX)
    MOV(RDI, IMM(12*16))

    VMOVUPD(ZMM(14), MEM(RDX, 0*16))
    VMOVUPD(ZMM(15), MEM(RDX, 4*16))
    VMOVUPD(ZMM(16), MEM(RDX, 8*16))
    ADD(RDX, RDI)

    /*****************************/
    /* gemm_output -= C * alpha  */
    /*****************************/
    SCALE_REG(ZMM(14) , ZMM(0), ZMM(1), ZMM(14))
    VSUBPD(ZMM(5), ZMM(14), ZMM(5))
    VMOVUPD(ZMM(14), MEM(RDX, 0*16))

    SCALE_REG(ZMM(15) , ZMM(0), ZMM(1), ZMM(15))
    VSUBPD(ZMM(6), ZMM(15), ZMM(6))
    VMOVUPD(ZMM(15), MEM(RDX, 4*16))

    SCALE_REG(ZMM(16) , ZMM(0), ZMM(1), ZMM(16))
    VSUBPD(ZMM(7), ZMM(16), ZMM(7))
    VMOVUPD(ZMM(16), MEM(RDX, 8*16))
    ADD(RDX, RDI)


    SCALE_REG(ZMM(14) , ZMM(0), ZMM(1), ZMM(14))
    VSUBPD(ZMM(11), ZMM(14), ZMM(11))
    VMOVUPD(ZMM(14), MEM(RDX, 0*16))

    SCALE_REG(ZMM(15) , ZMM(0), ZMM(1), ZMM(15))
    VSUBPD(ZMM(12), ZMM(15), ZMM(12))
    VMOVUPD(ZMM(15), MEM(RDX, 4*16))

    SCALE_REG(ZMM(16) , ZMM(0), ZMM(1), ZMM(16))
    VSUBPD(ZMM(13), ZMM(16), ZMM(13))
    VMOVUPD(ZMM(16), MEM(RDX, 8*16))
    ADD(RDX, RDI)


    SCALE_REG(ZMM(14) , ZMM(0), ZMM(1), ZMM(14))
    VSUBPD(ZMM(17), ZMM(14), ZMM(17))
    VMOVUPD(ZMM(14), MEM(RDX, 0*16))

    SCALE_REG(ZMM(15) , ZMM(0), ZMM(1), ZMM(15))
    VSUBPD(ZMM(18), ZMM(15), ZMM(18))
    VMOVUPD(ZMM(15), MEM(RDX, 4*16))

    SCALE_REG(ZMM(16) , ZMM(0), ZMM(1), ZMM(16))
    VSUBPD(ZMM(19), ZMM(16), ZMM(19))
    VMOVUPD(ZMM(16), MEM(RDX, 8*16))


    SCALE_REG(ZMM(14) , ZMM(0), ZMM(1), ZMM(14))
    VSUBPD(ZMM(23), ZMM(14), ZMM(23))
    VMOVUPD(ZMM(14), MEM(RDX, 0*16))

    SCALE_REG(ZMM(15) , ZMM(0), ZMM(1), ZMM(15))
    VSUBPD(ZMM(24), ZMM(15), ZMM(24))
    VMOVUPD(ZMM(15), MEM(RDX, 4*16))

    SCALE_REG(ZMM(16) , ZMM(0), ZMM(1), ZMM(16))
    VSUBPD(ZMM(25), ZMM(16), ZMM(25))
    VMOVUPD(ZMM(16), MEM(RDX, 8*16))


    //REGION - TRSM

    MOV(RAX, VAR(a11))
    LEA(RCX, MEM(RCX, RDI, 2))
    ADD(RCX, RDI)
    //iteration 0 -----------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (3+3*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (3+3*4)*16+8))
    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        SCALE_REG(ZMM(23), ZMM(0), ZMM(1), ZMM(23))
        SCALE_REG(ZMM(24), ZMM(0), ZMM(1), ZMM(24))
        SCALE_REG(ZMM(25), ZMM(0), ZMM(1), ZMM(25))
    #else
        VBROADCASTSD(ZMM(2), MEM(R11))
        VMULPD(ZMM(8), ZMM(0), ZMM(0))
        VFMADD231PD(ZMM(8), ZMM(1), ZMM(1))

        DIVIDE_COMPLEX(ZMM(23), ZMM(0), ZMM(1), ZMM(8))
        DIVIDE_COMPLEX(ZMM(24), ZMM(0), ZMM(1), ZMM(8))
        DIVIDE_COMPLEX(ZMM(25), ZMM(0), ZMM(1), ZMM(8))
    #endif
    VMOVUPD(MEM(RCX, 0*16), ZMM(23))
    VMOVUPD(MEM(RCX, 4*16), ZMM(24))
    VMOVUPD(MEM(RCX, 8*16), ZMM(25))
    SUB(RCX, RDI)

    //iteration 1 -----------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (2+3*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (2+3*4)*16+8))
    SCALE_REG(ZMM(23), ZMM(0), ZMM(1), ZMM(14))
    SCALE_REG(ZMM(24), ZMM(0), ZMM(1), ZMM(15))
    SCALE_REG(ZMM(25), ZMM(0), ZMM(1), ZMM(16))

    VSUBPD(ZMM(17), ZMM(17), ZMM(14))
    VSUBPD(ZMM(18), ZMM(18), ZMM(15))
    VSUBPD(ZMM(19), ZMM(19), ZMM(16))

    VBROADCASTSD(ZMM(0), MEM(RAX, (2+2*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (2+2*4)*16+8))
    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        SCALE_REG(ZMM(17), ZMM(0), ZMM(1), ZMM(17))
        SCALE_REG(ZMM(18), ZMM(0), ZMM(1), ZMM(18))
        SCALE_REG(ZMM(19), ZMM(0), ZMM(1), ZMM(19))
    #else
        VBROADCASTSD(ZMM(2), MEM(R11))
        VMULPD(ZMM(8), ZMM(0), ZMM(0))
        VFMADD231PD(ZMM(8), ZMM(1), ZMM(1))

        DIVIDE_COMPLEX(ZMM(17), ZMM(0), ZMM(1), ZMM(8))
        DIVIDE_COMPLEX(ZMM(18), ZMM(0), ZMM(1), ZMM(8))
        DIVIDE_COMPLEX(ZMM(19), ZMM(0), ZMM(1), ZMM(8))
    #endif
    VMOVUPD(MEM(RCX, 0*16), ZMM(17))
    VMOVUPD(MEM(RCX, 4*16), ZMM(18))
    VMOVUPD(MEM(RCX, 8*16), ZMM(19))
    SUB(RCX, RDI)

    //iteration 2 -----------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (1+3*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (1+3*4)*16+8))
    SCALE_REG(ZMM(23), ZMM(0), ZMM(1), ZMM(14))
    SCALE_REG(ZMM(24), ZMM(0), ZMM(1), ZMM(15))
    SCALE_REG(ZMM(25), ZMM(0), ZMM(1), ZMM(16))

    VBROADCASTSD(ZMM(0), MEM(RAX, (1+2*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (1+2*4)*16+8))
    SCALE_REG(ZMM(17), ZMM(0), ZMM(1), ZMM(20))
    SCALE_REG(ZMM(18), ZMM(0), ZMM(1), ZMM(21))
    SCALE_REG(ZMM(19), ZMM(0), ZMM(1), ZMM(22))
    VADDPD(ZMM(14), ZMM(14), ZMM(20))
    VADDPD(ZMM(15), ZMM(15), ZMM(21))
    VADDPD(ZMM(16), ZMM(16), ZMM(22))

    VSUBPD(ZMM(11), ZMM(11), ZMM(14))
    VSUBPD(ZMM(12), ZMM(12), ZMM(15))
    VSUBPD(ZMM(13), ZMM(13), ZMM(16))

    VBROADCASTSD(ZMM(0), MEM(RAX, (1+1*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (1+1*4)*16+8))
    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        SCALE_REG(ZMM(11), ZMM(0), ZMM(1), ZMM(11))
        SCALE_REG(ZMM(12), ZMM(0), ZMM(1), ZMM(12))
        SCALE_REG(ZMM(13), ZMM(0), ZMM(1), ZMM(13))
    #else
        VBROADCASTSD(ZMM(2), MEM(R11))
        VMULPD(ZMM(8), ZMM(0), ZMM(0))
        VFMADD231PD(ZMM(8), ZMM(1), ZMM(1))

        DIVIDE_COMPLEX(ZMM(11), ZMM(0), ZMM(1), ZMM(8))
        DIVIDE_COMPLEX(ZMM(12), ZMM(0), ZMM(1), ZMM(8))
        DIVIDE_COMPLEX(ZMM(13), ZMM(0), ZMM(1), ZMM(8))
    #endif
    VMOVUPD(MEM(RCX, 0*16), ZMM(11))
    VMOVUPD(MEM(RCX, 4*16), ZMM(12))
    VMOVUPD(MEM(RCX, 8*16), ZMM(13))
    SUB(RCX, RDI)

    //iteration 3 -----------------------------------
    VBROADCASTSD(ZMM(0), MEM(RAX, (0+3*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (0+3*4)*16+8))
    SCALE_REG(ZMM(23), ZMM(0), ZMM(1), ZMM(14))
    SCALE_REG(ZMM(24), ZMM(0), ZMM(1), ZMM(15))
    SCALE_REG(ZMM(25), ZMM(0), ZMM(1), ZMM(16))

    VBROADCASTSD(ZMM(0), MEM(RAX, (0+2*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (0+2*4)*16+8))
    SCALE_REG(ZMM(17), ZMM(0), ZMM(1), ZMM(20))
    SCALE_REG(ZMM(18), ZMM(0), ZMM(1), ZMM(21))
    SCALE_REG(ZMM(19), ZMM(0), ZMM(1), ZMM(22))
    VADDPD(ZMM(14), ZMM(14), ZMM(20))
    VADDPD(ZMM(15), ZMM(15), ZMM(21))
    VADDPD(ZMM(16), ZMM(16), ZMM(22))

    VBROADCASTSD(ZMM(0), MEM(RAX, (0+1*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (0+1*4)*16+8))
    SCALE_REG(ZMM(11), ZMM(0), ZMM(1), ZMM(20))
    SCALE_REG(ZMM(12), ZMM(0), ZMM(1), ZMM(21))
    SCALE_REG(ZMM(13), ZMM(0), ZMM(1), ZMM(22))
    VADDPD(ZMM(14), ZMM(14), ZMM(20))
    VADDPD(ZMM(15), ZMM(15), ZMM(21))
    VADDPD(ZMM(16), ZMM(16), ZMM(22))

    VSUBPD(ZMM(5), ZMM(5), ZMM(14))
    VSUBPD(ZMM(6), ZMM(6), ZMM(15))
    VSUBPD(ZMM(7), ZMM(7), ZMM(16))

    VBROADCASTSD(ZMM(0), MEM(RAX, (0+0*4)*16+0))
    VBROADCASTSD(ZMM(1), MEM(RAX, (0+0*4)*16+8))
    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        SCALE_REG(ZMM(5), ZMM(0), ZMM(1), ZMM(5))
        SCALE_REG(ZMM(6), ZMM(0), ZMM(1), ZMM(6))
        SCALE_REG(ZMM(7), ZMM(0), ZMM(1), ZMM(7))
    #else
        VBROADCASTSD(ZMM(2), MEM(R11))
        VMULPD(ZMM(8), ZMM(0), ZMM(0))
        VFMADD231PD(ZMM(8), ZMM(1), ZMM(1))

        VPERMILPD(ZMM(3), ZMM(5), IMM(0x55))
        VMULPD(ZMM(5), ZMM(5), ZMM(0))
        VMULPD(ZMM(3), ZMM(3), ZMM(1))
        VMULPD(ZMM(3), ZMM(3), ZMM(2))
        VFMADDSUB132PD(ZMM(5), ZMM(3), ZMM(31))
        VDIVPD(ZMM(5), ZMM(5), ZMM(8))

        VPERMILPD(ZMM(3), ZMM(6), IMM(0x55))
        VMULPD(ZMM(6), ZMM(6), ZMM(0))
        VMULPD(ZMM(3), ZMM(3), ZMM(1))
        VMULPD(ZMM(3), ZMM(3), ZMM(2))
        VFMADDSUB132PD(ZMM(6), ZMM(3), ZMM(31))
        VDIVPD(ZMM(6), ZMM(6), ZMM(8))

        VPERMILPD(ZMM(3), ZMM(7), IMM(0x55))
        VMULPD(ZMM(7), ZMM(7), ZMM(0))
        VMULPD(ZMM(3), ZMM(3), ZMM(1))
        VMULPD(ZMM(3), ZMM(3), ZMM(2))
        VFMADDSUB132PD(ZMM(7), ZMM(3), ZMM(31))
        VDIVPD(ZMM(7), ZMM(7), ZMM(8))
    #endif
    VMOVUPD(MEM(RCX, 0*16), ZMM(5))
    VMOVUPD(MEM(RCX, 4*16), ZMM(6))
    VMOVUPD(MEM(RCX, 8*16), ZMM(7))

// ENDREGION - TRSM

    MOV(RCX, VAR(c11))
    CMP(R10, IMM(16)) //CS == 1 IMPLIES ROW STORED
    JNZ(.ZCOLSTORED)

    LABEL(.ZROWSTORED)
        VMOVUPD(MEM(RCX        ), ZMM(5))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(6))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(7))
        ADD(RCX, R12)

        VMOVUPD(MEM(RCX        ), ZMM(11))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(12))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(13))
        ADD(RCX, R12)

        VMOVUPD(MEM(RCX        ), ZMM(17))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(18))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(19))
        ADD(RCX, R12)

        VMOVUPD(MEM(RCX        ), ZMM(23))
        VMOVUPD(MEM(RCX, R10, 4), ZMM(24))
        VMOVUPD(MEM(RCX, R10, 8), ZMM(25))

        JMP(.ZDONE)

    LABEL(.ZCOLSTORED)
        LEA(R11, MEM(R10, R10, 2))
        MOV(RDX, RCX)
        ADD(RCX, R12)
        STORE_REG_GEN(5) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(6) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(7)

        MOV(RDX, RCX)
        ADD(RCX, R12)
        STORE_REG_GEN(11) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(12) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(13)

        MOV(RDX, RCX)
        ADD(RCX, R12)
        STORE_REG_GEN(17) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(18) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(19)

        MOV(RDX, RCX)
        STORE_REG_GEN(23) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(24) LEA(RDX, MEM(RDX, R10, 4))
        STORE_REG_GEN(25)

    LABEL(.ZDONE)
    VZEROUPPER()

    END_ASM
    (
    : // output operands (none)
    : // input operands
        [a10]             "m" (a10),
        [k]               "m" (k),
        [b01]             "m" (b01),
        [a11]             "m" (a11),
        [b11]             "m" (b11),
        [c11]             "m" (c11),
        [rs_c]            "m" (rs_c),
        [cs_c]            "m" (cs_c),
        [alpha]           "m" (alpha),
        [neg_one_addr]    "m" (neg_one_addr),
        [one_addr]        "m" (one_addr)
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
