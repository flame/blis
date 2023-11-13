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

#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 4
#define TAIL_NITER 6

#define PREFETCH_A_L1(n, k) \
    PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*2*16 + (2*n+k)*(16)))
#define PREFETCH_B_L1(n, k) \
    PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*6*16 + (2*n+k)*(48)))

/*
 * A Registers:  YMM3
 * B Registers:  YMM0, YMM1, YMM2
 * C Registers:  YMM[4-15]
 */

#define LOOP_ALIGN ALIGN32

#define SUBITER(n) \
\
    PREFETCH_A_L1(n, 0) \
    VBROADCASTSD(YMM( 3), MEM(RAX,(4*n+ 0)*8)) \
    VFMADD231PD(YMM( 4), YMM(0), YMM(3)) \
    VFMADD231PD(YMM( 5), YMM(1), YMM(3)) \
    VFMADD231PD(YMM( 6), YMM(2), YMM(3)) \
    VBROADCASTSD(YMM( 3), MEM(RAX,(4*n+ 1)*8)) \
    VFMADD231PD(YMM( 7), YMM(0), YMM(3)) \
    VFMADD231PD(YMM( 8), YMM(1), YMM(3)) \
    VFMADD231PD(YMM( 9), YMM(2), YMM(3)) \
    \
    PREFETCH_B_L1(n, 0) \
    VBROADCASTSD(YMM( 3), MEM(RAX,(4*n+ 2)*8)) \
    VFMADD231PD(YMM(10), YMM(0), YMM(3)) \
    VFMADD231PD(YMM(11), YMM(1), YMM(3)) \
    VFMADD231PD(YMM(12), YMM(2), YMM(3)) \
    VBROADCASTSD(YMM( 3), MEM(RAX,(4*n+ 3)*8)) \
    VFMADD231PD(YMM(13), YMM(0), YMM(3)) \
    VFMADD231PD(YMM(14), YMM(1), YMM(3)) \
    VFMADD231PD(YMM(15), YMM(2), YMM(3)) \
    \
    VMOVAPD(YMM(0), MEM(RBX,(6*n+0)*16)) \
    VMOVAPD(YMM(1), MEM(RBX,(6*n+2)*16)) \
    VMOVAPD(YMM(2), MEM(RBX,(6*n+4)*16)) \

// used for division of complex number if TRSM_PREINV is disabled
static double negative[4] __attribute__((aligned(64)))
                                 = {-1, -1, -1, -1};

/**********************************************************/
/* Kernel : bli_zgemmtrsm_l_zen_asm_2x6                   */
/* It performs  A * X = alpha * B                         */
/* It is row preferred kernel, A and B are packed         */
/* C could be Row/Col/Gen Stored Matrix                   */
/* Registers are allocated as below                       */
/* Broadcast A :  YMM(3)                                  */
/* load B :  YMM(0, 1, 2)                                 */
/* Accumulation of B(real,imag)*Areal :                   */
/*       YMM(4-6,10-12)                                   */
/* Accumulation of B(real,imag)*Aimag :                   */
/*       YMM(7-9,13-15)                                   */
/* Computation of A(real,imag)*B(real,imag):              */
/*       YMM(4-6,10-12)                                   */
/**********************************************************/
void bli_zgemmtrsm_l_zen_asm_2x6
     (
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
    const int64_t k = k_;
    /*rowstride * size of one dcomplex element*/
    const int64_t rs_c = rs_c_*16;
    /*colstride * size of one dcomplex element*/
    const int64_t cs_c = cs_c_*16;
    const double* negPtr = &negative[0];


    BEGIN_ASM()

    VXORPD(YMM( 4), YMM( 4), YMM( 4))
    VXORPD(YMM( 5), YMM( 5), YMM( 5))
    VMOVAPD(YMM(6) , YMM(4))
    VMOVAPD(YMM(7) , YMM(4))
    VMOVAPD(YMM(8) , YMM(4))
    VMOVAPD(YMM(9) , YMM(4))
    VXORPD(YMM(10), YMM(10), YMM(10))
    VXORPD(YMM(11), YMM(11), YMM(11))
    VMOVAPD(YMM(12), YMM(4))
    VMOVAPD(YMM(13), YMM(4))
    VMOVAPD(YMM(14), YMM(4))
    VMOVAPD(YMM(15), YMM(4))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a10)) //load address of a
    MOV(RBX, VAR(b01)) //load address of b
    MOV(RCX, VAR(b11)) //load address of c
    MOV(R9, VAR(c11))  // load C for prefetch
    MOV(R11, VAR(negPtr))

    VMOVAPD(YMM(0), MEM(RBX, 0*16)) //pre-load b
    VMOVAPD(YMM(1), MEM(RBX, 2*16)) //pre-load b
    VMOVAPD(YMM(2), MEM(RBX, 4*16)) //pre-load b
    LEA(RBX, MEM(RBX,6*16)) //adjust b for pre-load

    MOV(R12, VAR(rs_c))
    MOV(R10, VAR(cs_c))

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    /************************************************************/
    /* Operation:                                               */
    /* SUBITER = (Ar, Ai)*(Br, Bi) = Ar*(Br, Bi) , Ai*(Br, Bi)  */
    /* Loop counts:                                             */
    /*          LOOP1:   k/4 - TAIL_NITER - 2                   */
    /*          LOOP2:   2                       <--prefetch_c  */
    /*          LOOP4:   TAIL_NITER                             */
    /************************************************************/
    SUB(RDI, IMM(2+TAIL_NITER))
    JLE(K_PREFETCH_C)

        LOOP_ALIGN
        LABEL(LOOP1)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*2*16))
            LEA(RBX, MEM(RBX,4*6*16))


        JNZ(LOOP1)

    LABEL(K_PREFETCH_C)

    ADD(RDI, IMM(2))
    JLE(K_TAIL_NITER)

        LOOP_ALIGN
        LABEL(LOOP2)

            PREFETCH(0, MEM(R9))
            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            PREFETCH(0, MEM(R9,64))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*2*16))
            LEA(RBX, MEM(RBX,4*6*16))
            LEA(R9, MEM(R9,R12,1))

        JNZ(LOOP2)

    LABEL(K_TAIL_NITER)

    ADD(RDI, IMM(0+TAIL_NITER))
    JLE(TAIL)

        LOOP_ALIGN
        LABEL(LOOP3)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*2*16))
            LEA(RBX, MEM(RBX,4*6*16))

        JNZ(LOOP3)

    LABEL(TAIL)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

        LOOP_ALIGN
        LABEL(TAIL_LOOP)

            SUB(RSI, IMM(1))
            SUBITER(0)
            LEA(RAX, MEM(RAX,2*16))
            LEA(RBX, MEM(RBX,6*16))

        JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    /**************************************************/
    /* Permute imag component register. Shuffle even  */
    /* and odd components                             */
    /* SRC: YMM7 =(Ai0*Br0, Ai0*Bi0, Ai0*Br1, Ai0*Bi1)*/
    /* DST: YMM7 =(Ai0*Bi0, Ai0*Br0, Ai0*Bi1, Ai0*Br1)*/
    /**************************************************/
    VPERMILPD(YMM( 7), YMM( 7), IMM(0x5))
    VPERMILPD(YMM( 8), YMM( 8), IMM(0x5))
    VPERMILPD(YMM( 9), YMM( 9), IMM(0x5))
    VPERMILPD(YMM(13), YMM(13), IMM(0x5))
    VPERMILPD(YMM(14), YMM(14), IMM(0x5))
    VPERMILPD(YMM(15), YMM(15), IMM(0x5))

    /***************************************************/
    /* SRC: YMM4 = (Ar0*Br0, Ar0*Bi0, Ar0*Br1, Ar0*Bi1)*/
    /* SRC: YMM7 = (Ai0*Bi0, Ai0*Br0, Ai0*Bi1, Ai0*Br1)*/
    /* DST: YMM4 =(Ar0*Br0-Ai0*Bi0, Ai0*Br0+Ar0*Bi0,   */
    /*             Ar0*Br1-Ai0*Bi1, Ai0*Br1+Ar0*Bi1)   */
    /***************************************************/
    VADDSUBPD(YMM(4), YMM(4), YMM(7))
    VADDSUBPD(YMM(5), YMM(5), YMM(8))
    VADDSUBPD(YMM(6), YMM(6), YMM(9))
    VADDSUBPD(YMM(10), YMM(10), YMM(13))
    VADDSUBPD(YMM(11), YMM(11), YMM(14))
    VADDSUBPD(YMM(12), YMM(12), YMM(15))

    /*Load alpha*/
    MOV(R9, VAR(alpha))
    VBROADCASTSD(YMM(7), MEM(R9))
    VBROADCASTSD(YMM(8), MEM(R9, 8))
    MOV(RDX, RCX)
    MOV(RDI, IMM(6*16))

    VMOVUPD(YMM(0), MEM(RDX, 0*16))
    VMOVUPD(YMM(1), MEM(RDX, 2*16))
    VMOVUPD(YMM(2), MEM(RDX, 4*16))
    ADD(RDX, RDI)

    /************************************************************************/
    /* gemm_output -= C * alpha                                             */
    /*                                                                      */
    /* Let  C * alpha = (a + ib) * (c + id)                                 */
    /*    (a + ib) * (c + id) = (ac - bd) + i(ad + bc)                      */
    /*                                                                      */
    /*Steps:                                                                */
    /*  YMM(0) = a0, b0, a1, b1                                             */
    /*  YMM(3) = b0, a0, b1, a1                                             */
    /*  YMM(0) = a0*c0, b0*c0, a1*c1, b1*c1                                 */
    /*  YMM(3) = b0*d0, a0*d0, b1*d1, a1*d1                                 */
    /*  YMM(0) = (a0c0 - b0d0), (b0c0 + a0d0), (a1c1 - b1d1), (b1c1 + a1d1) */
    /************************************************************************/
    VPERMILPD(YMM(3), YMM(0), IMM(0x5))
    VMULPD(YMM(0), YMM(0), YMM(7))    // a*c, b*c
    VMULPD(YMM(3), YMM(3), YMM(8))    // b*d, a*d
    VADDSUBPD(YMM(0), YMM(0), YMM(3)) // ac - bd, bc + ad
    VSUBPD(YMM(4), YMM(0), YMM(4))    // gemm_output - c * alpha

    VMOVUPD(YMM(0), MEM(RDX, 0*16))
    VPERMILPD(YMM(3), YMM(1), IMM(0x5))
    VMULPD(YMM(1), YMM(1), YMM(7))
    VMULPD(YMM(3), YMM(3), YMM(8))
    VADDSUBPD(YMM(1), YMM(1), YMM(3))
    VSUBPD(YMM(5), YMM(1), YMM(5))

    VMOVUPD(YMM(1), MEM(RDX, 2*16))
    VPERMILPD(YMM(3), YMM(2), IMM(0x5))
    VMULPD(YMM(2), YMM(2), YMM(7))
    VMULPD(YMM(3), YMM(3), YMM(8))
    VADDSUBPD(YMM(2), YMM(2), YMM(3))
    VSUBPD(YMM(6), YMM(2), YMM(6))

    VMOVUPD(YMM(2), MEM(RDX, 4*16))
    VPERMILPD(YMM(3), YMM(0), IMM(0x5))
    VMULPD(YMM(0), YMM(0), YMM(7))
    VMULPD(YMM(3), YMM(3), YMM(8))
    VADDSUBPD(YMM(0), YMM(0), YMM(3))
    VSUBPD(YMM(10), YMM(0), YMM(10))

    VPERMILPD(YMM(3), YMM(1), IMM(0x5))
    VMULPD(YMM(1), YMM(1), YMM(7))
    VMULPD(YMM(3), YMM(3), YMM(8))
    VADDSUBPD(YMM(1), YMM(1), YMM(3))
    VSUBPD(YMM(11), YMM(1), YMM(11))

    VPERMILPD(YMM(3), YMM(2), IMM(0x5))
    VMULPD(YMM(2), YMM(2), YMM(7))
    VMULPD(YMM(3), YMM(3), YMM(8))
    VADDSUBPD(YMM(2), YMM(2), YMM(3))
    VSUBPD(YMM(12), YMM(2), YMM(12))


    // REGION - TRSM
    MOV(RAX, VAR(a11))
    //iteration 0 -------------------------------------
    VBROADCASTSD(YMM(0), MEM(RAX, (0+0*2)*16+0))
    VBROADCASTSD(YMM(1), MEM(RAX, (0+0*2)*16+8))
    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        /****************************************************/
        /*  C = C * A11                                     */
        /*    (a + ib) * (c + id) = (ac - bd) + i(ad + bc)  */
        /****************************************************/
        VPERMILPD(YMM(3), YMM(4), IMM(0x5))
        VMULPD(YMM(4), YMM(4), YMM(0)) //a*c, b*c
        VMULPD(YMM(3), YMM(3), YMM(1)) //b*d, a*d
        VADDSUBPD(YMM(4), YMM(4), YMM(3)) // (ac - bd), (bc + ad)

        VPERMILPD(YMM(3), YMM(5), IMM(0x5))
        VMULPD(YMM(5), YMM(5), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VADDSUBPD(YMM(5), YMM(5), YMM(3))

        VPERMILPD(YMM(3), YMM(6), IMM(0x5))
        VMULPD(YMM(6), YMM(6), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VADDSUBPD(YMM(6), YMM(6), YMM(3))
    #else
        /************************************************************************/
        /*  C = C / A11                                                         */
        /*                                                                      */
        /* Let C / A11   =  (a + ib) / (c + id) =                               */
        /*     ((ac + bd) / (c^2 + d^2)) + i ((bc - ad) / (c^2+d^2))            */
        /*                                                                      */
        /*Steps:                                                                */
        /*  YMM(4) = a0, b0, a1, b1                                             */
        /*  YMM(3) = b0, a0, b1, a1                                             */
        /*  YMM(4) = a0*c0, b0*c0, a1*c1, b1*c1                                 */
        /*  YMM(3) = b0*d0, a0*d0, b1*d1, a1*d1                                 */
        /*  YMM(3) = -b0*d0, -a0*d0, -b1*d1, -a1*d1                             */
        /*  YMM(4) = (a0c0 - b0d0), (b0c0 + a0d0), (a1c1 - b1d1), (b1c1 + a1d1) */
        /*  YMM(4) = (a0c0 - b0d0) / (c^2 + d^2), (b0c0 + a0d0) / (c^2 + d^2),  */
        /*           (a1c1 - b1d1) / (c^2 + d^2), (b1c1 + a1d1 / (c^2 + d^2)    */
        /************************************************************************/
        VMOVUPD(YMM(2), MEM(R11)) // -1
        VMULPD(YMM(9), YMM(0), YMM(0))
        VFMADD231PD(YMM(9), YMM(1), YMM(1))

        VPERMILPD(YMM(3), YMM(4), IMM(0x5))
        VMULPD(YMM(4), YMM(4), YMM(0)) // a*c, b*c
        VMULPD(YMM(3), YMM(3), YMM(1)) // b*d, a*d
        VMULPD(YMM(3), YMM(3), YMM(2)) // -bd, -ad
        VADDSUBPD(YMM(4), YMM(4), YMM(3)) // ac + bd, bc - ad
        VDIVPD(YMM(4), YMM(4), YMM(9)) // (ac + bd) / (c^2 + d^2), (bc - ad) / (c^2 + d^2)

        VPERMILPD(YMM(3), YMM(5), IMM(0x5))
        VMULPD(YMM(5), YMM(5), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(5), YMM(5), YMM(3))
        VDIVPD(YMM(5), YMM(5), YMM(9))

        VPERMILPD(YMM(3), YMM(6), IMM(0x5))
        VMULPD(YMM(6), YMM(6), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(6), YMM(6), YMM(3))
        VDIVPD(YMM(6), YMM(6), YMM(9))
    #endif
    VMOVUPD(MEM(RCX, 0*16), YMM(4))
    VMOVUPD(MEM(RCX, 2*16), YMM(5))
    VMOVUPD(MEM(RCX, 4*16), YMM(6))
    ADD(RCX, RDI)

    //iteration 1 -------------------------------------

    VBROADCASTSD(YMM(0), MEM(RAX, (1+0*2)*16+0))
    VBROADCASTSD(YMM(1), MEM(RAX, (1+0*2)*16+8))

    VPERMILPD(YMM(3), YMM(4), IMM(0x5))
    VMULPD(YMM(2), YMM(4), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(7), YMM(2), YMM(3))

    VPERMILPD(YMM(3), YMM(5), IMM(0x5))
    VMULPD(YMM(2), YMM(5), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(8), YMM(2), YMM(3))

    VPERMILPD(YMM(3), YMM(6), IMM(0x5))
    VMULPD(YMM(2), YMM(6), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(9), YMM(2), YMM(3))

    VSUBPD(YMM(10), YMM(10), YMM(7))
    VSUBPD(YMM(11), YMM(11), YMM(8))
    VSUBPD(YMM(12), YMM(12), YMM(9))

    VBROADCASTSD(YMM(0), MEM(RAX, (1+1*2)*16+0))
    VBROADCASTSD(YMM(1), MEM(RAX, (1+1*2)*16+8))

    #ifdef BLIS_ENABLE_TRSM_PREINVERSION
        VPERMILPD(YMM(3), YMM(10), IMM(0x5))
        VMULPD(YMM(10), YMM(10), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VADDSUBPD(YMM(10), YMM(10), YMM(3))

        VPERMILPD(YMM(3), YMM(11), IMM(0x5))
        VMULPD(YMM(11), YMM(11), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VADDSUBPD(YMM(11), YMM(11), YMM(3))

        VPERMILPD(YMM(3), YMM(12), IMM(0x5))
        VMULPD(YMM(12), YMM(12), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VADDSUBPD(YMM(12), YMM(12), YMM(3))
    #else
        VMOVUPD(YMM(2), MEM(R11))
        VMULPD(YMM(9), YMM(0), YMM(0))
        VFMADD231PD(YMM(9), YMM(1), YMM(1))

        VPERMILPD(YMM(3), YMM(10), IMM(0x5))
        VMULPD(YMM(10), YMM(10), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(10), YMM(10), YMM(3))
        VDIVPD(YMM(10), YMM(10), YMM(9))

        VPERMILPD(YMM(3), YMM(11), IMM(0x5))
        VMULPD(YMM(11), YMM(11), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(11), YMM(11), YMM(3))
        VDIVPD(YMM(11), YMM(11), YMM(9))

        VPERMILPD(YMM(3), YMM(12), IMM(0x5))
        VMULPD(YMM(12), YMM(12), YMM(0))
        VMULPD(YMM(3), YMM(3), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(12), YMM(12), YMM(3))
        VDIVPD(YMM(12), YMM(12), YMM(9))
    #endif
    VMOVUPD(MEM(RCX, 0*16), YMM(10))
    VMOVUPD(MEM(RCX, 2*16), YMM(11))
    VMOVUPD(MEM(RCX, 4*16), YMM(12))

// ENDREGION - TRSM

    MOV(RAX, R12)
    MOV(RBX, R10)
    MOV(RCX, VAR(c11))

    CMP(RBX, IMM(16))
    JE(ROWUPDATE)

    LABEL(COLUPDATE)
        LEA(RDX, MEM(RCX, R12, 1))
        LEA(RDI, MEM(, R10, 2))

        VEXTRACTF128(XMM(3), YMM(4), IMM(0x1))
        VMOVUPD(MEM(RCX        ), XMM(4))
        VMOVUPD(MEM(RCX, R10, 1), XMM(3))
        ADD(RCX, RDI)

        VEXTRACTF128(XMM(3), YMM(5), IMM(0x1))
        VMOVUPD(MEM(RCX        ), XMM(5))
        VMOVUPD(MEM(RCX, R10, 1), XMM(3))
        ADD(RCX, RDI)

        VEXTRACTF128(XMM(3), YMM(6), IMM(0x1))
        VMOVUPD(MEM(RCX        ), XMM(6))
        VMOVUPD(MEM(RCX, R10, 1), XMM(3))


        VEXTRACTF128(XMM(3), YMM(10), IMM(0x1))
        VMOVUPD(MEM(RDX        ), XMM(10))
        VMOVUPD(MEM(RDX, R10, 1), XMM(3))
        ADD(RDX, RDI)

        VEXTRACTF128(XMM(3), YMM(11), IMM(0x1))
        VMOVUPD(MEM(RDX        ), XMM(11))
        VMOVUPD(MEM(RDX, R10, 1), XMM(3))
        ADD(RDX, RDI)

        VEXTRACTF128(XMM(3), YMM(12), IMM(0x1))
        VMOVUPD(MEM(RDX        ), XMM(12))
        VMOVUPD(MEM(RDX, R10, 1), XMM(3))
        JMP(END)


    LABEL(ROWUPDATE)
        LEA(RDX, MEM(RCX, R12, 1))

        VMOVUPD(MEM(RCX        ), YMM(4))
        VMOVUPD(MEM(RCX, R10, 2), YMM(5))
        VMOVUPD(MEM(RCX, R10, 4), YMM(6))

        VMOVUPD(MEM(RDX        ), YMM(10))
        VMOVUPD(MEM(RDX, R10, 2), YMM(11))
        VMOVUPD(MEM(RDX, R10, 4), YMM(12))
        JMP(END)

    LABEL(END)

    VZEROUPPER()


    END_ASM
    (
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
        [negPtr]    "m" (negPtr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
      "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6",
      "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12",
      "ymm13", "ymm14", "ymm15",
      "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
      "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12",
      "xmm13", "xmm14", "xmm15",
      "memory"
    )
}
