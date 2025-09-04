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

#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 4
#define TAIL_NITER 4
#define PREFETCH_A
// #define PREFETCH_B
// #define PREFETCH_A_NEXT
// #define PREFETCH_B_NEXT
#define PREFETCH_C              // perfetch c in middle loop over 2 iterations of k
// #define PREFETCH_C_SLOW      // prefetch c in middle loop over 4 iterations of k
// #define PREFETCH_C_SIMPL     // prefetch c before k loop


#ifdef PREFETCH_A
    #define PREFETCH_A_L1(n, k) \
        PREFETCH(0, MEM(RAX, A_L1_PREFETCH_DIST*2*16 + (2*n+k)*(16)))
#else
    #define PREFETCH_A_L1(n, k)
#endif

#ifdef PREFETCH_B
    #define PREFETCH_B_L1(n, k) \
        PREFETCH(0, MEM(RBX, B_L1_PREFETCH_DIST*6*16 + (6*n+(2*k))*(16)))
#else
    #define PREFETCH_B_L1(n, k)
#endif


/*
 * A Registers:  YMM3
 * B Registers:  YMM0, YMM1, YMM2
 * C Registers:  YMM[4-15]
 */

#define LOOP_ALIGN ALIGN32

#define SUBITER(n) \
\
    PREFETCH_A_L1(n, 0)\
    VBROADCASTSD(YMM(3), MEM(RAX,(4*n+0)*8)) \
    VFMADD231PD(YMM(4), YMM(0), YMM(3)) \
    VFMADD231PD(YMM(5), YMM(1), YMM(3)) \
    VFMADD231PD(YMM(6), YMM(2), YMM(3)) \
    VBROADCASTSD(YMM(3), MEM(RAX,(4*n+1)*8)) \
    VFMADD231PD(YMM(7), YMM(0), YMM(3)) \
    VFMADD231PD(YMM(8), YMM(1), YMM(3)) \
    VFMADD231PD(YMM(9), YMM(2), YMM(3)) \
    \
    PREFETCH_B_L1(n, 0)\
    VBROADCASTSD(YMM( 3), MEM(RAX,(4*n+2)*8)) \
    VFMADD231PD(YMM(10), YMM(0), YMM(3)) \
    VFMADD231PD(YMM(11), YMM(1), YMM(3)) \
    VFMADD231PD(YMM(12), YMM(2), YMM(3)) \
    VBROADCASTSD(YMM( 3), MEM(RAX,(4*n+3)*8)) \
    VFMADD231PD(YMM(13), YMM(0), YMM(3)) \
    VFMADD231PD(YMM(14), YMM(1), YMM(3)) \
    VFMADD231PD(YMM(15), YMM(2), YMM(3)) \
    \
    VMOVAPD(YMM(0), MEM(RBX,(6*n+0)*16)) \
    VMOVAPD(YMM(1), MEM(RBX,(6*n+2)*16)) \
    VMOVAPD(YMM(2), MEM(RBX,(6*n+4)*16)) \
    \


/**********************************************************/
/* Kernel : bli_zgemm_zen_asm_2x6                         */
/* It performs  C = C * beta + alpha * A * B              */
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
void bli_zgemm_zen_asm_2x6(
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

    BEGIN_ASM()

    VXORPD(YMM( 4), YMM( 4), YMM( 4))
    VXORPD(YMM( 5), YMM( 5), YMM( 5))
    VMOVAPD(YMM(6) , YMM(4))
    VMOVAPD(YMM(7) , YMM(4))
    VMOVAPD(YMM(8) , YMM(4))
    VMOVAPD(YMM(9) , YMM(4))
    VMOVAPD(YMM(10), YMM(4))
    VMOVAPD(YMM(11), YMM(4))
    VMOVAPD(YMM(12), YMM(4))
    VMOVAPD(YMM(13), YMM(4))
    VMOVAPD(YMM(14), YMM(4))
    VMOVAPD(YMM(15), YMM(4))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    MOV(RBX, VAR(b)) //load address of b
    MOV(RCX, VAR(c)) //load address of c

    #ifdef PREFETCH_C
        LEA(R9, MEM(RCX, 63)) // c for prefetch, first cache line
        LEA(R8, MEM(RCX, 95)) // c for prefetch, second cache line
    #endif


    VMOVAPD(YMM(0), MEM(RBX, 0*16)) //pre-load b
    VMOVAPD(YMM(1), MEM(RBX, 2*16)) //pre-load b
    VMOVAPD(YMM(2), MEM(RBX, 4*16)) //pre-load b
    LEA(RBX, MEM(RBX,6*16)) //adjust b for pre-load

    MOV(R12, VAR(rs_c))
    MOV(R10, VAR(cs_c))

    #if defined PREFETCH_A_NEXT || defined PREFETCH_B_NEXT
        MOV(RDI, RSI)
        IMUL(RDI, IMM(16*2)) // rdi = k * 16*2
    #endif

    #ifdef PREFETCH_A_NEXT
        LEA(R14, MEM(RAX, RDI, 1)) // r14(a_next) = A + (k*16*2)
    #endif

    #ifdef PREFETCH_B_NEXT
        IMUL(RDI, IMM(3)) // rdi = k * 16*6
        LEA(R15, MEM(RBX, RDI, 1)) // r15(b_next) = B + (k*16*6)
    #endif


    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))

    /************************************************************/
    /* Operation:                                               */
    /* SUBITER = (Ar, Ai)*(Br, Bi) = Ar*(Br, Bi) , Ai*(Br, Bi)  */
    /* Prefetch_C_SIMPLE:                                       */
    /*          LOOP1:   k/4 - TAIL_NITER                       */
    /*          LOOP2:   0                                      */
    /*          LOOP3:   0                                      */
    /*          LOOP4:   TAIL_NITER                             */
    /* PREFETCH_C_SLOW:                                         */
    /*          LOOP1:   k/4 - TAIL_NITER - 4                   */
    /*          LOOP2:   2                                      */
    /*          LOOP3:   2                                      */
    /*          LOOP4:   TAIL_NITER                             */
    /* PREFETCH_C:                                              */
    /*          LOOP1:   k/4 - TAIL_NITER - 2                   */
    /*          LOOP2:   2                                      */
    /*          LOOP3:   0                                      */
    /*          LOOP4:   TAIL_NITER                             */
    /************************************************************/
    #ifdef PREFETCH_C
        #ifdef PREFETCH_C_SIMPLE
            /* prefetch c over 1 iteration of k*/
            SUB(RDI, IMM(0+TAIL_NITER))
        #elif defined PREFETCH_C_SLOW
            /* prefetch c over 4 iterations of k*/
            SUB(RDI, IMM(4+TAIL_NITER))
        #else
            /* prefetch c over 2 iterations of k*/
            SUB(RDI, IMM(2+TAIL_NITER))
        #endif
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
                PREFETCH(1, MEM(R14, 64))
            #endif
            SUB(RDI, IMM(1))
            SUBITER(2)
            #ifdef PREFETCH_B_NEXT
                PREFETCH(1, MEM(R15, 64))
            #endif
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*2*16))
            LEA(RBX, MEM(RBX,4*6*16))
            #ifdef PREFETCH_A_NEXT
                LEA(R14, MEM(R14,128))
            #endif
            #ifdef PREFETCH_B_NEXT
                LEA(R15, MEM(R15,64))
            #endif

        JNZ(LOOP1)

    LABEL(K_PREFETCH_C)

#ifdef PREFETCH_C
#if defined PREFETCH_C_SIMPLE
    /*****************************/
    /* prefetch 2x6 of C at once */
    /*****************************/
    PREFETCH(0, MEM(R9))
    PREFETCH(0, MEM(R9, 31))
    PREFETCH(0, MEM(R9,R12, 1))
    PREFETCH(0, MEM(R9,R12, 1, 31))
    PREFETCH(0, MEM(R9,R12, 2))
    PREFETCH(0, MEM(R9,R12, 2, 31))
#else
    ADD(RDI, IMM(2))
    JLE(K_TAIL_NITER)

        LOOP_ALIGN
        LABEL(LOOP2)
            #ifdef PREFETCH_C
                PREFETCH(0, MEM(R9))
            #endif
            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            #ifndef PREFETCH_C_SLOW
                /************************************************/
                /* if prefetch is being done over 2 iterations, */
                /* prefetch 2 cache lines per iteration         */
                /* prefetch one row of C per iteration of Loop2 */
                /************************************************/
                PREFETCH(0, MEM(R9,31))
            #endif
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*2*16))
            LEA(RBX, MEM(RBX,4*6*16))
            #ifdef PREFETCH_C
                LEA(R9, MEM(R9,R12,1))
            #endif
        JNZ(LOOP2)

    LABEL(K_TAIL_NITER)

    #ifdef PREFETCH_C_SLOW
    ADD(RDI, IMM(2))
    JLE(K_TAIL_NITER_2)

        LOOP_ALIGN
        LABEL(LOOP3)
            #ifdef PREFETCH_C
                PREFETCH(0, MEM(R8))
            #endif
            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*2*16))
            LEA(RBX, MEM(RBX,4*6*16))
            #ifdef PREFETCH_C
                LEA(R8, MEM(R8,R12,1))
            #endif
        JNZ(LOOP3)
    LABEL(K_TAIL_NITER_2)

    #endif //PREFETCH_C_SLOW

#endif //PREFETCH_C_SIMPLE
    ADD(RDI, IMM(0+TAIL_NITER))
    JLE(TAIL)

        LOOP_ALIGN
        LABEL(LOOP4)

            SUBITER(0)
            SUBITER(1)
            SUB(RDI, IMM(1))
            SUBITER(2)
            SUBITER(3)

            LEA(RAX, MEM(RAX,4*2*16))
            LEA(RBX, MEM(RBX,4*6*16))

        JNZ(LOOP4)

#endif //PREFETCH_C

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

    VPERMILPD(YMM( 7), YMM( 7), IMM(0x5))
    VPERMILPD(YMM( 8), YMM( 8), IMM(0x5))
    VPERMILPD(YMM( 9), YMM( 9), IMM(0x5))
    VPERMILPD(YMM(13), YMM(13), IMM(0x5))
    VPERMILPD(YMM(14), YMM(14), IMM(0x5))
    VPERMILPD(YMM(15), YMM(15), IMM(0x5))

    VADDSUBPD(YMM(4), YMM(4), YMM(7))
    VADDSUBPD(YMM(5), YMM(5), YMM(8))
    VADDSUBPD(YMM(6), YMM(6), YMM(9))

    VADDSUBPD(YMM(10), YMM(10), YMM(13))
    VADDSUBPD(YMM(11), YMM(11), YMM(14))
    VADDSUBPD(YMM(12), YMM(12), YMM(15))

    /******************/
    /* scale by alpha */
    /******************/
    MOV(RAX, VAR(alpha))
    VBROADCASTSD(YMM(0), MEM(RAX))
    VBROADCASTSD(YMM(1), MEM(RAX, 8))

    VPERMILPD(YMM(3), YMM(4), IMM(0X5))
    VMULPD(YMM(4), YMM(4), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(4), YMM(4), YMM(3))

    VPERMILPD(YMM(3), YMM(5), IMM(0X5))
    VMULPD(YMM(5), YMM(5), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(5), YMM(5), YMM(3))

    VPERMILPD(YMM(3), YMM(6), IMM(0X5))
    VMULPD(YMM(6), YMM(6), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(6), YMM(6), YMM(3))

    // ROW 2
    VPERMILPD(YMM(3), YMM(10), IMM(0X5))
    VMULPD(YMM(10), YMM(10), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(10), YMM(10), YMM(3))

    VPERMILPD(YMM(3), YMM(11), IMM(0X5))
    VMULPD(YMM(11), YMM(11), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(11), YMM(11), YMM(3))

    VPERMILPD(YMM(3), YMM(12), IMM(0X5))
    VMULPD(YMM(12), YMM(12), YMM(0))
    VMULPD(YMM(3), YMM(3), YMM(1))
    VADDSUBPD(YMM(12), YMM(12), YMM(3))


    MOV(RBX, VAR(beta))
    VBROADCASTSD(YMM(1), MEM(RBX))
    VBROADCASTSD(YMM(2), MEM(RBX, 8))


    MOV(AL, VAR(beta_mul_type))
    CMP(AL, IMM(0))
    JE(.ZBETAZERO)

    CMP(R10, IMM(16)) //CS == 1 IMPLIES ROW STORED
    JNZ(.ZCOLSTORED)

    LABEL(.ZROWSTORED)
        LEA(RDX, MEM(RCX, R12, 1))

        // ROW 1
        VMOVUPD(YMM(0), MEM(RCX))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(4))
        VMOVUPD(MEM(RCX), YMM(0))

        VMOVUPD(YMM(0), MEM(RCX, R10, 2))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(5))
        VMOVUPD(MEM(RCX, R10, 2), YMM(0))

        VMOVUPD(YMM(0), MEM(RCX, R10, 4))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(6))
        VMOVUPD(MEM(RCX, R10, 4), YMM(0))

        //ROW 2
        VMOVUPD(YMM(0), MEM(RDX))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(10))
        VMOVUPD(MEM(RDX), YMM(0))

        VMOVUPD(YMM(0), MEM(RDX, R10, 2))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(11))
        VMOVUPD(MEM(RDX, R10, 2), YMM(0))

        VMOVUPD(YMM(0), MEM(RDX, R10, 4))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(12))
        VMOVUPD(MEM(RDX, R10, 4), YMM(0))

        JMP(.ZDONE)

    LABEL(.ZCOLSTORED)
        LEA(RDX, MEM(RCX, R12, 1))
        LEA(RDI, MEM(, R10, 2))

    	VMOVUPD(XMM(0), MEM(RCX        ))
        VMOVUPD(XMM(3), MEM(RCX, R10, 1))
        VINSERTF128(YMM(0), YMM(0), XMM(3), IMM(0x1))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(4))
        VEXTRACTF128(XMM(3), YMM(0), IMM(0x1))
        VMOVUPD(MEM(RCX        ), XMM(0))
        VMOVUPD(MEM(RCX, R10, 1), XMM(3))
        ADD(RCX, RDI)

        VMOVUPD(XMM(0), MEM(RCX        ))
        VMOVUPD(XMM(3), MEM(RCX, R10, 1))
        VINSERTF128(YMM(0), YMM(0), XMM(3), IMM(0x1))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(5))
        VEXTRACTF128(XMM(3), YMM(0), IMM(0x1))
        VMOVUPD(MEM(RCX        ), XMM(0))
        VMOVUPD(MEM(RCX, R10, 1), XMM(3))
        ADD(RCX, RDI)

        VMOVUPD(XMM(0), MEM(RCX        ))
        VMOVUPD(XMM(3), MEM(RCX, R10, 1))
        VINSERTF128(YMM(0), YMM(0), XMM(3), IMM(0x1))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(6))
        VEXTRACTF128(XMM(3), YMM(0), IMM(0x1))
        VMOVUPD(MEM(RCX        ), XMM(0))
        VMOVUPD(MEM(RCX, R10, 1), XMM(3))


        VMOVUPD(XMM(0), MEM(RDX        ))
        VMOVUPD(XMM(3), MEM(RDX, R10, 1))
        VINSERTF128(YMM(0), YMM(0), XMM(3), IMM(0x1))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(10))
        VEXTRACTF128(XMM(3), YMM(0), IMM(0x1))
        VMOVUPD(MEM(RDX        ), XMM(0))
        VMOVUPD(MEM(RDX, R10, 1), XMM(3))
        ADD(RDX, RDI)

        VMOVUPD(XMM(0), MEM(RDX        ))
        VMOVUPD(XMM(3), MEM(RDX, R10, 1))
        VINSERTF128(YMM(0), YMM(0), XMM(3), IMM(0x1))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(11))
        VEXTRACTF128(XMM(3), YMM(0), IMM(0x1))
        VMOVUPD(MEM(RDX        ), XMM(0))
        VMOVUPD(MEM(RDX, R10, 1), XMM(3))
        ADD(RDX, RDI)

        VMOVUPD(XMM(0), MEM(RDX        ))
        VMOVUPD(XMM(3), MEM(RDX, R10, 1))
        VINSERTF128(YMM(0), YMM(0), XMM(3), IMM(0x1))
        VPERMILPD(YMM(3), YMM(0), IMM(0x5))
        VMULPD(YMM(0), YMM(0), YMM(1))
        VMULPD(YMM(3), YMM(3), YMM(2))
        VADDSUBPD(YMM(0), YMM(0), YMM(3))
        VADDPD(YMM(0), YMM(0), YMM(12))
        VEXTRACTF128(XMM(3), YMM(0), IMM(0x1))
        VMOVUPD(MEM(RDX        ), XMM(0))
        VMOVUPD(MEM(RDX, R10, 1), XMM(3))
        ADD(RDX, RDI)


        JMP(.ZDONE)

    LABEL(.ZBETAZERO)
        CMP(R12, IMM(16))
        JNZ(.ZROWSTORBZ)

    LABEL(.ZCOLSTORBZ)
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
        JMP(.ZDONE)


    LABEL(.ZROWSTORBZ)
        LEA(RDX, MEM(RCX, R12, 1))

        VMOVUPD(MEM(RCX), YMM(4))
        VMOVUPD(MEM(RCX, R10, 2), YMM(5))
        VMOVUPD(MEM(RCX, R10, 4), YMM(6))

        VMOVUPD(MEM(RDX), YMM(10))
        VMOVUPD(MEM(RDX, R10, 2), YMM(11))
        VMOVUPD(MEM(RDX, R10, 4), YMM(12))



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
      [cs_c]          "m" (cs_c)
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
