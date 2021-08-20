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

#define BLIS_ASM_SYNTAX_INTEL
#include "bli_x86_asm_macros.h"

#define LOADMUL8x8(a,o,s1,s3,s5,s7, \
                   z0,z1,z2,z3,z4,z5,z6,z7) \
    \
    VMULPD(ZMM(z0), ZMM(31), MEM(a,     o)) \
    VMULPD(ZMM(z1), ZMM(31), MEM(a,s1,1,o)) \
    VMULPD(ZMM(z2), ZMM(31), MEM(a,s1,2,o)) \
    VMULPD(ZMM(z3), ZMM(31), MEM(a,s3,1,o)) \
    VMULPD(ZMM(z4), ZMM(31), MEM(a,s1,4,o)) \
    VMULPD(ZMM(z5), ZMM(31), MEM(a,s5,1,o)) \
    VMULPD(ZMM(z6), ZMM(31), MEM(a,s3,2,o)) \
    VMULPD(ZMM(z7), ZMM(31), MEM(a,s7,1,o))

#define LOADMUL8x8_MASK(a,o,s1,s3,s5,s7, \
                        z0,z1,z2,z3,z4,z5,z6,z7,k) \
    \
    VMULPD(ZMM(z0) MASK_KZ(k), ZMM(31), MEM(a,     o)) \
    VMULPD(ZMM(z1) MASK_KZ(k), ZMM(31), MEM(a,s1,1,o)) \
    VMULPD(ZMM(z2) MASK_KZ(k), ZMM(31), MEM(a,s1,2,o)) \
    VMULPD(ZMM(z3) MASK_KZ(k), ZMM(31), MEM(a,s3,1,o)) \
    VMULPD(ZMM(z4) MASK_KZ(k), ZMM(31), MEM(a,s1,4,o)) \
    VMULPD(ZMM(z5) MASK_KZ(k), ZMM(31), MEM(a,s5,1,o)) \
    VMULPD(ZMM(z6) MASK_KZ(k), ZMM(31), MEM(a,s3,2,o)) \
    VMULPD(ZMM(z7) MASK_KZ(k), ZMM(31), MEM(a,s7,1,o))

#define STORE8x8(a,o,s1,s3,s5,s7, \
                 z0,z1,z2,z3,z4,z5,z6,z7) \
    \
    VMOVUPD(MEM(a,     o), ZMM(z0)) \
    VMOVUPD(MEM(a,s1,1,o), ZMM(z1)) \
    VMOVUPD(MEM(a,s1,2,o), ZMM(z2)) \
    VMOVUPD(MEM(a,s3,1,o), ZMM(z3)) \
    VMOVUPD(MEM(a,s1,4,o), ZMM(z4)) \
    VMOVUPD(MEM(a,s5,1,o), ZMM(z5)) \
    VMOVUPD(MEM(a,s3,2,o), ZMM(z6)) \
    VMOVUPD(MEM(a,s7,1,o), ZMM(z7))

#define TRANSPOSE8x8(a0,a1,a2,a3,a4,a5,a6,a7, \
                     b0,b1,b2,b3,b4,b5,b6,b7) \
    \
    VUNPCKLPD(ZMM(b0), ZMM(a0), ZMM(a1)) \
    VUNPCKHPD(ZMM(b1), ZMM(a0), ZMM(a1)) \
    VUNPCKLPD(ZMM(b2), ZMM(a2), ZMM(a3)) \
    VUNPCKHPD(ZMM(b3), ZMM(a2), ZMM(a3)) \
    VUNPCKLPD(ZMM(b4), ZMM(a4), ZMM(a5)) \
    VUNPCKHPD(ZMM(b5), ZMM(a4), ZMM(a5)) \
    VUNPCKLPD(ZMM(b6), ZMM(a6), ZMM(a7)) \
    VUNPCKHPD(ZMM(b7), ZMM(a6), ZMM(a7)) \
    VSHUFF64X2(ZMM(a0), ZMM(b0), ZMM(b2), IMM(0x44)) \
    VSHUFF64X2(ZMM(a1), ZMM(b1), ZMM(b3), IMM(0x44)) \
    VSHUFF64X2(ZMM(a2), ZMM(b0), ZMM(b2), IMM(0xEE)) \
    VSHUFF64X2(ZMM(a3), ZMM(b1), ZMM(b3), IMM(0xEE)) \
    VSHUFF64X2(ZMM(a4), ZMM(b4), ZMM(b6), IMM(0x44)) \
    VSHUFF64X2(ZMM(a5), ZMM(b5), ZMM(b7), IMM(0x44)) \
    VSHUFF64X2(ZMM(a6), ZMM(b4), ZMM(b6), IMM(0xEE)) \
    VSHUFF64X2(ZMM(a7), ZMM(b5), ZMM(b7), IMM(0xEE)) \
    VSHUFF64X2(ZMM(b0), ZMM(a0), ZMM(a4), IMM(0x88)) \
    VSHUFF64X2(ZMM(b1), ZMM(a1), ZMM(a5), IMM(0x88)) \
    VSHUFF64X2(ZMM(b2), ZMM(a0), ZMM(a4), IMM(0xDD)) \
    VSHUFF64X2(ZMM(b3), ZMM(a1), ZMM(a5), IMM(0xDD)) \
    VSHUFF64X2(ZMM(b4), ZMM(a2), ZMM(a6), IMM(0x88)) \
    VSHUFF64X2(ZMM(b5), ZMM(a3), ZMM(a7), IMM(0x88)) \
    VSHUFF64X2(ZMM(b6), ZMM(a2), ZMM(a6), IMM(0xDD)) \
    VSHUFF64X2(ZMM(b7), ZMM(a3), ZMM(a7), IMM(0xDD))

//This is an array used for the scatter/gather instructions.
static int32_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

void bli_dpackm_knl_asm_8xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim_,
       dim_t            n_,
       dim_t            n_max_,
       double* restrict kappa_,
       double* restrict a_, inc_t inca_, inc_t lda_,
       double* restrict p_,              inc_t ldp_,
       cntx_t* restrict cntx
     )
{
    const int32_t* offsetPtr = &offsets[0];

    double*       a     = ( double* )a_;
    double*       p     = ( double* )p_;
    double*       kappa = ( double* )kappa_;
    const int64_t cdim  = cdim_;
    const int64_t mnr   = 8;
    const int64_t n     = n_;
    const int64_t n_max = n_max_;
    const int64_t inca  = inca_;
    const int64_t lda   = lda_;
    const int64_t ldp   = ldp_;

    if ( cdim == mnr )
    {

    BEGIN_ASM()

    MOV(RSI, VAR(n))
    MOV(RAX, VAR(a))
    MOV(RBX, VAR(inca))
    MOV(RCX, VAR(lda))
    MOV(R14, VAR(p))
    MOV(RDI, VAR(ldp))

    TEST(RSI, RSI)
    JZ(PACK8_DONE)

    LEA(RBX, MEM(,RBX,8))    //inca in bytes
    LEA(RCX, MEM(,RCX,8))    //lda in bytes
    LEA(RDI, MEM(,RDI,8))    //ldp in bytes
    LEA(R11, MEM(RDI,RDI,2)) //ldp*3
    LEA(R12, MEM(RDI,RDI,4)) //ldp*5
    LEA(R13, MEM(R11,RDI,4)) //ldp*7

    VBROADCASTSD(ZMM(31), VAR(kappa))

    CMP(RBX, IMM(8))
    JNE(PACK8_T)

    LABEL(PACK8_N)

        MOV(RDX, RSI)
        AND(RDX, IMM(7))
        SAR(RSI, IMM(3))
        JZ(PACK8_N_TAIL)

        LEA(R8,  MEM(RCX,RCX,2)) //lda*3
        LEA(R9,  MEM(RCX,RCX,4)) //lda*5
        LEA(R10, MEM(R8 ,RCX,4)) //lda*7

        LABEL(PACK8_N_LOOP)

            LOADMUL8x8(RAX,0,RCX,R8,R9,R10,0,1,2,3,4,5,6,7)
            STORE8x8(R14,0,RDI,R11,R12,R13,0,1,2,3,4,5,6,7)

            LEA(RAX, MEM(RAX,RCX,8))
            LEA(R14, MEM(R14,RDI,8))

            SUB(RSI, IMM(1))

        JNZ(PACK8_N_LOOP)

        TEST(RDX, RDX)
        JZ(PACK8_DONE)

        LABEL(PACK8_N_TAIL)

            VMULPD(ZMM(0), ZMM(31), MEM(RAX))
            VMOVUPD(MEM(R14), ZMM(0))

            LEA(RAX, MEM(RAX,RCX,1))
            LEA(R14, MEM(R14,RDI,1))

            SUB(RDX, IMM(1))

        JNZ(PACK8_N_TAIL)

        JMP(PACK8_DONE)

    LABEL(PACK8_T)

        CMP(RCX, IMM(8))
        JNE(PACK8_G)

        LEA(R8,  MEM(RBX,RBX,2)) //inca*3
        LEA(R9,  MEM(RBX,RBX,4)) //inca*5
        LEA(R10, MEM(R8 ,RBX,4)) //inca*7

        MOV(RDX, RSI)
        AND(RDX, IMM(7))
        SAR(RSI, IMM(3))
        JZ(PACK8_T_TAIL)

        LABEL(PACK8_T_LOOP)

            LOADMUL8x8(RAX,0,RBX,R8,R9,R10,0,1,2,3,4,5,6,7)
            TRANSPOSE8x8( 0, 1, 2, 3, 4, 5, 6, 7,
                         16,17,18,19,20,21,22,23)
            STORE8x8(R14,0,RDI,R11,R12,R13,16,17,18,19,20,21,22,23)

            LEA(RAX, MEM(RAX,RCX,8))
            LEA(R14, MEM(R14,RDI,8))

            SUB(RSI, IMM(1))

        JNZ(PACK8_T_LOOP)

        TEST(RDX, RDX)
        JZ(PACK8_DONE)

        LABEL(PACK8_T_TAIL)

        MOV(RSI, IMM(1))
        SHLX(RSI, RSI, RDX)
        SUB(RSI, IMM(1))
        KMOVW(K(1), ESI)  //mask for n%8 elements

        LOADMUL8x8_MASK(RAX,0,RBX,R8,R9,R10,0,1,2,3,4,5,6,7,1)
        TRANSPOSE8x8( 0, 1, 2, 3, 4, 5, 6, 7,
                      8, 9,10,11,12,13,14,15)

        VMOVUPD(MEM(R14      ), ZMM( 8))
        SUB(RDX, IMM(1))
        JZ(PACK8_DONE)
        VMOVUPD(MEM(R14,RDI,1), ZMM( 9))
        SUB(RDX, IMM(1))
        JZ(PACK8_DONE)
        VMOVUPD(MEM(R14,RDI,2), ZMM(10))
        SUB(RDX, IMM(1))
        JZ(PACK8_DONE)
        VMOVUPD(MEM(R14,R11,1), ZMM(11))
        SUB(RDX, IMM(1))
        JZ(PACK8_DONE)
        VMOVUPD(MEM(R14,RDI,4), ZMM(12))
        SUB(RDX, IMM(1))
        JZ(PACK8_DONE)
        VMOVUPD(MEM(R14,R12,1), ZMM(13))
        SUB(RDX, IMM(1))
        JZ(PACK8_DONE)
        VMOVUPD(MEM(R14,R11,2), ZMM(14))

        JMP(PACK8_DONE)

    LABEL(PACK8_G)

        VPBROADCASTD(ZMM(3), VAR(inca))
        MOV(RBX, VAR(offsetPtr))
        VPMULLD(YMM(0), YMM(3), MEM(RBX))

        LABEL(PACK8_G_LOOP)

            KXNORW(K(1), K(0), K(0))
            VGATHERDPD(ZMM(3) MASK_K(1), MEM(RAX,YMM(0),8))
            VMULPD(ZMM(3), ZMM(3), ZMM(31))
            VMOVUPD(MEM(R14), ZMM(3))

            LEA(RAX, MEM(RAX,RCX,1))
            LEA(R14, MEM(R14,RDI,1))

            SUB(RSI, IMM(1))

        JNZ(PACK8_G_LOOP)

    LABEL(PACK8_DONE)

    END_ASM(
        : //output operands
        : //input operands
          [n]         "m" (n),
          [kappa]     "m" (*kappa),
          [a]         "m" (a),
          [inca]      "m" (inca),
          [lda]       "m" (lda),
          [p]         "m" (p),
          [ldp]       "m" (ldp),
          [offsetPtr] "m" (offsetPtr)
        : //clobbers
          "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17",
          "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31",
          "rax", "rbx", "rcx", "rdx", "rdi", "rsi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "memory"
    )

	}
	else // if ( cdim < mnr )
	{
		bli_dscal2m_ex \
		( \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  ( trans_t )conja, \
		  cdim, \
		  n, \
		  kappa, \
		  a, inca, lda, \
		  p, 1,    ldp, \
		  cntx, \
		  NULL  \
		); \

		// if ( cdim < mnr )
		{
			const dim_t      i      = cdim;
			const dim_t      m_edge = mnr - i;
			const dim_t      n_edge = n_max;
			double* restrict p_edge = p + (i  )*1;

			bli_dset0s_mxn
			(
			  m_edge,
			  n_edge,
			  p_edge, 1, ldp
			);
		}
	}

	if ( n < n_max )
	{
		const dim_t      j      = n;
		const dim_t      m_edge = mnr;
		const dim_t      n_edge = n_max - j;
		double* restrict p_edge = p + (j  )*ldp;

		bli_dset0s_mxn
		(
		  m_edge,
		  n_edge,
		  p_edge, 1, ldp
		);
	}
}

void bli_dpackm_knl_asm_24xk
     (
       conj_t           conja,
       pack_t           schema,
       dim_t            cdim_,
       dim_t            n_,
       dim_t            n_max_,
       double* restrict kappa_,
       double* restrict a_, inc_t inca_, inc_t lda_,
       double* restrict p_,              inc_t ldp_,
       cntx_t* restrict cntx
     )
{
    const int32_t* offsetPtr = &offsets[0];

    double*       a     = ( double* )a_;
    double*       p     = ( double* )p_;
    double*       kappa = ( double* )kappa_;
    const int64_t cdim  = cdim_;
    const int64_t mnr   = 24;
    const int64_t n     = n_;
    const int64_t n_max = n_max_;
    const int64_t inca  = inca_;
    const int64_t lda   = lda_;
    const int64_t ldp   = ldp_;

    if ( cdim == mnr )
    {

    BEGIN_ASM()

        MOV(RSI, VAR(n))
        MOV(RAX, VAR(a))
        MOV(RBX, VAR(inca))
        MOV(RCX, VAR(lda))
        MOV(R15, VAR(p))
        MOV(RDI, VAR(ldp))

        LEA(RBX, MEM(,RBX,8))    //inca in bytes
        LEA(RCX, MEM(,RCX,8))    //lda in bytes
        LEA(RDI, MEM(,RDI,8))    //ldp in bytes
        LEA(R11, MEM(RDI,RDI,2)) //ldp*3
        LEA(R12, MEM(RDI,RDI,4)) //ldp*5
        LEA(R13, MEM(R11,RDI,4)) //ldp*7

        VBROADCASTSD(ZMM(31), VAR(kappa))

        TEST(RSI, RSI)
        JZ(PACK24_DONE)

        CMP(RBX, IMM(8))
        JNE(PACK24_T)

        LABEL(PACK24_N)

            SAR(RSI, IMM(3))
            JZ(PACK24_N_TAIL)

            LEA(R8,  MEM(RCX,RCX,2)) //lda*3
            LEA(R9,  MEM(RCX,RCX,4)) //lda*5
            LEA(R10, MEM(R8 ,RCX,4)) //lda*7

            LABEL(PACK24_N_LOOP)

                LOADMUL8x8(RAX,  0,RCX,R8,R9,R10, 0, 1, 2, 3, 4, 5, 6, 7)
                LOADMUL8x8(RAX, 64,RCX,R8,R9,R10, 8, 9,10,11,12,13,14,15)
                LOADMUL8x8(RAX,128,RCX,R8,R9,R10,16,17,18,19,20,21,22,23)
                STORE8x8(R15,  0,RDI,R11,R12,R13, 0, 1, 2, 3, 4, 5, 6, 7)
                STORE8x8(R15, 64,RDI,R11,R12,R13, 8, 9,10,11,12,13,14,15)
                STORE8x8(R15,128,RDI,R11,R12,R13,16,17,18,19,20,21,22,23)

                LEA(RAX, MEM(RAX,RCX,8))
                LEA(R15, MEM(R15,RDI,8))

                SUB(RSI, IMM(1))

            JNZ(PACK24_N_LOOP)

            LABEL(PACK24_N_TAIL)

            MOV(RSI, VAR(n))
            AND(RSI, IMM(7))
            TEST(RSI, RSI)
            JZ(PACK24_DONE)

            LABEL(PACK24_N_TAIL_LOOP)

                VMULPD(ZMM(0), ZMM(31), MEM(RAX,  0))
                VMULPD(ZMM(1), ZMM(31), MEM(RAX, 64))
                VMULPD(ZMM(2), ZMM(31), MEM(RAX,128))
                VMOVUPD(MEM(R15,  0), ZMM(0))
                VMOVUPD(MEM(R15, 64), ZMM(1))
                VMOVUPD(MEM(R15,128), ZMM(2))

                LEA(RAX, MEM(RAX,RCX,1))
                LEA(R15, MEM(R15,RDI,1))

                SUB(RSI, IMM(1))

            JNZ(PACK24_N_TAIL_LOOP)

            JMP(PACK24_DONE)

        LABEL(PACK24_T)

            CMP(RCX, IMM(8))
            JNE(PACK24_G)

            LEA(R8,  MEM(RBX,RBX,2)) //inca*3
            LEA(R9,  MEM(RBX,RBX,4)) //inca*5
            LEA(R10, MEM(R8 ,RBX,4)) //inca*7

            LEA(R14, MEM(RAX,RBX,8))
            LEA(RCX, MEM(R14,RBX,8))

            SAR(RSI, IMM(3))
            JZ(PACK24_T_TAIL)

            LABEL(PACK24_T_LOOP)

                LOADMUL8x8(RAX,0,RBX,R8,R9,R10, 0, 1, 2, 3, 4, 5, 6, 7)
                LOADMUL8x8(R14,0,RBX,R8,R9,R10, 8, 9,10,11,12,13,14,15)
                TRANSPOSE8x8( 0, 1, 2, 3, 4, 5, 6, 7,
                             16,17,18,19,20,21,22,23)
                STORE8x8(R15,  0,RDI,R11,R12,R13,16,17,18,19,20,21,22,23)
                LOADMUL8x8(RCX,0,RBX,R8,R9,R10, 0, 1, 2, 3, 4, 5, 6, 7)
                TRANSPOSE8x8( 8, 9,10,11,12,13,14,15,
                             16,17,18,19,20,21,22,23)
                STORE8x8(R15, 64,RDI,R11,R12,R13,16,17,18,19,20,21,22,23)
                TRANSPOSE8x8( 0, 1, 2, 3, 4, 5, 6, 7,
                             16,17,18,19,20,21,22,23)
                STORE8x8(R15,128,RDI,R11,R12,R13,16,17,18,19,20,21,22,23)

                LEA(RAX, MEM(RAX,64))
                LEA(R14, MEM(R14,64))
                LEA(RCX, MEM(RCX,64))
                LEA(R15, MEM(R15,RDI,8))

                SUB(RSI, IMM(1))

            JNZ(PACK24_T_LOOP)

            LABEL(PACK24_T_TAIL)

            MOV(RSI, VAR(n))
            AND(RSI, IMM(7))
            TEST(RSI, RSI)
            JZ(PACK24_DONE)

            MOV(R13, IMM(1))
            SHLX(R13, R13, RSI)
            SUB(R13, IMM(1))
            KMOVW(K(1), R13D)  //mask for n%8 elements

            LOADMUL8x8_MASK(RAX,0,RBX,R8,R9,R10, 0, 1, 2, 3, 4, 5, 6, 7,1)
            LOADMUL8x8_MASK(R14,0,RBX,R8,R9,R10, 8, 9,10,11,12,13,14,15,1)
            LOADMUL8x8_MASK(RCX,0,RBX,R8,R9,R10,16,17,18,19,20,21,22,23,1)
            TRANSPOSE8x8(16,17,18,19,20,21,22,23,
                         24,25,26,27,28,29,30,31)
            TRANSPOSE8x8( 8, 9,10,11,12,13,14,15,
                         16,17,18,19,20,21,22,23)
            TRANSPOSE8x8( 0, 1, 2, 3, 4, 5, 6, 7,
                          8, 9,10,11,12,13,14,15)

            VMOVUPD(MEM(R15,        0), ZMM( 8))
            VMOVUPD(MEM(R15,       64), ZMM(16))
            VMOVUPD(MEM(R15,      128), ZMM(24))
            SUB(RSI, IMM(1))
            JZ(PACK24_DONE)
            VMOVUPD(MEM(R15,RDI,1,  0), ZMM( 9))
            VMOVUPD(MEM(R15,RDI,1, 64), ZMM(17))
            VMOVUPD(MEM(R15,RDI,1,128), ZMM(25))
            SUB(RSI, IMM(1))
            JZ(PACK24_DONE)
            VMOVUPD(MEM(R15,RDI,2,  0), ZMM(10))
            VMOVUPD(MEM(R15,RDI,2, 64), ZMM(18))
            VMOVUPD(MEM(R15,RDI,2,128), ZMM(26))
            SUB(RSI, IMM(1))
            JZ(PACK24_DONE)
            VMOVUPD(MEM(R15,R11,1,  0), ZMM(11))
            VMOVUPD(MEM(R15,R11,1, 64), ZMM(19))
            VMOVUPD(MEM(R15,R11,1,128), ZMM(27))
            SUB(RSI, IMM(1))
            JZ(PACK24_DONE)
            VMOVUPD(MEM(R15,RDI,4,  0), ZMM(12))
            VMOVUPD(MEM(R15,RDI,4, 64), ZMM(20))
            VMOVUPD(MEM(R15,RDI,4,128), ZMM(28))
            SUB(RSI, IMM(1))
            JZ(PACK24_DONE)
            VMOVUPD(MEM(R15,R12,1,  0), ZMM(13))
            VMOVUPD(MEM(R15,R12,1, 64), ZMM(21))
            VMOVUPD(MEM(R15,R12,1,128), ZMM(29))
            SUB(RSI, IMM(1))
            JZ(PACK24_DONE)
            VMOVUPD(MEM(R15,R11,2,  0), ZMM(14))
            VMOVUPD(MEM(R15,R11,2, 64), ZMM(22))
            VMOVUPD(MEM(R15,R11,2,128), ZMM(30))

            JMP(PACK24_DONE)

        LABEL(PACK24_G)

            VPBROADCASTD(ZMM(3), VAR(inca))
            MOV(RBX, VAR(offsetPtr))
            VPMULLD(YMM(0), YMM(3), MEM(RBX, 0))
            VPMULLD(YMM(1), YMM(3), MEM(RBX,32))
            VPMULLD(YMM(2), YMM(3), MEM(RBX,64))

            LABEL(PACK24_G_LOOP)

                KXNORW(K(1), K(0), K(0))
                KXNORW(K(2), K(0), K(0))
                KXNORW(K(3), K(0), K(0))
                VGATHERDPD(ZMM(3) MASK_K(1), MEM(RAX,YMM(0),8))
                VGATHERDPD(ZMM(4) MASK_K(2), MEM(RAX,YMM(1),8))
                VGATHERDPD(ZMM(5) MASK_K(3), MEM(RAX,YMM(2),8))
                VMULPD(ZMM(3), ZMM(3), ZMM(31))
                VMULPD(ZMM(4), ZMM(4), ZMM(31))
                VMULPD(ZMM(5), ZMM(5), ZMM(31))
                VMOVUPD(MEM(R15,  0), ZMM(3))
                VMOVUPD(MEM(R15, 64), ZMM(4))
                VMOVUPD(MEM(R15,128), ZMM(5))

                LEA(RAX, MEM(RAX,RCX,1))
                LEA(R15, MEM(R15,RDI,1))

                SUB(RSI, IMM(1))

            JNZ(PACK24_G_LOOP)

        LABEL(PACK24_DONE)

    END_ASM(
        : //output operands
        : //input operands
          [n]         "m" (n),
          [kappa]     "m" (*kappa),
          [a]         "m" (a),
          [inca]      "m" (inca),
          [lda]       "m" (lda),
          [p]         "m" (p),
          [ldp]       "m" (ldp),
          [offsetPtr] "m" (offsetPtr)
        : //clobbers
          "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17",
          "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23",
          "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
          "zmm30", "zmm31",
          "rax", "rbx", "rcx", "rdi", "rsi",
          "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory"
    )

	}
	else // if ( cdim < mnr )
	{
		bli_dscal2m_ex \
		( \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  ( trans_t )conja, \
		  cdim, \
		  n, \
		  kappa, \
		  a, inca, lda, \
		  p, 1,    ldp, \
		  cntx, \
		  NULL  \
		); \

		// if ( cdim < mnr )
		{
			const dim_t      i      = cdim;
			const dim_t      m_edge = mnr - i;
			const dim_t      n_edge = n_max;
			double* restrict p_edge = p + (i  )*1;

			bli_dset0s_mxn
			(
			  m_edge,
			  n_edge,
			  p_edge, 1, ldp
			);
		}
	}

	if ( n < n_max )
	{
		const dim_t      j      = n;
		const dim_t      m_edge = mnr;
		const dim_t      n_edge = n_max - j;
		double* restrict p_edge = p + (j  )*ldp;

		bli_dset0s_mxn
		(
		  m_edge,
		  n_edge,
		  p_edge, 1, ldp
		);
	}
}
