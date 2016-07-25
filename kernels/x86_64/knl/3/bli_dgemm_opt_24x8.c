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
#include <assert.h>

#include "bli_avx512_macros.h"

#define UNROLL_K 1

#define PREFETCH_A_L2 0
#define PREFETCH_B_L2 0
#define L2_PREFETCH_DIST 16

#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 2

#define C_MIN_L2_ITERS 40 //C is not prefetched into L2 for k <= this
#define C_L1_ITERS 16 //number of iterations before the end to prefetch C into L1
                      //make sure there is an unrolled MAIN_LOOP_X for this number

#define LOOP_ALIGN ALIGN16

#define UPDATE_C_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX      )) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,RAX,1)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,RAX,2)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,RDI,1)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_BZ_FOUR_ROWS(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX      ), ZMM(R1)) \
    VMOVUPD(MEM(RCX,RAX,1), ZMM(R2)) \
    VMOVUPD(MEM(RCX,RAX,2), ZMM(R3)) \
    VMOVUPD(MEM(RCX,RDI,1), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,4))

#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VGATHERDPD(ZMM(3) MASK_K(1), MEM(RCX,YMM(2),8)) \
    VFMADD231PD(ZMM(NUM), ZMM(3), ZMM(1)) \
    VSCATTERDPD(MEM(RCX,YMM(2),8) MASK_K(2), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(NUM), ZMM(NUM), ZMM(0)) \
    VSCATTERDPD(MEM(RCX,YMM(2),8) MASK_K(1), ZMM(NUM)) \
    ADD(RCX, RAX)

#define PREFETCH_A_L1(n) \
\
    PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8)) \
    PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8+64)) \
    PREFETCH(0, MEM(RAX,(A_L1_PREFETCH_DIST+n)*24*8+128))

#if PREFETCH_A_L2
#undef PREFETCH_A_L2

#define PREFETCH_A_L2(n) \
\
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*24*8)) \
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*24*8+64)) \
    PREFETCH(1, MEM(RAX,(L2_PREFETCH_DIST+n)*24*8+128))

#else
#undef PREFETCH_A_L2
#define PREFETCH_A_L2(...)
#endif

#define PREFETCH_B_L1(n) PREFETCH(0, MEM(RBX,(B_L1_PREFETCH_DIST+n)*8*8))

#if PREFETCH_B_L2
#undef PREFETCH_B_L2

#define PREFETCH_B_L2(n) PREFETCH(1, MEM(RBX,(L2_PREFETCH_DIST+n)*8*8))

#else
#undef PREFETCH_B_L2
#define PREFETCH_B_L2(...)
#endif

//
// n: index in unrolled loop (for prefetching offsets)
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for A, except for offset
//
#define SUBITER(n,a,b,...) \
\
        VMOVAPD(ZMM(0), MEM(RBX,(n+1-1)*64)) \
\
        PREFETCH_A_L1(n) \
        PREFETCH_B_L1(n) \
        PREFETCH_A_L2(n) \
        PREFETCH_B_L2(n) \
\
        VFMADD231PD(ZMM( 8), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 0)*8)) \
        VFMADD231PD(ZMM( 9), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 1)*8)) \
        VFMADD231PD(ZMM(10), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 2)*8)) \
        VFMADD231PD(ZMM(11), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 3)*8)) \
        VFMADD231PD(ZMM(12), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 4)*8)) \
        VFMADD231PD(ZMM(13), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 5)*8)) \
        VFMADD231PD(ZMM(14), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 6)*8)) \
        VFMADD231PD(ZMM(15), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 7)*8)) \
        VFMADD231PD(ZMM(16), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 8)*8)) \
        VFMADD231PD(ZMM(17), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+ 9)*8)) \
        VFMADD231PD(ZMM(18), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+10)*8)) \
        VFMADD231PD(ZMM(19), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+11)*8)) \
        VFMADD231PD(ZMM(20), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+12)*8)) \
        VFMADD231PD(ZMM(21), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+13)*8)) \
        VFMADD231PD(ZMM(22), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+14)*8)) \
        VFMADD231PD(ZMM(23), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+15)*8)) \
        VFMADD231PD(ZMM(24), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+16)*8)) \
        VFMADD231PD(ZMM(25), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+17)*8)) \
        VFMADD231PD(ZMM(26), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+18)*8)) \
        VFMADD231PD(ZMM(27), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+19)*8)) \
        VFMADD231PD(ZMM(28), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+20)*8)) \
        VFMADD231PD(ZMM(29), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+21)*8)) \
        VFMADD231PD(ZMM(30), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+22)*8)) \
        VFMADD231PD(ZMM(31), ZMM(0), MEM_1TO8(__VA_ARGS__,((n%%4)*24+23)*8))

#define TAIL_LOOP(NAME) \
\
    LOOP_ALIGN \
    LABEL(NAME) \
\
        SUBITER(0,1,0,RAX) \
\
        VMOVAPD(ZMM(0), ZMM(1)) \
\
        ADD(RAX, IMM(24*8)) \
        ADD(RBX, IMM( 8*8)) \
\
        SUB(RDI, IMM(1)) \
\
    JNZ(NAME)

#define MAIN_LOOP_1(NAME) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RAX) \
\
        VMOVAPD(ZMM(0), ZMM(1)) \
\
        ADD(RAX, IMM(24*8)) \
        ADD(RBX, IMM( 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP)

#define MAIN_LOOP_2(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(1)) \
    SAR1(RSI) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RAX) \
        SUBITER(1,0,1,RAX) \
\
        ADD(RAX, IMM(2*24*8)) \
        ADD(RBX, IMM(2* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    SUBITER(0,1,0,RAX) \
\
    VMOVAPD(ZMM(0), ZMM(1)) \
\
    ADD(RAX, IMM(24*8)) \
    ADD(RBX, IMM( 8*8)) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_4(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(3)) \
    SAR(RSI, IMM(2)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RAX) \
        SUBITER(1,0,1,RAX) \
        SUBITER(2,1,0,RAX) \
        SUBITER(3,0,1,RAX) \
\
        ADD(RAX, IMM(4*24*8)) \
        ADD(RBX, IMM(4* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_8(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(7)) \
    SAR(RSI, IMM(3)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER(0,1,0,RAX) \
        SUBITER(1,0,1,RAX) \
        SUBITER(2,1,0,RAX) \
        SUBITER(3,0,1,RAX) \
        SUBITER(4,1,0,RAX,R8,1) \
        SUBITER(5,0,1,RAX,R8,1) \
        SUBITER(6,1,0,RAX,R8,1) \
        SUBITER(7,0,1,RAX,R8,1) \
\
        ADD(RAX, IMM(8*24*8)) \
        ADD(RBX, IMM(8* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_16(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(15)) \
    SAR(RSI, IMM(4)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER( 0,1,0,RAX) \
        SUBITER( 1,0,1,RAX) \
        SUBITER( 2,1,0,RAX) \
        SUBITER( 3,0,1,RAX) \
        SUBITER( 4,1,0,RAX,R8,1) \
        SUBITER( 5,0,1,RAX,R8,1) \
        SUBITER( 6,1,0,RAX,R8,1) \
        SUBITER( 7,0,1,RAX,R8,1) \
        SUBITER( 8,1,0,RAX,R8,2) \
        SUBITER( 9,0,1,RAX,R8,2) \
        SUBITER(10,1,0,RAX,R8,2) \
        SUBITER(11,0,1,RAX,R8,2) \
        SUBITER(12,1,0,RAX,R9,1) \
        SUBITER(13,0,1,RAX,R9,1) \
        SUBITER(14,1,0,RAX,R9,1) \
        SUBITER(15,0,1,RAX,R9,1) \
\
        ADD(RAX, IMM(16*24*8)) \
        ADD(RBX, IMM(16* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define MAIN_LOOP_32(NAME) \
\
    MOV(RDI, RSI) \
    AND(RDI, IMM(31)) \
    SAR(RSI, IMM(5)) \
    JZ(NAME##_TAIL) \
\
    LOOP_ALIGN \
    LABEL(NAME##_LOOP) \
\
        SUBITER( 0,1,0,RAX) \
        SUBITER( 1,0,1,RAX) \
        SUBITER( 2,1,0,RAX) \
        SUBITER( 3,0,1,RAX) \
        SUBITER( 4,1,0,RAX,R8,1) \
        SUBITER( 5,0,1,RAX,R8,1) \
        SUBITER( 6,1,0,RAX,R8,1) \
        SUBITER( 7,0,1,RAX,R8,1) \
        SUBITER( 8,1,0,RAX,R8,2) \
        SUBITER( 9,0,1,RAX,R8,2) \
        SUBITER(10,1,0,RAX,R8,2) \
        SUBITER(11,0,1,RAX,R8,2) \
        SUBITER(12,1,0,RAX,R9,1) \
        SUBITER(13,0,1,RAX,R9,1) \
        SUBITER(14,1,0,RAX,R9,1) \
        SUBITER(15,0,1,RAX,R9,1) \
        SUBITER(16,1,0,RAX,R8,4) \
        SUBITER(17,0,1,RAX,R8,4) \
        SUBITER(18,1,0,RAX,R8,4) \
        SUBITER(19,0,1,RAX,R8,4) \
        SUBITER(20,1,0,RAX,R10,1) \
        SUBITER(21,0,1,RAX,R10,1) \
        SUBITER(22,1,0,RAX,R10,1) \
        SUBITER(23,0,1,RAX,R10,1) \
        SUBITER(24,1,0,RAX,R9,2) \
        SUBITER(25,0,1,RAX,R9,2) \
        SUBITER(26,1,0,RAX,R9,2) \
        SUBITER(27,0,1,RAX,R9,2) \
        SUBITER(28,1,0,RAX,R11,1) \
        SUBITER(29,0,1,RAX,R11,1) \
        SUBITER(30,1,0,RAX,R11,1) \
        SUBITER(31,0,1,RAX,R11,1) \
\
        ADD(RAX, IMM(32*24*8)) \
        ADD(RBX, IMM(32* 8*8)) \
\
        SUB(RSI, IMM(1)) \
\
    JNZ(NAME##_LOOP) \
\
    TEST(RDI, RDI) \
    JZ(NAME##_DONE) \
\
    TAIL_LOOP(NAME##_TAIL) \
\
    LABEL(NAME##_DONE)

#define LOOP_K_(M,K) M##K
#define LOOP_K(M,K,NAME) LOOP_K_(M,K)(NAME)

#define MAIN_LOOP_L2 LOOP_K(MAIN_LOOP_,UNROLL_K,MAIN_LOOP_L2)
//#define MAIN_LOOP_L1 LOOP_K(MAIN_LOOP_,C_L1_ITERS,MAIN_LOOP_L1)
#define MAIN_LOOP_L1 LOOP_K(MAIN_LOOP_,1,MAIN_LOOP_L1)

//This is an array used for the scatter/gather instructions.
extern int32_t offsets[24];

//#define MONITORS
//#define LOOPMON
void bli_dgemm_opt_24x8(
                    dim_t            k,
                    double* restrict alpha,
                    double* restrict a,
                    double* restrict b,
                    double* restrict beta,
                    double* restrict c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t*       data,
                    cntx_t* restrict cntx
                  )
{
    /*
    for (dim_t i = 0;i < 24;i++)
    {
        for (dim_t j = 0;j < 8;j++)
        {
            c[i*rs_c+j*cs_c] *= *beta;
            for (dim_t p = 0;p < k;p++)
            {
                c[i*rs_c+j*cs_c] += (*alpha)*a[i+p*24]*b[j+p*8];
            }
        }
    }

    return;
    */

    const double * a_next = bli_auxinfo_next_a( data );
    const double * b_next = bli_auxinfo_next_b( data );

    const int32_t * offsetPtr = &offsets[0];

    uint64_t k64 = k;

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif
    
    __asm__ volatile
    (
#ifdef MONITORS
    RDTSC
    MOV(VAR(topl), EAX)
    MOV(VAR(toph), EDX)
#endif

    VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
    VMOVAPS(ZMM( 9), ZMM(8))
    VMOVAPS(ZMM(10), ZMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPS(ZMM(11), ZMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPS(ZMM(12), ZMM(8))
    VMOVAPS(ZMM(13), ZMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPS(ZMM(14), ZMM(8))   //VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b
    VMOVAPS(ZMM(15), ZMM(8))
    VMOVAPS(ZMM(16), ZMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPS(ZMM(17), ZMM(8))   //set up indexing information for prefetching C
    VMOVAPS(ZMM(18), ZMM(8))   //MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(19), ZMM(8))   //VBROADCASTSS(ZMM(4), VAR(rs_c))
    VMOVAPS(ZMM(20), ZMM(8))   //VMOVAPS(ZMM(2), MEM(RDI)) //at this point zmm2 contains (0...15)
    VMOVAPS(ZMM(21), ZMM(8))   //VPMULLD(ZMM(2), ZMM(2), ZMM(4)) //and now zmm2 contains (rs_c*0...15)
    VMOVAPS(ZMM(22), ZMM(8))   //VMOVAPS(YMM(3), MEM(RDI,64)) //at this point ymm3 contains (16...23)
    VMOVAPS(ZMM(23), ZMM(8))   //VPMULLD(YMM(3), YMM(3), YMM(4)) //and now ymm3 contains (rs_c*16...23)
    VMOVAPS(ZMM(24), ZMM(8))
    VMOVAPS(ZMM(25), ZMM(8))   //MOV(R8, IMM(4*24*8))     //offset for 4 iterations
    VMOVAPS(ZMM(26), ZMM(8))   //LEA(R9, MEM(R8,R8,2))    //*3
    VMOVAPS(ZMM(27), ZMM(8))   //LEA(R10, MEM(R8,R8,4))   //*5
    VMOVAPS(ZMM(28), ZMM(8))   //LEA(R11, MEM(R9,R8,4))   //*7
    VMOVAPS(ZMM(29), ZMM(8))
    VMOVAPS(ZMM(30), ZMM(8))
    VMOVAPS(ZMM(31), ZMM(8))

#ifdef MONITORS
    RDTSC
    MOV(VAR(midl), EAX)
    MOV(VAR(midh), EDX)
#endif

    //need 0+... to satisfy preprocessor
    //CMP(RSI, IMM(0+C_MIN_L2_ITERS))
    //JLE(PREFETCH_C_L1)

    //SUB(RSI, IMM(0+C_L1_ITERS))

    //prefetch C into L2
    //KXNORW(K(1), K(0), K(0))
    //KXNORW(K(2), K(0), K(0))
    //VSCATTERPFDPS(1, MEM(RCX,ZMM(2),8) MASK_K(1))
    //VSCATTERPFDPD(1, MEM(RCX,YMM(3),8) MASK_K(2))

    //MAIN_LOOP_L2

    //MOV(RSI, IMM(0+C_L1_ITERS))

    //LABEL(PREFETCH_C_L1)

    //prefetch C into L1
    //KXNORW(K(1), K(0), K(0))
    //KXNORW(K(2), K(0), K(0))
    //VSCATTERPFDPS(0, MEM(RCX,ZMM(2),8) MASK_K(1))
    //VSCATTERPFDPD(0, MEM(RCX,YMM(3),8) MASK_K(2))

    //MAIN_LOOP_L1

    LABEL(MAINLOOP)

        VMOVAPD(ZMM(0), MEM(RBX))

        VFMADD231PD(ZMM( 8), ZMM(0), MEM_1TO8(RAX, 0*8))
        VFMADD231PD(ZMM( 9), ZMM(0), MEM_1TO8(RAX, 1*8))
        VFMADD231PD(ZMM(10), ZMM(0), MEM_1TO8(RAX, 2*8))
        VFMADD231PD(ZMM(11), ZMM(0), MEM_1TO8(RAX, 3*8))
        VFMADD231PD(ZMM(12), ZMM(0), MEM_1TO8(RAX, 4*8))
        VFMADD231PD(ZMM(13), ZMM(0), MEM_1TO8(RAX, 5*8))
        VFMADD231PD(ZMM(14), ZMM(0), MEM_1TO8(RAX, 6*8))
        VFMADD231PD(ZMM(15), ZMM(0), MEM_1TO8(RAX, 7*8))
        VFMADD231PD(ZMM(16), ZMM(0), MEM_1TO8(RAX, 8*8))
        VFMADD231PD(ZMM(17), ZMM(0), MEM_1TO8(RAX, 9*8))
        VFMADD231PD(ZMM(18), ZMM(0), MEM_1TO8(RAX,10*8))
        VFMADD231PD(ZMM(19), ZMM(0), MEM_1TO8(RAX,11*8))
        VFMADD231PD(ZMM(20), ZMM(0), MEM_1TO8(RAX,12*8))
        VFMADD231PD(ZMM(21), ZMM(0), MEM_1TO8(RAX,13*8))
        VFMADD231PD(ZMM(22), ZMM(0), MEM_1TO8(RAX,14*8))
        VFMADD231PD(ZMM(23), ZMM(0), MEM_1TO8(RAX,15*8))
        VFMADD231PD(ZMM(24), ZMM(0), MEM_1TO8(RAX,16*8))
        VFMADD231PD(ZMM(25), ZMM(0), MEM_1TO8(RAX,17*8))
        VFMADD231PD(ZMM(26), ZMM(0), MEM_1TO8(RAX,18*8))
        VFMADD231PD(ZMM(27), ZMM(0), MEM_1TO8(RAX,19*8))
        VFMADD231PD(ZMM(28), ZMM(0), MEM_1TO8(RAX,20*8))
        VFMADD231PD(ZMM(29), ZMM(0), MEM_1TO8(RAX,21*8))
        VFMADD231PD(ZMM(30), ZMM(0), MEM_1TO8(RAX,22*8))
        VFMADD231PD(ZMM(31), ZMM(0), MEM_1TO8(RAX,23*8))

        ADD(RAX, IMM(24*8))
        ADD(RBX, IMM( 8*8))

        SUB(RSI, IMM(1))

    JNZ(MAINLOOP)

    LABEL(POSTACCUM)

#ifdef MONITORS
    RDTSC
    MOV(VAR(mid2l), EAX)
    MOV(VAR(mid2h), EDX)
#endif

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    // Check if C is row stride. If not, jump to the slow scattered update
    MOV(RAX, VAR(rs_c))
    MOV(RBX, VAR(cs_c))
    LEA(RDI, MEM(RAX,RAX,2))
    CMP(RBX, IMM(1))
    //JNE(SCATTEREDUPDATE)
    JMP(SCATTEREDUPDATE)

    VMOVQ(RDX, XMM(1))
    SAL1(RDX) //shift out sign bit
    //JZ(COLSTORBZ)

    UPDATE_C_FOUR_ROWS( 8, 9,10,11)
    UPDATE_C_FOUR_ROWS(12,13,14,15)
    UPDATE_C_FOUR_ROWS(16,17,18,19)
    UPDATE_C_FOUR_ROWS(20,21,22,23)
    UPDATE_C_FOUR_ROWS(24,25,26,27)
    UPDATE_C_FOUR_ROWS(28,29,30,31)

    JMP(END)

    LABEL(COLSTORBZ)

    UPDATE_C_BZ_FOUR_ROWS( 8, 9,10,11)
    UPDATE_C_BZ_FOUR_ROWS(12,13,14,15)
    UPDATE_C_BZ_FOUR_ROWS(16,17,18,19)
    UPDATE_C_BZ_FOUR_ROWS(20,21,22,23)
    UPDATE_C_BZ_FOUR_ROWS(24,25,26,27)
    UPDATE_C_BZ_FOUR_ROWS(28,29,30,31)

    JMP(END)

    LABEL(SCATTEREDUPDATE)

    MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(2), MEM(RDI))
    /* Note that this ignores the upper 32 bits in cs_c */
    VBROADCASTSS(ZMM(3), VAR(cs_c))
    VPMULLD(ZMM(2), ZMM(3), ZMM(2))

    VMOVQ(RDX, XMM(1))
    SAL1(RDX) //shift out sign bit
    //JZ(SCATTERBZ)

    UPDATE_C_ROW_SCATTERED( 8)
    UPDATE_C_ROW_SCATTERED( 9)
    UPDATE_C_ROW_SCATTERED(10)
    UPDATE_C_ROW_SCATTERED(11)
    UPDATE_C_ROW_SCATTERED(12)
    UPDATE_C_ROW_SCATTERED(13)
    UPDATE_C_ROW_SCATTERED(14)
    UPDATE_C_ROW_SCATTERED(15)
    UPDATE_C_ROW_SCATTERED(16)
    UPDATE_C_ROW_SCATTERED(17)
    UPDATE_C_ROW_SCATTERED(18)
    UPDATE_C_ROW_SCATTERED(19)
    UPDATE_C_ROW_SCATTERED(20)
    UPDATE_C_ROW_SCATTERED(21)
    UPDATE_C_ROW_SCATTERED(22)
    UPDATE_C_ROW_SCATTERED(23)
    UPDATE_C_ROW_SCATTERED(24)
    UPDATE_C_ROW_SCATTERED(25)
    UPDATE_C_ROW_SCATTERED(26)
    UPDATE_C_ROW_SCATTERED(27)
    UPDATE_C_ROW_SCATTERED(28)
    UPDATE_C_ROW_SCATTERED(29)
    UPDATE_C_ROW_SCATTERED(30)
    UPDATE_C_ROW_SCATTERED(31)

    JMP(END)

    LABEL(SCATTERBZ)

    UPDATE_C_BZ_ROW_SCATTERED( 8)
    UPDATE_C_BZ_ROW_SCATTERED( 9)
    UPDATE_C_BZ_ROW_SCATTERED(10)
    UPDATE_C_BZ_ROW_SCATTERED(11)
    UPDATE_C_BZ_ROW_SCATTERED(12)
    UPDATE_C_BZ_ROW_SCATTERED(13)
    UPDATE_C_BZ_ROW_SCATTERED(14)
    UPDATE_C_BZ_ROW_SCATTERED(15)
    UPDATE_C_BZ_ROW_SCATTERED(16)
    UPDATE_C_BZ_ROW_SCATTERED(17)
    UPDATE_C_BZ_ROW_SCATTERED(18)
    UPDATE_C_BZ_ROW_SCATTERED(19)
    UPDATE_C_BZ_ROW_SCATTERED(20)
    UPDATE_C_BZ_ROW_SCATTERED(21)
    UPDATE_C_BZ_ROW_SCATTERED(22)
    UPDATE_C_BZ_ROW_SCATTERED(23)
    UPDATE_C_BZ_ROW_SCATTERED(24)
    UPDATE_C_BZ_ROW_SCATTERED(25)
    UPDATE_C_BZ_ROW_SCATTERED(26)
    UPDATE_C_BZ_ROW_SCATTERED(27)
    UPDATE_C_BZ_ROW_SCATTERED(28)
    UPDATE_C_BZ_ROW_SCATTERED(29)
    UPDATE_C_BZ_ROW_SCATTERED(30)
    UPDATE_C_BZ_ROW_SCATTERED(31)

    LABEL(END)

#ifdef MONITORS
    RDTSC
    MOV(VAR(botl), EAX)
    MOV(VAR(both), EDX)
#endif
    : // output operands
#ifdef MONITORS
      [topl]  "=m" (topl),
      [toph]  "=m" (toph),
      [midl]  "=m" (midl),
      [midh]  "=m" (midh),
      [mid2l] "=m" (mid2l),
      [mid2h] "=m" (mid2h),
      [botl]  "=m" (botl),
      [both]  "=m" (both)
#endif
    : // input operands
      [k]         "m" (k64),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c),
      [a_next]    "m" (a_next),
      [b_next]    "m" (b_next),
      [offsetPtr] "m" (offsetPtr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    );

#ifdef LOOPMON
    printf("looptime = \t%d\n", bloopl - tloopl);
#endif
#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    dim_t mid = ((dim_t)midh << 32) | midl;
    dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
#endif
}
