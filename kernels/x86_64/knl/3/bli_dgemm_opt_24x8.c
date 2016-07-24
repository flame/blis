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

#define PREFETCH_A_L2 0
#define PREFETCH_B_L2 0
#define L2_PREFETCH_DIST 16

#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 2

#define C_MIN_L2_ITERS 40 //C is not prefetched into L2 for k <= this
#define C_L1_ITERS 10 //number of iterations before the end to prefetch C into L1

#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VGATHERDPD(ZMM(3) MASK_K(1), MEM(RCX,YMM(2),8)) \
    VFMADD231PD(ZMM(NUM), ZMM(3), ZMM(1)) \
    VSCATTERDPD(MEM(RCX,YMM(2),8) MASK_K(2), ZMM(NUM)) \
    ADD(RCX, RAX)

#define UPDATE_C_BZ_ROW_SCATTERED(NUM) \
\
    KXNORW(K(1), K(0), K(0)) \
    VSCATTERDPD(MEM(RCX,YMM(2),8) MASK_K(1), ZMM(NUM)) \
    ADD(RCX, RAX)

//This is an array used for the scatter/gather instructions.
extern int32_t offsets[16];

//#define MONITORS
//#define LOOPMON
void bli_dgemm_opt_30x8(
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

    VPXORD(ZMM(8), ZMM(8), ZMM(8))) //clear out registers
    VMOVAPS(ZMM( 9), ZMM(8))
    VMOVAPS(ZMM(10), ZMM(8))
    VMOVAPS(ZMM(11), ZMM(8))
    VMOVAPS(ZMM(12), ZMM(8))
    VMOVAPS(ZMM(13), ZMM(8))
    VMOVAPS(ZMM(14), ZMM(8))
    VMOVAPS(ZMM(15), ZMM(8))
    VMOVAPS(ZMM(16), ZMM(8))
    VMOVAPS(ZMM(17), ZMM(8))
    VMOVAPS(ZMM(18), ZMM(8))
    VMOVAPS(ZMM(19), ZMM(8))
    VMOVAPS(ZMM(20), ZMM(8))
    VMOVAPS(ZMM(21), ZMM(8))
    VMOVAPS(ZMM(22), ZMM(8))
    VMOVAPS(ZMM(23), ZMM(8))
    VMOVAPS(ZMM(24), ZMM(8))
    VMOVAPS(ZMM(25), ZMM(8))
    VMOVAPS(ZMM(26), ZMM(8))
    VMOVAPS(ZMM(27), ZMM(8))
    VMOVAPS(ZMM(28), ZMM(8))
    VMOVAPS(ZMM(29), ZMM(8))
    VMOVAPS(ZMM(30), ZMM(8))
    VMOVAPS(ZMM(31), ZMM(8))

    MOV(RSI, VAR(k)) //loop index
    MOV(RAX, VAR(a)) //load address of a
    ADD(RAX, IMM(15*8)) //offset a address to keep instructions <= 8 bytes
    MOV(RBX, VAR(b)) //load address of b
    VMOVAPD(ZMM(0), MEM(RBX)) //pre-load b
    MOV(RCX, VAR(c)) //load address of c

    //set up indexing information for prefetching C
    MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(2), MEM(RDI)) //at this point zmm2 contains (0...15)
    VBROADCASTSS(ZMM(3), MEM(cs_c))
    VPMULLD(ZMM(2), ZMM(2), ZMM(3)) //and now zmm2 contains (rs_c*0...15)
    MOV(RDX, IMM(0xFFFF)) //mask for prefetching (i.e. get all lines)

#ifdef MONITORS
    RDTSC
    MOV(VAR(midl), EAX)
    MOV(VAR(midh), EDX)
#endif

    //need 0+... to satisfy preprocessor
    CMP(RSI, IMM(0+C_MIN_L2_ITERS))
    JLE(PREFETCH_C_L1)

    MOV(RDI, IMM(0+C_L1_ITERS))
    SUB(RSI, RDI)

    //prefetch C into L2
    KMOV(K(1), RDX)
    KMOV(K(2), RDX)
    VSCATTERPFDPS(1, MEM(RCX,ZMM(2),8     ) MASK_K(1))
    VSCATTERPFDPD(1, MEM(RCX,YMM(2),8,16*8) MASK_K(2))

    LABEL(LOOP1)

        VMOVAPD(ZMM(1), MEM(RBX,8*8))

        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*8*24-15*8))
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*8*24- 7*8))
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*8*24+ 1*8))
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*8*8))

#if PREFETCH_A_L2
        PREFETCH(1, MEM(RAX,L2_PREFETCH_DIST*8*24-15*8))
        PREFETCH(1, MEM(RAX,L2_PREFETCH_DIST*8*24- 7*8))
        PREFETCH(1, MEM(RAX,L2_PREFETCH_DIST*8*24+ 1*8))
#endif

#if PREFETCH_B_L2
        PREFETCH(1, MEM(RBX,L2_PREFETCH_DIST*8*8))
#endif

        VFMADD231PD(ZMM( 8), ZMM(0), MEM_1TO8(RAX,-15*8))
        VFMADD231PD(ZMM( 9), ZMM(0), MEM_1TO8(RAX,-14*8))
        VFMADD231PD(ZMM(10), ZMM(0), MEM_1TO8(RAX,-13*8))
        VFMADD231PD(ZMM(11), ZMM(0), MEM_1TO8(RAX,-12*8))
        VFMADD231PD(ZMM(12), ZMM(0), MEM_1TO8(RAX,-11*8))
        VFMADD231PD(ZMM(13), ZMM(0), MEM_1TO8(RAX,-10*8))
        VFMADD231PD(ZMM(14), ZMM(0), MEM_1TO8(RAX, -9*8))
        VFMADD231PD(ZMM(15), ZMM(0), MEM_1TO8(RAX, -8*8))
        VFMADD231PD(ZMM(16), ZMM(0), MEM_1TO8(RAX, -7*8))
        VFMADD231PD(ZMM(17), ZMM(0), MEM_1TO8(RAX, -6*8))
        VFMADD231PD(ZMM(18), ZMM(0), MEM_1TO8(RAX, -5*8))
        VFMADD231PD(ZMM(19), ZMM(0), MEM_1TO8(RAX, -4*8))
        VFMADD231PD(ZMM(20), ZMM(0), MEM_1TO8(RAX, -3*8))
        VFMADD231PD(ZMM(21), ZMM(0), MEM_1TO8(RAX, -2*8))
        VFMADD231PD(ZMM(22), ZMM(0), MEM_1TO8(RAX, -1*8))
        VFMADD231PD(ZMM(23), ZMM(0), MEM_1TO8(RAX,  0*8))
        VFMADD231PD(ZMM(24), ZMM(0), MEM_1TO8(RAX,  1*8))
        VFMADD231PD(ZMM(25), ZMM(0), MEM_1TO8(RAX,  2*8))
        VFMADD231PD(ZMM(26), ZMM(0), MEM_1TO8(RAX,  3*8))
        VFMADD231PD(ZMM(27), ZMM(0), MEM_1TO8(RAX,  4*8))
        VFMADD231PD(ZMM(28), ZMM(0), MEM_1TO8(RAX,  5*8))
        VFMADD231PD(ZMM(29), ZMM(0), MEM_1TO8(RAX,  6*8))
        VFMADD231PD(ZMM(30), ZMM(0), MEM_1TO8(RAX,  7*8))
        VFMADD231PD(ZMM(31), ZMM(0), MEM_1TO8(RAX,  8*8))

        VMOVAPD(ZMM(0), ZMM(1))

        ADD(RAX, 24*8)
        ADD(RBX,  8*8)

        SUB(RSI, IMM(1))

    JNZ(LOOP1)

    MOV(RSI, RDI)

    LABEL(PREFETCH_C_L1)

    //prefetch C into L1
    KMOV(K(1), RDX)
    KMOV(K(2), RDX)
    VSCATTERPFDPS(0, MEM(RCX,ZMM(2),8     ) MASK_K(1))
    VSCATTERPFDPD(0, MEM(RCX,YMM(2),8,16*8) MASK_K(2))

    LABEL(LOOP2)

        VMOVAPD(ZMM(1), MEM(RBX,8*8))

        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*8*24-15*8))
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*8*24- 7*8))
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*8*24+ 1*8))
        PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*8*8))

#if PREFETCH_A_L2
        PREFETCH(1, MEM(RAX,L2_PREFETCH_DIST*8*24-15*8))
        PREFETCH(1, MEM(RAX,L2_PREFETCH_DIST*8*24- 7*8))
        PREFETCH(1, MEM(RAX,L2_PREFETCH_DIST*8*24+ 1*8))
#endif

#if PREFETCH_B_L2
        PREFETCH(1, MEM(RBX,L2_PREFETCH_DIST*8*8))
#endif

        VFMADD231PD(ZMM( 8), ZMM(0), MEM_1TO8(RAX,-15*8))
        VFMADD231PD(ZMM( 9), ZMM(0), MEM_1TO8(RAX,-14*8))
        VFMADD231PD(ZMM(10), ZMM(0), MEM_1TO8(RAX,-13*8))
        VFMADD231PD(ZMM(11), ZMM(0), MEM_1TO8(RAX,-12*8))
        VFMADD231PD(ZMM(12), ZMM(0), MEM_1TO8(RAX,-11*8))
        VFMADD231PD(ZMM(13), ZMM(0), MEM_1TO8(RAX,-10*8))
        VFMADD231PD(ZMM(14), ZMM(0), MEM_1TO8(RAX, -9*8))
        VFMADD231PD(ZMM(15), ZMM(0), MEM_1TO8(RAX, -8*8))
        VFMADD231PD(ZMM(16), ZMM(0), MEM_1TO8(RAX, -7*8))
        VFMADD231PD(ZMM(17), ZMM(0), MEM_1TO8(RAX, -6*8))
        VFMADD231PD(ZMM(18), ZMM(0), MEM_1TO8(RAX, -5*8))
        VFMADD231PD(ZMM(19), ZMM(0), MEM_1TO8(RAX, -4*8))
        VFMADD231PD(ZMM(20), ZMM(0), MEM_1TO8(RAX, -3*8))
        VFMADD231PD(ZMM(21), ZMM(0), MEM_1TO8(RAX, -2*8))
        VFMADD231PD(ZMM(22), ZMM(0), MEM_1TO8(RAX, -1*8))
        VFMADD231PD(ZMM(23), ZMM(0), MEM_1TO8(RAX,  0*8))
        VFMADD231PD(ZMM(24), ZMM(0), MEM_1TO8(RAX,  1*8))
        VFMADD231PD(ZMM(25), ZMM(0), MEM_1TO8(RAX,  2*8))
        VFMADD231PD(ZMM(26), ZMM(0), MEM_1TO8(RAX,  3*8))
        VFMADD231PD(ZMM(27), ZMM(0), MEM_1TO8(RAX,  4*8))
        VFMADD231PD(ZMM(28), ZMM(0), MEM_1TO8(RAX,  5*8))
        VFMADD231PD(ZMM(29), ZMM(0), MEM_1TO8(RAX,  6*8))
        VFMADD231PD(ZMM(30), ZMM(0), MEM_1TO8(RAX,  7*8))
        VFMADD231PD(ZMM(31), ZMM(0), MEM_1TO8(RAX,  8*8))

        VMOVAPD(ZMM(0), ZMM(1))

        ADD(RAX, 24*8)
        ADD(RBX,  8*8)

        SUB(RSI, IMM(1))

    JNZ(LOOP2)

    LABEL(POSTACCUM)

#ifdef MONITORS
    RDTSC
    MOV(VAR(mid2l), EAX)
    MOV(VAR(mid2h), EDX)
#endif

    VBROADCASTSD(ZMM(0), VAR(alpha))
    VBROADCASTSD(ZMM(1), VAR(beta))

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

    // Check if C is row stride. If not, jump to the slow scattered update
    MOV(RAX, VAR(rs_c))
    MOV(RBX, VAR(cs_c))
    LEA(RDI, MEM(RAX,RAX,2))
    CMP(RBX, IMM(1))
    JNE(SCATTEREDUPDATE)

    VMOVQ(RDX, XMM(1))
    TEST(RDX, RDX)
    JZ(COLSTORBZ)

    VFMADD231PD(ZMM( 8), ZMM(1), MEM(RCX      ))
    VFMADD231PD(ZMM( 9), ZMM(1), MEM(RCX,RAX,1))
    VFMADD231PD(ZMM(10), ZMM(1), MEM(RCX,RAX,2))
    VFMADD231PD(ZMM(11), ZMM(1), MEM(RCX,RDI,1))
    VMOVUPD(MEM(RCX      ), ZMM( 8))
    VMOVUPD(MEM(RCX,RAX,1), ZMM( 9))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(10))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(11))
    ADD(RCX, IMM(64))

    VFMADD231PD(ZMM(12), ZMM(1), MEM(RCX      ))
    VFMADD231PD(ZMM(13), ZMM(1), MEM(RCX,RAX,1))
    VFMADD231PD(ZMM(14), ZMM(1), MEM(RCX,RAX,2))
    VFMADD231PD(ZMM(15), ZMM(1), MEM(RCX,RDI,1))
    VMOVUPD(MEM(RCX      ), ZMM(12))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(13))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(14))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(15))
    ADD(RCX, IMM(64))

    VFMADD231PD(ZMM(16), ZMM(1), MEM(RCX      ))
    VFMADD231PD(ZMM(17), ZMM(1), MEM(RCX,RAX,1))
    VFMADD231PD(ZMM(18), ZMM(1), MEM(RCX,RAX,2))
    VFMADD231PD(ZMM(19), ZMM(1), MEM(RCX,RDI,1))
    VMOVUPD(MEM(RCX      ), ZMM(16))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(17))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(18))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(19))
    ADD(RCX, IMM(64))

    VFMADD231PD(ZMM(20), ZMM(1), MEM(RCX      ))
    VFMADD231PD(ZMM(21), ZMM(1), MEM(RCX,RAX,1))
    VFMADD231PD(ZMM(22), ZMM(1), MEM(RCX,RAX,2))
    VFMADD231PD(ZMM(23), ZMM(1), MEM(RCX,RDI,1))
    VMOVUPD(MEM(RCX      ), ZMM(20))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(21))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(22))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(23))
    ADD(RCX, IMM(64))

    VFMADD231PD(ZMM(24), ZMM(1), MEM(RCX      ))
    VFMADD231PD(ZMM(25), ZMM(1), MEM(RCX,RAX,1))
    VFMADD231PD(ZMM(26), ZMM(1), MEM(RCX,RAX,2))
    VFMADD231PD(ZMM(27), ZMM(1), MEM(RCX,RDI,1))
    VMOVUPD(MEM(RCX      ), ZMM(24))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(25))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(26))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(27))
    ADD(RCX, IMM(64))

    VFMADD231PD(ZMM(28), ZMM(1), MEM(RCX      ))
    VFMADD231PD(ZMM(29), ZMM(1), MEM(RCX,RAX,1))
    VFMADD231PD(ZMM(30), ZMM(1), MEM(RCX,RAX,2))
    VFMADD231PD(ZMM(31), ZMM(1), MEM(RCX,RDI,1))
    VMOVUPD(MEM(RCX      ), ZMM(28))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(29))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(30))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(31))
    ADD(RCX, IMM(64))

    JMP(END)

    LABEL(COLSTORBZ)

    VMOVUPD(MEM(RCX      ), ZMM( 8))
    VMOVUPD(MEM(RCX,RAX,1), ZMM( 9))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(10))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(11))
    ADD(RCX, IMM(64))

    VMOVUPD(MEM(RCX      ), ZMM(12))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(13))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(14))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(15))
    ADD(RCX, IMM(64))

    VMOVUPD(MEM(RCX      ), ZMM(16))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(17))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(18))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(19))
    ADD(RCX, IMM(64))

    VMOVUPD(MEM(RCX      ), ZMM(20))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(21))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(22))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(23))
    ADD(RCX, IMM(64))

    VMOVUPD(MEM(RCX      ), ZMM(24))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(25))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(26))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(27))
    ADD(RCX, IMM(64))

    VMOVUPD(MEM(RCX      ), ZMM(28))
    VMOVUPD(MEM(RCX,RAX,1), ZMM(29))
    VMOVUPD(MEM(RCX,RAX,2), ZMM(30))
    VMOVUPD(MEM(RCX,RDI,1), ZMM(31))
    ADD(RCX, IMM(64))

    JMP(END)

    LABEL(SCATTEREDUPDATE)

    MOV(RDI, VAR(offsetPtr))
    VMOVAPS(ZMM(2), MEM(RDI))
    /* Note that this ignores the upper 32 bits in cs_c */
    VPBROADCASTD(ZMM(3), RBX)
    VPMULLD(ZMM(2), ZMM(3), ZMM(2))

    VMOVQ(RDX, XMM(1))
    TEST(RDX, RDX)
    JZ(SCATTERBZ)

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
