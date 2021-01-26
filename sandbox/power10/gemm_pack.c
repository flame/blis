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

// Templates for different packing routine

#include "gemm_pack.h"

/*

    Details on bit16_dt vector data structure

    Vector X = [ X[0,0] X[0,1] X[1,0] X[1,1] X[2,0] X[2,1] X[3,0] X[3,1] ]
    Vector Y = [ Y[0,0] Y[0,1] Y[1,0] Y[1,1] Y[2,0] Y[2,1] Y[3,0] Y[3,1] ]

    These bit16_dt vectors represent a 4x2 matrix. Hence, in matrix form it 
    looks like the following:

    X = [ X[0,0] X[0,1] 
          X[1,0] X[1,1]
          X[2,0] X[2,1]
          X[3,0] X[3,1] ]

    The outer product instruction: xvbf16ger2 (bfloat16 outer product)

    Syntax: 

        xvbf16ger2 ACCUMULATOR A, VECTOR X, VECTOR Y

    Semantics:

        A = X * Y^T

    The generic packing routine would load 8 elements from the same column.
    This causes an issue since the instruction expects the vector to be a
    4x2 matrix where the data is packed in contiguous order. Thus, we must make 
    a packing routine that will interleave the matrix data. Making it so 
    that when we load the 8 contiguous elements from A, it will represent
    a 4x2 section of the matrix.

*/

#define k_even_apack_16(ir) \
            *adest++ = ap[ (i+ir)*rs_a + p_idx*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+1)*cs_a ];

#define k_odd_apack_16(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 2); \
            adest++;
    
#define pad_macro_16(dest_matrix) \
            memset(dest_matrix, 0, 4); \
            dest_matrix+=2; 

#define BIT16_PACK_A(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, A) \
    ( \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ) \
{ \
    int k_odd = k%2; \
    int p_idx; \
\
    DTYPE_IN* adest = apack; \
    for (int i=0; i<m; i+=MR) \
    { \
        int ib = bli_min(MR, m-i); \
        if (ib == MR) /* Full size column height */ \
        { \
            p_idx = 0; \
            for (int p=0; p<(k/2); p++) \
            {  \
                k_even_apack_16(0); \
                k_even_apack_16(1); \
                k_even_apack_16(2); \
                k_even_apack_16(3); \
                k_even_apack_16(4); \
                k_even_apack_16(5); \
                k_even_apack_16(6); \
                k_even_apack_16(7); \
                p_idx += 2; \
            } \
\
            /* In the case that k is odd, we must pad with 0s */ \
            if(k_odd) \
            { \
                k_odd_apack_16(0); \
                k_odd_apack_16(1); \
                k_odd_apack_16(2); \
                k_odd_apack_16(3); \
                k_odd_apack_16(4); \
                k_odd_apack_16(5); \
                k_odd_apack_16(6); \
                k_odd_apack_16(7); \
            } \
        } \
\
        else /* Not full size, pad with zeros */ \
        { \
            p_idx = 0; \
            for (int p=0; p<(k/2); p++) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_even_apack_16(ir); \
                } \
                for (int ir=ib; ir<MR; ir++) \
                { \
                    pad_macro_16(adest); \
                } \
                p_idx += 2; \
            } \
\
            if(k_odd) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_odd_apack_16(ir); \
                } \
                for (int ir=ib; ir<MR; ir++) \
                { \
                    pad_macro_16(adest); \
                } \
            } \
        } \
    } \
} 


#define k_even_bpack_16(jr) \
            *bdest++ = bp[ p_idx*rs_b     + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+1)*rs_b + (j+jr)*cs_b ]; \

#define k_odd_bpack_16(jr) \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 2); \
            bdest++; \

#define BIT16_PACK_B(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, B) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ) \
{ \
\
    int k_odd = k%2; \
    int p_idx; \
\
    DTYPE_IN* bdest = bpack; \
\
    for( int j=0; j<n; j += NR ) \
    { \
        int jb = bli_min(NR, n-j); \
\
        if ( jb == NR ) /* Full column width micro-panel.*/  \
        { \
            p_idx = 0; \
            for ( int p=0; p<(k/2); p++ ) \
            { \
                k_even_bpack_16(0); \
                k_even_bpack_16(1); \
                k_even_bpack_16(2); \
                k_even_bpack_16(3); \
                k_even_bpack_16(4); \
                k_even_bpack_16(5); \
                k_even_bpack_16(6); \
                k_even_bpack_16(7); \
                k_even_bpack_16(8); \
                k_even_bpack_16(9); \
                k_even_bpack_16(10); \
                k_even_bpack_16(11); \
                k_even_bpack_16(12); \
                k_even_bpack_16(13); \
                k_even_bpack_16(14); \
                k_even_bpack_16(15); \
                p_idx += 2; \
            } \
\
            /* In the case that k is odd, we must pad with 0s */ \
            if(k_odd) \
            { \
                k_odd_bpack_16(0); \
                k_odd_bpack_16(1); \
                k_odd_bpack_16(2); \
                k_odd_bpack_16(3); \
                k_odd_bpack_16(4); \
                k_odd_bpack_16(5); \
                k_odd_bpack_16(6); \
                k_odd_bpack_16(7); \
                k_odd_bpack_16(8); \
                k_odd_bpack_16(9); \
                k_odd_bpack_16(10); \
                k_odd_bpack_16(11); \
                k_odd_bpack_16(12); \
                k_odd_bpack_16(13); \
                k_odd_bpack_16(14); \
                k_odd_bpack_16(15); \
            } \
        } \
\
        else /* Not a full row size micro-panel.  We pad with zeroes. */ \
        { \
            p_idx = 0; \
            for ( int p=0; p<(k/2); p++ )  \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_even_bpack_16(jr); \
                } \
                for ( int jr=jb; jr<NR; jr++ ) \
                { \
                    pad_macro_16(bdest); \
                } \
                p_idx += 2; \
            } \
\
            if(k_odd) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_odd_bpack_16(jr); \
                } \
                for ( int jr=jb; jr<NR; jr++ ) \
                { \
                    pad_macro_16(bdest); \
                } \
            } \
        } \
    } \
};



/* 8 bit packing routines */

#define k_even_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + p_idx*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+1)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+2)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (p_idx+3)*cs_a ];

#define k_left3_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-3)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (k-2)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 1); \
            adest++;

#define k_left2_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-2)*cs_a ]; \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 2); \
            adest += 2;

#define k_left1_apack_8(ir) \
            *adest++ = ap[ (i+ir)*rs_a + (k-1)*cs_a ]; \
            memset(adest, 0, 3); \
            adest += 3;
    
#define pad_macro_8(dest_matrix) \
            memset(dest_matrix, 0, 4); \
            dest_matrix += 4;


#define BIT8_PACK_A(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, A) \
    ( \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ) \
{ \
    int k_left = k%4; \
    int k_iter = k/4; \
    int p_idx; \
\
    DTYPE_IN* adest = apack; \
\
    /* Each panel must be packed in this format */ \
    for (int i=0; i<m; i+=MR) \
    { \
        int ib = bli_min(MR, m-i); \
\
        if (ib == MR) /* Full size column height */ \
        { \
            p_idx = 0; \
            for (int p=0; p<k_iter; p++) \
            {  \
                k_even_apack_8(0); \
                k_even_apack_8(1); \
                k_even_apack_8(2); \
                k_even_apack_8(3); \
                k_even_apack_8(4); \
                k_even_apack_8(5); \
                k_even_apack_8(6); \
                k_even_apack_8(7); \
                p_idx += 4; \
            } \
\
            /* In the case that k is odd, we must pad with 0s */ \
            if(k_left==3) \
            { \
                k_left3_apack_8(0); \
                k_left3_apack_8(1); \
                k_left3_apack_8(2); \
                k_left3_apack_8(3); \
                k_left3_apack_8(4); \
                k_left3_apack_8(5); \
                k_left3_apack_8(6); \
                k_left3_apack_8(7); \
            } \
            else if(k_left==2) \
            { \
                k_left2_apack_8(0); \
                k_left2_apack_8(1); \
                k_left2_apack_8(2); \
                k_left2_apack_8(3); \
                k_left2_apack_8(4); \
                k_left2_apack_8(5); \
                k_left2_apack_8(6); \
                k_left2_apack_8(7); \
            } \
            else if(k_left==1) \
            { \
                k_left1_apack_8(0); \
                k_left1_apack_8(1); \
                k_left1_apack_8(2); \
                k_left1_apack_8(3); \
                k_left1_apack_8(4); \
                k_left1_apack_8(5); \
                k_left1_apack_8(6); \
                k_left1_apack_8(7); \
            } \
        } \
\
        else /* Not full size, pad with zeros */ \
        { \
            p_idx = 0; \
            for (int p=0; p<k_iter; p++) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_even_apack_8(ir); \
                } \
                for (int ir=ib; ir<MR; ir++) \
                { \
                    pad_macro_8(adest); \
                } \
                p_idx += 4; \
            } \
\
            if(k_left==3) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_left3_apack_8(ir); \
                } \
            } \
            else if(k_left==2) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_left2_apack_8(ir); \
                } \
            } \
            else if(k_left==1) \
            { \
                for (int ir=0; ir<ib; ir++) \
                { \
                    k_left1_apack_8(ir); \
                } \
            } \
            if(k_left!=0) \
            { \
                for (int ir=ib; ir<MR; ir++) { \
                    pad_macro_8(adest); \
                } \
            } \
        } \
    } \
}


#define k_even_bpack_8(jr) \
            *bdest++ = bp[ p_idx*rs_b     + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+1)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+2)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (p_idx+3)*rs_b + (j+jr)*cs_b ]; 

#define k_left3_bpack_8(jr) \
            *bdest++ = bp[ (k-3)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (k-2)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 1); \
            bdest++;

#define k_left2_bpack_8(jr) \
            *bdest++ = bp[ (k-2)*rs_b + (j+jr)*cs_b ]; \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 2); \
            bdest+=2;

#define k_left1_bpack_8(jr) \
            *bdest++ = bp[ (k-1)*rs_b + (j+jr)*cs_b ]; \
            memset(bdest, 0, 3); \
            bdest+=3;


#define BIT8_PACK_B(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, B) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ) \
{ \
    int k_left = k%4; \
    int k_iter = k/4; \
    int p_idx; \
\
    DTYPE_IN* bdest = bpack; \
\
    for( int j=0; j<n; j += NR ) \
    { \
        int jb = bli_min(NR, n-j); \
\
        if ( jb == NR ) /* Full column width micro-panel.*/ \
        { \
            p_idx = 0; \
            for ( int p=0; p<k_iter; p++ ) \
            { \
                k_even_bpack_8(0); \
                k_even_bpack_8(1); \
                k_even_bpack_8(2); \
                k_even_bpack_8(3); \
                k_even_bpack_8(4); \
                k_even_bpack_8(5); \
                k_even_bpack_8(6); \
                k_even_bpack_8(7); \
                k_even_bpack_8(8); \
                k_even_bpack_8(9); \
                k_even_bpack_8(10); \
                k_even_bpack_8(11); \
                k_even_bpack_8(12); \
                k_even_bpack_8(13); \
                k_even_bpack_8(14); \
                k_even_bpack_8(15); \
                p_idx += 4; \
            } \
\
            if(k_left==3) \
            { \
                k_left3_bpack_8(0); \
                k_left3_bpack_8(1); \
                k_left3_bpack_8(2); \
                k_left3_bpack_8(3); \
                k_left3_bpack_8(4); \
                k_left3_bpack_8(5); \
                k_left3_bpack_8(6); \
                k_left3_bpack_8(7); \
                k_left3_bpack_8(8); \
                k_left3_bpack_8(9); \
                k_left3_bpack_8(10); \
                k_left3_bpack_8(11); \
                k_left3_bpack_8(12); \
                k_left3_bpack_8(13); \
                k_left3_bpack_8(14); \
                k_left3_bpack_8(15); \
            } \
            else if(k_left==2) \
            { \
                k_left2_bpack_8(0); \
                k_left2_bpack_8(1); \
                k_left2_bpack_8(2); \
                k_left2_bpack_8(3); \
                k_left2_bpack_8(4); \
                k_left2_bpack_8(5); \
                k_left2_bpack_8(6); \
                k_left2_bpack_8(7); \
                k_left2_bpack_8(8); \
                k_left2_bpack_8(9); \
                k_left2_bpack_8(10); \
                k_left2_bpack_8(11); \
                k_left2_bpack_8(12); \
                k_left2_bpack_8(13); \
                k_left2_bpack_8(14); \
                k_left2_bpack_8(15); \
            } \
            else if(k_left==1)  \
            { \
                k_left1_bpack_8(0); \
                k_left1_bpack_8(1); \
                k_left1_bpack_8(2); \
                k_left1_bpack_8(3); \
                k_left1_bpack_8(4); \
                k_left1_bpack_8(5); \
                k_left1_bpack_8(6); \
                k_left1_bpack_8(7); \
                k_left1_bpack_8(8); \
                k_left1_bpack_8(9); \
                k_left1_bpack_8(10); \
                k_left1_bpack_8(11); \
                k_left1_bpack_8(12); \
                k_left1_bpack_8(13); \
                k_left1_bpack_8(14); \
                k_left1_bpack_8(15); \
            } \
        } \
\
        else /* Not a full row size micro-panel.  We pad with zeroes. */ \
        { \
            p_idx = 0; \
            for ( int p=0; p<k_iter; p++ ) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_even_bpack_8(jr); \
                } \
                for ( int jr=jb; jr<NR; jr++ ) \
                { \
                    pad_macro_8(bdest); \
                } \
                p_idx += 4; \
            } \
\
            if(k_left==3) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_left3_bpack_8(jr); \
                } \
            } \
            else if(k_left==2) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_left2_bpack_8(jr); \
                } \
            } \
            else if(k_left==1) \
            { \
                for ( int jr=0; jr<jb; jr++ ) \
                { \
                    k_left1_bpack_8(jr); \
                } \
            } \
            if (k_left!=0) \
            { \
                for ( int jr=jb; jr<NR; jr++ ) { \
                    pad_macro_8(bdest); \
                } \
            } \
        } \
    } \
}

////////////////////////////////////////////////////////////////////////////////
/*                            Packing Routines                                */
////////////////////////////////////////////////////////////////////////////////

/*

    Memory is byte-addressed. This results in two options when dealing with 
    int4. Either store 1 int4 value in a byte, or store 2 int4 values in 1 
    byte. The former is wasteful in storage, but it makes for a simpler
    packing routine. However, we want to not waste any storage if possible. 
    Therefore I went with the latter when designing my int4 kernel. 

    The int4 outerproduct instruction expects a 4x8 matrix in row major order 
    to be loaded into the vector. In order to achieve this 4x8 row major 
    matrix, we pack as many 4x8 panels from the src matrix into the pack matrix.

    To illustrate how my packing routine works:

    x0  x1  x2  x3  x4  x5  x6  x7
    x9  x10 x11 x12 x13 x14 x15 x16
    x17 x18 x19 x20 x21 x22 x23 x24
    x25 x26 x27 x28 x29 x30 x31 x32

    Assume we have a 4x8 matrix that is stored in column major order. Also 
    since we are dealing with int4 values, the values are stored as pairs 
    within a union struct. i.e. (x0, x9) are stored together in the same struct.
    
    Therefore in order to get the desired 4x8 row major matrix, we must go 
    through the first row of structs and grab the first int4 value and insert
    it into the appropriate spot in the pack matrix. This means that after 
    packing, (x0, x1) will be stored together in the same struct.

    This process then repeats until the entire src matrix is packed in these
    4x8 row major matrix panels. 

    To handle edge cases, the packing routine will fill in zeros where it is
    appropriate. 
    
*/

#include "i4_macros.h"

#define BIT4_PACK_A(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, A) \
    ( \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ) \
{ \
    int p_idx, k_left, k_iter; \
    DTYPE_IN* adest = apack; \
\
    k_left = k%8; \
    k_iter = k/8; \
\
    int i = 0; /* i is used for byte addressing */ \
    for(int int4_i=0; int4_i<m; int4_i+=MR) { /* pack panels */ \
\
        int ib = bli_min(MR, m-int4_i); \
        p_idx = 0; \
\
        if (ib == MR) { /* full size */ \
            for (int p=0; p<k_iter; p++) { \
                col_m_order_1(adest, ap, (i+0), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+0), rs_a, cs_a); \
                col_m_order_1(adest, ap, (i+1), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+1), rs_a, cs_a); \
                col_m_order_1(adest, ap, (i+2), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+2), rs_a, cs_a); \
                col_m_order_1(adest, ap, (i+3), rs_a, cs_a); \
                col_m_order_2(adest, ap, (i+3), rs_a, cs_a); \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                apad_col_kleft7(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 6) { \
                apad_col_kleft6(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 5) { \
                apad_col_kleft5(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 4) { \
                apad_col_kleft4(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 3) { \
                apad_col_kleft3(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 2) { \
                apad_col_kleft2(adest, ap, rs_a, cs_a); \
            } \
            else if(k_left == 1) { \
                apad_col_kleft1(adest, ap, rs_a, cs_a); \
            } \
        } \
\
        else { /* not full size */ \
            for (int p=0; p<k_iter; p++) { \
                for (int ir=0; ir<ib; ir++) { \
                    if (ir%2==0) { \
                        col_m_order_1(adest, ap, (i+ir/2), rs_a, cs_a); \
                    } \
                    else { \
                        col_m_order_2(adest, ap, (i+ir/2), rs_a, cs_a); \
                    } \
                } \
                for (int ir=ib; ir<MR; ir++) { \
                    zero_out_dest(adest); \
                } \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                edge7(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 6) { \
                edge6(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 5) { \
                edge5(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 4) { \
                edge4(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 3) { \
                edge3(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 2) { \
                edge2(adest, ap, i, ib, rs_a, cs_a); \
            } \
            else if(k_left == 1) { \
                edge1(adest, ap, i, ib, rs_a, cs_a); \
            } \
\
            /* fill in zeros when an edge case occurs */ \
            if(k_left!=0) \
            { \
                for (int ir=ib; ir<MR; ir++) \
                    zero_out_dest(adest); \
            } \
        } \
        i += (MR/2); \
    } \
}


#define BIT4_PACK_B(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, B) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ) \
{ \
\
    int p_idx, k_left, k_iter; \
    DTYPE_IN* bdest = bpack; \
\
    k_left = k%8; \
    k_iter = k/8; \
\
    int j = 0; \
\
    for(int int4_j=0; int4_j<n; int4_j+=NR) { /* pack panels */ \
        int jb = bli_min(NR, n-int4_j); \
\
        p_idx = 0; \
        if (jb == NR) { /* full size */ \
            for (int p=0; p<k_iter; p++) { \
                col_m_order_1(bdest, bp, (j+0), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+0), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+1), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+1), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+2), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+2), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+3), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+3), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+4), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+4), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+5), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+5), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+6), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+6), cs_b, rs_b); \
                col_m_order_1(bdest, bp, (j+7), cs_b, rs_b); \
                col_m_order_2(bdest, bp, (j+7), cs_b, rs_b); \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                bpad_col_kleft7(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 6) { \
                bpad_col_kleft6(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 5) { \
                bpad_col_kleft5(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 4) { \
                bpad_col_kleft4(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 3) { \
                bpad_col_kleft3(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 2) { \
                bpad_col_kleft2(bdest, bp, cs_b, rs_b); \
            } \
            else if(k_left == 1) { \
                bpad_col_kleft1(bdest, bp, cs_b, rs_b); \
            } \
        } \
        else { /* not full size */ \
            for (int p=0; p<k_iter; p++) { \
                for (int jr=0; jr<jb; jr++) { \
                    if (jr%2==0) { \
                        col_m_order_1(bdest, bp, (j+jr/2), cs_b, rs_b); \
                    } \
                    else { \
                        col_m_order_2(bdest, bp, (j+jr/2), cs_b, rs_b); \
                    } \
                } \
                for (int jr=jb; jr<NR; jr++) { \
                    zero_out_dest(bdest); \
                } \
                p_idx += 8; \
            } \
\
            /* handle edge cases if there are any */ \
            if(k_left == 7) { \
                edge7(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 6) { \
                edge6(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 5) { \
                edge5(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 4) { \
                edge4(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 3) { \
                edge3(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 2) { \
                edge2(bdest, bp, j, jb, cs_b, rs_b); \
            } \
            else if(k_left == 1) { \
                edge1(bdest, bp, j, jb, cs_b, rs_b); \
            } \
\
            /* fill in zeros when an edge case occurs */ \
            if(k_left!=0) \
            { \
                for (int ir=jb; ir<NR; ir++) \
                    zero_out_dest(bdest); \
            } \
        } \
        j += (NR/2); \
    } \
}



#define BIT16_PACK_ROUTINES(ch, DTYPE_IN) \
    BIT16_PACK_A(ch, DTYPE_IN); \
    BIT16_PACK_B(ch, DTYPE_IN);

#define BIT8_PACK_ROUTINES(ch, DTYPE_IN) \
    BIT8_PACK_A(ch, DTYPE_IN); \
    BIT8_PACK_B(ch, DTYPE_IN);

#define BIT4_PACK_ROUTINES(ch, DTYPE_IN) \
    BIT4_PACK_A(ch, DTYPE_IN); \
    BIT4_PACK_B(ch, DTYPE_IN);

BIT16_PACK_ROUTINES(sb, bfloat16);
BIT16_PACK_ROUTINES(i16, int16_t);
BIT16_PACK_ROUTINES(sh, float16);

BIT8_PACK_ROUTINES(i8, int8_t);

BIT4_PACK_ROUTINES(i4, nibbles);
