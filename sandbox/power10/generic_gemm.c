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

// Using the GENERIC_GEMM template, create GEMM functions for each datatype

#include "generic_gemm.h"
#include "gemm_pack.h"

#define GENERIC_GEMM(ch, DTYPE_IN, DTYPE_OUT, NEW_PB, MULT, UK_FUNC) \
\
void GEMM_PASTEMAC(ch) \
    ( \
        dim_t MR, dim_t NR, dim_t KC, dim_t NC, dim_t MC, \
        int m, int n, int k, \
        DTYPE_IN* restrict A, int rs_a, int cs_a, int A_align, \
        DTYPE_IN* restrict B, int rs_b, int cs_b, int B_align, \
        DTYPE_OUT* restrict C, int rs_c, int cs_c, \
        DTYPE_OUT* alpha, DTYPE_OUT* beta \
    ) \
{ \
    DTYPE_OUT zero  = 0.0; \
    DTYPE_OUT beta_  = *beta; \
    \
    DTYPE_IN * restrict btilde_sys = ( DTYPE_IN *) aligned_alloc( P10_PG_SIZE, B_align + KC * NC * sizeof( DTYPE_IN ) ); \
    DTYPE_IN * restrict atilde_sys = ( DTYPE_IN *) aligned_alloc( P10_PG_SIZE, A_align + MC * KC * sizeof( DTYPE_IN ) ); \
    \
    DTYPE_IN * restrict btilde_usr = ( DTYPE_IN *)((char *)btilde_sys + B_align); \
    DTYPE_IN * restrict atilde_usr = ( DTYPE_IN *)((char *)atilde_sys + A_align); \
    \
    const int rstep_c = MC*rs_c; \
    const int cstep_c = NC*cs_c; \
    \
    const int rstep_a = MC*rs_a; \
    const int cstep_a = KC*cs_a; \
    \
    const int rstep_b = KC*rs_b; \
    const int cstep_b = NC*cs_b; \
    \
    const int rstep_mt_c = MR*rs_c; \
    const int cstep_mt_c = NR*cs_c; \
    \
    DTYPE_OUT * restrict cblock = C; \
    DTYPE_IN  * restrict bblock = B; \
    \
    DTYPE_OUT tmp_cmicrotile[MR*NR];  \
    int   rs_ct = ( rs_c == 1 ? 1 : NR ); \
    int   cs_ct = ( rs_c == 1 ? MR : 1 ); \
    \
    for ( int jc=0; jc<n; jc+=NC ) \
    { \
        int jb = bli_min( NC, n-jc ); \
        DTYPE_IN * restrict apanel = A; \
        DTYPE_IN * restrict bpanel = bblock; \
        \
        for ( int pc=0; pc<k; pc+=KC ) \
        { \
            int pb = bli_min( KC, k-pc ); \
            ch ## _packB \
            (NR, pb, jb, bpanel, rs_b, cs_b, btilde_usr); \
            \
            int new_pb = NEW_PB; \
            const int a_ps = new_pb * (MULT * MR); \
            const int b_ps = new_pb * (MULT * NR); \
            \
            DTYPE_OUT * restrict cpanel = cblock; \
            DTYPE_IN  * restrict ablock = apanel; \
            \
            for ( int ic=0; ic<m; ic+=MC ) \
            { \
                int ib = bli_min( MC, m-ic ); \
                \
                ch ## _packA \
                ( MR, ib, pb, ablock, rs_a, cs_a, atilde_usr ); \
                \
                DTYPE_OUT * restrict cmicrotile_col = cpanel; \
                DTYPE_IN  * restrict bmicropanel = btilde_usr; \
                \
                for ( int jr=0; jr<jb; jr+=NR ) \
                { \
                    int jrb = bli_min( NR, jb-jr ); \
                    DTYPE_OUT * restrict cmicrotile = cmicrotile_col; \
                    DTYPE_IN  * restrict amicropanel = atilde_usr; \
                    \
                    for ( int ir=0; ir<ib; ir+=MR ) \
                    {    \
                        int irb = bli_min( MR, ib-ir ); \
                        \
                        if (jrb == NR && irb == MR) \
                            UK_FUNC (new_pb, alpha, amicropanel, bmicropanel, beta, cmicrotile, rs_c, cs_c, NULL, NULL); \
                        else \
                        { \
                            UK_FUNC (new_pb, alpha, amicropanel, bmicropanel, &zero, tmp_cmicrotile, rs_ct, cs_ct, NULL, NULL); \
                            \
                            for (int j=0; j<jrb;j++) \
                                for (int i=0; i<irb;i++)  \
                                    cmicrotile[i*rs_c + j*cs_c] = \
                                        beta_ * cmicrotile[i*rs_c + j*cs_c] + \
                                        tmp_cmicrotile[i*rs_ct + j*cs_ct]; \
                        } \
                        amicropanel += a_ps; \
                        cmicrotile += rstep_mt_c; \
                    } \
                    bmicropanel += b_ps; \
                    cmicrotile_col += cstep_mt_c; \
                } \
                ablock += rstep_a; \
                cpanel += rstep_c; \
            } \
            apanel += cstep_a; \
            bpanel += rstep_b; \
        } \
        cblock += cstep_c; \
        bblock += cstep_b; \
    } \
    free(btilde_sys); \
    free(atilde_sys); \
} 

GENERIC_GEMM( sb, bfloat16, float,   (pb/2 + pb%2), 2,  bli_sbgemm_power10_mma_8x16);
GENERIC_GEMM(i16,  int16_t,   int,   (pb/2 + pb%2), 2, bli_i16gemm_power10_mma_8x16);
GENERIC_GEMM( sh,  float16, float,   (pb/2 + pb%2), 2,  bli_shgemm_power10_mma_8x16); 
GENERIC_GEMM( i8,   int8_t,   int, (pb/4 + (pb%4>0)), 4,  bli_i8gemm_power10_mma_8x16);
GENERIC_GEMM( i4,  nibbles,   int, (pb/8 + (pb%8>0)), 8,  bli_i4gemm_power10_mma_8x16);
