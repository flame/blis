/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
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
#undef restrict

void bli_sgemm_8x8(
                    dim_t      k,
                    float*     alpha,
                    float*     a,
                    float*     b,
                    float*     beta,
                    float*     c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t* data
                  )
{
	/* Just call the reference implementation. */
	bli_sgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}




/*
 * Here is dgemm kernel for QPX. 
 * Instruction mix was divined by a statement in an email from John Gunnels when asked about the peak performance with a single thread:
 * "Achievable peak can either be:
 * 1) 12.8 GF 8 FMAs cycle * 1.6 GHz
 * 2) 8.53 GF Takes into account the instruction mix in DGEMM and the fact that you can only do an FMA or a load/store in a single cycle with just one thread
 * 3) 7.58 GF (2) + the fact that we can only issue 8 instructions in 9 cycles with one thread"
 *
 * Which I have taken to mean: 8.53 GFLOPS implies on average 5.33 flops/cycle. 
 * I know the kernel John uses is 8x8, so 16 flops per loop iteration. 
 * Thus there must be 24 total instructions per iteration because 16/24 = 5.33.
 *
 * Here, we have 6 loads per iteration. These are executed on a different pipeline from FMAs so
 * we could (maybe) theoretically hit 100% of peak with this instruction mix
*/

void bli_dgemm_8x8(
                    dim_t      k,
                    double*    alpha,
                    double*    a,
                    double*    b,
                    double*    beta,
                    double*    c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t* data
                  )

{
    //Registers for storing C.
    //4 4x4 subblocks of C, c00, c01, c10, c11
    //4 registers per subblock: a, b, c, d
    //There is an excel file that details which register ends up storing what
    vector4double c00a = vec_splats( 0.0 );
    vector4double c00b = vec_splats( 0.0 );
    vector4double c00c = vec_splats( 0.0 );
    vector4double c00d = vec_splats( 0.0 );

    vector4double c01a = vec_splats( 0.0 );
    vector4double c01b = vec_splats( 0.0 );
    vector4double c01c = vec_splats( 0.0 );
    vector4double c01d = vec_splats( 0.0 );

    vector4double c10a = vec_splats( 0.0 );
    vector4double c10b = vec_splats( 0.0 );
    vector4double c10c = vec_splats( 0.0 );
    vector4double c10d = vec_splats( 0.0 );

    vector4double c11a = vec_splats( 0.0 );
    vector4double c11b = vec_splats( 0.0 );
    vector4double c11c = vec_splats( 0.0 );
    vector4double c11d = vec_splats( 0.0 );

    vector4double b0a, b1a;
    vector4double b0b, b1b;
    vector4double a0, a1;

    for( dim_t i = 0; i < k; i++ )
    {
        b0a = vec_ld2a( 0 * sizeof(double), &b[8*i] );
        b0b = vec_ld2a( 2 * sizeof(double), &b[8*i] );
        b1a = vec_ld2a( 4 * sizeof(double), &b[8*i] );
        b1b = vec_ld2a( 6 * sizeof(double), &b[8*i] );

        a0  = vec_lda ( 0 * sizeof(double), &a[8*i] );
        a1  = vec_lda ( 4 * sizeof(double), &a[8*i] );
        
        c00a    = vec_xmadd ( b0a, a0, c00a );
        c00b    = vec_xxmadd( a0, b0a, c00b );
        c00c    = vec_xmadd ( b0b, a0, c00c );
        c00d    = vec_xxmadd( a0, b0b, c00d );

        c01a    = vec_xmadd ( b1a, a0, c01a );
        c01b    = vec_xxmadd( a0, b1a, c01b );
        c01c    = vec_xmadd ( b1b, a0, c01c );
        c01d    = vec_xxmadd( a0, b1b, c01d );

        c10a    = vec_xmadd ( b0a, a1, c10a );
        c10b    = vec_xxmadd( a1, b0a, c10b );
        c10c    = vec_xmadd ( b0b, a1, c10c );
        c10d    = vec_xxmadd( a1, b0b, c10d );

        c11a    = vec_xmadd ( b1a, a1, c11a );
        c11b    = vec_xxmadd( a1, b1a, c11b );
        c11c    = vec_xmadd ( b1b, a1, c11c );
        c11d    = vec_xxmadd( a1, b1b, c11d );
    }
    
    // Create patterns for permuting Cb and Cd
    vector4double pattern = vec_gpci( 01032 );

    vector4double AB;
    vector4double C = vec_splats( 0.0 );
    vector4double betav  = vec_lds( 0, beta );
    vector4double alphav = vec_lds( 0, alpha );
    double ct;
  
    //Macro to update 4 elements of C in a column.
    //REG is the register holding those 4 elements
    //ADDR is the address to write them to
    //OFFSET is the number of rows from ADDR to write to
#define UPDATE( REG, ADDR, OFFSET )     \
{                                       \
    ct = *(ADDR + (OFFSET + 0) * rs_c); \
    C = vec_insert( ct, C, 0 );         \
    ct = *(ADDR + (OFFSET + 1) * rs_c); \
    C = vec_insert( ct, C, 1 );         \
    ct = *(ADDR + (OFFSET + 2) * rs_c); \
    C = vec_insert( ct, C, 2 );         \
    ct = *(ADDR + (OFFSET + 3) * rs_c); \
    C = vec_insert( ct, C, 3 );         \
                                        \
    AB = vec_mul( REG, alphav );        \
    AB = vec_madd( C, betav, AB);       \
                                        \
    ct = vec_extract( AB, 0 );          \
    *(ADDR + (OFFSET + 0) * rs_c) = ct; \
    ct = vec_extract( AB, 1 );          \
    *(ADDR + (OFFSET + 1) * rs_c) = ct; \
    ct = vec_extract( AB, 2 );          \
    *(ADDR + (OFFSET + 2) * rs_c) = ct; \
    ct = vec_extract( AB, 3 );          \
    *(ADDR + (OFFSET + 3) * rs_c) = ct; \
}  
    //Update c00 and c10 sub-blocks
    UPDATE( c00a, c, 0 );
    UPDATE( c10a, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c00b, c00b, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10b, c10b, pattern );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    UPDATE( c00c, c, 0 );
    UPDATE( c10c, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c00d, c00d, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c10d, c10d, pattern );
    UPDATE( AB, c, 4 );

    //Update c01 and c11 sub-blocks
    c = c + cs_c;
    UPDATE( c01a, c, 0 );
    UPDATE( c11a, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c01b, c01b, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11b, c11b, pattern );
    UPDATE( AB, c, 4 );

    c = c + cs_c;
    UPDATE( c01c, c, 0 );
    UPDATE( c11c, c, 4 );

    c = c + cs_c;
    AB = vec_perm( c01d, c01d, pattern );
    UPDATE( AB, c, 0 );
    AB = vec_perm( c11d, c11d, pattern );
    UPDATE( AB, c, 4 );
}

void bli_dgemm_8x8_mt(
                       dim_t      k,
                       double*    alpha,
                       double*    a,
                       double*    b,
                       double*    beta,
                       double*    c, inc_t rs_c, inc_t cs_c,
                       auxinfo_t* data,
                       dim_t      tid
                     )
{
	bli_dgemm_8x8( k,
	               alpha, 
	               a,
	               b, beta, 
	               c, 
	               rs_c, cs_c,
	               data );
}

void bli_cgemm_8x8(
                    dim_t      k,
                    scomplex*  alpha,
                    scomplex*  a,
                    scomplex*  b,
                    scomplex*  beta,
                    scomplex*  c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t* data
                  )
{
	/* Just call the reference implementation. */
	bli_cgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}

void bli_zgemm_8x8(
                    dim_t      k,
                    dcomplex*  alpha,
                    dcomplex*  a,
                    dcomplex*  b,
                    dcomplex*  beta,
                    dcomplex*  c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t* data
                  )
{
	/* Just call the reference implementation. */
	bli_zgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}


void bli_sgemm_8x8_mt(
                       dim_t      k,
                       float*     alpha,
                       float*     a,
                       float*     b,
                       float*     beta,
                       float*     c, inc_t rs_c, inc_t cs_c,
                       auxinfo_t* data,
                       dim_t      t_id
                     )
{
	/* Just call the reference implementation. */
	bli_sgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}

void bli_cgemm_8x8_mt(
                       dim_t      k,
                       scomplex*  alpha,
                       scomplex*  a,
                       scomplex*  b,
                       scomplex*  beta,
                       scomplex*  c, inc_t rs_c, inc_t cs_c,
                       auxinfo_t* data,
                       dim_t      t_id
                     )
{
	/* Just call the reference implementation. */
	bli_cgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}

void bli_zgemm_8x8_mt(
                       dim_t      k,
                       dcomplex*  alpha,
                       dcomplex*  a,
                       dcomplex*  b,
                       dcomplex*  beta,
                       dcomplex*  c, inc_t rs_c, inc_t cs_c,
                       auxinfo_t* data,
                       dim_t      t_id
                     )
{
	/* Just call the reference implementation. */
	bli_zgemm_ref_mxn( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}
