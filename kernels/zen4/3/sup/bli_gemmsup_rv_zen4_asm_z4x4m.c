/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "immintrin.h"

#if defined __clang__
    #define UNROLL_LOOP()      _Pragma("clang loop unroll_count(4)")
    /*
    *   in clang, unroll_count(4) generates inefficient
    *   code compared to unroll(full) when loopCount = 4.
    */
    #define UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
#elif defined __GNUC__
    #define UNROLL_LOOP()      _Pragma("GCC unroll 4")
    #define UNROLL_LOOP_FULL()  _Pragma("GCC unroll 4")
#else
    #define UNROLL_LOOP()
    #define UNROLL_LOOP_FULL()
#endif

/*Set registers to zero which are used during fma operation*/
#define ZERO_REGISTERS() \
    c_reg[0] = _mm512_setzero_pd(); \
    c_reg[1] = _mm512_setzero_pd(); \
    c_reg[2] = _mm512_setzero_pd(); \
    c_reg[3] = _mm512_setzero_pd(); \
    c_imag_reg[0] = _mm512_setzero_pd(); \
    c_imag_reg[1] = _mm512_setzero_pd(); \
    c_imag_reg[2] = _mm512_setzero_pd(); \
    c_imag_reg[3] = _mm512_setzero_pd(); \

/*************************************************************/
/* Transpose contents of R0, R1, R2, R3 and store            */
/* the result to same register                               */
/* Transpose 4x4 register                                    */
/* Input c_reg0 = Ar0 Ai0 Ar1 Ai1 Ar2 Ai2 Ar3 Ai3            */
/* Input c_reg1 = Ar4 Ai4 Ar5 Ai5 Ar6 Ai6 Ar7 Ai7            */
/* Input c_reg2 = Ar8 Ai8 Ar9 Ai9 Ar10 Ai10 Ar11 Ai11        */
/* Input c_reg3 = Ar12 Ai12 Ar13 Ai13 Ar14 Ai14 Ar15 Ai15    */
/* Inter c_imag_reg0 = Ar0 Ai0 Ar2 Ai2 Ar4 Ai4 Ar6 Ai6       */
/* Inter c_imag_reg1 = Ar1 Ai1 Ar3 Ai3 Ar5 Ai5 Ar7 Ai7       */
/* Inter c_imag_reg2 = Ar8 Ai8 Ar10 Ai10 Ar12 Ai12 Ar14 Ai14 */
/* Inter c_imag_reg3 = Ar9 Ai9 Ar11 Ai11 Ar13 Ai13 Ar15 Ai15 */
/* Output c_reg0 = Ar0 Ai0 Ar4 Ai4 Ar8 Ai8 Ar12 Ai12         */
/* Output c_reg1 = Ar1 Ai1 Ar5 Ai5 Ar9 Ai9 Ar13 Ai13         */
/* Output c_reg2 = Ar2 Ai2 Ar6 Ai6 Ar10 Ai10 Ar14 Ai14       */
/* Output c_reg3 = Ar3 Ai3 Ar7 Ai7 Ar11 Ai11 Ar15 Ai15       */
/*************************************************************/
#define TRANSPOSE_4x4() \
    c_imag_reg[0] = _mm512_shuffle_f64x2(c_reg[0], c_reg[1], 0b10001000); \
    c_imag_reg[1] = _mm512_shuffle_f64x2(c_reg[0], c_reg[1], 0b11011101); \
    c_imag_reg[2] = _mm512_shuffle_f64x2(c_reg[2], c_reg[3], 0b10001000); \
    c_imag_reg[3] = _mm512_shuffle_f64x2(c_reg[2], c_reg[3], 0b11011101); \
    c_reg[0] = _mm512_shuffle_f64x2(c_imag_reg[0], c_imag_reg[2], 0b10001000); \
    c_reg[2] = _mm512_shuffle_f64x2(c_imag_reg[0], c_imag_reg[2], 0b11011101); \
    c_reg[1] = _mm512_shuffle_f64x2(c_imag_reg[1], c_imag_reg[3], 0b10001000); \
    c_reg[3] = _mm512_shuffle_f64x2(c_imag_reg[1], c_imag_reg[3], 0b11011101);

/****************************************/
/* Operation:                           */
/* c_reg = A(real) * B(real,imag)       */
/* c_imag_reg = A(imag) * B(real,imag)  */
/* Elements:                            */
/* MxK elements at a time               */
/* Inputs:                              */
/* b_reg = b_curr                       */
/* a_reg = a_curr->real                 */
/* a_reg = a_curr->imag                 */
/* Outputs:                             */
/* c_reg = b_reg * a_curr->real         */
/* c_imag_reg = b_reg * a_curr->imag    */
/****************************************/
#define GEMM_MxN(M,N) \
    UNROLL_LOOP() \
    for (dim_t j = 0; j < k; ++j) \
    { \
        b_reg = _mm512_maskz_loadu_pd(mask_n, b_curr); \
        b_curr += rs_b; \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            a_reg[ii] = _mm512_set1_pd(*( (double*)(a_curr + (rs_a * ii) ))); \
            c_reg[ii] = _mm512_fmadd_pd(a_reg[ii] , b_reg, c_reg[ii]); \
            a_reg[ii] = _mm512_set1_pd((a_curr + (rs_a * ii))->imag); \
            c_imag_reg[ii] = _mm512_fmadd_pd(a_reg[ii] ,  b_reg, c_imag_reg[ii]); \
        } \
        a_curr += cs_a; \
    }

/****************************************/
/* Store elements in col order          */
/* c_reg = Beta * C + Alpha * A * B     */
/* Elements:                            */
/* MxN elements at a time               */
/* Inputs:                              */
/* c_reg = b_reg * a_curr->real         */
/* c_imag_reg = b_reg * a_curr->imag    */
/* Intermediate:                        */
/* c_reg = c_reg +/- c_imag_reg         */
/* Transpose 4x4 elements in c_reg      */
/* Output:                              */
/* c_reg = Beta * C(real,imag) +        */
/* Alpha * A(real,imag) * B(real,imag)  */
/****************************************/
#define STORE_COL(M, N) \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
       a_reg[ii] = _mm512_permute_pd(c_imag_reg[ii], 0b01010101); \
       c_reg[ii] = _mm512_fmaddsub_pd(c_reg[ii], one_reg, a_reg[ii]);  \
    } \
    TRANSPOSE_4x4() \
    if ((((beta->real) == 0) && (beta->imag) == 0) ) { STORE_COL_BZ(M, N) } \
    else \
    { \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < N; ++ii) \
        { \
            SCALE_ALPHA_COL(M) \
            SCALE_BETA(mask_n, cs_c) \
            _mm512_mask_storeu_pd(c + cs_c * ii, mask_n, c_reg[ii]); \
        } \
    } \

/****************************************/
/* Operation:                           */
/* Scale reg with alpha value and       */
/* store elements in col major order    */
/* where Beta = 0                       */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Input:                               */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_COL_BZ(M, N) \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < N; ++ii) \
    { \
        SCALE_ALPHA_COL(M) \
        _mm512_mask_storeu_pd(c + cs_c * ii, mask_n, c_reg[ii]); \
    } \

/****************************************/
/* Operation:                           */
/* 1. Load C register based on the mask */
/* and scale it with beta               */
/* 2. Scale A*B result with alpha value */
/* 3. Add results from step1 & step2    */
/* 4. Transpose and store results in    */
/*    in col major order                */
/* 5. Output update is done only for    */
/*    lower traingular matrix           */
/* NOTE:                                */
/* Mask value is set to 1 if the        */
/* element exist else it is set to 0    */
/* For m=1, mask = 2 to store real and  */
/* imag component                       */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Input:                               */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Beta * C +                   */
/*         Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_COL_LOWER(M, N) \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
       a_reg[ii] = _mm512_permute_pd(c_imag_reg[ii], 0b01010101); \
       c_reg[ii] = _mm512_fmaddsub_pd(c_reg[ii], one_reg, a_reg[ii]);  \
    } \
    TRANSPOSE_4x4() \
    if ((((beta->real) == 0) && (beta->imag) == 0) ) { STORE_COL_LOWER_BZ(M, N) } \
    else \
    { \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < N; ++ii) \
        { \
            SCALE_ALPHA_COL(M) \
            mask_n = ((1 << ((n_rem*2) - (ii*2))) -1) << (ii*2); \
            SCALE_BETA(mask_n, cs_c) \
            _mm512_mask_storeu_pd(c + cs_c * ii, mask_n, c_reg[ii]); \
        } \
    } \

/****************************************/
/* Operation:                           */
/* Scale reg with alpha value and store */
/* number of elements based on the mask */
/* in col major order where Beta = 0    */
/* Output update is done only for       */
/* lower traingular matrix              */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Input:                               */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_COL_LOWER_BZ(M, N) \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < N; ++ii) \
    { \
        SCALE_ALPHA_COL(M) \
        mask_n = ((1 << ((n_rem*2) - (ii*2))) - 1) << (ii*2); \
        _mm512_mask_storeu_pd(c + cs_c * ii, mask_n, c_reg[ii]); \
    } \

/****************************************/
/* Operation:                           */
/* 1. Load C register based on the mask */
/* and scale it with beta               */
/* 2. Scale A*B result with alpha value */
/* 3. Add results from step1 & step2    */
/* 4. Transpose and store results in    */
/*    in col major order                */
/* 5. Output update is done only for    */
/*    upper traingular matrix           */
/* NOTE:                                */
/* Mask value is set to 1 if the        */
/* element exist else it is set to 0    */
/* For m=1, mask = 2 to store real and  */
/* imag component                       */
/* Elements:                            */
/* MxN elements at a time               */
/* Inputs:                              */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Beta * C +                   */
/*         Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_COL_UPPER(M, N) \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
       a_reg[ii] = _mm512_permute_pd(c_imag_reg[ii], 0b01010101); \
       c_reg[ii] = _mm512_fmaddsub_pd(c_reg[ii], one_reg, a_reg[ii]);  \
    } \
    TRANSPOSE_4x4() \
    if ((((beta->real) == 0) && (beta->imag) == 0) ) { STORE_COL_UPPER_BZ(M, N) } \
    else \
    { \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < N; ++ii) \
        { \
            SCALE_ALPHA_COL(M) \
            mask_n = (1 << ((ii+1)*2)) - 1; \
            SCALE_BETA(mask_n, cs_c) \
            _mm512_mask_storeu_pd(c + cs_c * ii, mask_n, c_reg[ii]); \
        } \
    } \

/****************************************/
/* Operation:                           */
/* Scale reg with alpha value and store */
/* number of elements based on the mask */
/* in col major order where Beta = 0    */
/* Output update is done only for       */
/* upper traingular matrix              */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Inputs:                              */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_COL_UPPER_BZ(M, N) \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < N; ++ii) \
    { \
        SCALE_ALPHA_COL(M) \
        mask_n = (1 << (((ii+1)*2))) - 1; \
        _mm512_mask_storeu_pd(c + cs_c * ii, mask_n, c_reg[ii]); \
    } \

/****************************************/
/* Operation:                           */
/* Scale reg with alpha value and       */
/* store elements in row major order    */
/* where Beta = 0                       */
/* Elements:                            */
/* Mx4 elements at a time               */
/* Input:                               */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_ROW_BZ(M, N) \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
        SCALE_ALPHA(M) \
        _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
    } \

/****************************************/
/* Store elements in row major order    */
/* Elements:                            */
/* Mx4 elements at a time               */
/* Inputs:                              */
/* c_reg = b_reg * a_curr->real         */
/* c_imag_reg = b_reg * a_curr->imag    */
/* Intermediate:                        */
/* c_reg = c_reg +/- c_imag_reg         */
/* Output:                              */
/* c_reg = Beta * C(real,imag) +        */
/* Alpha * A(real,imag) * B(real,imag)  */
/****************************************/
#define STORE_ROW(M, N) \
    if ((((beta->real) == 0) && (beta->imag) == 0) ) { STORE_ROW_BZ(M, N) } \
    else \
    { \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            SCALE_ALPHA(M) \
            SCALE_BETA(mask_n, rs_c) \
            _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
        } \
    } \

/****************************************/
/* Scale A * B matrix with alpha value  */
/* Elements:                            */
/* 4 elements at a time                 */
/* Inputs:                              */
/* c_reg = b_reg * a_curr->real         */
/* c_imag_reg = b_reg * a_curr->imag    */
/* Output:                              */
/* c_reg = Alpha * A(real,imag) *       */
/*         B(real,imag)                 */
/****************************************/
#define SCALE_ALPHA(M)\
    a_reg[ii] = _mm512_permute_pd(c_imag_reg[ii], 0b01010101); \
    c_reg[ii] = _mm512_fmaddsub_pd(c_reg[ii], one_reg, a_reg[ii]);  \
    c_imag_reg[ii] = _mm512_permute_pd(c_reg[ii], 0b01010101); \
    c_reg[ii] = _mm512_mul_pd(c_reg[ii], alpha_reg);  \
    c_imag_reg[ii] = _mm512_mul_pd(c_imag_reg[ii], alpha_imag_reg);  \
    c_reg[ii] = _mm512_fmaddsub_pd(c_reg[ii], one_reg, c_imag_reg[ii]);  \

/****************************************/
/* Scale A * B matrix with alpha value  */
/* Elements:                            */
/* 4 elements at a time                 */
/* Input:                              */
/* c_reg = A * B                        */
/* Output:                              */
/* c_reg = Alpha * A(real,imag) *       */
/*         B(real,imag)                 */
/****************************************/
#define SCALE_ALPHA_COL(M)\
    c_imag_reg[ii] = _mm512_permute_pd(c_reg[ii], 0b01010101); \
    c_reg[ii] = _mm512_mul_pd(c_reg[ii], alpha_reg);  \
    c_imag_reg[ii] = _mm512_mul_pd(c_imag_reg[ii], alpha_imag_reg);  \
    c_reg[ii] = _mm512_fmaddsub_pd(c_reg[ii], one_reg, c_imag_reg[ii]);  \

/****************************************/
/* Scale C matrix with beta value       */
/* Elements:                            */
/* 4 elements at a time                 */
/* Mask is set based on M elements      */
/* Output :                             */
/* c_reg = Beta * C                     */
/****************************************/
#define SCALE_BETA(mask_n, stride) \
    a_reg[ii] = _mm512_maskz_loadu_pd(mask_n, c + (stride * ii)); \
    c_imag_reg[ii] = _mm512_permute_pd(a_reg[ii], 0b01010101); \
    a_reg[ii] = _mm512_mul_pd(a_reg[ii], beta_reg); \
    c_imag_reg[ii] = _mm512_mul_pd(c_imag_reg[ii], beta_imag_reg); \
    a_reg[ii] = _mm512_fmaddsub_pd(a_reg[ii], one_reg, c_imag_reg[ii]);  \
    c_reg[ii] = _mm512_add_pd(a_reg[ii], c_reg[ii]);  \

/****************************************/
/* Operation:                           */
/* 1. Load C register based on the mask */
/* and scale it with beta               */
/* 2. Scale A*B result with alpha value */
/* 3. Add results from step1 & step2    */
/* 4. Transpose and store results in    */
/*    in row major order                */
/* 5. Output update is done only for    */
/*    lower traingular matrix           */
/* NOTE:                                */
/* Mask value is set to 1 if the        */
/* element exist else it is set to 0    */
/* For m=1, mask = 2 to store real and  */
/* imag component                       */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Input:                               */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Beta * C +                   */
/*         Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_ROW_LOWER(M, N) \
    if ((((beta->real) == 0) && (beta->imag) == 0) ) { STORE_ROW_LOWER_BZ(M, N) } \
    else \
    { \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            SCALE_ALPHA(M) \
            mask_n = (1 << ((ii+1)*2)) - 1; \
            SCALE_BETA(mask_n, rs_c) \
            _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
        } \
    } \

/****************************************/
/* Operation:                           */
/* Scale reg with alpha value and store */
/* number of elements based on the mask */
/* in row major order where Beta = 0    */
/* Output update is done only for       */
/* lower traingular matrix              */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Input:                               */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_ROW_LOWER_BZ(M, N) \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
        SCALE_ALPHA(M) \
        mask_n = (1 << ((ii+1)*2)) - 1; \
        _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
    } \

/****************************************/
/* Operation:                           */
/* Scale reg with alpha value and store */
/* number of elements based on the mask */
/* in row major order where Beta = 0    */
/* Output update is done only for       */
/* upper traingular matrix              */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Inputs:                              */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_ROW_UPPER(M, N) \
    if ((((beta->real) == 0) && (beta->imag) == 0) ) { STORE_ROW_UPPER_BZ(M, N) } \
    else \
    { \
        UNROLL_LOOP_FULL() \
        for(dim_t ii = 0; ii < M; ++ii) \
        { \
            SCALE_ALPHA(M) \
            mask_n = ((1 << ((n_rem*2) - (ii*2))) - 1) << (ii*2); \
            SCALE_BETA(mask_n, rs_c) \
            _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
        } \
    } \

/****************************************/
/* Operation:                           */
/* Scale reg with alpha value and store */
/* number of elements based on the mask */
/* in row major order where Beta = 0    */
/* Output update is done only for       */
/* upper traingular matrix              */
/* Elements:                            */
/* Nx4 elements at a time               */
/* Inputs:                              */
/* c_reg = A(real, imag) * B(real, img) */
/* Output:                              */
/* c_reg = Alpha * A(real, imag) *      */
/*         B(real, img)                 */
/****************************************/
#define STORE_ROW_UPPER_BZ(M, N) \
    UNROLL_LOOP_FULL() \
    for(dim_t ii = 0; ii < M; ++ii) \
    { \
        SCALE_ALPHA(M) \
        mask_n = (((1 << ((n_rem*2) - (ii*2)))) - 1) << (ii*2); \
        _mm512_mask_storeu_pd(c + (rs_c * ii), mask_n, c_reg[ii]); \
    } \

/****************************************/
/* Perform C = C * Beta + Alpha * A * B */
/* Below functions are categorised based*/
/* on row/col order and upper/lower     */
/* 1. Calculate n_rem for 4x4 blocks    */
/* 2. Set AVX register to zero which    */
/*    are used during fma operation     */
/* 3. a_curr is pointer to matrix A,    */
/*   updated based on m and panel stride*/
/* 4. Mask is required for fringe case  */
/*    if n_rem=1, mask_n = 0011b, 1real */
/*    and 1complex elements to be       */
/*    accessed/stored                   */
/*    if n_rem=2, mask_n = 1111b, since */
/*    2real and 2complex elements to be */
/*    accessed/stored                   */
/* 5. Perfom A*B                        */
/* 6. Store Beta*C + Alpha*A*B in to C  */
/****************************************/
#define MAIN_LOOP_ROW(M) \
    n_rem = n % 4; \
    if (n_rem == 0) n_rem = 4; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem*2)) - 1; \
    GEMM_MxN(M, n_rem) \
    STORE_ROW(M, n_rem) \
    c += 4 * rs_c; \

#define MAIN_LOOP_COL(M) \
    n_rem = n % 4; \
    if (n_rem == 0) n_rem = 4; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem*2)) - 1; \
    GEMM_MxN(M, n_rem) \
    mask_n = (1 << (M*2)) - 1; \
    STORE_COL(M, n_rem) \
    c += 4 * rs_c; \

#define MAIN_LOOP_LOWER_DIAG_ROW(M) \
    n_rem = n % 4; \
    if (n_rem == 0) n_rem = 4; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem*2)) - 1; \
    GEMM_MxN(M, n_rem) \
    STORE_ROW_LOWER(M, n_rem) \
    c += 4 * rs_c; \

#define MAIN_LOOP_LOWER_DIAG_COL(M) \
    n_rem = n % 4; \
    if (n_rem == 0) n_rem = 4; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem*2)) - 1; \
    GEMM_MxN(M, n_rem) \
    STORE_COL_LOWER(M, n_rem) \
    c += 4 * rs_c; \

#define MAIN_LOOP_UPPER_DIAG_ROW(M) \
    n_rem = n % 4; \
    if (n_rem == 0) n_rem = 4; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem*2)) - 1; \
    GEMM_MxN(M, n_rem) \
    STORE_ROW_UPPER(M, n_rem) \
    c += 4 * rs_c; \

#define MAIN_LOOP_UPPER_DIAG_COL(M) \
    n_rem = n % 4; \
    if (n_rem == 0) n_rem = 4; \
    ZERO_REGISTERS() \
    b_curr = b; \
    a_curr = a + i * ps_a; \
    mask_n = (1 << (n_rem*2)) - 1; \
    GEMM_MxN(M, n_rem) \
    STORE_COL_UPPER(M, n_rem) \
    c += 4 * rs_c; \

/****************************************/
/* Perform GEMMT operations             */
/* C matrix is row major matrix         */
/* Kernel size is 4x4                   */
/* For fringe cases, mask load/store    */
/* instruction is used                  */
/****************************************/
void bli_zgemmsup_rv_zen4_asm_4x4m_row
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[4];
    __m512d c_imag_reg[4];
    __m512d a_reg[4];
    __m512d b_reg;
    __m512d one_reg = _mm512_set1_pd(1);
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 4;
    dim_t m_rem = m % 4;
    dcomplex *a_curr, *b_curr, *c = c_;

    /*Load real and complex value of alpha*/
    __m512d alpha_reg = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_reg = _mm512_set1_pd(alpha->imag);

    /*Load real and complex value of beta*/
    __m512d beta_reg = _mm512_set1_pd(beta->real);
    __m512d beta_imag_reg = _mm512_set1_pd(beta->imag);

    dim_t i =0;

    /*4x4 block is handled here*/
    for (i = 0; i < m_main; i++)
    {
       MAIN_LOOP_ROW(4);
    }

    /*Fringe blocks are handled here*/
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_ROW(1); break;
        case 2:
            MAIN_LOOP_ROW(2); break;
        case 3:
            MAIN_LOOP_ROW(3); break;
    }

}

/****************************************/
/* Perform GEMMT operations             */
/* C matrix is col major matrix         */
/* Kernel size is 4x4                   */
/* For fringe cases, mask load/store    */
/* instruction is used                  */
/****************************************/
void bli_zgemmsup_rv_zen4_asm_4x4m_col
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[4];
    __m512d c_imag_reg[4];
    __m512d a_reg[4];
    __m512d b_reg;
    __m512d one_reg = _mm512_set1_pd(1);
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 4;
    dim_t m_rem = m % 4;
    dcomplex *a_curr, *b_curr, *c = c_;

    /*Load real and complex value of alpha*/
    __m512d alpha_reg = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_reg = _mm512_set1_pd(alpha->imag);

    /*Load real and complex value of beta*/
    __m512d beta_reg = _mm512_set1_pd(beta->real);
    __m512d beta_imag_reg = _mm512_set1_pd(beta->imag);

    dim_t i =0;
    /*4x4 block is handled here*/
    for (i = 0; i < m_main; i++)
    {
       MAIN_LOOP_COL(4);
    }

    /*Fringe blocks are handled here*/
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_COL(1); break;
        case 2:
            MAIN_LOOP_COL(2); break;
        case 3:
            MAIN_LOOP_COL(3); break;
    }

}

void bli_zgemmsup_rv_zen4_asm_4x4m
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    /* C is row stored*/
    if (cs_c == 1) {
        bli_zgemmsup_rv_zen4_asm_4x4m_row
                (
                    conja,
                    conjb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a, cs_a,
                    b,     rs_b, cs_b,
                    beta,
                    c_,     rs_c,     cs_c,
                    data,
                    cntx );
    }else{
        /* C is col stored*/
        bli_zgemmsup_rv_zen4_asm_4x4m_col
                (
                    conja,
                    conjb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a, cs_a,
                    b,     rs_b, cs_b,
                    beta,
                    c_,     rs_c,     cs_c,
                    data,
                    cntx );
    }
}

/****************************************/
/* Perform GEMMT operations             */
/* C matrix is row major matrix         */
/* Only lower portion below diagonal    */
/* elements are updated                 */
/* Kernel size is 4x4                   */
/* For fringe cases, mask load/store    */
/* instruction is used                  */
/****************************************/
void bli_zgemmsup_rv_zen4_asm_4x4m_lower_row
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[4];
    __m512d c_imag_reg[4];
    __m512d a_reg[4];
    __m512d b_reg;
    __m512d one_reg = _mm512_set1_pd(1);
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 4;
    dim_t m_rem = m % 4;
    dcomplex *a_curr,*b_curr, *c = c_;

    /*Load real and complex value of alpha*/
    __m512d alpha_reg = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_reg = _mm512_set1_pd(alpha->imag);

    /*Load real and complex value of beta*/
    __m512d beta_reg = _mm512_set1_pd(beta->real);
    __m512d beta_imag_reg = _mm512_set1_pd(beta->imag);

    dim_t i = 0;
    /*4x4 block is handled here*/
    for (i = 0; i < m_main; i++)
    {
        MAIN_LOOP_LOWER_DIAG_ROW(4);
    }

    /*Fringe blocks are handled here*/
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_LOWER_DIAG_ROW(1); break;
        case 2:
            MAIN_LOOP_LOWER_DIAG_ROW(2); break;
        case 3:
            MAIN_LOOP_LOWER_DIAG_ROW(3); break;
    }
}

/****************************************/
/* Perform GEMMT operations             */
/* C matrix is col major matrix         */
/* Only lower portion below diagonal    */
/* elements are updated                 */
/* Kernel size is 4x4                   */
/* For fringe cases, mask load/store    */
/* instruction is used                  */
/****************************************/
void bli_zgemmsup_rv_zen4_asm_4x4m_lower_col
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[4];
    __m512d c_imag_reg[4];
    __m512d a_reg[4];
    __m512d b_reg;
    __m512d one_reg = _mm512_set1_pd(1);
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 4;
    dim_t m_rem = m % 4;
    dcomplex *a_curr,*b_curr, *c = c_;

    /*Load real and complex value of alpha*/
    __m512d alpha_reg = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_reg = _mm512_set1_pd(alpha->imag);

    /*Load real and complex value of beta*/
    __m512d beta_reg = _mm512_set1_pd(beta->real);
    __m512d beta_imag_reg = _mm512_set1_pd(beta->imag);

    dim_t i = 0;
    /*4x4 block is handled here*/
    for (i = 0; i < m_main; i++)
    {
        MAIN_LOOP_LOWER_DIAG_COL(4);
    }

    /*Fringe blocks are handled here*/
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_LOWER_DIAG_COL(1); break;
        case 2:
            MAIN_LOOP_LOWER_DIAG_COL(2); break;
        case 3:
            MAIN_LOOP_LOWER_DIAG_COL(3); break;
    }
}

void bli_zgemmsup_rv_zen4_asm_4x4m_lower
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    /* C is row stored*/
   if (cs_c == 1) {
        bli_zgemmsup_rv_zen4_asm_4x4m_lower_row
                (
                    conja,
                    conjb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a, cs_a,
                    b,     rs_b, cs_b,
                    beta,
                    c_,     rs_c,     cs_c,
                    data,
                    cntx );
    }else{
        /* C is col stored*/
        bli_zgemmsup_rv_zen4_asm_4x4m_lower_col
                (
                    conja,
                    conjb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a, cs_a,
                    b,     rs_b, cs_b,
                    beta,
                    c_,     rs_c,     cs_c,
                    data,
                    cntx );
    }
}

/****************************************/
/* Perform GEMMT operations             */
/* C matrix is row major matrix         */
/* Only upper portion above diagonal    */
/* elements are updated                 */
/* Kernel size is 4x4                   */
/* For fringe cases, mask load/store    */
/* instruction is used                  */
/****************************************/
void bli_zgemmsup_rv_zen4_asm_4x4m_upper_row
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[4];
    __m512d c_imag_reg[4];
    __m512d a_reg[4];
    __m512d b_reg;
    __m512d one_reg = _mm512_set1_pd(1);
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 4;
    dim_t m_rem = m % 4;
    dcomplex *a_curr, *b_curr, *c = c_;

    /*Load real and complex value of alpha*/
    __m512d alpha_reg = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_reg = _mm512_set1_pd(alpha->imag);

    /*Load real and complex value of beta*/
    __m512d beta_reg = _mm512_set1_pd(beta->real);
    __m512d beta_imag_reg = _mm512_set1_pd(beta->imag);

    dim_t i = 0;
    /*4x4 block is handled here*/
    for (i = 0; i < m_main; i++)
    {
        MAIN_LOOP_UPPER_DIAG_ROW(4);
    }

    /*Fringe blocks are handled here*/
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_UPPER_DIAG_ROW(1); break;
        case 2:
            MAIN_LOOP_UPPER_DIAG_ROW(2); break;
        case 3:
            MAIN_LOOP_UPPER_DIAG_ROW(3); break;
    }
}

/****************************************/
/* Perform GEMMT operations             */
/* C matrix is col major matrix         */
/* Only upper portion above diagonal    */
/* elements are updated                 */
/* Kernel size is 4x4                   */
/* For fringe cases, mask load/store    */
/* instruction is used                  */
/****************************************/
void bli_zgemmsup_rv_zen4_asm_4x4m_upper_col
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    uint64_t ps_a = bli_auxinfo_ps_a( data );
    __m512d c_reg[4];
    __m512d c_imag_reg[4];
    __m512d a_reg[4];
    __m512d b_reg;
    __m512d one_reg = _mm512_set1_pd(1);
    __mmask8 mask_n;
    dim_t n_rem;
    dim_t m_main = m / 4;
    dim_t m_rem = m % 4;
    dcomplex *a_curr, *b_curr, *c = c_;

    /*Load real and complex value of alpha*/
    __m512d alpha_reg = _mm512_set1_pd(alpha->real);
    __m512d alpha_imag_reg = _mm512_set1_pd(alpha->imag);

    /*Load real and complex value of beta*/
    __m512d beta_reg = _mm512_set1_pd(beta->real);
    __m512d beta_imag_reg = _mm512_set1_pd(beta->imag);

    dim_t i = 0;
    /*4x4 block is handled here*/
    for (i = 0; i < m_main; i++)
    {
        MAIN_LOOP_UPPER_DIAG_COL(4);
    }

    /*Fringe blocks are handled here*/
    switch (m_rem)
    {
        case 1:
            MAIN_LOOP_UPPER_DIAG_COL(1); break;
        case 2:
            MAIN_LOOP_UPPER_DIAG_COL(2); break;
        case 3:
            MAIN_LOOP_UPPER_DIAG_COL(3); break;
    }
}

void bli_zgemmsup_rv_zen4_asm_4x4m_upper
      (
        conj_t              conja,
        conj_t              conjb,
        dim_t               m,
        dim_t               n,
        dim_t               k,
        dcomplex*    restrict alpha,
        dcomplex*    restrict a, inc_t rs_a, inc_t cs_a,
        dcomplex*    restrict b, inc_t rs_b, inc_t cs_b,
        dcomplex*    restrict beta,
        dcomplex*    restrict c_, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
      )
{
    /* C is row stored*/
    if (cs_c == 1) {
        bli_zgemmsup_rv_zen4_asm_4x4m_upper_row
                (
                    conja,
                    conjb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a, cs_a,
                    b,     rs_b, cs_b,
                    beta,
                    c_,     rs_c,     cs_c,
                    data,
                    cntx );
    }else{
        /* C is col stored*/
        bli_zgemmsup_rv_zen4_asm_4x4m_upper_col
                (
                    conja,
                    conjb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a, cs_a,
                    b,     rs_b, cs_b,
                    beta,
                    c_,     rs_c,     cs_c,
                    data,
                    cntx );
    }
}
