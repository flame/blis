/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "immintrin.h"
#include "blis.h"


// #define INTERLEAVE_LOAD_FMA // enable interleave load and FMA for A matrix

#ifdef INTERLEAVE_LOAD_FMA
    #define N_REGS NR           // if interleave load is enabled, n_roll registers are needed, 1 for each row
    #define FMA_OFFSET col      // offset at which fma result is stored in case interleave load and fma is enabled

#else
    #define N_REGS 1            // if interleave is not enabled, only one register is needed to store all FMA results.
    #define FMA_OFFSET 0        // if interleave not enabled, offset=0 so that all accumulations are stored in same register.

#endif

// Macros to unroll loop.
#if defined __clang__
    #define UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
#elif defined __GNUC__
    #define UNROLL_LOOP_FULL()
#else
    #define UNROLL_LOOP_FULL()
#endif


/*
 *   If INTERLEAVE_LOAD_FMA is enabled, for each column of A matrix, fresh set of registers are
 *   used for A matrix load and accumulation.
 *   Therefore after all the FMAs are computed, we need to accumulate the results
 *   of all registers into one register.
 *   
 *   This macro accumulates all the FMA registers into one register.
 * 
 * Assumptions:
 *   NR is defined as one of the loop count
 *   num_loads_per_MR is defined as MR / ELEM_PER_REGS
 *   is_fringe boolean is defined which is true when if MR is not a perfect multiple of ELEM_PER_REGS
 *   sv[num_loads_per_MR * NR] is defined as an array of vector registers which holds the FMA accumulations.
 *   ch is defined as the character type for the operation (e.g., d for double).
 *   PASTECH is a macro that concatenates the given strings to form the function name.
 *   PASTECH(_mm512_add_p, ch) is a macro that generates the appropriate AVX512 instruction for addition based on the character type.
 *   sv[num_loads_per_MR * NR] is the register that holds the final accumulation result.
 *   sv[0]  += sv[1] + sv[2] + ... + sv[NR-1]
 *   sv[NR] += sv[NR+1] + sv[NR+2] + ... + sv[2*NR-1]
 *   :
 *   sv[(num_loads_per_MR * NR)] += sv[(num_loads_per_MR * NR) + 1] + sv[(num_loads_per_MR * NR) + 2] + ... + sv[(num_loads_per_MR * NR) + (NR-1)]
 */
#ifdef INTERLEAVE_LOAD_FMA
    // accumulate result if interleave load is enabled.
    #define HANDLE_INTERLEAVE_LOAD_FMA(ch)                                                                                \
        if( NR > 1 || num_loads_per_MR > 1 || is_fringe )                                                                 \
        {                                                                                                                 \
            UNROLL_LOOP_FULL()                                                                                            \
            for(dim_t j = 1; j < NR; ++j)                                                                                 \
            {                                                                                                             \
                /* Add jth row  accumulation results to 1st row results. */                                               \
                UNROLL_LOOP_FULL()                                                                                        \
                for(dim_t row_reg = 0; row_reg < num_loads_per_MR; ++row_reg )                                            \
                {                                                                                                         \
                    sv[(row_reg * NR)] = PASTECH(_mm512_add_p, ch)(sv[(row_reg * NR)], sv[(row_reg * NR) + j]);           \
                }                                                                                                         \
                                                                                                                          \
                if(is_fringe)                                                                                             \
                {                                                                                                         \
                    sv[(num_loads_per_MR * NR)] = PASTECH(_mm512_add_p, ch)(sv[(num_loads_per_MR * NR)],                  \
                                                                            sv[(num_loads_per_MR * NR) + j]);             \
                }                                                                                                         \
            }                                                                                                             \
        }
#else
    #define HANDLE_INTERLEAVE_LOAD_FMA(ch) // Placeholder definition.

#endif


// # region ST N kernel

/*
 *  Calculate thread local values a_start and y_start
 *  For single thread the values a_start and y_start are set to global a and y
 *  thread_start is not used in single thread case, so it is defined as void.
 *  This macro is used to avoid unused variable warning.
 */
#define INIT_A_Y(ctype, MR)          \
    ctype* restrict y_start = y;     \
    ctype* restrict a_start = a;     \
    (void) thread_start;            // avoid warning for unused variable

/*
 *  For single thread, this macros is a placeholder
 */
#define PARALLELIZE_LOOP(ch) \
    (void) nt;                  // avoid warning for unused variable

/*
 *  Definition to generate function name for micro kernel of size MR, NR
 *  Expanded function name will be: bli_<ch>gemv_n_block_<MR>_<NR>_<st/mt>
 *  Example: bli_dgemv_n_block_40_2_st
 */
#define  GENTFUNC_GEMVS(ctype, ch, MR, NR, threading)       \
    PASTEMAC6(ch, gemv_n_block_, MR, _, NR, _, threading)

/*
 *  This function is used to generate code for n-kernel of size MR, NR.
 *  n-kernel implies innermost loop is along the n-direction (columns) of matrix A
 *
 *  Assumptions:
 *          ( m % MR ) == 0, m is a multiple of MR
 *          ( n % NR ) == 0, n is a multiple of NR
 *  ELEM_PER_REG is the number of elements that can fit in a SIMD register.
 *  load happen along the columns of A matrix.
 *  fringe indicates if MR is not a perfect multiple of ELEM_PER_REG.
 */
 /*
 * |y0| += | a00 a01 |   |    |
   |y1|    | a10 a11 |   | x0 |
   |y2|    | a20 a21 | * | x1 |
   |y3|    | a30 a31 |   |    |

    To create a chain of independent instructions
    we can use multiple set of registers to hold the results of FMA operations.
    * For example, if MR = 4 and NR = 2, we can use 2 sets of registers
    * Final we perform operation like axpyf, where f the fusing factor = NR.
    * Example this one set
   |y0|  +=  | a00 | * x0
   |y1|      | a10 |
   |y2|      | a20 |
   |y3|      | a30 |

   * This is another set
   |y0|  +=  | a01 | * x1
   |y1|      | a11 |
   |y2|      | a21 |
   |y3|      | a31 |
   * finally we combine results from both these sets
   * using HANDLE_INTERLEAVE_LOAD_FMA macro
    where a is the matrix, x is the vector and y is the result vector.
 */
#define GENTFUNC_GEMV(ctype, ch, MR, NR, threading )                                                                                  \
void GENTFUNC_GEMVS (ctype,  ch, MR, NR, threading )                                                                                  \
     (                                                                                                                                \
       trans_t transa,                                                                                                                \
       conj_t  conjx,                                                                                                                 \
       dim_t   m,                                                                                                                     \
       dim_t   n,                                                                                                                     \
       ctype* alpha,                                                                                                                  \
       ctype* a, inc_t rs_a, inc_t cs_a,                                                                                              \
       ctype* x, inc_t incx,                                                                                                          \
       ctype* beta,                                                                                                                   \
       ctype* y, inc_t incy,                                                                                                          \
       cntx_t* cntx                                                                                                                   \
     )                                                                                                                                \
{                                                                                                                                     \
    const dim_t ELEM_SIZE         = sizeof(ctype);                                                                                    \
    const dim_t ELEM_PER_REG      = ( BLIS_SIMD_SIZE / ELEM_SIZE ); /* Elements per register             */                           \
    const dim_t num_loads_per_MR  = (  MR / ELEM_PER_REG );         /* Number of loads needed for one MR */                           \
    const bool  is_fringe         = ( (MR % ELEM_PER_REG) != 0 );   /* is partial load needed            */                           \
    dim_t MR_left                 = (  m  % ELEM_PER_REG ) ;        /* num elements for partial load     */                           \
                                                                                                                                      \
    ctype *x_temp = x;                                                                                                                \
    dim_t nt      = 1;                                                                                                                \
                                                                                                                                      \
   /* if MT enabled, Macro launches OpenMP threads */                                                                                 \
    PARALLELIZE_LOOP(ch)                                                                                                              \
    {                                                                                                                                 \
        dim_t job_per_thread = m;                                                                                                     \
        dim_t thread_start   = 0;                                                                                                     \
                                                                                                                                      \
        /*  create local copies of av(A vector), xv(X vector) and sv(sum vector) for each thread */                                   \
        /*  av: A vector                                                                         */                                   \
        /*  xv: X vector                                                                         */                                   \
        /*  sv: sum vector                                                                       */                                   \
                                                                                                                                      \
                                                                                                                                      \
        PASTECH(__m512, ch) av[num_loads_per_MR * N_REGS + 1];                                                                        \
        PASTECH(__m512, ch) sv[num_loads_per_MR * N_REGS + 1];                                                                        \
                                                                                                                                      \
        PASTECH(__m512, ch) xv[NR + MR];                                                                                              \
                                                                                                                                      \
        /* Initialize A, Y for current thread */                                                                                      \
        INIT_A_Y(ctype, MR)                                                                                                           \
                                                                                                                                      \
        /* loop over M in steps of MR */                                                                                              \
        for(dim_t i = 0; i < job_per_thread; i += MR)                                                                                 \
        {                                                                                                                             \
            /* compute thread local A, Y, X*/                                                                                         \
            /* Partition along the rows, a_local = a_start + i * rs_a */                                                              \
            ctype* a_local = a_start + (0 * cs_a) + i * rs_a;                                                                         \
            ctype* y_local = y_start + i * incy;                                                                                      \
            ctype* x_local = x_temp;                                                                                                  \
            UNROLL_LOOP_FULL()                                                                                                        \
            for(dim_t j = 0; j <  (N_REGS * num_loads_per_MR); ++j)                                                                   \
            {                                                                                                                         \
                /* set all accumulator vectors to zero */                                                                             \
                sv[j] = PASTECH(_mm512_setzero_p, ch)();                                                                              \
            }                                                                                                                         \
            if(is_fringe)                                                                                                             \
            {                                                                                                                         \
                /* if MR % ELEM_PER_REG != 0, one extra register to process these elements */                                         \
                sv[num_loads_per_MR] = PASTECH(_mm512_setzero_p, ch)();                                                               \
            }                                                                                                                         \
            UNROLL_LOOP_FULL()                                                                                                        \
            /* loop over n in steps of NR */                                                                                          \
            for(dim_t j = 0; j < n; j += NR)                                                                                          \
            {                                                                                                                         \
                UNROLL_LOOP_FULL()                                                                                                    \
                /* loop over each column in NR */                                                                                     \
                for ( dim_t col = 0; col < NR; ++col )                                                                                \
                {                                                                                                                     \
                    xv[col] = PASTECH(_mm512_set1_p, ch)( (*alpha) * (*x_local) );                                                    \
                    x_local += incx;                                                                                                  \
                    UNROLL_LOOP_FULL()                                                                                                \
                    /* loop over MR in steps of elems_per_register */                                                                 \
                    for(dim_t row_reg = 0; row_reg < num_loads_per_MR; ++row_reg)                                                     \
                    {                                                                                                                 \
                        av[(row_reg*N_REGS)+FMA_OFFSET] =                                                                             \
                       PASTECH(_mm512_loadu_p, ch)( a_local + (j + col) * cs_a + ((row_reg * ELEM_PER_REG) * rs_a));                  \
                        sv[(row_reg*N_REGS)+FMA_OFFSET] =                                                                             \
                        PASTECH(_mm512_fmadd_p, ch)( xv[col], av[(row_reg * N_REGS) + FMA_OFFSET],                                    \
                                                              sv[(row_reg * N_REGS) + FMA_OFFSET] );                                  \
                    }                                                                                                                 \
                    if(is_fringe)                                                                                                     \
                    {                                                                                                                 \
                        av[(num_loads_per_MR*N_REGS)+FMA_OFFSET] =                                                                    \
                        PASTECH(_mm512_maskz_loadu_p, ch)( (1 << (MR_left)) - 1,   a_local + (j + col) * cs_a +                       \
                                                                ((num_loads_per_MR * ELEM_PER_REG) * rs_a));                          \
                        sv[(num_loads_per_MR*N_REGS)+FMA_OFFSET] =                                                                    \
                        PASTECH(_mm512_fmadd_p, ch)( xv[col], av[(num_loads_per_MR * N_REGS) + FMA_OFFSET],                           \
                                                              sv[(num_loads_per_MR * N_REGS) + FMA_OFFSET] );                         \
                    }                                                                                                                 \
                }                                                                                                                     \
            }                                                                                                                         \
            HANDLE_INTERLEAVE_LOAD_FMA(ch)                                                                                            \
            PASTECH(__m512, ch) beta_ = PASTECH(_mm512_set1_p, ch)( *beta );                                                          \
            UNROLL_LOOP_FULL()                                                                                                        \
            /* scale by beta */                                                                                                       \
            for(dim_t row_reg = 0; row_reg < num_loads_per_MR; ++row_reg)                                                             \
            {                                                                                                                         \
                if ( !bli_deq0( *beta ) )                                                                                             \
                {                                                                                                                     \
                    xv[row_reg]            = PASTECH(_mm512_loadu_p, ch)(y_local + row_reg * incy * ELEM_PER_REG);                    \
                    sv[(row_reg * N_REGS)] = PASTECH(_mm512_fmadd_p, ch)(xv[row_reg], beta_, sv[(row_reg * N_REGS)]);                 \
                }                                                                                                                     \
                PASTECH(_mm512_storeu_p, ch)((ctype*)(y_local + row_reg * incy * ELEM_PER_REG ), sv[(row_reg * N_REGS)]);             \
            }                                                                                                                         \
            if(is_fringe)                                                                                                             \
            {                                                                                                                         \
                if ( !bli_deq0( *beta ) )                                                                                             \
                {                                                                                                                     \
                    xv[num_loads_per_MR] =                                                                                            \
                    PASTECH(_mm512_maskz_loadu_p, ch)((1 << (MR_left)) - 1, y_local + num_loads_per_MR * incy * ELEM_PER_REG);        \
                    sv[(num_loads_per_MR * N_REGS)] =                                                                                 \
                    PASTECH(_mm512_fmadd_p, ch)(xv[num_loads_per_MR], beta_, sv[(num_loads_per_MR * N_REGS)]);                        \
                }                                                                                                                     \
                PASTECH(_mm512_mask_storeu_p, ch)((ctype*)(y_local + num_loads_per_MR * incy * ELEM_PER_REG ),                        \
                                                          (1 << (MR_left)) - 1, sv[(num_loads_per_MR * N_REGS)]);                     \
            }                                                                                                                         \
        }                                                                                                                             \
    }                                                                                                                                 \
} // End of gemv kernel function


// Define a macro for gemv function pointer
#define GENT_GEMV_FPTR(ctype, ch)                         \
    typedef void (*PASTECH(ch, gemv_ker))                 \
    (                                                     \
        trans_t transa,                                   \
        conj_t conjx,                                     \
        dim_t m,                                          \
        dim_t n,                                          \
        ctype *alpha,                                     \
        ctype *a, inc_t rs_a, inc_t cs_a,                 \
        ctype *x, inc_t incx,                             \
        ctype *beta,                                      \
        ctype *y, inc_t incy,                             \
        cntx_t * cntx                                     \
    );

/* Example */
//typedef void (*dgemv_ker) (
//                            trans_t transa,
//                            conj_t conjx,
//                            dim_t m,
//                            dim_t n,
//                            double* alpha,
//                            double* a, inc_t rs_a, inc_t cs_a,
//                            double* x, inc_t incx,
//                            double* beta,
//                            double* y, inc_t incy,
//                            cntx_t* cntx
//                         );
GENT_GEMV_FPTR(double, d);

/*
*  Macro which defines various kernel functions for different sizes of MR and NR.
*  Primary Kernel , MR = 40 and NR = 2, ELES_PER_REGS = 8.
*  Secondary Kernels MR = {4,3,2,1,0} * ELEMS_PER_REGS + {0,1}* (ELEM_PER_REGS - 1)
*   NR = {0,1}
*
*  Generate a static array of function pointer pointing to different n-kernels.
*  The function pointer array is indexed by the index of MR and NR.
*  The function pointer array is used to call the correct kernel for given size.

*/
#define GENERATE_KERNELS(threading)                                                         \
                                                                                            \
    GENTFUNC_GEMV(double, d, 40, 2, threading); GENTFUNC_GEMV(double, d, 40, 1, threading); \
    GENTFUNC_GEMV(double, d, 39, 2, threading); GENTFUNC_GEMV(double, d, 39, 1, threading); \
    GENTFUNC_GEMV(double, d, 32, 2, threading); GENTFUNC_GEMV(double, d, 32, 1, threading); \
    GENTFUNC_GEMV(double, d, 31, 2, threading); GENTFUNC_GEMV(double, d, 31, 1, threading); \
    GENTFUNC_GEMV(double, d, 24, 2, threading); GENTFUNC_GEMV(double, d, 24, 1, threading); \
    GENTFUNC_GEMV(double, d, 23, 2, threading); GENTFUNC_GEMV(double, d, 23, 1, threading); \
    GENTFUNC_GEMV(double, d, 16, 2, threading); GENTFUNC_GEMV(double, d, 16, 1, threading); \
    GENTFUNC_GEMV(double, d, 15, 2, threading); GENTFUNC_GEMV(double, d, 15, 1, threading); \
    GENTFUNC_GEMV(double, d,  8, 2, threading); GENTFUNC_GEMV(double, d,  8, 1, threading); \
    GENTFUNC_GEMV(double, d,  7, 2, threading); GENTFUNC_GEMV(double, d,  7, 1, threading); \
    \
    \
    static dgemv_ker PASTECH(dgemv_n_ker_fp_, threading)[10][2] =                                     \
    {                                                                                                 \
        { GENTFUNC_GEMVS(double, d, 40, 2, threading), GENTFUNC_GEMVS(double, d, 40, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d,  8, 2, threading), GENTFUNC_GEMVS(double, d,  8, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d, 16, 2, threading), GENTFUNC_GEMVS(double, d, 16, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d, 24, 2, threading), GENTFUNC_GEMVS(double, d, 24, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d, 32, 2, threading), GENTFUNC_GEMVS(double, d, 32, 1, threading) }, \
        \
        { GENTFUNC_GEMVS(double, d,  7, 2, threading), GENTFUNC_GEMVS(double, d,  7, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d, 15, 2, threading), GENTFUNC_GEMVS(double, d, 15, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d, 23, 2, threading), GENTFUNC_GEMVS(double, d, 23, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d, 31, 2, threading), GENTFUNC_GEMVS(double, d, 31, 1, threading) }, \
        { GENTFUNC_GEMVS(double, d, 39, 2, threading), GENTFUNC_GEMVS(double, d, 39, 1, threading) }  \
    };                                                                                                \

/*
* Here we generate the kernels for single thread.
* and defines a function pointer array which holds pointers to
* kernels for all combinations of MR and NR.
*/
GENERATE_KERNELS(st)


// Macro for Wrapper function to call correct kernels for given m, & n
/*
* Define a macro for the function that calls the correct kernel
* based on the sizes of m and n
* example: bli_dgemv_n_zen4_int_MRxNR_st ()
*/
/*
* This operation computes y = beta*y + alpha*A*x, where A is an m x n matrix,
* x is a vector of length n, and y is a vector of length m.
* The matrix A is partitioned into four regions, and the kernel is called
* for each region.
*/
/*
                             n
          <------------------------------------------------------>
          <------------- multiple of NR -------------><--n_left-->
        ^ +------------------------------------------+-----------+
        | |                                          |           |
        | |            m >= MR && n >= NR            | m >= MR   |
        | |                                          | && n % NR |
m       | |              multiple of MR              |           |
        | |                                          |           |
        | |         Region 1                         | Region 2  |
        | |                                          |           |
        v +------------------------------------------+-----------+
      ^   |            n >= NR && m % MR             | m % MR    |
m_left|   |            Region 3                      | && n % NR |
      v   +------------------------------------------+-----------+
*/

#define GENT_GEMV_CALLER(ctype, ch, MR, NR, direction, threading)                        \
void PASTEMAC4(ch, gemv_, direction, PASTECH4(_zen4_int_, MR, x, NR, _ ), threading)     \
     (                                                                                   \
       trans_t transa,                                                                   \
       conj_t  conjx,                                                                    \
       dim_t   m,                                                                        \
       dim_t   n,                                                                        \
       ctype* alpha,                                                                     \
       ctype* a, inc_t rs_a, inc_t cs_a,                                                 \
       ctype* x, inc_t incx,                                                             \
       ctype* beta,                                                                      \
       ctype* y, inc_t incy,                                                             \
       cntx_t* cntx                                                                      \
     )                                                                                   \
{                                                                                        \
    /* Checks are added because this function can be called directly from other BLAS     \
     * routines like GEMM                                                                \
     */                                                                                  \
    if ( (( rs_a != 1 ) && ( cs_a != 1 )) || transa != BLIS_NO_TRANSPOSE || incy != 1 )  \
    {                                                                                    \
        PASTEMAC(ch, gemv_zen_ref) (                                                     \
                                      transa,                                            \
                                      m,                                                 \
                                      n,                                                 \
                                      alpha,                                             \
                                      a, rs_a, cs_a,                                     \
                                      x, incx,                                           \
                                      beta,                                              \
                                      y, incy,                                           \
                                      NULL                                               \
                                   );                                                    \
                                                                                         \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)                                      \
        return;                                                                          \
    }                                                                                    \
                                                                                         \
    const dim_t elem_per_reg = BLIS_SIMD_SIZE / sizeof(ctype);                           \
                                                                                         \
    bool is_m_left = ((m % MR) % elem_per_reg) >= 1 ? true : false;                      \
    dim_t m_idx = (m % MR) / elem_per_reg +                                              \
                  (is_m_left * MR/elem_per_reg);                                         \
    dim_t n_idx = (n % NR);                                                              \
    ctype one = 1;                                                                       \
                                                                                         \
    /* Region 1 */                                                                       \
    if(m >= MR && n >= NR)                                                               \
    {                                                                                    \
        PASTECH4(ch, gemv_, direction, _ker_fp_, threading)[0][0]                        \
        (                                                                                \
            transa,                                                                      \
            conjx,                                                                       \
            ((dim_t)( m / MR )) * MR,                                                    \
            ((dim_t)( n / NR )) * NR,                                                    \
            alpha,                                                                       \
            a, rs_a, cs_a,                                                               \
            x, incx,                                                                     \
            beta,                                                                        \
            y, incy,                                                                     \
            cntx                                                                         \
        );                                                                               \
    }                                                                                    \
                                                                                         \
    /* Region 2 */                                                                       \
    if (m >= MR && n % NR)                                                               \
    {   /* example: bli_dgemv_n_ker_fp_st[][] */                                         \
        PASTECH4(ch, gemv_, direction, _ker_fp_, threading)[0][n_idx]                    \
        (                                                                                \
            transa,                                                                      \
            conjx,                                                                       \
            ((dim_t)( m / MR )) * MR,                                                    \
            n % NR,                                                                      \
            alpha,                                                                       \
            a + (((dim_t)( n / NR )) * NR * cs_a), rs_a, cs_a,                           \
            x + (((dim_t)( n / NR )) * NR * incx), incx,                                 \
            n > NR ? &one : beta,                                                        \
            y, incy,                                                                     \
            cntx                                                                         \
        );                                                                               \
    }                                                                                    \
                                                                                         \
     /* Region 3 */                                                                      \
    if (n >= NR && m % MR)                                                               \
    {                                                                                    \
        PASTECH4(ch, gemv_, direction, _ker_fp_, threading)[m_idx][0]                    \
        (                                                                                \
            transa,                                                                      \
            conjx,                                                                       \
            m % MR,                                                                      \
            ((dim_t)( n / NR )) * NR,                                                    \
            alpha,                                                                       \
            a + (((dim_t)( m / MR )) * MR * rs_a), rs_a, cs_a,                           \
            x, incx,                                                                     \
            beta,                                                                        \
            y + ((dim_t)( m / MR )) * MR * incy, incy,                                   \
            cntx                                                                         \
        );                                                                               \
    }                                                                                    \
                                                                                         \
    /* Region 4 */                                                                       \
    if (m % MR && n % NR)                                                                \
    {                                                                                    \
        PASTECH4(ch, gemv_, direction, _ker_fp_, threading)[m_idx][n_idx]                \
        (                                                                                \
            transa,                                                                      \
            conjx,                                                                       \
            m % MR,                                                                      \
            n % NR,                                                                      \
            alpha,                                                                       \
            a + (((dim_t)( m / MR )) * MR * rs_a) +                                      \
                         ((dim_t)( n / NR )) * NR * cs_a, rs_a, cs_a,                    \
            x + (((dim_t)( n / NR )) * NR * incx), incx,                                 \
            n > NR ? &one : beta,                                                        \
            y + ((dim_t)( m / MR )) * MR * incy, incy,                                   \
            cntx                                                                         \
        );                                                                               \
    }                                                                                    \
}                                                                                        \

// Call the interface kernel
GENT_GEMV_CALLER(double, d, 40, 2, n, st);

// #endregion ST n kernels

// #region MT n kernels

// This section is for multi-threaded n-kernels.


// set A, Y for each thread
#ifdef BLIS_ENABLE_OPENMP

/* Redefine INIT_A_Y */

#undef INIT_A_Y

#define INIT_A_Y(ctype, MR)                                             \
    const dim_t tid      = omp_get_thread_num();                        \
    const dim_t nt_real  = omp_get_num_threads();                       \
                                                                        \
    /* dim_t thread_start = 0;  */                                      \
    /* This work-distribution should be replaced by bli_thread_range */ \
    bli_normfv_thread_partition( m, nt_real, &thread_start,             \
                                 &job_per_thread, MR, incy, tid );      \
                                                                        \
    ctype* restrict y_start = y + thread_start;                         \
    ctype* restrict a_start = a + (thread_start / incy) * rs_a;         \
    (void) nt_real;      // avoid warning for unused variable


// Redefine PARALLELIZE_LOOP
#undef PARALLELIZE_LOOP


#define PARALLELIZE_LOOP(ch)                                              \
   /* Find optimal number of threads */                                   \
   /* This function name should be changed later                          \
    * to bli_optimalthreads_l2                                            \
   */                                                                     \
    bli_nthreads_l2                                                       \
    (                                                                     \
        BLIS_GEMV_KER,                                                    \
        PASTEMAC(ch,type),                                                \
        BLIS_NO_TRANSPOSE,                                                \
        bli_arch_query_id(),                                              \
        m,                                                                \
        n,                                                                \
        &nt                                                               \
    );                                                                    \
    _Pragma("omp parallel num_threads(nt)")


/* Generate multi-thread version of all bli_chgemv_n_MR_NR_mt kernels */
GENERATE_KERNELS(mt)

/*  Call the multi-thread interface kernel */
GENT_GEMV_CALLER(double, d, 40, 2, n, mt); //bli_dgemv_n_zen4_int_40x2_mt

#endif
// #endregion MT N kernels




// #region ST M kernels
#undef GENTFUNC_GEMVS
#undef GENTFUNC_GEMV

/*
 * Macro to generate the name for a GEMV M-kernel function.
 * The name follows the pattern: bli_<ch>gemv_m_block_<MR>_<NR>
 * Example: bli_dgemv_m_block_40_8
 */
#define GENTFUNC_GEMVS(ctype, ch, MR, NR) \
    PASTEMAC4(ch, gemv_m_block_, MR, _, NR)

/*
 * Macro to generate a GEMV M-kernel function.
 * An M-kernel processes the matrix in a column-major fashion, meaning the
 * innermost loop iterates along the M dimension (rows). This is generally
 * optimal for matrices stored in column-major format or when the N
 * dimension is small.
 *
 * Parameters:
 *   ctype    - C data type (e.g., double, float)
 *   ch       - Character identifier for the data type (e.g., 'd' for double)
 *   MR       - The micro-kernel row dimension (register block size for M).
 *   NR       - The micro-kernel column dimension (register block size for N).
 *
 * SIMD Algorithm:
 * 1. The outer loop iterates through the columns of matrix A in steps of NR.
 * 2. For each block of NR columns:
 *    a. The corresponding NR elements of vector x are multiplied by alpha
 *       and broadcast into NR separate SIMD registers (xv).
 * 3. The inner loops iterate through the rows of the current block of A in
 *    steps of MR.
 * 4. For each block of MR rows:
 *    a. The corresponding MR elements of vector y are loaded into SIMD
 *       registers (yv). This is done in chunks of EPR (Elements Per Register).
 *    b. For each of the NR columns, the corresponding MR elements of A are
 *       loaded and a Fused Multiply-Add (FMA) operation is performed with
 *       the broadcasted x value and the y register.
 *       yv[k] = FMA(av[l], xv[l], yv[k])
 *    c. The updated yv registers are stored back to the y vector.
 * 5. Special handling for fringe cases (where m is not a multiple of MR,
 *    or MR is not a multiple of EPR) is done using masked loads and stores.
 */
#define GENTFUNC_GEMV(ctype, ch, MR, NR)                                                         \
void GENTFUNC_GEMVS(ctype, ch, MR, NR)                                                           \
     (                                                                                           \
       trans_t transa,                                                                           \
       conj_t  conjx,                                                                            \
       dim_t   m,                                                                                \
       dim_t   n,                                                                                \
       ctype* alpha,                                                                             \
       ctype* a, inc_t rs_a, inc_t cs_a,                                                         \
       ctype* x, inc_t incx,                                                                     \
       ctype* beta,                                                                              \
       ctype* y, inc_t incy,                                                                     \
       cntx_t* cntx                                                                              \
     )                                                                                           \
{                                                                                                \
    const dim_t ELEM_SIZE = sizeof(ctype);                                                       \
    const dim_t EPR       = BLIS_SIMD_SIZE / ELEM_SIZE; /* Elements per SIMD register */         \
    ctype* restrict abuf = a;                                        /* A buffer */              \
    ctype* restrict xbuf = x;                                        /* x buffer */              \
    ctype* restrict ybuf = y;                                        /* y buffer */              \
                                                                                                 \
    const dim_t num_y_full = MR / EPR;                  /* Number of full Y registers */         \
    const dim_t MR_left    = m % EPR;                   /* Number of left over elements < MR */  \
    const dim_t num_y_regs = num_y_full + ((MR_left == 0) ? 0 : 1); /* Number of Y registers */  \
                                                                                                 \
    PASTECH(__m512, ch) xv[NR];          /* X vector registers */                                \
    PASTECH(__m512, ch) yv[num_y_regs];  /* Y vector registers */                                \
    PASTECH(__m512, ch) av[NR];          /* A vector registers */                                \
                                                                                                 \
    if( (m/MR) < 1) /* if m < MR, set m to MR */                                                 \
    {                                                                                            \
        m = MR;                                                                                  \
    }                                                                                            \
                                                                                                 \
    for(dim_t i = 0; i < (n / NR); ++i)                                                          \
    {                                                                                            \
        ybuf = y;                             /* set ybuf to start of y */                       \
        abuf = a + (NR * i * cs_a);           /* abuf points to i*NRth column of A */            \
        for(dim_t j = 0; j < NR; ++j)                                                            \
        {                                                                                        \
            /* broadcast alpha * x[j] */                                                         \
            xv[j] = PASTECH(_mm512_set1_p, ch)( (*alpha) * (*(xbuf + j * incx)) );               \
        }                                                                                        \
                                                                                                 \
        for( dim_t j = 0; j < (m / MR); ++j)      /* In steps of MR */                           \
        {                                                                                        \
            for(dim_t k = 0; k < num_y_full; ++k) /* for each load register in MR */             \
            {                                                                                    \
                /* load &ybuf[k * EPR] into yv[k] */                                             \
                yv[k] = PASTECH(_mm512_loadu_p, ch)(ybuf + k * EPR);                             \
                /* NR times we load A and perform FMA with x & y */                              \
                                                                                                 \
                for(dim_t l = 0; l < NR; ++l)                                                    \
                {                                                                                \
                    /* load &abuf[k*EPR*rs_a + l*cs_a] into av[l] */                             \
                    av[l] = PASTECH(_mm512_loadu_p, ch)(abuf + k*EPR*rs_a + l*cs_a);             \
                    /* perform FMA with av[l], xv[l], yv[k] */                                   \
                    /* yv[k] = yv[k] + av[l] * xv[l] */                                          \
                    yv[k] = PASTECH(_mm512_fmadd_p, ch)(av[l], xv[l], yv[k]);                    \
                }                                                                                \
                /* store yv[k] into &ybuf[k*EPR*incy] */                                         \
                PASTECH(_mm512_storeu_p, ch)( ybuf + k*EPR*incy, yv[k] );                        \
            }                                                                                    \
                                                                                                 \
            if (MR_left)                                                                         \
            {                                                                                    \
                yv[num_y_full] = PASTECH(_mm512_maskz_loadu_p, ch)((1 << (MR_left)) - 1,         \
                                                                   ybuf + num_y_full * EPR);     \
                for(dim_t l = 0; l < NR; ++l)                                                    \
                {                                                                                \
                    av[l] = PASTECH(_mm512_maskz_loadu_p, ch)((1 << (MR_left))-1,                \
                                                abuf + num_y_full * EPR * rs_a + l * cs_a);      \
                    yv[num_y_full] = PASTECH(_mm512_fmadd_p, ch)(av[l], xv[l], yv[num_y_full]);  \
                }                                                                                \
                PASTECH(_mm512_mask_storeu_p, ch)( ybuf + num_y_full * EPR * incy,               \
                             (1 << (MR_left)) - 1, yv[num_y_full]);                              \
            }                                                                                    \
            ybuf += MR*incy;                                                                     \
            abuf += MR*rs_a;                                                                     \
        }                                                                                        \
        xbuf += NR*incx;                                                                         \
    }                                                                                            \
} // End of function GENTFUNC_GEMV



GENTFUNC_GEMV(double, d, 40, 8); GENTFUNC_GEMV(double, d, 40, 7); GENTFUNC_GEMV(double, d, 40, 6); GENTFUNC_GEMV(double, d, 40, 5); GENTFUNC_GEMV(double, d, 40, 4); GENTFUNC_GEMV(double, d, 40, 3); GENTFUNC_GEMV(double, d, 40, 2); GENTFUNC_GEMV(double, d, 40, 1);
GENTFUNC_GEMV(double, d, 39, 8); GENTFUNC_GEMV(double, d, 39, 7); GENTFUNC_GEMV(double, d, 39, 6); GENTFUNC_GEMV(double, d, 39, 5); GENTFUNC_GEMV(double, d, 39, 4); GENTFUNC_GEMV(double, d, 39, 3); GENTFUNC_GEMV(double, d, 39, 2); GENTFUNC_GEMV(double, d, 39, 1);
GENTFUNC_GEMV(double, d, 32, 8); GENTFUNC_GEMV(double, d, 32, 7); GENTFUNC_GEMV(double, d, 32, 6); GENTFUNC_GEMV(double, d, 32, 5); GENTFUNC_GEMV(double, d, 32, 4); GENTFUNC_GEMV(double, d, 32, 3); GENTFUNC_GEMV(double, d, 32, 2); GENTFUNC_GEMV(double, d, 32, 1);
GENTFUNC_GEMV(double, d, 31, 8); GENTFUNC_GEMV(double, d, 31, 7); GENTFUNC_GEMV(double, d, 31, 6); GENTFUNC_GEMV(double, d, 31, 5); GENTFUNC_GEMV(double, d, 31, 4); GENTFUNC_GEMV(double, d, 31, 3); GENTFUNC_GEMV(double, d, 31, 2); GENTFUNC_GEMV(double, d, 31, 1);
GENTFUNC_GEMV(double, d, 24, 8); GENTFUNC_GEMV(double, d, 24, 7); GENTFUNC_GEMV(double, d, 24, 6); GENTFUNC_GEMV(double, d, 24, 5); GENTFUNC_GEMV(double, d, 24, 4); GENTFUNC_GEMV(double, d, 24, 3); GENTFUNC_GEMV(double, d, 24, 2); GENTFUNC_GEMV(double, d, 24, 1);
GENTFUNC_GEMV(double, d, 23, 8); GENTFUNC_GEMV(double, d, 23, 7); GENTFUNC_GEMV(double, d, 23, 6); GENTFUNC_GEMV(double, d, 23, 5); GENTFUNC_GEMV(double, d, 23, 4); GENTFUNC_GEMV(double, d, 23, 3); GENTFUNC_GEMV(double, d, 23, 2); GENTFUNC_GEMV(double, d, 23, 1);
GENTFUNC_GEMV(double, d, 16, 8); GENTFUNC_GEMV(double, d, 16, 7); GENTFUNC_GEMV(double, d, 16, 6); GENTFUNC_GEMV(double, d, 16, 5); GENTFUNC_GEMV(double, d, 16, 4); GENTFUNC_GEMV(double, d, 16, 3); GENTFUNC_GEMV(double, d, 16, 2); GENTFUNC_GEMV(double, d, 16, 1);
GENTFUNC_GEMV(double, d, 15, 8); GENTFUNC_GEMV(double, d, 15, 7); GENTFUNC_GEMV(double, d, 15, 6); GENTFUNC_GEMV(double, d, 15, 5); GENTFUNC_GEMV(double, d, 15, 4); GENTFUNC_GEMV(double, d, 15, 3); GENTFUNC_GEMV(double, d, 15, 2); GENTFUNC_GEMV(double, d, 15, 1);
GENTFUNC_GEMV(double, d, 8, 8); GENTFUNC_GEMV(double, d, 8, 7); GENTFUNC_GEMV(double, d, 8, 6); GENTFUNC_GEMV(double, d, 8, 5); GENTFUNC_GEMV(double, d, 8, 4); GENTFUNC_GEMV(double, d, 8, 3); GENTFUNC_GEMV(double, d, 8, 2); GENTFUNC_GEMV(double, d, 8, 1);
GENTFUNC_GEMV(double, d, 7, 8); GENTFUNC_GEMV(double, d, 7, 7); GENTFUNC_GEMV(double, d, 7, 6); GENTFUNC_GEMV(double, d, 7, 5); GENTFUNC_GEMV(double, d, 7, 4); GENTFUNC_GEMV(double, d, 7, 3); GENTFUNC_GEMV(double, d, 7, 2); GENTFUNC_GEMV(double, d, 7, 1);


/*
 * Function pointer array for the single-threaded GEMV M-kernels.
 *
 * This 2D array serves as a fast lookup table to select the appropriate
 * M-kernel at runtime based on the remainder dimensions (m % MR, n % NR).
 * The GENT_GEMV_CALLER function uses `m_idx` and `n_idx` to index into this
 * array and dispatch the correct kernel for the fringe regions of the matrix.
 *
 * The array is indexed as `dgemv_m_ker_fp[m_idx][n_idx]`.
 * - The first dimension (m_idx) corresponds to different MR values.
 * - The second dimension (n_idx) corresponds to different NR values (1 to 7, with 8 as the full case).
 *
 * The kernels are ordered to provide optimal coverage for various matrix sizes.
 */
static dgemv_ker dgemv_m_ker_fp[10][8] =
{
    { GENTFUNC_GEMVS(double, d, 40, 8), GENTFUNC_GEMVS(double, d, 40, 1), GENTFUNC_GEMVS(double, d, 40, 2), GENTFUNC_GEMVS(double, d, 40, 3), GENTFUNC_GEMVS(double, d, 40, 4), GENTFUNC_GEMVS(double, d, 40, 5), GENTFUNC_GEMVS(double, d, 40, 6), GENTFUNC_GEMVS(double, d, 40, 7) },
    { GENTFUNC_GEMVS(double, d,  8, 8), GENTFUNC_GEMVS(double, d,  8, 1), GENTFUNC_GEMVS(double, d,  8, 2), GENTFUNC_GEMVS(double, d,  8, 3), GENTFUNC_GEMVS(double, d,  8, 4), GENTFUNC_GEMVS(double, d,  8, 5), GENTFUNC_GEMVS(double, d,  8, 6), GENTFUNC_GEMVS(double, d,  8, 7) },
    { GENTFUNC_GEMVS(double, d, 16, 8), GENTFUNC_GEMVS(double, d, 16, 1), GENTFUNC_GEMVS(double, d, 16, 2), GENTFUNC_GEMVS(double, d, 16, 3), GENTFUNC_GEMVS(double, d, 16, 4), GENTFUNC_GEMVS(double, d, 16, 5), GENTFUNC_GEMVS(double, d, 16, 6), GENTFUNC_GEMVS(double, d, 16, 7) },
    { GENTFUNC_GEMVS(double, d, 24, 8), GENTFUNC_GEMVS(double, d, 24, 1), GENTFUNC_GEMVS(double, d, 24, 2), GENTFUNC_GEMVS(double, d, 24, 3), GENTFUNC_GEMVS(double, d, 24, 4), GENTFUNC_GEMVS(double, d, 24, 5), GENTFUNC_GEMVS(double, d, 24, 6), GENTFUNC_GEMVS(double, d, 24, 7) },
    { GENTFUNC_GEMVS(double, d, 32, 8), GENTFUNC_GEMVS(double, d, 32, 1), GENTFUNC_GEMVS(double, d, 32, 2), GENTFUNC_GEMVS(double, d, 32, 3), GENTFUNC_GEMVS(double, d, 32, 4), GENTFUNC_GEMVS(double, d, 32, 5), GENTFUNC_GEMVS(double, d, 32, 6), GENTFUNC_GEMVS(double, d, 32, 7) },
    { GENTFUNC_GEMVS(double, d,  7, 8), GENTFUNC_GEMVS(double, d,  7, 1), GENTFUNC_GEMVS(double, d,  7, 2), GENTFUNC_GEMVS(double, d,  7, 3), GENTFUNC_GEMVS(double, d,  7, 4), GENTFUNC_GEMVS(double, d,  7, 5), GENTFUNC_GEMVS(double, d,  7, 6), GENTFUNC_GEMVS(double, d,  7, 7) },
    { GENTFUNC_GEMVS(double, d, 15, 8), GENTFUNC_GEMVS(double, d, 15, 1), GENTFUNC_GEMVS(double, d, 15, 2), GENTFUNC_GEMVS(double, d, 15, 3), GENTFUNC_GEMVS(double, d, 15, 4), GENTFUNC_GEMVS(double, d, 15, 5), GENTFUNC_GEMVS(double, d, 15, 6), GENTFUNC_GEMVS(double, d, 15, 7) },
    { GENTFUNC_GEMVS(double, d, 23, 8), GENTFUNC_GEMVS(double, d, 23, 1), GENTFUNC_GEMVS(double, d, 23, 2), GENTFUNC_GEMVS(double, d, 23, 3), GENTFUNC_GEMVS(double, d, 23, 4), GENTFUNC_GEMVS(double, d, 23, 5), GENTFUNC_GEMVS(double, d, 23, 6), GENTFUNC_GEMVS(double, d, 23, 7) },
    { GENTFUNC_GEMVS(double, d, 31, 8), GENTFUNC_GEMVS(double, d, 31, 1), GENTFUNC_GEMVS(double, d, 31, 2), GENTFUNC_GEMVS(double, d, 31, 3), GENTFUNC_GEMVS(double, d, 31, 4), GENTFUNC_GEMVS(double, d, 31, 5), GENTFUNC_GEMVS(double, d, 31, 6), GENTFUNC_GEMVS(double, d, 31, 7) },
    { GENTFUNC_GEMVS(double, d, 39, 8), GENTFUNC_GEMVS(double, d, 39, 1), GENTFUNC_GEMVS(double, d, 39, 2), GENTFUNC_GEMVS(double, d, 39, 3), GENTFUNC_GEMVS(double, d, 39, 4), GENTFUNC_GEMVS(double, d, 39, 5), GENTFUNC_GEMVS(double, d, 39, 6), GENTFUNC_GEMVS(double, d, 39, 7) }
};

// static dgemv_ker dgemv_m_ker_fp[8][4] = 
// {
//     { GENTFUNC_GEMVS(double, d, 32, 4), GENTFUNC_GEMVS(double, d, 32, 1), GENTFUNC_GEMVS(double, d, 32, 2), GENTFUNC_GEMVS(double, d, 32, 3) },
//     { GENTFUNC_GEMVS(double, d,  8, 4), GENTFUNC_GEMVS(double, d,  8, 1), GENTFUNC_GEMVS(double, d,  8, 2), GENTFUNC_GEMVS(double, d,  8, 3) },
//     { GENTFUNC_GEMVS(double, d, 16, 4), GENTFUNC_GEMVS(double, d, 16, 1), GENTFUNC_GEMVS(double, d, 16, 2), GENTFUNC_GEMVS(double, d, 16, 3) },
//     { GENTFUNC_GEMVS(double, d, 24, 4), GENTFUNC_GEMVS(double, d, 24, 1), GENTFUNC_GEMVS(double, d, 24, 2), GENTFUNC_GEMVS(double, d, 24, 3) },
//     { GENTFUNC_GEMVS(double, d,  7, 4), GENTFUNC_GEMVS(double, d,  7, 1), GENTFUNC_GEMVS(double, d,  7, 2), GENTFUNC_GEMVS(double, d,  7, 3) },
//     { GENTFUNC_GEMVS(double, d, 15, 4), GENTFUNC_GEMVS(double, d, 15, 1), GENTFUNC_GEMVS(double, d, 15, 2), GENTFUNC_GEMVS(double, d, 15, 3) },
//     { GENTFUNC_GEMVS(double, d, 23, 4), GENTFUNC_GEMVS(double, d, 23, 1), GENTFUNC_GEMVS(double, d, 23, 2), GENTFUNC_GEMVS(double, d, 23, 3) },
//     { GENTFUNC_GEMVS(double, d, 31, 4), GENTFUNC_GEMVS(double, d, 31, 1), GENTFUNC_GEMVS(double, d, 31, 2), GENTFUNC_GEMVS(double, d, 31, 3) },
// };




#undef GENT_GEMV_CALLER

/*
 * Macro to generate a wrapper function for the GEMV M-kernels.
 *
 * This macro creates a function that acts as an interface to the M-kernels.
 * It intelligently partitions the matrix A into up to four regions and
 * calls the most appropriate micro-kernel for each region from a function
 * pointer table. This strategy handles any matrix dimensions efficiently.
 *
 * The generated function name follows the pattern:
 * bli_<ch>gemv_<direction>_zen4_int_<MR>x<NR>_st
 * Example: bli_dgemv_m_zen4_int_40x8_st
 *
 * Partitioning Strategy:
 *   The matrix is divided based on the largest multiples of MR and NR
 *   that fit within the matrix dimensions m and n.
 *
 *                             n
 *          <---------------------------------------------------->
 *          <------------- n/NR * NR ---------------><-- n%NR -->
 *        ^ +----------------------------------------+-----------+
 *        | |                                        |           |
 *   m/MR | |                Region 1                | Region 2  |
 *   * MR | |            (m >= MR, n >= NR)          | (m >= MR) |
 *        | |                                        |           |
 *        v +----------------------------------------+-----------+
 *   m%MR ^ |                Region 3                | Region 4  |
 *        | |              (n >= NR)                 | (corner)  |
 *        v +----------------------------------------+-----------+
 *
 *
 * Parameters:
 *   ctype     - C data type (e.g., double, float).
 *   ch        - Character identifier for the data type (e.g., 'd').
 *   MR        - The primary register block size for the M dimension.
 *   NR        - The primary register block size for the N dimension.
 *   direction - The kernel type, which is 'm' for M-kernels.
 *
 * Functionality:
 * 1. Performs input validation:
 *    - Checks if the matrix storage format is supported (row or column major).
 *    - Checks for no-transpose operation and unit increment for y.
 *    - Falls back to a reference implementation if checks fail.
 * 2. Handles the beta scaling (y = beta*y) if beta is not one.
 * 3. Calculates indices (m_idx, n_idx) to select the correct fringe kernel
 *    from the function pointer array.
 * 4. Calls the appropriate kernels for each of the four regions.
 */
#define GENT_GEMV_CALLER(ctype, ch, MR, NR, direction)                                       \
void PASTEMAC3(ch, gemv_, direction, PASTECH4(_zen4_int_,MR,x,NR,_st))                       \
     (                                                                                       \
       trans_t transa,                                                                       \
       conj_t  conjx,                                                                        \
       dim_t   m,                                                                            \
       dim_t   n,                                                                            \
       ctype* alpha,                                                                         \
       ctype* a, inc_t rs_a, inc_t cs_a,                                                     \
       ctype* x, inc_t incx,                                                                 \
       ctype* beta,                                                                          \
       ctype* y, inc_t incy,                                                                 \
       cntx_t* cntx                                                                          \
     )                                                                                       \
{                                                                                            \
     AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)                                            \
    if ( (( rs_a != 1 ) && ( cs_a != 1 )) || transa != BLIS_NO_TRANSPOSE || incy != 1)       \
    {                                                                                        \
        PASTEMAC(ch, gemv_zen_ref)                                                           \
        (                                                                                    \
          transa,                                                                            \
          m,                                                                                 \
          n,                                                                                 \
          alpha,                                                                             \
          a, rs_a, cs_a,                                                                     \
          x, incx,                                                                           \
          beta,                                                                              \
          y, incy,                                                                           \
          NULL                                                                               \
        );                                                                                   \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)                                          \
        return;                                                                              \
    }                                                                                        \
                                                                                             \
    const dim_t elem_per_reg = BLIS_SIMD_SIZE / sizeof(ctype);                               \
                                                                                             \
    bool is_m_left = ((m % MR) % elem_per_reg) >= 1 ? true : false;                          \
    dim_t m_idx = (m % MR) / elem_per_reg +                                                  \
                  (is_m_left * MR/elem_per_reg);                                             \
    dim_t n_idx = (n % NR);                                                                  \
    ctype one   = 1;                                                                         \
                                                                                             \
    if (*beta != 1)                                                                          \
    {                                                                                        \
        PASTEMAC(ch, scalv_zen4_int)                                                   \
        (                                                                                    \
          BLIS_NO_CONJUGATE,                                                                 \
          m,                                                                                 \
          beta,                                                                              \
          y, incy,                                                                           \
          cntx                                                                               \
        );                                                                                   \
    }                                                                                        \
                                                                                             \
    /* Region 1 */                                                                           \
    if(m >= MR && n >= NR)                                                                   \
    {                                                                                        \
        PASTECH3(ch, gemv_, direction, _ker_fp)[0][0]                                        \
        (                                                                                    \
            transa,                                                                          \
            conjx,                                                                           \
            ((dim_t)( m / MR )) * MR,                                                        \
            ((dim_t)( n / NR )) * NR,                                                        \
            alpha,                                                                           \
            a, rs_a, cs_a,                                                                   \
            x, incx,                                                                         \
            beta,                                                                            \
            y, incy,                                                                         \
            cntx                                                                             \
        );                                                                                   \
    }                                                                                        \
                                                                                             \
    /* Region 2 */                                                                           \
    if (m >= MR && n % NR)                                                                   \
    {                                                                                        \
        PASTECH3(ch, gemv_, direction, _ker_fp)[0][n_idx]                                    \
        (                                                                                    \
            transa,                                                                          \
            conjx,                                                                           \
            ((dim_t)( m / MR )) * MR,                                                        \
            n % NR,                                                                          \
            alpha,                                                                           \
            a + (((dim_t)( n / NR )) * NR * cs_a), rs_a, cs_a,                               \
            x + (((dim_t)( n / NR )) * NR * incx), incx,                                     \
            &one,                                                                            \
            y, incy,                                                                         \
            cntx                                                                             \
        );                                                                                   \
    }                                                                                        \
                                                                                             \
    /* Region 3 */                                                                           \
    if (n >= NR && m % MR)                                                                   \
    {                                                                                        \
        PASTECH3(ch, gemv_, direction, _ker_fp)[m_idx][0]                                    \
        (                                                                                    \
            transa,                                                                          \
            conjx,                                                                           \
            m % MR,                                                                          \
            ((dim_t)( n / NR )) * NR,                                                        \
            alpha,                                                                           \
            a + (((dim_t)( m / MR )) * MR * rs_a), rs_a, cs_a,                               \
            x, incx,                                                                         \
            beta,                                                                            \
            y + ((dim_t)( m / MR )) * MR * incy, incy,                                       \
            cntx                                                                             \
        );                                                                                   \
    }                                                                                        \
                                                                                             \
    /* Region 4 */                                                                           \
    if (m % MR && n % NR)                                                                    \
    {                                                                                        \
        PASTECH3(ch, gemv_, direction, _ker_fp)[m_idx][n_idx]                                \
        (                                                                                    \
            transa,                                                                          \
            conjx,                                                                           \
            m % MR,                                                                          \
            n % NR,                                                                          \
            alpha,                                                                           \
            a + (((dim_t)( m / MR )) * MR * rs_a) + ((dim_t)( n / NR )) * NR * cs_a, rs_a, cs_a, \
            x + (((dim_t)( n / NR )) * NR * incx), incx,                                         \
            &one,                                                                                \
            y + ((dim_t)( m / MR )) * MR * incy, incy,                                           \
            cntx                                                                                 \
        );                                                                                       \
    }                                                                                            \
}                                                                                                \
// End of GENT_GEMV_CALLER macro


/*
 * Generate the main single-threaded GEMV M-kernel interface function.
 * This will create the function `bli_dgemv_m_zen4_int_40x8_st`.
 */
GENT_GEMV_CALLER(double, d, 40, 8, m);

// #endregion ST M kernels

// #region MT M kernels


#define MR 40
#define NR 8

#ifdef BLIS_ENABLE_OPENMP

    /*
     * Multi-threaded GEMV M-kernel with division along both M and N dimensions
     *
     * This kernel implements a sophisticated parallelization strategy that divides
     * the matrix A into NTxNT blocks, where NT is the number of threads. Each
     * thread works on a different block in a checkerboard or diagonal pattern,
     * ensuring that no two threads access the same row or column simultaneously.
     * This approach maximizes parallelism but is currently not used due to
     * performance issues from false sharing, especially on multi-CCD systems.
     *
     * Parallelization Strategy:
     * - The matrix A is partitioned into a grid of NTxNT blocks.
     * - The N dimension is divided among threads using bli_thread_vector_partition.
     * - The M dimension is also divided into NT parts.
     * - The work is distributed in a checkerboard pattern. In each iteration,
     *   a thread `tid` processes the block at `(m_idx, n_idx)`, where `m_idx`
     *   is calculated to shift diagonally in each step: `m_idx = (tid + iteration) % NT`.
     * - This diagonal shifting ensures that for any given iteration, all threads
     *   are working on different rows and columns.
     * - An OpenMP barrier (`#pragma omp barrier`) is used after each iteration
     *   to synchronize all threads before they move to the next set of blocks.
     *
     * Visualizing the Parallelization (4 Threads, 4x4 Blocks):
     *
     * Iteration 1:
     *   T0 works on B(0,0)
     *   T1 works on B(1,1)
     *   T2 works on B(2,2)
     *   T3 works on B(3,3)
     *
     * Iteration 2:
     *   T0 works on B(1,0) -> shifts to B(0,1)
     *   T1 works on B(2,1) -> shifts to B(1,2)
     *   T2 works on B(3,2) -> shifts to B(2,3)
     *   T3 works on B(0,3) -> shifts to B(3,0)
     *
     * ...and so on.
     *
     * The diagram above illustrates this shifting work distribution. Each color
     * represents a thread, and you can see how they move to a new row and
     * column in each iteration.
     *
     * Performance Considerations:
     * - Theoretical maximum parallelism but suffers from false sharing.
     * - Cache line conflicts between threads accessing adjacent memory.
     * - Poor performance on multi-CCD systems due to NUMA effects.
     */
    void
    bli_dgemv_m_zen4_int_40x8_mt_Mdiv_Ndiv(
        trans_t transa,
        conj_t conjx,
        dim_t m,
        dim_t n,
        double *alpha,
        double *a, inc_t rs_a, inc_t cs_a,
        double *x, inc_t incx,
        double *beta,
        double *y, inc_t incy,
        cntx_t *cntx)
{
    dim_t nt   = 1;
    double one = 1;

    /* Determine optimal number of threads for this operation */
    bli_nthreads_l2
    (
        BLIS_GEMV_KER,
        BLIS_DOUBLE,   //PASTEMAC(d,type),
        BLIS_NO_TRANSPOSE,
        bli_arch_query_id(),
        m,
        n,
        &nt
    );

    if ( m <= nt || n <= nt)
    {
        bli_dgemv_m_zen4_int_40x8_mt_Mdiv
        (
            transa, conjx, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, cntx
        );
        return;
    }

    /* Handle beta scaling if beta != 1 */
    if (*beta != 1)
    {
        bli_dscalv_zen4_int
        (
          BLIS_NO_CONJUGATE,
          m,
          beta,
          y, incy,
          cntx
        );
    }
    double* beta_ = &one;

    /* Parallel region with work distribution across both dimensions */
    _Pragma("omp parallel num_threads(nt)")
    {
        /* Work distribution variables */
        dim_t job_per_thread = m;
        dim_t thread_start   = 0;

        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();

        /* Partition work along N dimension (columns) */
        bli_thread_vector_partition( n, nt_real, &thread_start, &job_per_thread, tid );

        /* Calculate block size for M dimension */
        dim_t m_part = (m / nt_real);
        if (m % nt_real) ++m_part;

        /* Calculate number of full iterations */
        dim_t num_itr_full = (nt_real  + (m - (m_part * nt_real)));

        /* Process matrix in blocks with checkerboard distribution */
        for(dim_t i = 0; i < m ; i += m_part)
        {
            /* Calculate which block this thread should process */
            dim_t m_idx = ((tid + (i/m_part)) % nt_real);
            dim_t m_curr = m_idx <= (num_itr_full - 1) ? m_part : m_part-1;
            dim_t m_offset = m_idx * (m_part-1);
            dim_t m_job = (m - i >= m_curr ? m_curr : m - i);

            /* Adjust offset for edge cases */
            if (num_itr_full >= m_idx+1)
            {
                m_offset += m_idx;
            }
            else
            {
                m_offset += num_itr_full;
            }

            /* Call single-threaded kernel for this thread's block */
            bli_dgemv_m_zen4_int_40x8_st
            (
                transa,
                conjx,
                m_job,
                job_per_thread,
                alpha,
                a + (thread_start * cs_a) + (m_offset * rs_a), rs_a, cs_a,
                x + (thread_start * incx), incx,
                beta_,
                y + m_offset * incy, incy,
                cntx
            );

            /* Synchronize all threads before next iteration */
            #pragma omp barrier
        }
    }
}

/*
 * Multi-threaded GEMV M-kernel with division along M dimension only
 *
 * This kernel divides the work along the M dimension (rows) only, which is
 * the preferred approach for most matrix sizes. Each thread processes a
 * contiguous block of rows, which provides good cache locality and minimizes
 * false sharing.
 *
 * Parallelization Strategy:
 * - Divide matrix A along rows into NT contiguous blocks
 * - Each thread processes a block of consecutive rows
 * - Simple and effective parallelization approach
 * - Good cache locality and minimal false sharing
 *
 * Performance Characteristics:
 * - Excellent cache performance due to contiguous memory access
 * - Minimal false sharing between threads
 * - Good scalability for tall matrices
 * - Preferred approach for most practical use cases
 *
 * Algorithm:
 * 1. Validate input parameters and fall back to reference if needed
 * 2. Calculate optimal number of threads
 * 3. Partition work along M dimension (rows)
 * 4. Each thread calls single-threaded kernel for its portion
 * 5. No synchronization needed as threads work on disjoint memory regions
 */
void bli_dgemv_m_zen4_int_40x8_mt_Mdiv
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       double* alpha,
       double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx,
       double* beta,
       double* y, inc_t incy,
       cntx_t* cntx
     )
{
    /* Input validation and fallback to reference implementation if needed */
    if ( (( rs_a != 1 ) && ( cs_a != 1 )) || transa != BLIS_NO_TRANSPOSE || incy != 1)
    {
        PASTEMAC(d, gemv_zen_ref)
        (
          transa,
          m,
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          y, incy,
          NULL
        );
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
        return;
    }

    dim_t nt = 1;
    /* Determine optimal number of threads */
    bli_nthreads_l2
    (
        BLIS_GEMV_KER,
        PASTEMAC(d,type),
        BLIS_NO_TRANSPOSE,
        bli_arch_query_id(),
        m,
        n,
        &nt
    );

    /* Parallel region with work distribution along M dimension */
    _Pragma("omp parallel num_threads(nt)")
    {
        /* Work distribution variables */
        dim_t job_per_thread = m;
        dim_t thread_start   = 0;

        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();

        /* Partition work along M dimension (rows) */
        bli_thread_vector_partition( m, nt_real, &thread_start, &job_per_thread, tid );

        /* Call single-threaded kernel for this thread's portion */
        bli_dgemv_m_zen4_int_40x8_st
        (
            transa,
            conjx,
            job_per_thread,
            n,
            alpha,
            a +  thread_start * rs_a, rs_a, cs_a,
            x , incx,
            beta,
            y + thread_start * incy, incy,
            cntx
        );
    }
}

/*
 * Multi-threaded GEMV M-kernel with division along N dimension
 *
 * This kernel divides the work along the N dimension (columns) and uses
 * temporary memory to store partial results from each thread. This approach
 * is used for very wide matrices or extremely large matrices where
 * N-division provides better load balancing.
 *
 * Parallelization Strategy:
 * - Divide matrix A along columns into NT blocks
 * - Each thread processes a subset of columns
 * - Use temporary memory to avoid false sharing
 * - Accumulate results from all threads at the end
 *
 * Memory Management:
 * - Allocates temporary memory for partial results
 * - Each thread (except thread 0) writes to its own temporary buffer
 * - Thread 0 writes directly to the output vector
 * - Results are accumulated at the end using vector addition
 *
 * Performance Considerations:
 * - Requires additional memory allocation (m * incy * nt doubles)
 * - Good for very wide matrices where N >> M
 * - May not be optimal for all matrix sizes due to memory overhead
 * - Fallback to M-division if memory allocation fails
 *
 * Algorithm:
 * 1. Validate input parameters and fall back to reference if needed
 * 2. Calculate optimal number of threads
 * 3. Allocate temporary memory for partial results
 * 4. Partition work along N dimension (columns)
 * 5. Each thread processes its columns and writes to temporary buffer
 * 6. Accumulate results from all threads using vector addition
 * 7. Release temporary memory
 */
void bli_dgemv_m_zen4_int_40x8_mt_Ndiv
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       double* alpha,
       double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx,
       double* beta,
       double* y, inc_t incy,
       cntx_t* cntx
     )
{
    /* Input validation and fallback to reference implementation if needed */
    if ( (( rs_a != 1 ) && ( cs_a != 1 )) || transa != BLIS_NO_TRANSPOSE || incy != 1)
    {
        PASTEMAC(d, gemv_zen_ref)
        (
          transa,
          m,
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          y, incy,
          NULL
        );
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
        return;
    }

    dim_t nt = 1;
    /* Determine optimal number of threads */
    bli_nthreads_l2
    (
        BLIS_GEMV_KER,
        PASTEMAC(d,type),
        BLIS_NO_TRANSPOSE,
        bli_arch_query_id(),
        m,
        n,
        &nt
    );

    /* Allocate temporary memory for partial results */
    rntm_t rntm;
    double* temp_mem;
    bli_rntm_init_from_global( &rntm );
    bli_rntm_set_num_threads_only( 1, &rntm );
    bli_pba_rntm_set_pba( &rntm );
    mem_t local_mem_buf = {0};

    // Total tmporary memory needed = NT (to store job_per_thread) +
    // m * incy * (nt-1)  ( memory of size Y for each thread except thread 0 )
    dim_t temp_mem_size = nt + (m * incy * (nt-1));
    bli_pba_acquire_m
    (
        &rntm,
        (temp_mem_size *sizeof(double)),
        BLIS_BITVAL_BUFFER_FOR_GEN_USE,
        &local_mem_buf
    );

    /* Fallback to reference implementation if memory allocation fails */
    if ( !bli_mem_is_alloc(&local_mem_buf) )
    {
        PASTEMAC(d, gemv_zen_ref)
        (
          transa,
          m,
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          y, incy,
          NULL
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
        return;
    }

    temp_mem = bli_mem_buffer(&local_mem_buf);
    if( temp_mem == NULL )
        nt = 1;
    if (local_mem_buf.size < temp_mem_size *sizeof(double))
    {
        nt = 1;
        if (bli_mem_is_alloc( &local_mem_buf ))
        {
            bli_pba_release(&rntm, &local_mem_buf);
        }
    }

    /* Fallback to M-division if single thread or insufficient memory */
    if (nt == 1)
    {
        bli_dgemv_m_zen4_int_40x8_mt_Mdiv
        (
            transa, conjx, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, cntx
        );
        return;
    }

    /* Initialize temporary memory to zero */
    double zero = 0;
    memset(temp_mem, 0, temp_mem_size * sizeof(double));

    /* Parallel region with work distribution along N dimension */
    _Pragma("omp parallel num_threads(nt)")
    {
        /* Work distribution variables */
        dim_t job_per_thread = m;
        dim_t thread_start   = 0;

        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();

        /* Partition work along N dimension (columns) */
        bli_thread_vector_partition( n, nt_real, &thread_start, &job_per_thread, tid );
        double* mem   = y;
        double* beta_ = beta;

        /* Use temporary memory for all threads except the first */
        if (tid != 0)
        {
            mem = temp_mem + nt + (m * incy * (tid-1));
            beta_ = &zero;
        }
        *(temp_mem + tid) = job_per_thread;

        /* Call single-threaded kernel for this thread's portion */
        bli_dgemv_m_zen4_int_40x8_st
        (
            transa,
            conjx,
            m,
            job_per_thread,
            alpha,
            a +  thread_start * cs_a, rs_a, cs_a,
            x + thread_start * incx, incx,
            beta_,
            mem, incy,
            cntx
        );
    }

    /* Accumulate results from all threads */
    for(dim_t i = 1; i < nt; ++i)
    {
        if ( *(temp_mem + i) == 0 ) continue;
        bli_daddv_zen4_int
        (
            BLIS_NO_CONJUGATE,
            m,
            temp_mem + nt + (m * incy * (i-1)), incy,
            y, incy,
            cntx
        );
    }

    /* Release temporary memory */
    if (bli_mem_is_alloc( &local_mem_buf ))
    {
        bli_pba_release(&rntm, &local_mem_buf);
    }
}
#endif

// #endregion MT M kernels

// Main routine which calls appropriate kernel based on the size of the matrix A
// y = alpha * A * x + beta * y
// where A is m x n matrix, x is n vector and y is m vector.
void bli_dgemv_n_zen4_int (
                            trans_t transa,
                            conj_t  conjx,
                            dim_t   m,
                            dim_t   n,
                            double* alpha,
                            double* a, inc_t rs_a, inc_t cs_a,
                            double* x, inc_t incx,
                            double* beta,
                            double* y, inc_t incy,
                            cntx_t* cntx
                        )
{
    void (*ker_ft) ( trans_t, 
                      conj_t, 
                      dim_t, 
                      dim_t, 
                      double*, 
                      double*,
                      inc_t, 
                      inc_t, 
                      double*, 
                      inc_t, 
                      double*, 
                      double*, 
                      inc_t, cntx_t* ) = NULL;

// If AOCL_DYNAMIC is enabled, call ST kernels for small sizes.
#if (defined(AOCL_DYNAMIC) || (defined(BLIS_ENABLE_OPENMP)))
    dim_t size = m * n;
#endif
#ifdef AOCL_DYNAMIC
    if ( size < 95000 )
    {
        // we call sequential GEMV
        if ( m <= 46 )
        {
            ker_ft = bli_dgemv_n_zen4_int_40x2_st;
        }
        else if ( n < 8 )
        {
            ker_ft = bli_dgemv_n_zen4_int_32x8_st;
        }
        else
        {
            ker_ft = bli_dgemv_m_zen4_int_40x8_st;
        }
    }
    else
#endif

    {
#ifdef BLIS_ENABLE_OPENMP
        if ( m < 1250 || size >= (700000 * 128))
        {
            ker_ft = bli_dgemv_m_zen4_int_40x8_mt_Ndiv;
        }
        else
        {
            ker_ft = bli_dgemv_m_zen4_int_40x8_mt_Mdiv;
        }
#else
        if ( m <= 46 )
        {
            ker_ft = bli_dgemv_n_zen4_int_40x2_st;
        }
        else if ( n < 8 )
        {
            ker_ft = bli_dgemv_n_zen4_int_32x8_st;
        }
        else
        {
            ker_ft = bli_dgemv_m_zen4_int_40x8_st;
        }
#endif
    }

    // Use 32x8 kernel when transa = "C" or "H"
    // and if incy != 1, which uses packing to handle non unit stride y
    if ( incy != 1 || transa != BLIS_NO_TRANSPOSE)
    {
        ker_ft = bli_dgemv_n_zen4_int_32x8_st;
    }
    ker_ft
    (
        transa,
        conjx,
        m,
        n,
        alpha,
        a, rs_a, cs_a,
        x, incx,
        beta,
        y, incy,
        cntx
    );
}// end of function bli_dgemv_n_zen4_int

