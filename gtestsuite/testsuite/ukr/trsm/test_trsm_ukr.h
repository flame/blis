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

#pragma once

#include "level3/trsm/trsm.h"
#include "blis.h"
#include "level3/ref_trsm.h"
#include "inc/check_error.h"
#include "common/testing_helpers.h"
#include <stdexcept>
#include <algorithm>
#include "level3/trsm/test_trsm.h"



template<typename T, typename FT>
static void test_trsm_ukr( FT ukr_fp, char storage, char uploa, char diaga,
                          gtint_t m, gtint_t n, gtint_t k, T alpha,
                          gtint_t ldc_inc, double thresh)
{
    gtint_t lda = m, ldb = n;
    gtint_t ldc = ldc_inc;

    // Allocate memory for A10(k*lda) and A11(m*lda)
    T* a10 = (T*)malloc( (k+m) * lda * sizeof(T) ); //col major
    // Allocate memory for A01(k*ldb) and B11(m*ldb)
    T* b01 = (T*)aligned_alloc(BLIS_HEAP_STRIDE_ALIGN_SIZE, (k+m) * ldb * sizeof(T)); //row major
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    init_mat( a10, uploa, 'c', 'n', 3, 10, m, (k+m), lda);
    init_mat( b01, uploa, 'r', 'n', 3, 10, n, (k+m), ldb);
    // Get A11(A10 + sizeof(A01)) and B11(B10 + sizeof(B10))
    T* a11  = a10 + (k*lda);
    T* b11  = b01 + (k*ldb);

    // make A11 triangular for trsm
    testinghelpers::make_triangular<T>( 'c', uploa, m, a11, lda );

    T* c, *c_ref;
    gtint_t rs_c, cs_c, rs_c_ref, cs_c_ref;
    gtint_t size_c, size_c_ref;

    // allocate memory for C according to the storage scheme
    if (storage == 'r' || storage == 'R')
    {
        ldc += n;
        rs_c = ldc, cs_c = 1;
        rs_c_ref = rs_c, cs_c_ref = cs_c;
        size_c = ldc * m * sizeof(T), size_c_ref = ldc * m * sizeof(T);
        c_ref = (T*)malloc( size_c_ref );
        c     = (T*)malloc( size_c );
    }
    else if (storage == 'c' || storage == 'C')
    {
        ldc += m;
        cs_c = ldc, rs_c = 1;
        rs_c_ref = rs_c, cs_c_ref = cs_c;
        size_c = ldc * n * sizeof(T), size_c_ref = ldc * n * sizeof(T);
        c_ref = (T*)malloc( size_c_ref );
        c     = (T*)malloc( size_c );
    }
    else
    {
        ldc += m;
        rs_c_ref = 1, cs_c_ref = ldc;
        rs_c = ldc, cs_c = ldc*ldc;
        size_c = ldc * n * ldc * sizeof(T), size_c_ref = ldc * n * 1   * sizeof(T);
        c_ref = (T*)malloc( size_c_ref );
        c     = (T*)malloc( size_c );
    }
    memset(c,     0, size_c);
    memset(c_ref, 0, size_c_ref);

    // copy contents of B11 to C and C_ref
    for (gtint_t i = 0; i < m; ++i)
    {
        for (gtint_t j = 0; j < n; ++j)
        {
            c[j*cs_c + i*rs_c] = b11[i*ldb + j];
            c_ref[j*cs_c_ref + i*rs_c_ref] = b11[i*ldb + j];
        }
    }

    // make A11 diagonal dominant
    for (gtint_t i =0;i< m; i++)
    {
        a11[i+i*lda] = T{float(m)}*a11[i+i*lda];
    }

    if (diaga == 'u' || diaga == 'U')
    {
        for (gtint_t i =0;i< m; i++)
        {
            a11[i+i*lda] = 1;
        }
    }

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    ukr_fp
    (
        k,
        &alpha,
        a10, a11,
        b01, b11,
        c,
        rs_c, cs_c,
        nullptr, nullptr
    );

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
    // compensate for the trsm per-inversion
    for (gtint_t i =0;i< m; i++)
    {
        a11[i+i*lda] = 1/a11[i+i*lda];
    }
#endif

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    if (storage == 'c' || storage == 'C')
    {
        testinghelpers::ref_gemm<T>( storage, 'n', 't', m, n, k, -1,
                                a10, lda, b01, ldb, alpha, c_ref, ldc);
        testinghelpers::ref_trsm<T>( storage, 'l', uploa, 'n', diaga, m, n, 1, a11,
                                lda, c_ref, ldc );
    }
    else if (storage == 'r' || storage == 'R')// row major
    {
        testinghelpers::ref_gemm<T>( storage, 't', 'n', m, n, k, -1,
                                a10, lda, b01, ldb, alpha, c_ref, ldc);

        // convert col major A11 to row Major for TRSM
        T temp = 0;
        for(gtint_t i = 0; i < m; ++i)
        {
            for(gtint_t j = i; j< m; ++j)
            {
                temp = a11[i+j*lda];
                a11[i+j*lda] = a11[j+i*lda];
                a11[j+i*lda] = temp;
            }
        }

        testinghelpers::ref_trsm<T>( storage, 'l', uploa, 'n', diaga, m, n, 1, a11,
                                lda, c_ref, ldc );
    }
    else
    {
        testinghelpers::ref_gemm<T>( 'c', 'n', 't', m, n, k, -1,
                                a10, lda, b01, ldb, alpha, c_ref, ldc);
        testinghelpers::ref_trsm<T>( 'c', 'l', uploa, 'n', diaga, m, n, 1, a11,
                                lda, c_ref, ldc );

        T* c_ref_gs = (T*)malloc( ldc * n * 1   * sizeof(T) );
        memset(c_ref_gs, 0, ldc * n * 1   * sizeof(T));


        for (gtint_t i = 0; i < m; ++i)
        {
            for (gtint_t j = 0; j < n; ++j)
            {
                c_ref_gs[i*rs_c_ref + j*cs_c_ref] = c[i*rs_c + j*cs_c];
            }
        }
        free(c);
        c = c_ref_gs;
    }

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, n, c, c_ref, ldc, thresh );

    free(a10);
    free(b01);
    free(c);
    free(c_ref);
}