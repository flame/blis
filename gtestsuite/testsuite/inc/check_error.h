/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "blis.h"
#include <gtest/gtest.h>
#include "common/testing_helpers.h"

template <typename T>
void computediff( T blis_sol, T ref )
{
    ASSERT_EQ(blis_sol, ref) << "ref = " << ref << "    blis_sol = " << blis_sol;
}

template <typename T>
void computediff( T blis_sol, T ref, double thresh )
{
    ASSERT_TRUE(testinghelpers::getError(ref, blis_sol) < thresh)
                                        << "ref="<< ref <<"   blis_sol=" <<blis_sol 
                                        << "   Err=" << testinghelpers::getError(ref, blis_sol)
                                        <<"   thresh=" << thresh;
}

template <typename T>
void computediff( gtint_t n, T *blis_sol, T *ref, gtint_t incy )
{
    gtint_t idx;
    gtint_t i;
    for( idx = 0 ; idx < n ; idx++ )
    {
        i = (incy > 0) ? (idx * incy) : ( - ( n - idx - 1 ) * incy );
        ASSERT_EQ(ref[i], blis_sol[i]) << "blis_sol[" << i << "]="<< blis_sol[i] <<"   ref[" << i << "]=" << ref[i];
    }
}

template <typename T>
void computediff( gtint_t n, T *blis_sol, T *ref, gtint_t incy, double thresh )
{
    gtint_t idx;
    gtint_t i;
    for( idx = 0 ; idx < n ; idx++ )
    {
        i = (incy > 0) ? (idx * incy) : ( - ( n - idx - 1 ) * incy );
        ASSERT_TRUE(testinghelpers::getError(ref[i], blis_sol[i]) < thresh)
                    << "blis_sol[" << i << "]="<< blis_sol[i]
                    <<"   ref[" << i << "]=" << ref[i]
                    <<"   Err=" << testinghelpers::getError(ref[i], blis_sol[i])
                    <<"   thresh=" << thresh;
    }
}

template <typename T>
void computediff(char storage, gtint_t m, gtint_t n, T *blis_sol, T *ref, gtint_t ld, double thresh )
{
    gtint_t i,j;
    gtint_t rs,cs;
    rs=cs=1;
    if( (storage == 'c') || (storage == 'C') )
        cs = ld ;
    else
        rs = ld ;

    for( i = 0 ; i < m ; i++ )
    {
        for( j = 0 ; j < n ; j++ )
        {
            gtint_t idx = (i*rs + j*cs);
            auto av = blis_sol[ idx ];
            auto xv = ref[ idx ];
            ASSERT_TRUE(testinghelpers::getError(av, xv) < thresh)
                            << "blis_sol[" << i <<","<< j << "]="<< av
                            <<"   ref[" << i <<","<< j << "]=" << xv
                            << "   relErr=" << testinghelpers::getError(av, xv)
                            <<"   thresh=" << thresh;
        }
    }
}