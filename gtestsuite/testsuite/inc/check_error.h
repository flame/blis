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