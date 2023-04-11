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

#include "common/testing_basics.h"
#include "common/type_info.h"

namespace testinghelpers {

/**
 * Function that tests the compatibility of integer types.
 */
void int_compatibility(){
#if TEST_BLIS_TYPED
    static_assert(sizeof(gtint_t)==sizeof(dim_t),"Mismatch of integer types.");
#else
    static_assert(sizeof(gtint_t)==sizeof(f77_int),"Mismatch of integer types.");
#endif
}

void char_to_blis_trans( char trans, trans_t* blis_trans )
{
    if      ( trans == 'n' || trans == 'N' ) *blis_trans = BLIS_NO_TRANSPOSE;
    else if ( trans == 't' || trans == 'T' ) *blis_trans = BLIS_TRANSPOSE;
    else if ( trans == 'c' || trans == 'C' ) *blis_trans = BLIS_CONJ_TRANSPOSE;
    else if ( trans == 'h' || trans == 'H' )
    {
        throw std::invalid_argument("Error in file src/common/testing_basics.cpp in function char_to_blis_trans(): trans == 'h'. "
                    "To test BLIS-typed interface for this parameter be aware that this is not "
                    "a BLAS/CBLAS option. In BLAS/CBLAS interface 'c' is conjugate transpose (Hermitian), "
                    "while in BLIS_typed this would be 'h'. "
                    "To implement this option, please modify ref_*.cpp to use the correct matrix.");
    }
}

void char_to_blis_conj( char conj, conj_t* blis_conj )
{
    if      ( conj == 'n' || conj == 'N' ) *blis_conj = BLIS_NO_CONJUGATE;
    else if ( conj == 'c' || conj == 'C' ) *blis_conj = BLIS_CONJUGATE;
}

void char_to_blis_side( char side, side_t* blis_side )
{
    if      ( side == 'l' || side == 'L' ) *blis_side = BLIS_LEFT;
    else if ( side == 'r' || side == 'R' ) *blis_side = BLIS_RIGHT;
}

void char_to_blis_uplo( char uplo, uplo_t* blis_uplo )
{
    if      ( uplo == 'l' || uplo == 'L' ) *blis_uplo = BLIS_LOWER;
    else if ( uplo == 'u' || uplo == 'U' ) *blis_uplo = BLIS_UPPER;
}

void char_to_blis_diag( char diag, diag_t* blis_diag )
{
    if      ( diag == 'n' || diag == 'N' ) *blis_diag = BLIS_NONUNIT_DIAG;
    else if ( diag == 'u' || diag == 'U' ) *blis_diag = BLIS_UNIT_DIAG;
}

void char_to_cblas_order( char order, CBLAS_ORDER *cblas_order )
{
    if      ( order == 'c' || order == 'C' ) *cblas_order = CblasColMajor;
    else if ( order == 'r' || order == 'R' ) *cblas_order = CblasRowMajor;

}

void char_to_cblas_trans( char trans, CBLAS_TRANSPOSE *cblas_trans )
{
    if      ( trans == 'n' || trans == 'N' ) *cblas_trans = CBLAS_TRANSPOSE::CblasNoTrans;
    else if ( trans == 't' || trans == 'T' ) *cblas_trans = CBLAS_TRANSPOSE::CblasTrans;
    else if ( trans == 'c' || trans == 'C' ) *cblas_trans = CBLAS_TRANSPOSE::CblasConjTrans;
}

void char_to_cblas_uplo( char uplo, CBLAS_UPLO *cblas_uplo )
{
    if      ( uplo == 'l' || uplo == 'L' ) *cblas_uplo = CblasLower;
    else if ( uplo == 'u' || uplo == 'U' ) *cblas_uplo = CblasUpper;
}

void char_to_cblas_diag( char diag, CBLAS_DIAG *cblas_diag )
{
    if      ( diag == 'n' || diag == 'N' ) *cblas_diag = CblasNonUnit;
    else if ( diag == 'u' || diag == 'U' ) *cblas_diag = CblasUnit;
}

void char_to_cblas_side( char side, CBLAS_SIDE *cblas_side )
{
    if      ( side == 'l' || side == 'L' ) *cblas_side = CblasLeft;
    else if ( side == 'r' || side == 'R' ) *cblas_side = CblasRight;
}

/**
 * @brief Returns the size of a buffer which has strides.
 *
 * @param n length of vector
 * @param incx increment
 * @return gtint_t dimension of the buffer that stored a vector with length n and increment incx
 */
gtint_t buff_dim( gtint_t n, gtint_t incx ) {
    return (n*std::abs(incx) - (std::abs(incx)-1));
}

gtint_t matsize( char storage, char trans, gtint_t m, gtint_t n, gtint_t ldm )
{
    gtint_t km;
    if( (storage == 'c') || (storage == 'C') ) {
        /*Column_Major*/
        km  = chktrans( trans ) ? m : n ;
    }
    else {
        /*Row_Major*/
        km  = chktrans( trans ) ? n : m ;
    }
    return (km*ldm);
}

/**
 * Returns the leading dimension of a matrix depending on the storage type,
 * whether it is transpose or not, and the size of rows and columns.
 *
 * @param storage specifies the storage format of matrix in memory.
 * @param trns    specifies the form of given matrix.
 * @param m       specifies the number of rows of given matrix.
 * @param n       specifies the number of columns of given matrix.
 * @param inc     specifies the increment of the leading dimension.
*/
gtint_t get_leading_dimension( char storage, char trans, gtint_t m, gtint_t n, gtint_t inc )
{
    gtint_t lda;
    if( (storage == 'c') || (storage == 'C') ) //column-major order
    {
        if ((trans == 'n')||(trans == 'N'))
            lda = std::max(gtint_t(1),m) + inc;
        else
            lda = std::max(gtint_t(1),n) + inc;
    }
    else //row-major order
    {
        if ((trans == 'n')||(trans == 'N'))
            lda = std::max(gtint_t(1),n) + inc;
        else
            lda = std::max(gtint_t(1),m) + inc;
    }
    return lda;
}

/**
 * If T is real, returns NaN.
 * If T is complex, returns {NaN, 0.0}
*/
template<typename T>
T getNaN()
{
    using RT = typename testinghelpers::type_info<T>::real_type;
    if constexpr (testinghelpers::type_info<T>::is_real)
        return std::numeric_limits<RT>::quiet_NaN();
    else
        return T{std::numeric_limits<RT>::quiet_NaN(), 0};
}
template float getNaN<float>();
template double getNaN<double>();
template scomplex getNaN<scomplex>();
template dcomplex getNaN<dcomplex>();

/**
 * If T is real, returns inf.
 * If T is complex, returns {inf, 0.0}
*/
template<typename T>
T getInf()
{
    using RT = typename testinghelpers::type_info<T>::real_type;
    if constexpr (testinghelpers::type_info<T>::is_real)
        return std::numeric_limits<RT>::infinity();
    else
        return T{std::numeric_limits<RT>::infinity(), 0};
}
template float getInf<float>();
template double getInf<double>();
template scomplex getInf<scomplex>();
template dcomplex getInf<dcomplex>();



bool chktrans( char trns )
{
    return (!(trns=='n'));
}

bool chknotrans( char trns )
{
    trans_t trans;
    char_to_blis_trans( trns, &trans );
    return ( bool )
	       ( ( trans & BLIS_TRANS_BIT ) == BLIS_BITVAL_NO_TRANS );
}

bool chkconjtrans( char trns )
{
    trans_t trans;
    char_to_blis_trans( trns, &trans );
    return ( bool )
	       ( ( ( trans & BLIS_CONJ_BIT ) & ( trans & BLIS_TRANS_BIT ) ) == BLIS_BITVAL_CONJ_TRANS );
}

bool chktransconj( char trns )
{
    trans_t trans;
    char_to_blis_trans( trns, &trans );
    return ( bool )
        ( ( trans & BLIS_CONJ_BIT ) == BLIS_BITVAL_CONJ );
}

bool chkconj( char conjx )
{
    conj_t conj;
    char_to_blis_conj( conjx, &conj );
    return ( bool )
        ( ( conj & BLIS_CONJ_BIT ) == BLIS_BITVAL_CONJ );
}

bool chkupper( char uplo )
{
    uplo_t uploa;
    char_to_blis_uplo( uplo, &uploa );
    return ( bool ) ( uploa == BLIS_UPPER );
}

bool chklower( char uplo )
{
    uplo_t uploa;
    char_to_blis_uplo( uplo, &uploa );
    return ( bool ) ( uploa == BLIS_LOWER );
}

bool chkunitdiag( char diag )
{
    diag_t diaga;
    char_to_blis_diag( diag, &diaga );
    return ( bool ) ( diaga == BLIS_BITVAL_UNIT_DIAG );
}

bool chknonunitdiag( char diag )
{
    diag_t diaga;
    char_to_blis_diag( diag, &diaga );
    return ( bool ) ( diaga == BLIS_BITVAL_NONUNIT_DIAG );
}

bool chksideleft( char mside )
{
    side_t  side;
    char_to_blis_side( mside, &side );
    return ( bool ) ( side == BLIS_LEFT );
}

bool chksideright( char mside )
{
    side_t  side;
    char_to_blis_side( mside, &side );
    return ( bool ) ( side == BLIS_RIGHT );
}

void swap_dims_with_trans( char trans,
                           gtint_t  m,  gtint_t  n,  gtint_t  rs,  gtint_t  cs,
                           gtint_t* mt, gtint_t* nt, gtint_t* rst, gtint_t* cst )
{
    if ( chktrans( trans ) ) { *mt = n; *nt = m; *rst = cs; *cst = rs; }
    else                     { *mt = m; *nt = n; *rst = rs; *cst = cs; }
}

void swap_strides_with_trans( char trans,
                              gtint_t  rs,  gtint_t  cs,
                              gtint_t* rst, gtint_t* cst )
{
    if ( chktrans( trans ) ) {*rst = cs; *cst = rs; }
    else                     {*rst = rs; *cst = cs; }
}

void swap_dims( gtint_t* x, gtint_t* y )
{
    gtint_t temp = *x;
    *x = *y;
    *y = temp;
}

void set_dims( char trans, gtint_t m, gtint_t n, gtint_t* mt, gtint_t* nt )
{
   if ( chktrans( trans ) ) { *mt = n; *nt = m; }
   else                     { *mt = m; *nt = n; }
}

void set_dim_with_side( char side, gtint_t m, gtint_t n, gtint_t* dim )
{
    if ( chksideleft( side ) ) *dim = m;
    else                       *dim = n;
}

} //end of namespace testinghelpers
