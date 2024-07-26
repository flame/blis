/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdio.h>
#include <iostream>
#include <sstream>
#include "cblas.h"
#include "common/type_info.h"

namespace testinghelpers {

void char_to_blis_trans( char trans, trans_t* blis_trans );
void char_to_blis_conj( char conj, conj_t* blis_conj );
void char_to_blis_side( char side, side_t* blis_side );
void char_to_blis_uplo( char uplo, uplo_t* blis_uplo );
void char_to_blis_diag( char diag, diag_t* blis_diag );

void char_to_cblas_order( char order, CBLAS_ORDER* cblas_order );
void char_to_cblas_trans( char trans, CBLAS_TRANSPOSE* cblas_trans );
void char_to_cblas_uplo( char uplo, CBLAS_UPLO* cblas_uplo );
void char_to_cblas_diag( char diag, CBLAS_DIAG* cblas_diag );
void char_to_cblas_side( char side, CBLAS_SIDE* cblas_side );

/**
 * @brief Returns the size of a buffer which has strides.
 *
 * @param n length of vector
 * @param incx increment
 * @return gtint_t dimension of the buffer that stored a vector with length n and increment incx
 */
gtint_t buff_dim(gtint_t n, gtint_t incx);

/**
 * @brief Returns the size of matrix.
 *
 * @param storage specifies the storage format of matrix in memory.
 * @param trns    specifies the form of given matrix.
 * @param m       specifies the number of rows of given matrix.
 * @param n       specifies the number of columns of given matrix.
 * @param ldm     specifies the leading dimension of given matrix.
  * @return gtint_t  Size of the matrix for dimension (m,n) and strides(rs,cs).
 */
gtint_t matsize(char storage, char trans, gtint_t m, gtint_t n, gtint_t ldm );

/**
 * Returns the leading dimension of a matrix depending on the storage type,
 * whether it is transpose or not, and the size of rows and columns, and the stride.
 *
 * @param storage specifies the storage format of matrix in memory.
 * @param trns    specifies the form of given matrix.
 * @param m       specifies the number of rows of given matrix.
 * @param n       specifies the number of columns of given matrix.
 * @param inc     specifies the increment of the leading dimension.
 * @param stride  specifies the stride between two "continuous" elements in the matrix.
*/
gtint_t get_leading_dimension( char storage, char trans, gtint_t m, gtint_t n, gtint_t inc, gtint_t stride = 1 );

/**
 * If T is real, returns NaN.
 * If T is complex, returns {NaN, 0.0}
*/
template<typename T>
T getNaN();

/**
 * If T is real, returns NaN.
 * If T is complex, returns {NaN, NaN}
*/
template<typename T>
T getNaNNaN();

/**
 * If T is real, returns inf.
 * If T is complex, returns {inf, 0.0}
*/
template<typename T>
T getInf();

/**
 * If T is real, returns inf.
 * If T is complex, returns {inf, inf}
*/
template<typename T>
T getInfInf();

/**
 * If T is real, returns extval.
 * If T is complex, returns {extval, extval}
 * where extval = NaN or Inf
*/
template<typename T>
T aocl_extreme();

/**
 * @brief Returns the conjugate of a scalar x.
 *
 * @tparam T float, double, scomplex, dcomplex
 * @param x scalar of type T
 * @return T conjugate of x
 */
template<typename T>
static T conj(T &x){
    if constexpr (testinghelpers::type_info<T>::is_real)
        return x;
    else
        return {x.real, -x.imag};
}

template <typename T>
void conj(T* x, gtint_t len, gtint_t inx)
{
    gtint_t i, ix;
    ix = 0;
    for( i = 0 ; i < len ; i++ )
    {
      x[ix] = conj<T>(x[ix]);
      ix = ix + inx;
    }
    return;
}

template <typename T>
void conj(char storage, T* X, gtint_t m, gtint_t n, gtint_t ldm )
{
    gtint_t i,j;
    gtint_t rs, cs;
    rs=cs=1;
    if( (storage == 'c') || (storage == 'C') )
        cs = ldm ;
    else
        rs = ldm ;

    for( i = 0 ; i < m ; i++ )
    {
        for( j = 0 ; j < n ; j++ )
        {
            X[i*rs + j*cs] = conj<T>( X[i*rs + j*cs] );
        }
    }
    return;
}

template<typename T>
static void initone(T &x) {
    if constexpr (testinghelpers::type_info<T>::is_real)
        x = 1.0;
    else
        x = {1.0, 0.0};
}

template<typename T>
static void initzero(T &x) {
    if constexpr (testinghelpers::type_info<T>::is_real)
        x = 0.0;
    else
        x = {0.0, 0.0};
}

template<typename T>
static void alphax( gtint_t n, T alpha, T *xp, gtint_t incx )
{
    gtint_t i = 0;
    gtint_t ix = 0;
    for(i = 0 ; i < n ; i++) {
        xp[ix] = (alpha * xp[ix]);
        // use absolute value of incx to ensure
        // correctness when incx < 0
        ix = ix + std::abs(incx);
    }
}

template<typename T>
static T ONE() {
    if constexpr (testinghelpers::type_info<T>::is_real)
        return 1.0;
    else
        return {1.0, 0.0};
}

template<typename T>
static T ZERO() {
    if constexpr (testinghelpers::type_info<T>::is_real)
        return 0.0;
    else
        return {0.0, 0.0};
}

/**
 * @brief Returns the boolean form of a trans value.
 *
 * @param trans specifies the form of matrix stored in memory.
 * @return boolean of the transform of the matrix.
 */
bool chktrans( char trans );
bool chknotrans( char trans );
bool chkconjtrans( char trans );
bool chktransconj( char trans );
bool chkconj( char trans );


/**
 * @brief Returns the boolean form of a matrix triangular form.
 *
 * @param uplo specifies whether matrix is upper or lower triangular stored in memory.
 * @return boolean of the triangular form of the matrix.
 */
bool is_upper_triangular( char uplo );
bool is_lower_triangular( char uplo );

/**
 * @brief Returns the boolean form of a matrix unit/non-unit diagonal form.
 *
 * @param diag specifies whether matrix is unit or non-unit diagonal form.
 * @return boolean of the diagonal form of the matrix.
 */
bool chkunitdiag( char diag );
bool chknonunitdiag( char diag );

/**
 * @brief Returns the boolean form of a matrix left/right side.
 *
 * @param side specifies whether matrix is left or right side form.
 * @return boolean of the side of the matrix.
 */
bool chksideleft( char side );
bool chksideright( char side );

/**
 * @brief swap the dimensions and strides of the matrix based on trans
 *
 * @param trans specifies the form of matrix stored in memory.
 * @param m       specifies the number of rows of given matrix.
 * @param n       specifies the number of columns of given matrix.
 * @param rs      specifies the row stride of given matrix.
 * @param cs      specifies the column stride of given matrix.
 * @param mt      pointer to the row number of given matrix.
 * @param nt      pointer to the column number of given matrix.
 * @param rst     pointer to the row stride of given matrix.
 * @param cst     pointer to the column stride of given matrix.
 */
void swap_dims_with_trans( char trans,
                           gtint_t  m,  gtint_t  n,  gtint_t  rs,  gtint_t  cs,
                           gtint_t* mt, gtint_t* nt, gtint_t* rst, gtint_t* cst );
/**
 * @brief swap the strides of the matrix based on trans
 *
 * @param trans specifies the form of matrix stored in memory.
 * @param rs      specifies the row stride of given matrix.
 * @param cs      specifies the column stride of given matrix.
 * @param rst     pointer to the row stride of given matrix.
 * @param cst     pointer to the column stride of given matrix.
 */
void swap_strides_with_trans( char trans,
                                     gtint_t  rs,  gtint_t  cs,
                                     gtint_t* rst, gtint_t* cst );

/**
 * @brief swap the dimensions
 *
 * @param trans specifies the form of matrix stored in memory.
 * @param x     pointer to the dimension of given vector/matrix.
 * @param y     pointer to the dimension of given vector/matrix.
 */
void swap_dims( gtint_t* x, gtint_t* y );

/**
 * @brief set the dimension of the matrix based on trans
 *
 * @param trans specifies the form of matrix stored in memory.
 * @param m       specifies the number of rows of given matrix.
 * @param n       specifies the number of columns of given matrix.
 * @param mt      pointer to the row number of given matrix.
 * @param nt      pointer to the column number of given matrix.
 */
void set_dims( char trans, gtint_t m, gtint_t n, gtint_t* mt, gtint_t* nt );

/**
 * @brief set the dimension of the matrix based on side
 *
 * @param side    specifies the side of matrix selected in memory.
 * @param m       specifies the number of rows of given matrix.
 * @param n       specifies the number of columns of given matrix.
 * @param dim     pointer to the dimension based on side.
 */
void set_dim_with_side( char side, gtint_t m, gtint_t n, gtint_t* dim );

/**
 * ==========================================================================
 * MKHERM
 * Make an n x n matrix A explicitly Hermitian by copying the conjugate
 * of the triangle specified by uploa to the opposite triangle. Imaginary
 * components of diagonal elements are explicitly set to zero.
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 * @param[in] storage specifies the storage format of matrix in memory.
 * @param[in] uplo    specifies upper or lower triangular part of A is used.
 * @param[in] n       specifies the number of rows & columns of square matrix.
 * @param[in] a       specifies pointer which points to the first element of a.
 * @param[in] ld      specifies leading dimension for a given matrix.
 */
template<typename T>
void make_herm( char storage, char uplo, gtint_t n, T* a, gtint_t ld );

/**
 * ==========================================================================
 * MKSYMM
 * Make an n x n matrix A explicitly symmetric by copying the triangle
 * specified by uploa to the opposite triangle.
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 * @param[in] storage specifies the storage format of matrix in memory.
 * @param[in] uplo    specifies upper or lower triangular part of A is used.
 * @param[in] n       specifies the number of rows & columns of square matrix.
 * @param[in] a       specifies pointer which points to the first element of a.
 * @param[in] ld      specifies leading dimension for a given matrix.
 */
template<typename T>
void make_symm( char storage, char uplo, gtint_t n, T* a, gtint_t ld );

/**
 * ==========================================================================
 * MKTRIM
 * Make an n x n matrix A explicitly triangular by preserving the triangle
 * specified by uploa and zeroing the elements in the opposite triangle.
 * It is assumed that the diagonal offset of A is zero
 * ==========================================================================
 * @param[in] storage specifies the storage format of matrix in memory.
 * @param[in] uplo    specifies upper or lower triangular part of A is used.
 * @param[in] n       specifies the number of rows & columns of square matrix.
 * @param[in] a       specifies pointer which points to the first element of a.
 * @param[in] ld      specifies leading dimension for a given matrix.
 */
template<typename T>
void make_triangular( char storage, char uplo, gtint_t n, T* a, gtint_t ld );

/**
 * ==========================================================================
 * MKDIAG
 * Make an m x n matrix A, which adds a scalar value to
 * every element along an arbitrary diagonal of a matrix.
 * It is assumed that the diagonal offset of A is zero
 * ==========================================================================
 * @param[in] storage specifies the storage format of matrix in memory.
 * @param[in] m       specifies the number of rows of a given matrix.
 * @param[in] n       specifies the number of columns of a given matrix.
 * @param[in] alpha   specifies the value to set diagonal elements.
 * @param[in] a       specifies pointer which points to the first element of a.
 * @param[in] ld      specifies leading dimension for a given matrix.
 */
template<typename T>
void make_diag( char storage, gtint_t m, gtint_t n, T alpha, T *a, gtint_t ld );

/**
 * print vector of length  n
 * @param[in] n    specifies the length of the given vector.
 * @param[in] a    specifies pointer which points to the first element of a.
 * @param[in] incx specifies storage spacing between elements of a.
 */
template<typename T>
void print_vector( gtint_t n, T *x, gtint_t incx);

/**
 * print matrix of size m x n
 * @param[in] storage specifies the storage format of matrix in memory.
 * @param[in] m       specifies the number of rows of given matrix.
 * @param[in] n       specifies the number of columns of given matrix.
 * @param[in] a       specifies pointer which points to the first element of a.
 * @param[in] ld      specifies leading dimension for a given matrix.
 */
template<typename T>
void print_matrix( char storage, gtint_t m, gtint_t n, T *a, gtint_t ld);

/**
 * @brief returns a string with the correct NaN/Inf for printing
 *
 * @tparam T gtint_t, float, double, scomplex, dcomplex.
 * @param exval exception value for setting the string.
 */
template<typename T>
std::string get_value_string( T exval );

} //end of namespace testinghelpers
