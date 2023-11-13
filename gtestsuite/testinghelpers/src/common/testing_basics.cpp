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
#include "common/complex_helpers.h"

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
            lda = (std::max)(gtint_t(1),m) + inc;
        else
            lda = (std::max)(gtint_t(1),n) + inc;
    }
    else //row-major order
    {
        if ((trans == 'n')||(trans == 'N'))
            lda = (std::max)(gtint_t(1),n) + inc;
        else
            lda = (std::max)(gtint_t(1),m) + inc;
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

bool is_upper_triangular( char uplo )
{
    uplo_t uploa;
    char_to_blis_uplo( uplo, &uploa );
    return ( bool ) ( uploa == BLIS_UPPER );
}

bool is_lower_triangular( char uplo )
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

template<typename T>
static void set_imag_zero(T &x){
        x = {x.real, 0.0};
}

/**
 * ==========================================================================
 * MKHERM
 * Make an n x n matrix A explicitly Hermitian by copying the conjugate
 * of the triangle specified by uploa to the opposite triangle. Imaginary
 * components of diagonal elements are explicitly set to zero.
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 */
template<typename T>
void make_herm( char storage, char uplo, gtint_t n, T* a, gtint_t ld )
{
    gtint_t rs,cs;
    rs=cs=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        cs = ld ;
    else
        rs = ld ;

    bool uploa = testinghelpers::is_upper_triangular( uplo );

    if( uploa ) {
        gtint_t i, j;
        for ( j = 0; j < ( n-1) ; j++ )
        {
            for ( i = (j+1) ; i < n ; i++ )
            {
                a[i*rs + j*cs] = testinghelpers::conj<T>(a[i*cs + j*rs]);
            }
        }
    }
    else
    {
        gtint_t i, j;
        for ( j = 1; j <  n ; j++ )
        {
            for ( i = 0 ; i < j  ; i++ )
            {
                a[i*rs + j*cs] = testinghelpers::conj<T>(a[i*cs + j*rs]);
            }
        }
    }
    if constexpr (testinghelpers::type_info<T>::is_complex) {
        gtint_t i;
        for ( i = 0; i < n ; i++ )
        {
            set_imag_zero<T>(a[i*rs + i*cs]);
        }
    }
}
template void make_herm<float>( char, char, gtint_t, float *, gtint_t );
template void make_herm<double>( char, char, gtint_t, double *, gtint_t );
template void make_herm<scomplex>( char, char, gtint_t, scomplex *, gtint_t );
template void make_herm<dcomplex>( char, char, gtint_t, dcomplex *, gtint_t );

/**
 * ==========================================================================
 * MKSYMM
 * Make an n x n matrix A explicitly symmetric by copying the triangle
 * specified by uploa to the opposite triangle.
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 */
template<typename T>
void make_symm( char storage, char uplo, gtint_t n, T* a, gtint_t ld )
{
    gtint_t rs,cs;
    rs=cs=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        cs = ld ;
    else
        rs = ld ;

    bool uploa = testinghelpers::is_upper_triangular( uplo );

   /* Toggle uplo so that it refers to the unstored triangle. */
    if( uploa ) {
        gtint_t i, j;
        for ( j = 0; j < ( n-1) ; j++ )
        {
            for ( i = (j+1) ; i < n ; i++ )
            {
                a[i*rs + j*cs] = a[i*cs + j*rs];
            }
        }
    }
    else
    {
        gtint_t i, j;
        for ( j = 1; j <  n ; j++ )
        {
            for ( i = 0 ; i < j  ; i++ )
            {
                a[i*rs + j*cs] = a[i*cs + j*rs];
            }
        }
    }
}
template void make_symm<float>( char, char, gtint_t, float *, gtint_t );
template void make_symm<double>( char, char, gtint_t, double *, gtint_t );
template void make_symm<scomplex>( char, char, gtint_t, scomplex *, gtint_t );
template void make_symm<dcomplex>( char, char, gtint_t, dcomplex *, gtint_t );

/**
 * ==========================================================================
 * MKTRIM
 * Make an n x n matrix A explicitly triangular by preserving the triangle
 * specified by uploa and zeroing the elements in the opposite triangle.
 * It is assumed that the diagonal offset of A is zero
 * ==========================================================================
 */
template<typename T>
void make_triangular( char storage, char uplo, gtint_t n, T* a, gtint_t ld )
{
    gtint_t rs,cs;
    rs=cs=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        cs = ld ;
    else
        rs = ld ;

    if ( n < 0 )
        return;

    bool uploa = testinghelpers::is_upper_triangular( uplo );
    T zero;
    testinghelpers::initzero<T>(zero);

   /* Toggle uplo so that it refers to the unstored triangle. */
    if( !uploa ) {
        gtint_t i, j;
        for ( j = 1; j <  n ; j++ )
        {
            for ( i = 0 ; i < j  ; i++ )
            {
                a[i*rs + j*cs] = zero;
            }
        }
    }
    else
    {
        gtint_t i, j;
        for ( j = 0; j < ( n-1) ; j++ )
        {
            for ( i = (j+1) ; i < n ; i++ )
            {
                a[i*rs + j*cs] = zero;
            }
        }
    }
}
template void make_triangular<float>( char, char, gtint_t, float *, gtint_t );
template void make_triangular<double>( char, char, gtint_t, double *, gtint_t );
template void make_triangular<scomplex>( char, char, gtint_t, scomplex *, gtint_t );
template void make_triangular<dcomplex>( char, char, gtint_t, dcomplex *, gtint_t );

/**
 * ==========================================================================
 * MKDIAG
 * Make an m x n matrix A, which adds a scalar value to
 * every element along an arbitrary diagonal of a matrix.
 * It is assumed that the diagonal offset of A is zero
 * ==========================================================================
 */
template<typename T>
void make_diag( char storage, gtint_t m, gtint_t n, T alpha, T *a, gtint_t ld )
{
    gtint_t rs,cs;
    rs=cs=1;

    if( (storage == 'c') || (storage == 'C') )
        cs = ld ;
    else
        rs = ld ;

    /* a = mn x mn   */
    gtint_t mn   = (std::min)( n , m );

    gtint_t i;
    gtint_t inca = rs + cs ;
    T *ap        = a;
    gtint_t ia   = 0;
    for ( i = 0; i < mn; i++ )
    {
        ap[ia] = (alpha + ap[ia]);
        ia = ia + inca;
    }
}
template void make_diag<float>( char, gtint_t, gtint_t, float, float *, gtint_t );
template void make_diag<double>( char, gtint_t, gtint_t, double, double *, gtint_t );
template void make_diag<scomplex>( char, gtint_t, gtint_t, scomplex, scomplex *, gtint_t );
template void make_diag<dcomplex>( char, gtint_t, gtint_t, dcomplex, dcomplex *, gtint_t );

/**
 * print scalar value
 * @param[in] x    specifies the value.
 * @param[in] spec specifies the format specifer.
 */
template<typename T>
void print_scalar( T x, const char *spec ) {
    if constexpr (testinghelpers::type_info<T>::is_real)
        printf(spec, x);
    else {
        printf( spec, x.real );
        if(x.imag < 0)    printf( "-" );
        else              printf( "+" );
        printf( spec, abs(x.imag) );
        printf( " " );
    }
}
template void print_scalar<float>( float x, const char * );
template void print_scalar<double>( double x, const char * );
template void print_scalar<scomplex>( scomplex x, const char * );
template void print_scalar<dcomplex>( dcomplex x, const char * );

/**
 * print vector of length  n
 * @param[in] vec  specifies the vector name
 * @param[in] n    specifies the length of the given vector.
 * @param[in] a    specifies pointer which points to the first element of a.
 * @param[in] incx specifies storage spacing between elements of a.
 * @param[in] spec specifies the format specifer.
 */
template<typename T>
void print_vector( const char *vec, gtint_t n, T *x, gtint_t incx, const char *spec )
{
    gtint_t i, idx;
    T val;
    std::cout << "Vector " << vec << std::endl;
    for ( i = 0; i < n; i++ )
    {
        idx = (incx > 0) ? (i * incx) : ( - ( n - i - 1 ) * incx );
        val = x[idx];
        print_scalar<T>(val,spec);
        printf( " " );
    }
    printf( "\n\n" );
}
template void print_vector<float>( const char *vec, gtint_t, float *, gtint_t, const char * );
template void print_vector<double>( const char *vec, gtint_t, double *, gtint_t, const char * );
template void print_vector<scomplex>( const char *vec, gtint_t, scomplex *, gtint_t, const char * );
template void print_vector<dcomplex>( const char *vec, gtint_t, dcomplex *, gtint_t, const char * );

/**
 * print matrix of size m x n
 * @param[in] mat     specifies the matrix name
 * @param[in] storage specifies the storage format of matrix in memory.
 * @param[in] m       specifies the number of rows of given matrix.
 * @param[in] n       specifies the number of columns of given matrix.
 * @param[in] a       specifies pointer which points to the first element of a.
 * @param[in] ld      specifies leading dimension for a given matrix.
 * @param[in] spec    specifies the format specifer.
 */
template<typename T>
void print_matrix( const char *mat, char storage, gtint_t m, gtint_t n, T *a, gtint_t ld, const char *spec )
{
    gtint_t rs,cs;
    rs=cs=1;
    T val;
    if( (storage == 'c') || (storage == 'C') )
        cs = ld ;
    else
        rs = ld ;

    gtint_t i, j;
    std::cout << "Matrix " << mat << std::endl;
    for ( i = 0; i < m; i++ )
    {
        for ( j = 0; j < n; j++ )
        {
            val = a[i*rs + j*cs];
            print_scalar<T>(val,spec);
            printf( " " );
        }
        printf( "\n" );
    }
    printf( "\n" );
}
template void print_matrix<float>( const char *mat, char, gtint_t, gtint_t, float *, gtint_t, const char * );
template void print_matrix<double>( const char *mat, char, gtint_t, gtint_t, double *, gtint_t, const char * );
template void print_matrix<scomplex>( const char *mat, char, gtint_t, gtint_t, scomplex *, gtint_t, const char * );
template void print_matrix<dcomplex>( const char *mat, char, gtint_t, gtint_t, dcomplex *, gtint_t, const char * );


/*
    Helper function that returns a string based on the value that is passed
    The return values are as follows :
    If datatype is real : "nan", "inf"/"minus_inf", "value", where "value"
    is the string version of the value that is passed, if it is not nan/inf/-inf.

    If the datatype is complex : The string is concatenated with both the real and
    imaginary components values, based on analysis done separately to each of them
    (similar to real datatype).
*/
template<typename T>
std::string get_value_string(T exval)
{
  std::string exval_str;
  if constexpr (testinghelpers::type_info<T>::is_real)
  {
    if(std::isnan(exval))
      exval_str = "nan";
    else if(std::isinf(exval))
      exval_str = (exval >= 0) ? "inf" : "minus_inf";
    else
      exval_str = ( exval >= 0) ? std::to_string(int(exval)) : "minus_" + std::to_string(int(std::abs(exval)));
  }
  else
  {
    if(std::isnan(exval.real))
    {
      exval_str = "nan";
      if(std::isinf(exval.imag))
        exval_str = exval_str + "pi" + ((exval.imag >= 0) ? "inf" : "minus_inf");
      else
        exval_str = exval_str + "pi" + ((exval.imag >= 0)? std::to_string(int(exval.imag)) : "m" + std::to_string(int(std::abs(exval.imag))));
    }
    else if(std::isnan(exval.imag))
    {
      if(std::isinf(exval.real))
        exval_str = ((exval.real >= 0) ? "inf" : "minus_inf");
      else
        exval_str = ((exval.real >= 0)? std::to_string(int(exval.real)) : "m" + std::to_string(int(std::abs(exval.real))));
      exval_str = exval_str + "pinan";
    }
    else if(std::isinf(exval.real))
    {
      exval_str = ((exval.real >= 0) ? "inf" : "minus_inf");
      if(std::isnan(exval.imag))
        exval_str = exval_str + "pinan";
      else
        exval_str = exval_str + "pi" + ((exval.imag >= 0)? std::to_string(int(exval.imag)) : "m" + std::to_string(int(std::abs(exval.imag))));
    }
    else if(std::isinf(exval.imag))
    {
      if(std::isnan(exval.real))
        exval_str = "nan";
      else
        exval_str = ((exval.real >= 0)? std::to_string(int(exval.real)) : "m" + std::to_string(int(std::abs(exval.real))));

      exval_str = exval_str + ((exval.imag >= 0) ? "inf" : "minus_inf");
    }
    else
    {
        exval_str = ((exval.real >= 0)? std::to_string(int(exval.real)) : "m" + std::to_string(int(std::abs(exval.real))));
        exval_str = exval_str + "pi" + ((exval.imag >= 0)? std::to_string(int(exval.imag)) : "m" + std::to_string(int(std::abs(exval.imag))));
    }
  }
  return exval_str;
}
template std::string testinghelpers::get_value_string( float );
template std::string testinghelpers::get_value_string( double );
template std::string testinghelpers::get_value_string( scomplex );
template std::string testinghelpers::get_value_string( dcomplex );

} //end of namespace testinghelpers
