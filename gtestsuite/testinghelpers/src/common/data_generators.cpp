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

#include <random>
#include "common/testing_helpers.h"

namespace testinghelpers {
namespace datagenerators {

/***************************************************
 *             Floating Point Generators
****************************************************/
/**
 * @brief Returns a random fp type (float, double, scomplex, dcomplex)
 *        that lies in the range [from, to].
 *
 * @param[in, out] alpha the random fp
 */
template<typename T>
void getfp(int from, int to, T* alpha)
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                              generator(94);
    std::uniform_real_distribution<real_T>    distr(from, to);
    if constexpr (testinghelpers::type_info<T>::is_real)
        *alpha = distr(generator);
    else
        *alpha = {distr(generator), distr(generator)};
}

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 */
template<typename T>
void getfp(int from, int to, gtint_t n, gtint_t incx, T* x)
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    T* chi;
    std::mt19937                              generator(94);
    std::uniform_real_distribution<real_T>    distr(from, to);
    for ( gtint_t i = 0; i < n; ++i )
    {
        chi = x + i*std::abs(incx);
        if constexpr (testinghelpers::type_info<T>::is_real)
            *chi = distr(generator);
        else
            *chi = {distr(generator), distr(generator)};
    }
}

template<typename T>
void getfp(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda )
{
    T*    a_begin;
    gtint_t inca;
    gtint_t n_iter;
    gtint_t n_elem;
    gtint_t j;

    // Initialize with optimal values for column-major storage.
    inca   = 1;
    n_iter = n;
    n_elem = m;

    // An optimization: if A is row-major, then let's access the matrix by
    // rows instead of by columns for increased spatial locality.
    if( (storage == 'r') || (storage == 'R') )
    {
        swap_dims( &n_iter, &n_elem );
        swap_dims( &lda, &inca );
    }

    for ( j = 0; j < n_iter; j++ )
    {
        a_begin = a + j*lda;
        getfp<T>( from, to, n_elem, inca, a_begin );
    }
}

template<typename T>
void getfp(int from, int to, char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda )
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                              generator(1994);
    std::uniform_real_distribution<real_T>    distr(from, to);

    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }

    if((storage == 'c') || (storage == 'C'))
    {
        for(gtint_t i=0; i<m; i++)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                    a[i+j*lda] = real_T(distr(generator));
                else
                    a[i+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        for(gtint_t j=0; j<n; j++)
        {
            for(gtint_t i=0; i<m; i++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                    a[j+i*lda] = real_T(distr(generator));
                else
                    a[j+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
            }
        }
    }
}

/***************************************************
 *              Integer Generators
****************************************************/
/**
 * @brief Returns a random integer converted to an fp type (float, double, scomplex, dcomplex)
 *        that lies in the range [from, to].
 *
 * @param[in, out] alpha the random fp
 */
template<typename T>
void getint(int from, int to, T* alpha)
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                          generator(94);
    std::uniform_int_distribution<int>    distr(from, to);
    if constexpr (testinghelpers::type_info<T>::is_real)
        *alpha = real_T(distr(generator));
    else
        *alpha = {real_T(distr(generator)), real_T(distr(generator))};
}
/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that are integers and follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 */
template<typename T>
void getint(int from, int to, gtint_t n, gtint_t incx, T* x)
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    T* chi;
    std::mt19937                          generator(94);
    std::uniform_int_distribution<int>    distr(from, to);
    for ( gtint_t i = 0; i < n; ++i )
    {
        chi = x + i*std::abs(incx);
        if constexpr (testinghelpers::type_info<T>::is_real)
            *chi = real_T(distr(generator));
        else
            *chi = {real_T(distr(generator)), real_T(distr(generator))};
    }
}

template<typename T>
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda )
{
    T*    a_begin;
    gtint_t inca;
    gtint_t n_iter;
    gtint_t n_elem;
    gtint_t j;

    // Initialize with optimal values for column-major storage.
    inca   = 1;
    n_iter = n;
    n_elem = m;

    // An optimization: if A is row-major, then let's access the matrix by
    // rows instead of by columns for increased spatial locality.
    if( (storage == 'r') || (storage == 'R') )
    {
        swap_dims( &n_iter, &n_elem );
        swap_dims( &lda, &inca );
    }

    for ( j = 0; j < n_iter; j++ )
    {
        a_begin = a + j*lda;
        getint<T>( from, to, n_elem, inca, a_begin );
    }
}

/// @brief
/// @tparam T
/// @param from
/// @param to
/// @param storage
/// @param m
/// @param n
/// @param a
/// @param transa
/// @param lda
template<typename T>
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda )
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                          generator(1994);
    std::uniform_int_distribution<int>    distr(from, to);

    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }

    if((storage == 'c') || (storage == 'C'))
    {
        for(gtint_t i=0; i<m; i++)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                    a[i+j*lda] = real_T(distr(generator));
                else
                    a[i+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        for(gtint_t j=0; j<n; j++)
        {
            for(gtint_t i=0; i<m; i++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                    a[j+i*lda] = real_T(distr(generator));
                else
                    a[j+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
            }
        }
    }
}

template<typename T>
void randomgenerators( int from, int to, T* alpha, char datatype ) {

    if( (datatype == 'i') ||(datatype == 'I') )
        getint<T>( from, to, alpha );
    else /*if( (datatype == 'f') ||(datatype == 'F') ) */
        getfp<T>( from, to, alpha );
}

template<typename T>
void randomgenerators(int from, int to, gtint_t n, gtint_t incx, T* x, char datatype ) {

    if( (datatype == 'i') ||(datatype == 'I') )
        getint<T>( from, to, n, incx, x );
    else /*if( (datatype == 'f') ||(datatype == 'F') ) */
        getfp<T>( from, to, n, incx, x );
}

template<typename T>
void randomgenerators( int from, int to, char storage, gtint_t m, gtint_t n,
     T* a, gtint_t lda, char datatype ) {

    if( (datatype == 'i') ||(datatype == 'I') )
        getint<T>( from, to, storage, m, n, a, lda );
    else /*if( (datatype == 'f') ||(datatype == 'F') ) */
        getfp<T>( from, to, storage, m, n, a, lda );
}

template<typename T>
void randomgenerators( int from, int to, char storage, gtint_t m, gtint_t n,
     T* a, char transa, gtint_t lda, char datatype ) {

    if( (datatype == 'i') ||(datatype == 'I') )
        getint<T>( from, to, storage, m, n, a, transa, lda );
    else /*if( (datatype == 'f') ||(datatype == 'F') ) */
        getfp<T>( from, to, storage, m, n, a, transa, lda );
}

template<typename T>
void randomgenerators(int from, int to, char storage, char uplo, gtint_t k,
                    T* a, gtint_t lda, char datatype) {
    randomgenerators<T>(from, to, storage, k, k, a, lda, datatype);
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<k; j++)
        {
            for(gtint_t i=0; i<k; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[i+j*lda] = T{0};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[i+j*lda] = T{0};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
    else
    {
        for(gtint_t i=0; i<k; i++)
        {
            for(gtint_t j=0; j<k; j++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[j+i*lda] = T{0};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[j+i*lda] = T{0};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
}

} //end of namespace datagenerators

template<typename T>
std::vector<T> get_random_matrix(int from, int to, char storage, char trans, gtint_t m, gtint_t n,
                    gtint_t lda, char datatype)
{
    std::vector<T> a(matsize(storage, trans, m, n, lda));
    testinghelpers::datagenerators::randomgenerators<T>( from, to, storage, m, n, a.data(), trans, lda, datatype );
    return a;
}
template<typename T>
std::vector<T> get_random_matrix(int from, int to, char storage, char uplo, gtint_t k, gtint_t lda, char datatype)
{
    // Create matrix for the given sizes.
    std::vector<T> a( testinghelpers::matsize( storage, 'n', k, k, lda ) );
    testinghelpers::datagenerators::randomgenerators( from, to, storage, uplo, k, a.data(), lda, datatype );
    return a;
}

template<typename T>
std::vector<T> get_random_vector(int from, int to, gtint_t n, gtint_t incx, char datatype)
{
    // Create vector for the given sizes.
    std::vector<T> x( testinghelpers::buff_dim(n, incx) );
    testinghelpers::datagenerators::randomgenerators( from, to, n, incx, x.data(), datatype );
    return x;
}



template<typename T>
void set_vector( gtint_t n, gtint_t incx, T* x, T value )
{
    T* chi;
    for ( gtint_t i = 0; i < n; ++i )
    {
        chi = x + i*std::abs(incx);
        *chi = value ;
    }
}

template<typename T>
void set_matrix( char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, T value )
{
    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }

    if((storage == 'c') || (storage == 'C'))
    {
        for( gtint_t i = 0 ; i < m ; i++ )
        {
            for( gtint_t j = 0 ; j < n ; j++ )
            {
                a[i+j*lda] = value ;
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        for( gtint_t j = 0 ; j < n ; j++ )
        {
            for( gtint_t i = 0 ; i < m ; i++ )
            {
                a[j+i*lda] = value ;
            }
        }
    }
}

template<typename T>
std::vector<T> get_vector( gtint_t n, gtint_t incx, T value )
{
    // Create vector for the given sizes.
    std::vector<T> x( testinghelpers::buff_dim(n, incx) );
    testinghelpers::set_vector( n, incx, x.data(), value );
    return x;
}

template<typename T>
std::vector<T> get_matrix( char storage, char trans, gtint_t m, gtint_t n, gtint_t lda, T value )
{
    std::vector<T> a( matsize( storage, trans, m, n, lda ) );
    testinghelpers::set_matrix<T>( storage, m, n, a.data(), trans, lda, value );
    return a;
}

} //end of namespace testinghelpers

// Explicit template instantiations
template void testinghelpers::datagenerators::randomgenerators<float>(int, int, float*, char);
template void testinghelpers::datagenerators::randomgenerators<double>(int, int, double*, char);
template void testinghelpers::datagenerators::randomgenerators<scomplex>(int, int, scomplex*, char);
template void testinghelpers::datagenerators::randomgenerators<dcomplex>(int, int, dcomplex*, char);

template void testinghelpers::datagenerators::randomgenerators<float>(int, int, gtint_t, gtint_t, float*, char);
template void testinghelpers::datagenerators::randomgenerators<double>(int, int, gtint_t, gtint_t, double*, char);
template void testinghelpers::datagenerators::randomgenerators<scomplex>(int, int, gtint_t, gtint_t, scomplex*, char);
template void testinghelpers::datagenerators::randomgenerators<dcomplex>(int, int, gtint_t, gtint_t, dcomplex*, char);

template void testinghelpers::datagenerators::randomgenerators<float>(int, int, char, gtint_t, gtint_t, float*, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<double>(int, int, char, gtint_t, gtint_t, double*, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<scomplex>(int, int, char, gtint_t, gtint_t, scomplex*, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<dcomplex>(int, int, char, gtint_t, gtint_t, dcomplex*, gtint_t, char);

template void testinghelpers::datagenerators::randomgenerators<float>(int, int, char, gtint_t, gtint_t, float*, char, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<double>(int, int, char, gtint_t, gtint_t, double*, char, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<scomplex>(int, int, char, gtint_t, gtint_t, scomplex*, char, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<dcomplex>(int, int, char, gtint_t, gtint_t, dcomplex*, char, gtint_t, char);

template void testinghelpers::datagenerators::randomgenerators<float>(int, int, char, char, gtint_t, float*, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<double>(int, int, char, char, gtint_t, double*, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<scomplex>(int, int, char, char, gtint_t, scomplex*, gtint_t, char);
template void testinghelpers::datagenerators::randomgenerators<dcomplex>(int, int, char, char, gtint_t, dcomplex*, gtint_t, char);

template std::vector<float> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, gtint_t, char);
template std::vector<double> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, gtint_t, char);
template std::vector<scomplex> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, gtint_t, char);
template std::vector<dcomplex> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, gtint_t, char);

template std::vector<float> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, char);
template std::vector<double> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, char);
template std::vector<scomplex> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, char);
template std::vector<dcomplex> testinghelpers::get_random_matrix(int, int, char, char, gtint_t, gtint_t, char);

template std::vector<float> testinghelpers::get_random_vector(int, int, gtint_t, gtint_t, char);
template std::vector<double> testinghelpers::get_random_vector(int, int, gtint_t, gtint_t, char);
template std::vector<scomplex> testinghelpers::get_random_vector(int, int, gtint_t, gtint_t, char);
template std::vector<dcomplex> testinghelpers::get_random_vector(int, int, gtint_t, gtint_t, char);

template std::vector<float> testinghelpers::get_vector(gtint_t, gtint_t, float);
template std::vector<double> testinghelpers::get_vector(gtint_t, gtint_t, double);
template std::vector<scomplex> testinghelpers::get_vector(gtint_t, gtint_t, scomplex);
template std::vector<dcomplex> testinghelpers::get_vector(gtint_t, gtint_t, dcomplex);

template std::vector<float> testinghelpers::get_matrix( char, char, gtint_t, gtint_t, gtint_t, float );
template std::vector<double> testinghelpers::get_matrix( char, char, gtint_t, gtint_t, gtint_t, double );
template std::vector<scomplex> testinghelpers::get_matrix( char, char, gtint_t, gtint_t, gtint_t, scomplex );
template std::vector<dcomplex> testinghelpers::get_matrix( char, char, gtint_t, gtint_t, gtint_t, dcomplex );
