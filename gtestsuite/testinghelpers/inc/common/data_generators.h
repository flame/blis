/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

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

// Setting an enum class to make random data generation more robust.
enum class ElementType {FP, INT};
// Define a static variable to be used as the default argument in
// the generators, depending on CMake configuration.
#ifdef BLIS_INT_ELEMENT_TYPE
// Integer random values will be used in testing.
static const ElementType GenericET = ElementType::INT;
#else
// Floating-point random values will be used in testing.
static const ElementType GenericET = ElementType::FP;
#endif

/***************************************************
 *             Floating Point Generators
****************************************************/
/**
 * @brief Returns a random fp type (float, double, scomplex, dcomplex)
 *        that lies in the range [from, to].
 *
 * @param[in, out] alpha the random fp
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, T1* alpha)
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
    std::mt19937                              generator(94);
    std::uniform_real_distribution<real_T>    distr(from, to);
    if constexpr (testinghelpers::type_info<T1>::is_real)
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
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, gtint_t n, gtint_t incx, T1* x)
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
    T1* chi;
    std::mt19937                              generator(94);
    std::uniform_real_distribution<real_T>    distr(from, to);
    for ( gtint_t i = 0; i < n; ++i )
    {
        chi = x + i*std::abs(incx);
        if constexpr (testinghelpers::type_info<T1>::is_real)
            *chi = distr(generator);
        else
            *chi = {distr(generator), distr(generator)};
    }
}

/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimentions of matrix A
 * @param[in, out] a the random fp matrix A 
 * @param[in] lda leading dimension of matrix A
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, gtint_t lda )
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
    std::mt19937                              generator(1994);
    std::uniform_real_distribution<real_T>    distr(from, to);

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
        for(gtint_t i=0; i<n_elem; i++)
        {
            if constexpr (testinghelpers::type_info<T1>::is_real)
                a[j+i*inca] = real_T(distr(generator));
            else
                a[j+i*inca] = {real_T(distr(generator)), real_T(distr(generator))};
        }
    }
}
/**
 * @brief Returns a random fp vector (float, double, scomplex, dcomplex)
 *        with elements that follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimentions of matrix A 
 * @param[in, out] a the random fp matrix A
 * @param[in] trans transposition of matrix A
 * @param[in] lda leading dimension of matrix A
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, char transa, gtint_t lda )
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
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
                if constexpr (testinghelpers::type_info<T1>::is_real)
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
                if constexpr (testinghelpers::type_info<T1>::is_real)
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

/**
 * @brief Returns a random fp matrix (float, double, scomplex, dcomplex)
 *        with elements that are integers and follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimentions of matrix A
 * @param[in, out] a the random fp matrix A 
 * @param[in] lda leading dimension of matrix A
 */
template<typename T>
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda )
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                          generator(94);
    std::uniform_int_distribution<int>    distr(from, to);

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
        for(gtint_t i=0; i<n_elem; i++)
        {
            if constexpr (testinghelpers::type_info<T>::is_real)
                a[j+i*inca] = real_T(distr(generator));
            else
                a[j+i*inca] = {real_T(distr(generator)), real_T(distr(generator))};
        }
    }
}

/**
 * @brief Returns a random fp matrix (float, double, scomplex, dcomplex)
 *        with elements that are integers and follow a uniform distribution in the range [from, to].
 * @param[in] storage storage type of matrix A, row or column major
 * @param[in] m, n dimentions of matrix A 
 * @param[in, out] a the random fp matrix A
 * @param[in] trans transposition of matrix A
 * @param[in] lda leading dimension of matrix A
 */
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

template<typename T1, typename T2, typename T3>
void randomgenerators(T2 from, T3 to, gtint_t n, gtint_t incx, T1* x, ElementType datatype = GenericET) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, n, incx, x );
    else
        getfp<T1>( from, to, n, incx, x );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, gtint_t m, gtint_t n,
     T1* a, gtint_t lda, ElementType datatype = GenericET ) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, storage, m, n, a, lda );
    else
        getfp<T1>( from, to, storage, m, n, a, lda );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, gtint_t m, gtint_t n,
     T1* a, char transa, gtint_t lda, ElementType datatype = GenericET ) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, storage, m, n, a, transa, lda );
    else
        getfp<T1>( from, to, storage, m, n, a, transa, lda );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, char uplo, gtint_t k,
                    T1* a, gtint_t lda, ElementType datatype = GenericET ) {
    randomgenerators<T1>(from, to, storage, k, k, a, lda, datatype);
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<k; j++)
        {
            for(gtint_t i=0; i<k; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[i+j*lda] = T1{0};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[i+j*lda] = T1{0};
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
                    if(i>j) a[j+i*lda] = T1{0};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[j+i*lda] = T1{0};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
}

} //end of namespace datagenerators

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_matrix(T2 from, T3 to, char storage, char trans, gtint_t m, gtint_t n,
                    gtint_t lda, datagenerators::ElementType datatype = datagenerators::GenericET)
{
    std::vector<T1> a(matsize(storage, trans, m, n, lda));
    testinghelpers::datagenerators::randomgenerators<T1>( from, to, storage, m, n, a.data(), trans, lda, datatype );
    return a;
}

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_matrix(T2 from, T3 to, char storage, char uplo, gtint_t k, gtint_t lda, datagenerators::ElementType datatype = datagenerators::GenericET )
{
    // Create matrix for the given sizes.
    std::vector<T1> a( testinghelpers::matsize( storage, 'n', k, k, lda ) );
    testinghelpers::datagenerators::randomgenerators<T1>( from, to, storage, uplo, k, a.data(), lda, datatype );
    return a;
}

template<typename T1, typename T2, typename T3>
std::vector<T1> get_random_vector(T2 from, T3 to, gtint_t n, gtint_t incx, datagenerators::ElementType datatype = datagenerators::GenericET)
{
    // Create vector for the given sizes.
    std::vector<T1> x( testinghelpers::buff_dim(n, incx) );
    testinghelpers::datagenerators::randomgenerators<T1>( from, to, n, incx, x.data(), datatype );
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

template<typename T>
void set_ev_mat( char storage, char trns, gtint_t ld, gtint_t i, gtint_t j, T exval, T* m )
{
    // Setting the exception values on the indices passed as arguments
    if ( storage == 'c' || storage == 'C' )
    {
      if ( trns == 'n' || trns == 'N' )
        m[i + j*ld] = exval;
      else
        m[j + i*ld] = exval;
    }
    else
    {
      if ( trns == 'n' || trns == 'N' )
        m[i*ld + j] = exval;
      else
        m[j*ld + i] = exval;
    }
}

} //end of namespace testinghelpers
