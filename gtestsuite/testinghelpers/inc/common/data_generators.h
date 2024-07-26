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

#include <random>
#include <type_traits>
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

    if (incx != 1)
    {
        // First initialize all elements in vector to unusual value to help
        // catch if intervening elements have been incorrectly used or modified.
        for ( gtint_t i = 0; i < testinghelpers::buff_dim(n, incx); ++i )
        {
            chi = x + i;
            *chi = T1{-1.2345e38};
        }
    }

    // Generate the values from the uniform distribution that
    // the BLAS routine should read and/or modify.
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
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, gtint_t lda, gtint_t stridea = 1 )
{
    using real_T = typename testinghelpers::type_info<T1>::real_type;
    std::mt19937                              generator(1994);
    std::uniform_real_distribution<real_T>    distr(from, to);

    if((storage == 'c') || (storage == 'C'))
    {
        if (m > 0)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if constexpr (testinghelpers::type_info<T1>::is_real)
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T1{-1.2345e38};

                        a[i*stridea+j*lda] = real_T(distr(generator));
                    }
                    a[(m-1)*stridea+j*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T1{-1.2345e38};

                        a[i*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(m-1)*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t i=(m-1)*stridea+1; i<lda; i++)
                {
                    a[i+j*lda] = T1{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t j=0; j<n; j++)
            {
                for(gtint_t i=0; i<lda; i++)
                {
                    a[i+j*lda] = T1{-1.2345e38};
                }
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        if (n > 0)
        {
            for(gtint_t i=0; i<m; i++)
            {
                if constexpr (testinghelpers::type_info<T1>::is_real)
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T1{-1.2345e38};

                        a[j*stridea+i*lda] = real_T(distr(generator));
                    }
                    a[(n-1)*stridea+i*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T1{-1.2345e38};

                        a[j*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(n-1)*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t j=(n-1)*stridea+1; j<lda; j++)
                {
                    a[j+i*lda] = T1{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t i=0; i<m; i++)
            {
                for(gtint_t j=0; j<lda; j++)
                {
                    a[j+i*lda] = T1{-1.2345e38};
                }
            }
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
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T1, typename T2, typename T3>
void getfp(T2 from, T3 to, char storage, gtint_t m, gtint_t n, T1* a, char transa, gtint_t lda, gtint_t stridea = 1 )
{
    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }
    getfp<T1>( from, to, storage, m, n, a, lda, stridea );
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

    if (incx != 1)
    {
        // First initialize all elements in vector to unusual value to help
        // catch if intervening elements have been incorrectly used or modified.
        for ( gtint_t i = 0; i < testinghelpers::buff_dim(n, incx); ++i )
        {
            chi = x + i;
            *chi = T{-1.2345e38};
        }
    }

    // Generate the values from the uniform distribution that
    // the BLAS routine should read and/or modify.
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
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T>
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, gtint_t stridea = 1 )
{
    using real_T = typename testinghelpers::type_info<T>::real_type;
    std::mt19937                          generator(94);
    std::uniform_int_distribution<int>    distr(from, to);

    if((storage == 'c') || (storage == 'C'))
    {
        if (m > 0)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T{-1.2345e38};

                        a[i*stridea+j*lda] = real_T(distr(generator));
                    }
                    a[(m-1)*stridea+j*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t i=0; i<m-1; i++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[i*stridea+p+j*lda] = T{-1.2345e38};

                        a[i*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(m-1)*stridea+j*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t i=(m-1)*stridea+1; i<lda; i++)
                {
                    a[i+j*lda] = T{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t j=0; j<n; j++)
            {
                for(gtint_t i=0; i<lda; i++)
                {
                    a[i+j*lda] = T{-1.2345e38};
                }
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        if (n > 0)
        {
            for(gtint_t i=0; i<m; i++)
            {
                if constexpr (testinghelpers::type_info<T>::is_real)
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T{-1.2345e38};

                        a[j*stridea+i*lda] = real_T(distr(generator));
                    }
                    a[(n-1)*stridea+i*lda] = real_T(distr(generator));
                }
                else
                {
                    for(gtint_t j=0; j<n-1; j++)
                    {
                        for(gtint_t p=1; p<stridea; p++)
                            a[j*stridea+p+i*lda] = T{-1.2345e38};

                        a[j*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                    }
                    a[(n-1)*stridea+i*lda] = {real_T(distr(generator)), real_T(distr(generator))};
                }
                for(gtint_t j=(n-1)*stridea+1; j<lda; j++)
                {
                    a[j+i*lda] = T{-1.2345e38};
                }
            }
        }
        else
        {
            for(gtint_t i=0; i<m; i++)
            {
                for(gtint_t j=0; j<lda; j++)
                {
                    a[j+i*lda] = T{-1.2345e38};
                }
            }
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
 * @param[in] stridea stride between two "continuous" elements in matrix A
 */
template<typename T>
void getint(int from, int to, char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, gtint_t stridea = 1 )
{
    if( chktrans( transa )) {
       swap_dims( &m, &n );
    }
    getint<T>( from, to, storage, m, n, a, lda, stridea );
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
     T1* a, gtint_t lda, gtint_t stridea = 1, ElementType datatype = GenericET ) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, storage, m, n, a, lda, stridea );
    else
        getfp<T1>( from, to, storage, m, n, a, lda, stridea );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, gtint_t m, gtint_t n,
     T1* a, char transa, gtint_t lda, gtint_t stridea = 1, ElementType datatype = GenericET ) {

    if( datatype == ElementType::INT )
        getint<T1>( from, to, storage, m, n, a, transa, lda, stridea );
    else
        getfp<T1>( from, to, storage, m, n, a, transa, lda, stridea );
}

template<typename T1, typename T2, typename T3>
void randomgenerators( T2 from, T3 to, char storage, char uplo, gtint_t k,
                    T1* a, gtint_t lda, ElementType datatype = GenericET ) {
    testinghelpers::datagenerators::randomgenerators<T1>(from, to, storage, k, k, a, lda, 1, datatype);
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<k; j++)
        {
            for(gtint_t i=0; i<k; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[i+j*lda] = T1{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[i+j*lda] = T1{2.987e38};
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
                    if(i>j) a[j+i*lda] = T1{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[j+i*lda] = T1{2.987e38};
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
                    gtint_t lda, gtint_t stridea = 1, datagenerators::ElementType datatype = datagenerators::GenericET)
{
    std::vector<T1> a(matsize(storage, trans, m, n, lda));
    testinghelpers::datagenerators::randomgenerators<T1>( from, to, storage, m, n, a.data(), trans, lda, stridea, datatype );
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

    if (incx != 1)
    {
        // First initialize all elements in vector to unusual value to help
        // catch if intervening elements have been incorrectly used or modified.
        for ( gtint_t i = 0; i < testinghelpers::buff_dim(n, incx); ++i )
        {
            chi = x + i;
            *chi = T{-1.2345e38};
        }
    }

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
        for( gtint_t j = 0 ; j < n ; j++ )
        {
            for( gtint_t i = 0 ; i < m ; i++ )
            {
                a[i+j*lda] = value ;
            }
            for(gtint_t i=m; i<lda; i++)
            {
                a[i+j*lda] = T{-1.2345e38};
            }
        }
    }
    else if( (storage == 'r') || (storage == 'R') )
    {
        for( gtint_t i = 0 ; i < m ; i++ )
        {
            for( gtint_t j = 0 ; j < n ; j++ )
            {
                a[j+i*lda] = value ;
            }
            for(gtint_t j=n; j<lda; j++)
            {
                a[j+i*lda] = T{-1.2345e38};
            }
        }
    }
}

template<typename T>
void set_matrix( char storage, gtint_t n, T* a, char uplo, gtint_t lda, T value )
{
    testinghelpers::set_matrix<T>(storage, n, n, a, 'n', lda, value );
    if( (storage=='c')||(storage=='C') )
    {
        for(gtint_t j=0; j<n; j++)
        {
            for(gtint_t i=0; i<n; i++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[i+j*lda] = T{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[i+j*lda] = T{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
            }
        }
    }
    else
    {
        for(gtint_t i=0; i<n; i++)
        {
            for(gtint_t j=0; j<n; j++)
            {
                if( (uplo=='u')||(uplo=='U') )
                {
                    if(i>j) a[j+i*lda] = T{2.987e38};
                }
                else if ( (uplo=='l')||(uplo=='L') )
                {
                    if (i<j) a[j+i*lda] = T{2.987e38};
                }
                else
                    throw std::runtime_error("Error in common/data_generators.cpp: side must be 'u' or 'l'.");
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

/*
    Function to set few values of a matrix to values relative to DBL_MAX/DBL_MIN
    These values are used to create overflow and underflow scenarios
*/
template<typename T>
void set_overflow_underflow_mat(char storage, char trns, gtint_t ld, gtint_t i, gtint_t j, T* a, gtint_t mode, gtint_t input_range)
{
    /* Calculate index where overflow/underflow values need to be inserted */
    gtint_t indexA = 0;

    if ( storage == 'c' || storage == 'C' )
    {
      if ( trns == 'n' || trns == 'N' )
      {
        indexA = i + j*ld;
      }
      else
      {
        indexA = j + i*ld;
      }
    }
    else
    {
      if ( trns == 'n' || trns == 'N' )
      {
        indexA = i*ld + j;
      }
      else
      {
        indexA = j*ld + i;
      }
    }

    using RT = typename testinghelpers::type_info<T>::real_type;
    std::vector<int> exponent(12);

    if (std::is_same<RT, double>::value)
    {
      exponent = {23, 203, 18, 180, 123, 130, 185, 178, 108, 158, 185, 220};
    }
    else if (std::is_same<RT, float>::value)
    {
      exponent = {3, 20, 8, 2, 30, 28, 8, 10, 33, 24, 8, 22};
    }

    T limits_val;

    /* When mode is set to 0, values relative to DBL_MAX are inserted into the input matrices */
    if(mode == 0)
    {
        limits_val = (std::numeric_limits<RT>::max)();
        switch(input_range)
        {
            case -1:
                     a[0] = limits_val/ pow(10, exponent[0]);
                     a[indexA] = limits_val/ pow(10, exponent[1]);
                     break;

            case 0:
                     a[0] = -(limits_val/ pow(10, exponent[4]));
                     a[indexA] = -(limits_val/ pow(10, exponent[5]));
                     break;

            case 1:
                     a[0] = limits_val/ pow(10, exponent[8]);
                     a[indexA] = limits_val/ pow(10, exponent[9]);
        }
    }
    /* When mode is set to 1, values relative to DBL_MIN are inserted into the input matrices*/
    else
    {
        limits_val = (std::numeric_limits<RT>::min)();
        switch(input_range)
        {
            case -1:
                     a[0] = limits_val * pow(10, exponent[0]);
                     a[indexA] = limits_val * pow(10, exponent[1]);
                     break;

            case 0:
                     a[0] = -(limits_val * pow(10, exponent[4]));
                     a[indexA] = -(limits_val * pow(10, exponent[5]));
                     break;

            case 1:
                     a[0] = limits_val * pow(10, exponent[8]);
                     a[indexA] = limits_val * pow(10, exponent[9]);
        }

    }
}

} //end of namespace testinghelpers
