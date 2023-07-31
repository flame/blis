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

#include <random>
#include "common/type_info.h"

namespace testinghelpers {
namespace datagenerators {

/***************************************************
 *              Random Generators
****************************************************/
/**
 * @brief Returns a random int/float converted to an fp type (float, double, scomplex, dcomplex)
 *        that lies in the range [from, to].
 *
 * @param[in, out] alpha the random fp
 */
template<typename T>
void randomgenerators(int from, int to, T* alpha, char fp);

/**
 * @brief Returns a random vector (float, double, scomplex, dcomplex)
 *        with elements that are integers or floats, depending on char, and follow a uniform distribution in the range [from, to].
 * @param[in] n length of vector x
 * @param[in] incx increments of vector x
 * @param[in, out] x the random fp vector
 * @param[in] fp if fp=='i' the elements will have random integer values.
 *               if fp=='f' the elements will have random float values.
 */
template<typename T>
void randomgenerators(int from, int to, gtint_t n, gtint_t incx, T* x, char fp = BLIS_ELEMENT_TYPE);

template<typename T>
void randomgenerators(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, char fp = BLIS_ELEMENT_TYPE);

template<typename T>
void randomgenerators(int from, int to, char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, char fp = BLIS_ELEMENT_TYPE);

template<typename T>
void randomgenerators(int from, int to, char storage, char uplo, gtint_t m,
                    T* a, gtint_t lda, char fp = BLIS_ELEMENT_TYPE );
} //end of namespace datagenerators

template<typename T>
std::vector<T> get_random_matrix(int from, int to, char storage, char trans, gtint_t m, gtint_t n,
                    gtint_t lda, char datatype = BLIS_ELEMENT_TYPE );

template<typename T>
std::vector<T> get_random_matrix(int from, int to, char storage, char uplo, gtint_t k,
                    gtint_t lda, char datatype = BLIS_ELEMENT_TYPE );

template<typename T>
std::vector<T> get_random_vector(int from, int to, gtint_t n, gtint_t incx,char datatype = BLIS_ELEMENT_TYPE);

template<typename T>
std::vector<T> get_vector( gtint_t n, gtint_t incx, T value );

template<typename T>
std::vector<T> get_matrix( char storage, char trans, gtint_t m, gtint_t n, gtint_t lda, T value );

template<typename T>
void set_vector( gtint_t n, gtint_t incx, T* x, T value );

template<typename T>
void set_matrix( char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, T value );

// Function template to set the exception value exval on matrix m, at indices (i, j)
// In case of transposition, this function internally swaps the indices, and thus they can be
// passed without swapping on the instantiator.
template<typename T>
void set_ev_mat( char storage, char trns, gtint_t ld, gtint_t i, gtint_t j, T exval, T* m );

} //end of namespace testinghelpers
