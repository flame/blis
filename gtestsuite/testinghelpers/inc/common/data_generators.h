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
void randomgenerators(int from, int to, gtint_t n, gtint_t incx, T* x, char fp);

template<typename T>
void randomgenerators(int from, int to, char storage, gtint_t m, gtint_t n, T* a, gtint_t lda, char fp);

template<typename T>
void randomgenerators(int from, int to, char storage, gtint_t m, gtint_t n, T* a, char transa, gtint_t lda, char fp);

template<typename T>
void randomgenerators(int from, int to, char storage, char uplo, gtint_t m,
                    T* a, gtint_t lda, char datatype);
} //end of namespace datagenerators

template<typename T>
std::vector<T> get_random_matrix(int from, int to, char storage, char trans, gtint_t m, gtint_t n,
                    gtint_t lda, char datatype);

template<typename T>
std::vector<T> get_random_matrix(int from, int to, char storage, char uplo, gtint_t k,
                    gtint_t lda, char datatype);

template<typename T>
std::vector<T> get_random_vector(int from, int to, gtint_t n, gtint_t incx, char datatype);

} //end of namespace testinghelpers