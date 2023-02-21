#pragma once

#include "blis.h"
#include "common/testing_helpers.h"

/**
 * @brief Overload bli_*normfv() functions using typed_nrm2.
 *        Will be used in testing and especially in TYPED_TESTs.
 *        Computes the Euclidean norm of x.
 * @param[in] n vector length
 * @param[in] x pointer which points to the first element of x
 * @param[in] incx increment of x
 * @return the Euclidean norm of x
 */

template<typename T, typename Treal>
static Treal nrm2_(gtint_t n, T* x, gtint_t incx){
    if constexpr (std::is_same<T, float>::value)
        return snrm2_( &n, x, &incx );
    else if constexpr (std::is_same<T, double>::value)
        return dnrm2_( &n, x, &incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        return scnrm2_( &n, x, &incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        return dznrm2_( &n, x, &incx );
    else
      throw std::runtime_error("Error in testsuite/level1/nrm2.h: Invalid typename in nrm2_().");
}

template<typename T, typename Treal>
static Treal cblas_nrm2(gtint_t n, T* x, gtint_t incx){
    if constexpr (std::is_same<T, float>::value)
        return cblas_snrm2( n, x, incx );
    else if constexpr (std::is_same<T, double>::value)
        return cblas_dnrm2( n, x, incx );
    else if constexpr (std::is_same<T, scomplex>::value)
        return cblas_scnrm2( n, x, incx );
    else if constexpr (std::is_same<T, dcomplex>::value)
        return cblas_dznrm2( n, x, incx );
    else
      throw std::runtime_error("Error in testsuite/level1/nrm2.h: Invalid typename in cblas_nrm2().");
}

template<typename T, typename Treal>
static Treal typed_nrm2(gtint_t n, T* x, gtint_t incx){
    Treal nrm;
    if constexpr (std::is_same<T, float>::value)
        bli_snormfv(n, x, incx, &nrm);
    else if constexpr (std::is_same<T, double>::value)
        bli_dnormfv(n, x, incx, &nrm);
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cnormfv(n, x, incx, &nrm);
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_znormfv(n, x, incx, &nrm);
    else
      throw std::runtime_error("Error in testsuite/level1/nrm2.h: Invalid typename in cblas_nrm2().");
    return nrm;
}

template<typename T, typename Treal>
static Treal nrm2(gtint_t n, T* x, gtint_t incx)
{
#ifdef TEST_BLAS
    return nrm2_<T, Treal>(n, x, incx);
#elif TEST_CBLAS
    return cblas_nrm2<T, Treal>(n, x, incx);
#elif TEST_BLIS_TYPED
    return typed_nrm2<T, Treal>(n, x, incx);
#else
    throw std::runtime_error("Error in testsuite/level1/axpyv.h: No interfaces are set to be tested.");
#endif
}