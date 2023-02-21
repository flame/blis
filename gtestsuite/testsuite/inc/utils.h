#pragma once

#pragma once
#include "blis.h"
#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * MKHERM 
 * Make an m x m matrix A explicitly Hermitian by copying the conjugate 
 * of the triangle specified by uploa to the opposite triangle. Imaginary
 * components of diagonal elements are explicitly set to zero. 
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 */
template<typename T>
static void mkherm( char storage, char uplo, gtint_t n, T* ap, gtint_t lda )
{
    uplo_t  uploa;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_smkherm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dmkherm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cmkherm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zmkherm( uploa, n, ap, rsa, csa );
    else

        throw std::runtime_error("Error in utils.h: Invalid typename in mkherm().");
}

/*
 * ==========================================================================
 * MKSYMM 
 * Make an m x m matrix A explicitly symmetric by copying the triangle 
 * specified by uploa to the opposite triangle. 
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 */

template<typename T>
static void mksymm( char storage, char uplo, gtint_t n, T* ap, gtint_t lda )
{
    uplo_t  uploa;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_smksymm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dmksymm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cmksymm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zmksymm( uploa, n, ap, rsa, csa );
    else

        throw std::runtime_error("Error in utils.h: Invalid typename in mksymm().");
}

/*
 * ==========================================================================
 * MKTRIM 
 * Make an m x m matrix A explicitly triangular by preserving the triangle 
 * specified by uploa and zeroing the elements in the opposite triangle. 
 * It is assumed that the diagonal offset of A is zero
 * ==========================================================================
 */
template<typename T>
static void mktrim( char storage, char uplo, gtint_t n, T* ap, gtint_t lda )
{
    uplo_t  uploa;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_smktrim( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dmktrim( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cmktrim( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zmktrim( uploa, n, ap, rsa, csa );
    else

        throw std::runtime_error("Error in utils.h: Invalid typename in mktrim().");
}