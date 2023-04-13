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
#include "blis.h"
#include <gtest/gtest.h>
#include "common/testing_helpers.h"

/**
 * This file includes the functionality used to determine correctness of the results.
 * We compare the results component-wise, meaning that for two scalars, we compare the scalars,
 * and for two vectors or matrices we compare each element of the vector or matrix respectively.
 *
 * We have two separate cases:
 * 1) it's meaningful to have NaNs and/or Infs at the results,
 *    because we are testing the correct propagation of those values.
 *    In this case the results could be either extreme values, or f.p. values,
 *    and we need to be able to compare all of those following the rules below:
 *    - if there are NaNs/Infs, check if both reference and blis solution has
 *      NaN/Inf accordingly. Remember that for Infs we need to check the sign as well.
 *    - if there are no NaNs/Infs, either do the bitwise comparison (if that's desired),
 *      or call into getError() function. getError() will check if reference is less
 *      than one, in which case will compute the absolute error, otherwise compute the
 *      relative error.
 *    - for complex numbers, we need to check for all possible combinations of NaN/Inf/FP
 *      for real and imaginary parts. So that will be a combination of the two steps above.
 * 2) it's not meaningful to check for NaNs and Infs and we expect only FP values in the
 *    results. In this case, we either do a bitwise comparison or call into getError() directly.
 *
 * Note that all operations with a NaN/Inf will lead to either comparison with a NaN, or
 * inf < thresh, which always return false; so NumericalComparisonFPOnly() will return failure.
 * So, for the case where we do not expect NaN/Infs, we want to fail if NaN and Infs are
 * present so that we do not have tests passing when it doesn't make sense to do so.
 * For an example of such case, think of a triangular solver with zeros on the diagonal.
 *
*/

// Enum used to do the correct printing depending on what we aim to compare.
enum ObjType {
    SCALAR,
    VECTOR,
    MATRIX
};
// Enum used to do the correct comparison for NaNs, depending on whether we
// compare the real or the imaginary component.
enum ComplexPart {
    REAL,
    IMAGINARY
};

// Helper class to be used to pass info into the Comparators.
struct ComparisonHelper{
    double threshold;
    ObjType object_type;
    gtint_t i; // used to print vector/matrix elements that we compare
    gtint_t j; // used to print matrix elements that we compare
    bool binary_comparison; // By default compare using relative error or absolute error approach.
    bool nan_inf_check; //By default do not check for NaNs and Infs.

    // Constructor for the case of binary_comparison, no threshold.
    ComparisonHelper(ObjType object_type) : threshold(-13.0),
                                            object_type(object_type),
                                            i(-11),
                                            j(-11),
                                            binary_comparison(false),
                                            nan_inf_check(false) {};
    // Constructor for the generic case where theshold is used.
    ComparisonHelper(ObjType object_type, double threshold) : threshold(threshold),
                                                              object_type(object_type),
                                                              i(-11),
                                                              j(-11),
                                                              binary_comparison(false),
                                                              nan_inf_check(false) {};
};

// Generic comparison of f.p. numbers that doesn't check for NaN's and Infs:
template<typename T>
testing::AssertionResult NumericalComparisonFPOnly(const char* blis_sol_char,
                                             const char* ref_sol_char,
                                             const char* comp_helper_char,
                                             const T blis_sol,
                                             const T ref_sol,
                                             const ComparisonHelper comp_helper,
                                             const std::string error_message)
{
    if (comp_helper.binary_comparison)
    {
        if (blis_sol == ref_sol) return testing::AssertionSuccess();
        return testing::AssertionFailure() << error_message;
    }
    else {
        double error = testinghelpers::getError(blis_sol,ref_sol);
        if (error < comp_helper.threshold) return testing::AssertionSuccess();
        return testing::AssertionFailure() << error_message
                                           << ",    thesh = " << comp_helper.threshold
                                           << ",    error = " << error;
    }
}

// NaN/Inf comparison for real numbers
template<typename T>
testing::AssertionResult NumericalComparisonRealNaNInf(const char* blis_sol_char,
                                             const char* ref_sol_char,
                                             const char* comp_helper_char,
                                             const T blis_sol,
                                             const T ref_sol,
                                             const ComparisonHelper comp_helper,
                                             const std::string error_message)
{
    // if both are NaN return SUCCESS
    if ((std::isnan(ref_sol)) && (std::isnan(blis_sol)))
        return testing::AssertionSuccess();
    // if only one of them is NaN, return FAILURE
    else if ((std::isnan(ref_sol)) || (std::isnan(blis_sol)))
        return testing::AssertionFailure() << error_message;
    // if both are inf check the sign
    else if ((std::isinf(ref_sol)) && (std::isinf(blis_sol)))
    {
        // check the sign of infs
        if( ref_sol == blis_sol ) return testing::AssertionSuccess();
        // both are infs but have different signs, return FAILURE.
        else return testing::AssertionFailure() << error_message;
    }
    // if only one of them is Inf
    else if ((std::isinf(ref_sol)) || (std::isinf(blis_sol)))
        return testing::AssertionFailure() << error_message;
    // If neither reference nor BLIS sol is NaN/Inf do simple comparison, based on relative or absolute error.
    else return NumericalComparisonFPOnly<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, error_message);
}

// Comparison for complex numbers in the case of NaNs.
// Will be re-used for comparison of real and imaginary components.
template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
testing::AssertionResult NumericalComparisonNaN(const char* blis_sol_char,
                                             const char* ref_sol_char,
                                             const char* comp_helper_char,
                                             const T blis_sol,
                                             const T ref_sol,
                                             const ComparisonHelper comp_helper,
                                             const ComplexPart complex_part,
                                             const std::string error_message)
{
    // Assign values to intermediate variables as if we are comparing the real part.
    RT ref_sol_1 = ref_sol.real, ref_sol_2 = ref_sol.imag, blis_sol_1 = blis_sol.real, blis_sol_2 = blis_sol.imag;
    // if we are comparing based on the imaginary part update the values.
    if (complex_part == IMAGINARY)
    {
        ref_sol_2 = ref_sol.real;
        ref_sol_1 = ref_sol.imag;
        blis_sol_2 = blis_sol.real;
        blis_sol_1 = blis_sol.imag;
    }
    // Check if the both parts are NaNs.
    if ((std::isnan(ref_sol_1)) && (std::isnan(blis_sol_1)))
        // Check second part for equality based on real NaN/Inf comparison.
        return NumericalComparisonRealNaNInf<RT>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol_2, ref_sol_2, comp_helper, error_message);
    // if only one of the first parts is NaN
    return testing::AssertionFailure() << error_message;
}

// Comparison for complex numbers in the case of Infs.
// Will be re-used for comparison of real and imaginary components.
template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
testing::AssertionResult NumericalComparisonInf(const char* blis_sol_char,
                                             const char* ref_sol_char,
                                             const char* comp_helper_char,
                                             const T blis_sol,
                                             const T ref_sol,
                                             const ComparisonHelper comp_helper,
                                             const ComplexPart complex_part,
                                             const std::string error_message)
{
    // Assign values to intermediate variables as if we are comparing the real part.
    RT ref_sol_1 = ref_sol.real, ref_sol_2 = ref_sol.imag, blis_sol_1 = blis_sol.real, blis_sol_2 = blis_sol.imag;
    // if we are comparing based on the imaginary part update the values.
    if (complex_part == IMAGINARY)
    {
        ref_sol_2 = ref_sol.real;
        ref_sol_1 = ref_sol.imag;
        blis_sol_2 = blis_sol.real;
        blis_sol_1 = blis_sol.imag;
    }
    // check if both of the first parts are inf
    if ((std::isinf(ref_sol_1)) && (std::isinf(blis_sol_1)))
    {
        // check the sign of infs
        if( ref_sol_1 == blis_sol_1 )
            // Check second part for equality based on real NaN/Inf comparison.
            return NumericalComparisonRealNaNInf<RT>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol_2, ref_sol_2, comp_helper, error_message);
        // if both are infs but have different signs, return FAILURE.
        else return testing::AssertionFailure() << error_message;
    }
    // if only one of them is Inf
    return testing::AssertionFailure() << error_message;
}

// Comparisons that take into account the presence of NaNs and Infs:
template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
testing::AssertionResult NumericalComparison(const char* blis_sol_char,
                                             const char* ref_sol_char,
                                             const char* comp_helper_char,
                                             const T blis_sol,
                                             const T ref_sol,
                                             const ComparisonHelper comp_helper)
{
    // Base error message used for scalar values
    std::string error_message = blis_sol_char;
                error_message += " = " + testinghelpers::to_string(blis_sol) + ",   ";
                error_message += ref_sol_char;
                error_message += " = " + testinghelpers::to_string(ref_sol);
    // If we are comparing a vector, update error message to include the current index
    if(comp_helper.object_type == VECTOR)
        error_message += ", i = " + std::to_string(comp_helper.i);
    // If we are comparing a matrix, update error message to include the current indices
    else if(comp_helper.object_type == MATRIX)
        error_message += ", i = " + std::to_string(comp_helper.i) + ", j = " + std::to_string(comp_helper.j);

    // Check if NaN/Inf comparison is necessary and if so, proceed.
    // Otherwise, call numerical comparison only, without considering NaNs and Infs.
    if (comp_helper.nan_inf_check)
    {
        if constexpr (testinghelpers::type_info<T>::is_real)
            return NumericalComparisonRealNaNInf<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, error_message);
        // If it's complex we need to check real and imaginary parts.
        else
        {
            // Check if any of the real parts is NaN, and if so, call into NaN comparator.
            if ((std::isnan(ref_sol.real)) || (std::isnan(blis_sol.real)))
            {
                ComplexPart complex_part = REAL;
                return NumericalComparisonNaN<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, complex_part, error_message);
            }
            // Check if any of the imag parts is NaN, and if so, call into NaN comparator.
            else if ((std::isnan(ref_sol.imag)) || (std::isnan(blis_sol.imag)))
            {
                ComplexPart complex_part = IMAGINARY;
                return NumericalComparisonNaN<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, complex_part, error_message);
            }
            // Check if any of the real parts is Inf, and if so, call into Inf comparator.
            else if ((std::isinf(ref_sol.real)) || (std::isinf(blis_sol.real)))
            {
                ComplexPart complex_part = REAL;
                return NumericalComparisonInf<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, complex_part, error_message);
            }
            // Check if any of the imag parts is NaN or Inf, and if so, call into Inf comparator.
            else if ((std::isinf(ref_sol.imag)) || (std::isinf(blis_sol.imag)))
            {
                ComplexPart complex_part = IMAGINARY;
                return NumericalComparisonInf<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, complex_part, error_message);
            }
            // If neither reference nor BLIS sol is NaN or Inf, or if NaN/Inf checks are not necessary,
            // do simple comparison, based on relative or absolute error.
            else
                return NumericalComparisonFPOnly<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, error_message);
        }
    }
    // If NaN/Inf checks are not necessary, do simple comparison, based on relative or absolute error.
    else
        return NumericalComparisonFPOnly<T>(blis_sol_char, ref_sol_char, comp_helper_char, blis_sol, ref_sol, comp_helper, error_message);
}

/**
 * Binary comparison of two scalars.
 */
template <typename T>
void computediff( T blis_sol, T ref_sol, bool nan_inf_check = false )
{
    ComparisonHelper comp_helper(SCALAR);
    comp_helper.binary_comparison = true;
    comp_helper.nan_inf_check = nan_inf_check;

    ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol, ref_sol, comp_helper);
}

/**
 * Relative comparison of two scalars, using a threshold.
 */
template <typename T>
void computediff( T blis_sol, T ref_sol, double thresh, bool nan_inf_check = false )
{
    ComparisonHelper comp_helper(SCALAR, thresh);    
    comp_helper.nan_inf_check = nan_inf_check;
    ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol, ref_sol, comp_helper);
}

/**
 * Binary comparison of two vectors with length n and increment inc.
 */
template <typename T>
void computediff( gtint_t n, T *blis_sol, T *ref_sol, gtint_t inc, bool nan_inf_check = false )
{
    gtint_t abs_inc = std::abs(inc);
    ComparisonHelper comp_helper(VECTOR);
    comp_helper.nan_inf_check = nan_inf_check;
    comp_helper.binary_comparison = true;

    // In case inc is negative in a call to BLIS APIs, we just access it from the end to the beginning,
    // so practically nothing changes. Access from beginning to end to optimize memory operations.
    for (gtint_t i = 0; i < n; i++)
    {
        comp_helper.i = i;
        ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*abs_inc], ref_sol[i*abs_inc], comp_helper) << "inc = " << inc;
        // Go through elements that are part of the array that should not have been modified by the
        // call to a BLIS API. Use the bitwise comparison for this case.
        if (i < n-1)
        {
            for (gtint_t j = 1; j < abs_inc; j++)
            {
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*abs_inc + j], ref_sol[i*abs_inc + j], comp_helper) << "inc = " << inc << " This element is expected to not be modified.";
            }
        }
    }
}

/**
 * Relative comparison of two vectors with length n and increment inc.
 */
template <typename T>
void computediff( gtint_t n, T *blis_sol, T *ref_sol, gtint_t inc, double thresh, bool nan_inf_check = false )
{
    gtint_t abs_inc = std::abs(inc);
    ComparisonHelper comp_helper(VECTOR, thresh);
    comp_helper.nan_inf_check = nan_inf_check;

    // In case inc is negative in a call to BLIS APIs, we just access it from the end to the beginning,
    // so practically nothing changes. Access from beginning to end to optimize memory operations.
    for (gtint_t i = 0; i < n; i++)
    {
        comp_helper.i = i;
        ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*abs_inc], ref_sol[i*abs_inc], comp_helper) << "inc = " << inc;
        // Go through elements that are part of the array that should not have been modified by the
        // call to a BLIS API. Use the bitwise comparison for this case.
        if (i < n-1)
        {
            for (gtint_t j = 1; j < abs_inc; j++)
            {
                comp_helper.binary_comparison = true;
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*abs_inc + j], ref_sol[i*abs_inc + j], comp_helper) << "inc = " << inc << " This element is expected to not be modified.";
            }
            comp_helper.binary_comparison = false;
        }
    }
}

/**
 * Binary comparison of two matrices with dimensions m-by-n and leading dimension ld.
 */
template <typename T>
void computediff(char storage, gtint_t m, gtint_t n, T *blis_sol, T *ref_sol, gtint_t ld, bool nan_inf_check = false )
{
    gtint_t i,j;
    ComparisonHelper comp_helper(MATRIX);
    comp_helper.nan_inf_check = nan_inf_check;
    comp_helper.binary_comparison = true;
    // Loop for column-major order
    if( (storage == 'c') || (storage == 'C') )
    {
        for( j = 0 ; j < n ; j++ )
        {
            // First iterate through the elements of the arrays that are part of the matrix
            // and are expected to be modified by a call to BLIS APIs.
            for( i = 0 ; i < m ; i++ )
            {
                comp_helper.i = i;
                comp_helper.j = j;
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i + j*ld], ref_sol[i + j*ld], comp_helper);
            }
            // Now iterate through the rest of elements in memory space that are not part of the matrix,
            // so we use binary comparison to verify that are exactly the same as the reference.
            // Since to get create the data we use a copy to initialize the reference results, those
            // elements are expected to identical.
            for (i = m; i < ld; i++)
            {
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i + j*ld], ref_sol[i + j*ld], comp_helper) << "This element is expected to not be modified.";
            }
        }
    }
    // Loop for row-major order
    else
    {
        for( i = 0 ; i < m ; i++ )
        {
            // First iterate through the elements of the arrays that are part of the matrix
            // and are expected to be modified by a call to BLIS APIs.
            for( j = 0 ; j < n ; j++ )
            {
                comp_helper.i = i;
                comp_helper.j = j;
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*ld + j], ref_sol[i*ld + j], comp_helper);
            }
            // Now iterate through the rest of elements in memory space that are not part of the matrix,
            // so we use binary comparison to verify that are exactly the same as the reference.
            // Since to get create the data we use a copy to initialize the reference results, those
            // elements are expected to identical.
            for (j = n; j < ld; j++)
            {
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*ld + j], ref_sol[i*ld + j], comp_helper) << "This element is expected to not be modified.";
            }
        }
    }
}

/**
 * Relative comparison of two matrices with dimensions m-by-n and leading dimension ld.
 */
template <typename T>
void computediff(char storage, gtint_t m, gtint_t n, T *blis_sol, T *ref_sol, gtint_t ld, double thresh, bool nan_inf_check = false )
{
    gtint_t i,j;
    ComparisonHelper comp_helper(MATRIX, thresh);
    comp_helper.nan_inf_check = nan_inf_check;

    // Loop for column-major order
    if( (storage == 'c') || (storage == 'C') )
    {
        for( j = 0 ; j < n ; j++ )
        {
            // First iterate through the elements of the arrays that are part of the matrix
            // and are expected to be modified by a call to BLIS APIs.
            for( i = 0 ; i < m ; i++ )
            {
                comp_helper.i = i;
                comp_helper.j = j;
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i + j*ld], ref_sol[i + j*ld], comp_helper);
            }
            // Now iterate through the rest of elements in memory space that are not part of the matrix,
            // so we use binary comparison to verify that are exactly the same as the reference.
            // Since to get create the data we use a copy to initialize the reference results, those
            // elements are expected to identical.
            comp_helper.binary_comparison = true;
            for (i = m; i < ld; i++)
            {
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i + j*ld], ref_sol[i + j*ld], comp_helper) << "This element is expected to not be modified.";
            }
            // Disable binary comparison before we go through the next column.
            comp_helper.binary_comparison = false;
        }
    }
    // Loop for row-major order
    else
    {
        for( i = 0 ; i < m ; i++ )
        {
            // First iterate through the elements of the arrays that are part of the matrix
            // and are expected to be modified by a call to BLIS APIs.
            for( j = 0 ; j < n ; j++ )
            {
                comp_helper.i = i;
                comp_helper.j = j;
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*ld + j], ref_sol[i*ld + j], comp_helper);
            }
            // Now iterate through the rest of elements in memory space that are not part of the matrix,
            // so we use binary comparison to verify that are exactly the same as the reference.
            // Since to get create the data we use a copy to initialize the reference results, those
            // elements are expected to identical.
            comp_helper.binary_comparison = true;
            for (j = n; j < ld; j++)
            {
                ASSERT_PRED_FORMAT3(NumericalComparison<T>, blis_sol[i*ld + j], ref_sol[i*ld + j], comp_helper) << "This element is expected to not be modified.";
            }
            // Disable binary comparison before we go through the next column.
            comp_helper.binary_comparison = false;
        }
    }
}
