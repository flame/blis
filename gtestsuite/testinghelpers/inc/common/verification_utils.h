/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#include "common/testing_basics.h"
#include "common/crc_utils.h"
#include "common/binary_output_utils.h"

namespace testinghelpers {
namespace verification {

/**
 * @brief Collect CRC checksums and binary output for vector (or scalar) data.
 *
 * Computes CRC-32 checksums for both BLIS and reference buffers and records
 * them as GTest properties. Optionally writes both buffers to binary files.
 * Scalars are treated as vectors of length 1.
 * This function does not compare the data; comparison is done separately.
 *
 * Each feature is compiled in only when the corresponding CMake option is ON:
 *   -DENABLE_CRC=ON           -> CRC checksums
 *   -DENABLE_BINARY_OUTPUT=ON -> binary file output
 */
template<typename T>
inline void collect_vector_data(const std::string& var_name,
                               const T* blis_data, const T* ref_data,
                               size_t total_elements) {
#ifdef ENABLE_CRC
    crc_utils::calculate_and_print_crc(
        crc_utils::OUTPUT, blis_data, total_elements, var_name);
    crc_utils::calculate_and_print_crc(
        crc_utils::REFERENCE, ref_data, total_elements, var_name);
#endif
#ifdef ENABLE_BINARY_OUTPUT
    binary_output_utils::write_comparison_outputs(
        var_name, blis_data, ref_data, total_elements,
        binary_output_utils::get_current_test_name());
#endif
#if !defined(ENABLE_CRC) && !defined(ENABLE_BINARY_OUTPUT)
    (void)var_name; (void)blis_data; (void)ref_data; (void)total_elements;
#endif
}

/**
 * @brief Collect CRC checksums and binary output for matrix data.
 *
 * Uses storage-aware CRC that respects column-major / row-major layout.
 * See collect_vector_data() for compile-time feature gating.
 */
template<typename T>
inline void collect_matrix_data(const std::string& var_name,
                               const T* blis_data, const T* ref_data,
                               size_t m, size_t n, size_t ld,
                               char storage) {
#ifdef ENABLE_CRC
    crc_utils::calculate_and_print_matrix_crc_with_storage(
        crc_utils::OUTPUT, blis_data, m, n, ld, storage, var_name);
    crc_utils::calculate_and_print_matrix_crc_with_storage(
        crc_utils::REFERENCE, ref_data, m, n, ld, storage, var_name);
#endif
#ifdef ENABLE_BINARY_OUTPUT
    size_t total_elements = matsize(storage, 'n', m, n, ld);
    binary_output_utils::write_comparison_outputs(
        var_name, blis_data, ref_data, total_elements,
        binary_output_utils::get_current_test_name());
#endif
#if !defined(ENABLE_CRC) && !defined(ENABLE_BINARY_OUTPUT)
    (void)var_name; (void)blis_data; (void)ref_data;
    (void)m; (void)n; (void)ld; (void)storage;
#endif
}

} // namespace verification
} // namespace testinghelpers
