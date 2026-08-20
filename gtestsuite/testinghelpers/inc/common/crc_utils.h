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

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <gtest/gtest.h>
#include "common/testing_helpers.h"

namespace testinghelpers {
namespace crc_utils {

/*
 * CRC-32 polynomial 0xAD0424F3, chosen for good distribution across
 * floating-point data patterns in BLAS operations.
 * Reference: Philip Koopman, Carnegie Mellon University
 * https://users.ece.cmu.edu/~koopman/crc/
 */
inline constexpr uint32_t CRC32_POLYNOMIAL = 0xad0424f3;

enum CrcDataType {
    OUTPUT = 0,
    REFERENCE = 1
};

#ifdef ENABLE_CRC
inline constexpr bool CRC_ENABLED = true;
#else
inline constexpr bool CRC_ENABLED = false;
#endif

/**
 * @brief Number of uint32_t blocks needed to represent one element of type T.
 * Compile-time check ensures T is uint32_t-aligned.
 */
template<typename T>
constexpr gtint_t get_uint32_factor() {
    static_assert(sizeof(T) >= sizeof(uint32_t),
        "CRC requires types at least as wide as uint32_t");
    static_assert(sizeof(T) % sizeof(uint32_t) == 0,
        "CRC requires types whose size is a multiple of sizeof(uint32_t)");
    return static_cast<gtint_t>(sizeof(T) / sizeof(uint32_t));
}

/**
 * @brief Read a uint32_t from a byte buffer at the given index.
 * Uses memcpy to avoid strict-aliasing violations when the source
 * buffer type is float, double, or complex.
 */
inline uint32_t read_uint32(const void* base, size_t index) {
    uint32_t val;
    std::memcpy(&val,
                static_cast<const unsigned char*>(base) + index * sizeof(uint32_t),
                sizeof(uint32_t));
    return val;
}

/**
 * @brief Single-value binary division step for CRC-32 calculation.
 * Processes one uint32_t value against the running remainder using
 * bit-by-bit modulo-2 binary division.
 */
inline uint32_t binary_division_uint32(uint32_t a, uint32_t remainder, uint32_t polynomial)
{
    for(gtint_t z = 0; z < 32; z++)
    {
        if((a & 0x80000000) == 0x80000000)
        {
            remainder = remainder | 1;
        }
        if((remainder & 0x80000000) == 0x80000000)
        {
            remainder = remainder ^ polynomial;
        }
        a = a << 1;
        remainder = remainder << 1;
    }
    return remainder;
}

/**
 * @brief Generate CRC-32 checksum for a vector of m elements.
 *
 * The algorithm interprets the buffer as an array of uint32_t blocks
 * and performs bit-by-bit modulo-2 binary division with CRC32_POLYNOMIAL.
 */
template<typename T>
inline uint32_t generate_crc_vector(gtint_t m, const T* A)
{
    if (m <= 0 || A == nullptr) return 0;

    constexpr gtint_t mul = get_uint32_factor<T>();
    const uint32_t polynomial = CRC32_POLYNOMIAL;

    if (m * mul < 1) return 0;

    uint32_t remainder = read_uint32(A, 0);
    if((remainder & 0x80000000) == 0x80000000)
        remainder = remainder ^ polynomial;
    remainder = remainder << 1;

    for(gtint_t idx = 1; idx < (m * mul); idx++)
    {
        remainder = binary_division_uint32(read_uint32(A, idx), remainder, polynomial);
    }
    return remainder;
}

/**
 * @brief Generate CRC-32 checksum for a matrix with specified storage layout.
 *
 * For column-major ('c'/'C'), iterates columns (j) then rows (i).
 * For row-major, iterates rows (i) then columns (j).
 * Uses A(i,j) convention: i = row index, j = column index.
 */
template<typename T>
inline uint32_t generate_crc_matrix_with_storage(gtint_t m, gtint_t n, const T* A,
                                                 gtint_t lda, char storage)
{
    if (m <= 0 || n <= 0 || A == nullptr) return 0;

    constexpr gtint_t mul = get_uint32_factor<T>();
    const uint32_t polynomial = CRC32_POLYNOMIAL;

    uint32_t remainder = read_uint32(A, 0);
    if((remainder & 0x80000000) == 0x80000000)
        remainder = remainder ^ polynomial;
    remainder = remainder << 1;

    bool first = true;

    if((storage == 'c') || (storage == 'C'))
    {
        for(gtint_t j = 0; j < n; j++)
        {
            for(gtint_t i = 0; i < (m * mul); i++)
            {
                if(first)
                {
                    first = false;
                    continue;
                }
                remainder = binary_division_uint32(
                    read_uint32(A, (j * lda * mul) + i), remainder, polynomial);
            }
        }
    }
    else
    {
        for(gtint_t i = 0; i < m; i++)
        {
            for(gtint_t j = 0; j < (n * mul); j++)
            {
                if(first)
                {
                    first = false;
                    continue;
                }
                remainder = binary_division_uint32(
                    read_uint32(A, (i * lda * mul) + j), remainder, polynomial);
            }
        }
    }
    return remainder;
}

/**
 * @brief Format a CRC value as hex, print it to stdout, and record it as a GTest property.
 */
inline void format_and_record_crc(CrcDataType crc_type, uint32_t crc,
                                  const std::string& var_name) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(8) << crc;
    std::string crc_str = oss.str();

    const char* type_label = (crc_type == OUTPUT) ? "Output CRC    " : "Reference CRC ";
    if (!var_name.empty())
        std::cout << "             " << type_label << "[" << var_name << "]: " << crc_str << std::endl;
    else
        std::cout << "             " << type_label << ": " << crc_str << std::endl;

    std::string property_name = (crc_type == OUTPUT) ? "output_crc" : "reference_crc";
    if (!var_name.empty())
        property_name = var_name + "_" + property_name;
    ::testing::Test::RecordProperty(property_name, crc_str.c_str());
}

/**
 * @brief Calculate CRC of vector data, print it, and record as a GTest property.
 */
template<typename T>
void calculate_and_print_crc(CrcDataType crc_type, const T* data, size_t count,
                             const std::string& var_name = "") {
    if constexpr (CRC_ENABLED) {
        uint32_t crc = generate_crc_vector(static_cast<gtint_t>(count), data);
        format_and_record_crc(crc_type, crc, var_name);
    }
}

/**
 * @brief Calculate CRC of matrix data with storage layout, print it, and record as a GTest property.
 */
template<typename T>
void calculate_and_print_matrix_crc_with_storage(CrcDataType crc_type, const T* data,
                                                 size_t m, size_t n, size_t lda,
                                                 char storage,
                                                 const std::string& var_name = "") {
    if constexpr (CRC_ENABLED) {
        uint32_t crc = generate_crc_matrix_with_storage(
            static_cast<gtint_t>(m), static_cast<gtint_t>(n),
            data, static_cast<gtint_t>(lda), storage);
        format_and_record_crc(crc_type, crc, var_name);
    }
}

} // namespace crc_utils
} // namespace testinghelpers
