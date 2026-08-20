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

#include <fstream>
#include <filesystem>
#include <algorithm>
#include <string>
#include <iostream>
#include <gtest/gtest.h>

namespace testinghelpers {
namespace binary_output_utils {

inline constexpr size_t MAX_OUTPUT_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB
inline constexpr const char* DEFAULT_OUTPUT_DIR = "blis_test_outputs";

#ifdef ENABLE_BINARY_OUTPUT
inline constexpr bool BINARY_OUTPUT_ENABLED = true;
#else
inline constexpr bool BINARY_OUTPUT_ENABLED = false;
#endif

/**
 * @brief Get current test name from Google Test framework.
 */
inline std::string get_current_test_name() {
    const ::testing::TestInfo* const test_info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    if (test_info) {
        std::string test_name = test_info->test_suite_name();
        test_name += "_" + std::string(test_info->name());
        return test_name;
    }
    return "unknown_test";
}

inline bool ensure_output_directory(const std::string& dir_path) {
    try {
        std::filesystem::create_directories(dir_path);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Could not create directory " << dir_path
                  << ": " << e.what() << std::endl;
        return false;
    }
}

inline std::string generate_binary_filename(const std::string& var_name,
                                            const std::string& test_name,
                                            const std::string& suffix = "") {
    std::string safe_test_name = test_name;
    const std::string invalid_chars = "/:*?\"<>|\\";
    for (char c : invalid_chars) {
        std::replace(safe_test_name.begin(), safe_test_name.end(), c, '_');
    }

    std::string filename = safe_test_name + "_" + var_name;
    if (!suffix.empty()) {
        filename += suffix;
    }
    filename += ".bin";
    return filename;
}

inline bool check_disk_space(const std::string& file_path, size_t required_size) {
    try {
        auto space_info = std::filesystem::space(
            std::filesystem::path(file_path).parent_path());
        return space_info.available >= required_size;
    } catch (const std::exception&) {
        return true;
    }
}

/**
 * @brief Write binary data to file. Only runs when compiled with -DENABLE_BINARY_OUTPUT=ON.
 * @return true if write was successful or output is disabled, false on error.
 */
template<typename T>
bool write_binary_output(const std::string& var_name, const T* data, size_t count,
                         const std::string& test_name, const std::string& suffix = "",
                         const std::string& output_dir = DEFAULT_OUTPUT_DIR) {
    if (!BINARY_OUTPUT_ENABLED) {
        return true;
    }

    if (count > SIZE_MAX / sizeof(T)) {
        std::cerr << "Warning: Integer overflow computing binary output size for "
                  << var_name << ". Skipping." << std::endl;
        return false;
    }

    size_t total_size = count * sizeof(T);

    if (total_size > MAX_OUTPUT_SIZE_BYTES) {
        std::cerr << "Warning: Requested binary output size (" << total_size
                  << " bytes) exceeds maximum allowed (" << MAX_OUTPUT_SIZE_BYTES
                  << " bytes). Skipping binary output for " << var_name << "." << std::endl;
        return false;
    }

    if (!ensure_output_directory(output_dir)) {
        return false;
    }

    std::filesystem::path filepath =
        std::filesystem::path(output_dir) / generate_binary_filename(var_name, test_name, suffix);

    if (!check_disk_space(filepath.string(), total_size)) {
        std::cerr << "Warning: Insufficient disk space for binary output. Required: "
                  << total_size << " bytes. Skipping binary output for " << var_name << "." << std::endl;
        return false;
    }

    try {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return false;
        }

        file.write(reinterpret_cast<const char*>(data), total_size);

        if (!file.good()) {
            std::cerr << "Error: Failed to write data to file: " << filepath << std::endl;
            return false;
        }

        file.close();
        std::cout << "Binary output written to: " << filepath
                  << " (" << total_size << " bytes)" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error: Exception while writing binary output to "
                  << filepath << ": " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Write both BLIS and reference outputs for comparison.
 *
 * Write failures are recorded as GTest properties so they appear in JSON
 * output, but do not cause the test to fail -- binary dumps are a
 * diagnostic aid and I/O errors must not affect test verdicts.
 */
template<typename T>
void write_comparison_outputs(const std::string& var_name,
                              const T* blis_data, const T* ref_data,
                              size_t count,
                              const std::string& test_name,
                              const std::string& output_dir = DEFAULT_OUTPUT_DIR) {
    bool blis_ok = write_binary_output(var_name, blis_data, count, test_name, "_blis", output_dir);
    bool ref_ok  = write_binary_output(var_name, ref_data, count, test_name, "_ref", output_dir);
    if (!blis_ok || !ref_ok) {
        std::string err;
        if (!blis_ok) err += "blis_write_failed";
        if (!blis_ok && !ref_ok) err += " ";
        if (!ref_ok)  err += "ref_write_failed";
        std::cout << "Warning: binary output write failed for " << var_name
                  << " (" << err << ")" << std::endl;
        ::testing::Test::RecordProperty(var_name + "_binary_output_error", err);
    }
}

} // namespace binary_output_utils
} // namespace testinghelpers
