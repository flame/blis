/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include "common/type_info.h"

namespace testinghelpers {
    class ProtectedBuffer
    {
    private:
        static const size_t REDZONE_SIZE = 1;
        void* redzone_1   = nullptr;
        void* redzone_2   = nullptr;
        void* mem         = nullptr;
        bool is_mem_test  = false;

        /**
         * ==========================================================================
         * get_mem
         * returns a aligned or unaligned buffer of size "size"
         * ==========================================================================
         * @param[in] size         specifies the size of the buffer to be allocated.
         * @param[in] is_aligned   specifies if the buffer needs to be aligned or not.
         */
        static void* get_mem(dim_t, bool);

    public:
        void* greenzone_1 = nullptr;
        void* greenzone_2 = nullptr;

        ProtectedBuffer(dim_t size, bool is_aligned = false, bool is_mem_test = false);
        ~ProtectedBuffer();

        static void handle_mem_test_fail(int signal);

        /**
         * Adds signal handler for segmentation fault.
         */
        static void start_signal_handler();

        /**
         * Removes signal handler for segmentation fault.
         */
        static void stop_signal_handler();
    };
}
