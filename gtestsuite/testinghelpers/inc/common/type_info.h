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

// Set the integer type that we use in testing depending on the CMake option.
#if INT_SIZE == 32
    using gtint_t = int32_t;
    using ugtint_t = uint32_t;
#elif INT_SIZE == 64
    using gtint_t = int64_t;
    using ugtint_t = uint64_t;
#endif

namespace testinghelpers {
    // type_info<T>::real_type will return the real type of T.
    // If T is float or double, real_type is float or double respectivelly.
    // If T is scomplex or dcomplex, real_type is float or double respectivelly.
    template<typename T>
    struct type_info {
        using real_type = T;
        static constexpr bool is_complex = false;
        static constexpr bool is_real = true;
    };

    template<>
    struct type_info<scomplex> {
        using real_type = float;
        static constexpr bool is_complex = true;
        static constexpr bool is_real = false;
    };

    template<>
    struct type_info<dcomplex> {
        using real_type = double;
        static constexpr bool is_complex = true;
        static constexpr bool is_real = false;
    };
} //end of namespace testinghelpers
