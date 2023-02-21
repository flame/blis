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