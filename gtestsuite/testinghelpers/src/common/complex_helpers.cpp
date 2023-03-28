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
#include <complex>
#include "common/complex_helpers.h"

namespace std {
    // Overload std::abs to work with scomplex and dcomplex.
    float abs(const scomplex x)
    {
        std::complex<float> y{x.real, x.imag};
        return std::abs(y);
    }
    double abs(const dcomplex x)
    {
        std::complex<double> y{x.real, x.imag};
        return std::abs(y);
    }
    // Overload the stream operator to be able to print scomplex in error messages.
    ostream& operator<<(ostream& os, const scomplex& x)
    {
        os << "(" << x.real << ", " << x.imag <<")";
        return os;
    }
    ostream& operator<<(ostream& os, const dcomplex& x)
    {
        os << "(" << x.real << ", " << x.imag <<")";
        return os;
    }
}

// Operator overloading for scomplex and dcomplex types.
scomplex operator+(const scomplex x, const scomplex y)
{
    return scomplex{x.real+y.real, x.imag+y.imag};
}
dcomplex operator+(const dcomplex x, const dcomplex y)
{
    return dcomplex{x.real+y.real, x.imag+y.imag};
}

scomplex operator-(const scomplex x, const scomplex y)
{
    return scomplex{x.real-y.real, x.imag-y.imag};
}
dcomplex operator-(const dcomplex x, const dcomplex y)
{
    return dcomplex{x.real-y.real, x.imag-y.imag};
}

scomplex operator*(const scomplex x, const scomplex y)
{
    return scomplex{(( x.real * y.real ) - ( x.imag * y.imag )),(( x.real * y.imag ) + ( x.imag * y.real ))};
}
dcomplex operator*(const dcomplex x, const dcomplex y)
{
    return dcomplex{(( x.real * y.real ) - ( x.imag * y.imag )),(( x.real * y.imag ) + ( x.imag * y.real ))};
}

bool operator== (const scomplex x, const scomplex y)
{
    return {(x.real==y.real) && (x.imag==y.imag)};
}
bool operator== (const dcomplex x, const dcomplex y)
{
    return {(x.real==y.real) && (x.imag==y.imag)};
}

bool operator!= (const scomplex x, const scomplex y)
{
    return {!((x.real==y.real) && (x.imag==y.imag))};
}
bool operator!= (const dcomplex x, const dcomplex y)
{
    return {!((x.real==y.real) && (x.imag==y.imag))};
}
