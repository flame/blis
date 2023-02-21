#include "common/complex_helpers.h"

namespace std {
    // Overload std::abs to work with scomplex and dcomplex.
    float abs(const scomplex x)
    {
        return sqrt(x.real*x.real + x.imag*x.imag);
    }
    double abs(const dcomplex x)
    {
        return sqrt(x.real*x.real + x.imag*x.imag);
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
