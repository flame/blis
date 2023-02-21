#pragma once

#include <iostream>
#include "common/type_info.h"

namespace std {
    // Overload std::abs to work with scomplex and dcomplex.
    float abs(const scomplex x);
    double abs(const dcomplex x);
    // Overload the stream operator to be able to print scomplex in error messages.
    ostream& operator<<(ostream& os, const scomplex& x);
    ostream& operator<<(ostream& os, const dcomplex& x);
}

// Operator overloading for scomplex and dcomplex types.
scomplex operator+(const scomplex x, const scomplex y);
dcomplex operator+(const dcomplex x, const dcomplex y);

scomplex operator-(const scomplex x, const scomplex y);
dcomplex operator-(const dcomplex x, const dcomplex y);

scomplex operator*(const scomplex x, const scomplex y);
dcomplex operator*(const dcomplex x, const dcomplex y);

bool operator== (const scomplex x, const scomplex y);
bool operator== (const dcomplex x, const dcomplex y);

bool operator!= (const scomplex x, const scomplex y);
bool operator!= (const dcomplex x, const dcomplex y);