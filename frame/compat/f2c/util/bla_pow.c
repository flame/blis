/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 1990 - 1997, AT&T, Lucent Technologies and Bellcore.
   Copyright (C) 2014, The University of Texas at Austin

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

#include "blis.h"

#ifdef BLIS_ENABLE_BLAS

double bla_pow_ri( const bla_real *ap, const bla_integer *bp )
{
	return pow( ( bla_double )*ap, ( bla_double )*bp );
}

double bla_pow_di( const bla_double *ap, const bla_integer *bp )
{
	return pow( ( bla_double )*ap, ( bla_double )*bp );
}

double bla_pow_dd( const bla_double *ap, const bla_double *bp )
{
	return pow( ( bla_double )*ap, ( bla_double )*bp );
}

void bla_pow_ci( bla_scomplex *p, const bla_scomplex *a, const bla_integer *b )
{
	bla_dcomplex p1, a1;

	a1.real = a->real;
	a1.imag = a->imag;

	bla_pow_zi( &p1, &a1, b );

	p->real = p1.real;
	p->imag = p1.imag;
}

void bla_pow_zz( bla_dcomplex *r, const bla_dcomplex *a, const bla_dcomplex *b )
{
	double logr, logi, x, y;

	logr = log( hypot( a->real, a->imag ) );
	logi = atan2( a->imag, a->real );

	x = exp( logr * b->real - logi * b->imag );
	y = logr * b->imag + logi * b->real;

	r->real = x * cos(y);
	r->imag = x * sin(y);
}

void bla_pow_zi( bla_dcomplex *p, const bla_dcomplex *a, const bla_integer *b )
{
	bla_integer n;
	unsigned long u;
	bla_double t;
	bla_dcomplex q, x;
	bla_dcomplex one = { 1.0, 0.0 };

	n = *b;
	q.real = 1.0;
	q.imag = 0.0;

	if ( n == 0 )
		goto done;
	if ( n < 0 )
	{
		n = -n;
		bla_z_div( &x, &one, a );
	}
	else
	{
		x.real = a->real;
		x.imag = a->imag;
	}

	for ( u = n; ; )
	{
		if ( u & 01 )
		{
			t = q.real * x.real - q.imag * x.imag;
			q.imag = q.real * x.imag + q.imag * x.real;
			q.real = t;
		}
		if ( u >>= 1 )
		{
			t = x.real * x.real - x.imag * x.imag;
			x.imag = 2.0 * x.real * x.imag;
			x.real = t;
		}
		else break;
	}
	done:
	p->imag = q.imag;
	p->real = q.real;
}

#endif

