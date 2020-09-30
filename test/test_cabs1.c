/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"

#include <time.h>

#define FLOAT

#ifdef BLIS_ENABLE_CBLAS
    #define CHECK_CBLAS // Macro to test cblas interface of the function cblas?cabs1
#endif

#ifdef CHECK_CBLAS
#include "cblas.h"
#endif

int main(int argc, char**argv)
{
  /* initialize random seed: */
  srand (time(NULL));

  int   r, n_repeats;
  n_repeats = 5;

#ifdef FLOAT

  float z_abs = 0.0f;

#else

  double z_abs = 0.0;

#endif

  for ( r = 0; r < n_repeats; ++r )
    {

#ifdef FLOAT

      float maxRandVal = 1000.0f;
      scomplex inp;
      inp.real = ((float)rand()/(float)(RAND_MAX)) * maxRandVal - maxRandVal/2;
      inp.imag = ((float)rand()/(float)(RAND_MAX)) * maxRandVal - maxRandVal/2;

#ifdef BLIS
      printf( "data_scabs1_BLIS");
#else
      printf( "data_scabs1_%s", BLAS );
#endif

#else

      double maxRandVal = 1000.0;
      dcomplex inp;
      inp.real = ((double)rand()/(double)(RAND_MAX)) * maxRandVal - maxRandVal/2;
      inp.imag = ((double)rand()/(double)(RAND_MAX)) * maxRandVal - maxRandVal/2;

#ifdef BLIS
      printf( "data_dcabs1_BLIS: ");
#else
      printf( "data_dcabs1_%s: ", BLAS );
#endif

#endif

#ifdef FLOAT

#ifdef CHECK_CBLAS
      z_abs = cblas_scabs1( &inp );
#else
      z_abs = scabs1_( &inp );
#endif

#else

#ifdef CHECK_CBLAS
      z_abs = cblas_dcabs1( &inp );
#else
      z_abs = dcabs1_( &inp );
#endif

#endif

      printf(" z = %lf%+lfi, cabs1(z) = %lf \n", inp.real, inp.imag, z_abs);
    }

  return 0;
}
