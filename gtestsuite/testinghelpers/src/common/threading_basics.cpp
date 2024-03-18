/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "common/testing_basics.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void vary_num_threads()
{
	// In nested parallelism testing, vary number of threads depending
	// on thread id in enclosing parallel region. Select which method
	// to use based on existing choice of OMP_NUM_THREADS, BLIS_NUM_THREADS
	// or BLIS_*_NT.

	bool nt_set   = false;
	bool ways_set = false;

#ifdef _OPENMP
	int me = omp_get_thread_num();
#else
        int me = 0;
#endif

	// Try to read BLIS_NUM_THREADS first.
	gtint_t nt = bli_env_get_var( "BLIS_NUM_THREADS", -1 );

	// If BLIS_NUM_THREADS was not set, try to read BLIS_NT.
	if ( nt == -1 ) nt = bli_env_get_var( "BLIS_NT", -1 );

	// Read the environment variables for the number of threads (ways of
	// parallelism) for each individual loop.
	gtint_t jc = bli_env_get_var( "BLIS_JC_NT", -1 );
	gtint_t pc = bli_env_get_var( "BLIS_PC_NT", -1 );
	gtint_t ic = bli_env_get_var( "BLIS_IC_NT", -1 );
	gtint_t jr = bli_env_get_var( "BLIS_JR_NT", -1 );
	gtint_t ir = bli_env_get_var( "BLIS_IR_NT", -1 );

	if ( nt > 0 ) { nt_set   = true; }
	if ( jc > 0 ) { ways_set = true; }
	if ( pc > 0 ) { ways_set = true; }
	if ( ic > 0 ) { ways_set = true; }
	if ( jr > 0 ) { ways_set = true; }
	if ( ir > 0 ) { ways_set = true; }

	if ( ways_set == TRUE )
	{
		jc = 2*me+1;
		pc = 1;
		ic = me+1;
		jr = 1;
		ir = 1;

		bli_thread_set_ways( jc, pc, ic, jr, ir );
	}
	else if ( nt_set == TRUE )
	{
		bli_thread_set_num_threads(2*me+1);
	}
#ifdef _OPENMP
	else
	{
		omp_set_num_threads(2*me+1);
	}
#endif
}
