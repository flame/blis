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

#include "blis.h"

#if !defined (BLIS_ENABLE_MULTITHREADING) || defined (BLIS_ENABLE_PTHREADS)

void bli_l3_compute_thread_decorator
     (
       l3computeint_t func,
       opid_t         family,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       cntx_t*        cntx,
       rntm_t*        rntm
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);

    const dim_t n_threads = 1;
    array_t* restrict array = bli_sba_checkout_array( n_threads );
    bli_sba_rntm_set_pool( 0, array, rntm );
    bli_pba_rntm_set_pba( rntm );

    {
        rntm_t* restrict rntm_p = rntm;
        const dim_t tid = 0;

        // This optimization allows us to use one of the global thrinfo_t
        // objects for single-threaded execution rather than grow one from
        // scratch. The key is that bli_thrinfo_sup_grow(), which is called
        // from within the variants, will immediately return if it detects
        // that the thrinfo_t* passed into it is either
        // &BLIS_GEMM_SINGLE_THREADED or &BLIS_PACKM_SINGLE_THREADED.
        thrinfo_t* thread = &BLIS_GEMM_SINGLE_THREADED;

        ( void )tid;

        func
        (
          a,
          b,
          beta,
          c,
          cntx,
          rntm_p,
          thread
        );
    }

    bli_sba_checkin_array( array );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);

}

#endif
