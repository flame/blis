/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

// -- gemmt specific function
bool bli_cntx_gemmtsup_thresh_is_met_zen( obj_t* a, obj_t* b, obj_t* c, cntx_t* cntx )
{
    num_t       dt          =   bli_obj_dt( c );
    dim_t       n           =   bli_obj_length( c );
    dim_t       k           =   bli_obj_width_after_trans( a );
    rntm_t rntm;

    bli_rntm_init_from_global( &rntm );

    // Query the number of threads from rntm object.
    const dim_t n_threads   =   bli_rntm_num_threads( &rntm );

    if( bli_is_double( dt ))
    {
        if( n_threads == 16)
        {
            /*Push sizes for n<1200 into SUP path*/
            if ( n < 1200 )                 return TRUE;
            /*For 1200<n<1600 and n/k <5 SUP is performing better than Native
              n/k >5 , With packing , Native path performance is better */
            if ( n < 1600 && (n / k) < 5)   return TRUE;
        }
        else
        {
            if ( n < 800 )       return TRUE;
            if ( (k / n ) > 50 ) return TRUE;
        }
        return FALSE;
    }
    else if ( bli_is_dcomplex( dt ) )
    {
        if ( n < 100 )   return TRUE;
        else             return FALSE;
    }
    else
        return bli_cntx_l3_sup_thresh_is_met( a, b, c, cntx );
}

// -- syrk specific function
bool bli_cntx_syrksup_thresh_is_met_zen( obj_t* a, obj_t* b, obj_t* c, cntx_t* cntx )
{
    num_t dt = bli_obj_dt( c );

    dim_t n = bli_obj_length( c );
    dim_t k = bli_obj_width_after_trans( a );

    stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );

    if( bli_is_double( dt ) )
    {
        if( ( stor_id == BLIS_RRC ) || ( stor_id == BLIS_CCR ) )
        {
            if( n < 140) return TRUE;
            else if( ( n < 200 ) && ( k < 100 ) ) return TRUE;
            else if( ( n <= 450 ) && ( k < 50 ) ) return TRUE;
            else return FALSE;
        }
        else
        {
            if( n <= 432 ) return TRUE;
            else return FALSE;
        }
    }
    else
        return bli_cntx_l3_sup_thresh_is_met( a, b, c, cntx );
}
