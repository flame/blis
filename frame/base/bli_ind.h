/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

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

#ifndef BLIS_IND_H
#define BLIS_IND_H

// level-3 induced method management
#include "bli_l3_ind.h"

void   bli_ind_init( void );
void   bli_ind_finalize( void );

BLIS_EXPORT_BLIS void    bli_ind_enable( ind_t method );
BLIS_EXPORT_BLIS void    bli_ind_disable( ind_t method );
BLIS_EXPORT_BLIS void    bli_ind_disable_all( void );

BLIS_EXPORT_BLIS void    bli_ind_enable_dt( ind_t method, num_t dt );
BLIS_EXPORT_BLIS void    bli_ind_disable_dt( ind_t method, num_t dt );
BLIS_EXPORT_BLIS void    bli_ind_disable_all_dt( num_t dt );

BLIS_EXPORT_BLIS void    bli_ind_oper_enable_only( opid_t oper, ind_t method, num_t dt );

BLIS_EXPORT_BLIS bool    bli_ind_oper_is_impl( opid_t oper, ind_t method );
BLIS_EXPORT_BLIS ind_t   bli_ind_oper_find_avail( opid_t oper, num_t dt );
BLIS_EXPORT_BLIS char*   bli_ind_oper_get_avail_impl_string( opid_t oper, num_t dt );

char*  bli_ind_get_impl_string( ind_t method );
num_t  bli_ind_map_cdt_to_index( num_t dt );


#endif

