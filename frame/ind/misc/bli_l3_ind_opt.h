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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#ifndef BLIS_L3_IND_OPT_H
#define BLIS_L3_IND_OPT_H

#define bli_l3_ind_recast_1m_params( dt_exec, schema_a, c, \
                                     m, n, k, \
                                     pd_a, ps_a, \
                                     pd_b, ps_b, \
                                     rs_c, cs_c ) \
{ \
	obj_t beta; \
\
	/* Detach the beta scalar from c so that we can test its imaginary
	   component. */ \
	bli_obj_scalar_detach( c, &beta ); \
\
	/* If beta is in the real domain, and c is row- or column-stored,
	   then we may proceed with the optimization. */ \
	if ( bli_obj_imag_equals( &beta, &BLIS_ZERO ) && \
	     !bli_is_gen_stored( rs_c, cs_c ) ) \
	{ \
		dt_exec = bli_datatype_proj_to_real( dt_exec ); \
\
		if ( bli_is_1e_packed( schema_a ) ) \
		{ \
			m    *= 2; \
			n    *= 1; \
			k    *= 2; \
			pd_a *= 2; ps_a *= 2; \
			pd_b *= 1; ps_b *= 2; \
			rs_c *= 1; cs_c *= 2; \
		} \
		else /* if ( bli_is_1r_packed( schema_a ) ) */ \
		{ \
			m    *= 1; \
			n    *= 2; \
			k    *= 2; \
			pd_a *= 1; ps_a *= 2; \
			pd_b *= 2; ps_b *= 2; \
			rs_c *= 2; cs_c *= 1; \
		} \
	} \
}

#endif
