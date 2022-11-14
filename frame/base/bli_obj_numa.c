/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

//
// UPDATE November 2022 - Leick Robinson
//
// To improve performance under NUMA architectures, added new versions of the
// object creation functions to facilitate the creation of objects with
// interleaved memory. Those new functions are:
//
// - bli_obj_create_interleaved(),
// - bli_obj_create_conf_to_interleaved()
//
// Two new helper functions were also added:
//
// - bli_obj_interleave_buffer()
// - bli_interleave_mem()
//


void bli_obj_create_interleaved
     (
       num_t  dt,
       dim_t  m,
       dim_t  n,
       inc_t  rs,
       inc_t  cs,
       obj_t* obj
     )
{
	// Create an object with its memory interleaved across the NUMA regions.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_obj_create_interleaved(): " );
	#endif

	bli_obj_create( dt, m, n, rs, cs, obj );

#ifdef BLIS_ENABLE_OPENMP
	bli_obj_interleave_buffer( rs, cs, 1, obj );
#endif
}


void bli_obj_create_conf_to_interleaved
     (
       obj_t* s,
       obj_t* d
     )
{
	// Create an object that is conformal to an existing object while also
	// having its memory interleaved across the NUMA regions.

	const num_t dt = bli_obj_dt( s );
	const dim_t m  = bli_obj_length( s );
	const dim_t n  = bli_obj_width( s );
	const inc_t rs = bli_obj_row_stride( s );
	const inc_t cs = bli_obj_col_stride( s );

	bli_obj_create_interleaved( dt, m, n, rs, cs, d );
}


#ifdef BLIS_ENABLE_OPENMP

//
// UPDATE November 2022 - Leick Robinson
//
// Touch the object's buffer in a pattern that will result in the memory being
// interleaved across the NUMA zones.
// This utilizes the Linux first-touch policy.
// Note: Currently only implemented for 1 or 2 NUMA zones.
//       If 2 NUMA zones, assumes that the first and last thread are in
//       different zones.
//
void bli_obj_interleave_buffer
     (
       inc_t  rs,
       inc_t  cs,
       inc_t  is,
       obj_t* obj
     )
{
	dim_t  n_elem = 0;
	dim_t  m, n;
	siz_t  elem_size;
	siz_t  buffer_size;
	void*  p;

	// Query the dimensions of the object
	m = bli_obj_length( obj );
	n = bli_obj_width( obj );

	// Query the size of one element.
	elem_size = bli_obj_elem_size( obj );

	// Adjust the strides, if needed, before doing anything else
	bli_adjust_strides( m, n, elem_size, &rs, &cs, &is );

	// Determine the size of the buffer
	if ( m == 0 || n == 0 )
	{
		// For empty objects, set n_elem to zero. Row and column strides
		// should remain unchanged (because alignment is not needed).
		n_elem = 0;
	}
	else
	{
		// The number of elements to allocate is given by the distance from
		// the element with the lowest address (usually {0, 0}) to the element
		// with the highest address (usually {m-1, n-1}), plus one for the
		// highest element itself.
		n_elem = (m-1) * bli_abs( rs ) + (n-1) * bli_abs( cs ) + 1;
	}

	// Handle the special case where imaginary stride is larger than
	// normal.
	if ( bli_obj_is_complex( obj ) )
	{
		// Notice that adding is/2 works regardless of whether the
		// imaginary stride is unit, something between unit and
		// 2*n_elem, or something bigger than 2*n_elem.
		n_elem = bli_abs( is ) / 2 + n_elem;
	}

	// Compute the size of the total buffer to be allocated, which includes
	// padding if the leading dimension was increased for alignment purposes.
	buffer_size = ( siz_t )n_elem * elem_size;

	p = bli_obj_buffer( obj );

	// Interleave the buffer
	bli_interleave_mem( p, buffer_size );
}


void bli_interleave_mem
     (
       void* buf,
       siz_t buf_size
     )
{
	const size_t min_page_size = 4096;

	// Find the maximum page size that could encompass this data.
	size_t max_page_size = 1;
	while ( buf_size > max_page_size )
	{
		if ( ( max_page_size << 1 ) == 0 ) { break; }
		max_page_size = max_page_size << 1;
	}

	// Try to get the number of threads that BLIS will be using.
	rntm_t dummy_rntm;
	bli_rntm_init_from_global( &dummy_rntm );

	dim_t jc = bli_rntm_jc_ways( &dummy_rntm );
	dim_t pc = bli_rntm_pc_ways( &dummy_rntm );
	dim_t ic = bli_rntm_ic_ways( &dummy_rntm );
	dim_t jr = bli_rntm_jr_ways( &dummy_rntm );
	dim_t ir = bli_rntm_ir_ways( &dummy_rntm );

	dim_t nt = jc * pc * ic * jr * ir;

	//printf( "rntm nt %d jc %d pc %d ic %d jr %d ir %d\n",
	//        nt, jc, pc, ic, jr, ir );

	// Use all threads.
	_Pragma( "omp parallel" )
	{
		char* charbuf = ( char* )buf;
		char* buf_end = charbuf + buf_size;

		dim_t max_nt = omp_get_num_threads();
		dim_t tid    = omp_get_thread_num();

		// Check to see if we got a "sane" value for the number of threads to be
		// used by BLIS.
		if ( nt <= 0 )
		{
			// If the number of threads in the rntm_t is zero or negative, we'll
			// use the max number of threads obtained via omp instead.
			nt = max_nt;
		}

		size_t max_align_mask = max_page_size - 1;
		char* prev_max_aligned = ( char* )(    ( uintptr_t )charbuf &
		                                    ~( ( uintptr_t )max_align_mask )
		                                  );

		size_t stride = max_page_size;
		size_t start  = 0;

		#if 0
		if ( tid == 0 )
		{
			printf( "make_interleaved: buf %p prev_max_aligned %p\n",
			        charbuf, prev_max_aligned );
			printf( "nt = %d\n", nt );
		}
		#endif

		for ( size_t sz_b = max_page_size; sz_b >= min_page_size;
		             sz_b = sz_b / 2 )
		{
			if ( tid == 0 )
			{
				// Thread in first NUMA zone.
				for ( char* p = prev_max_aligned + start; p < buf_end;
				            p += stride )
				{
					if ( p >= charbuf ) { *p = 'a'; }
				}

				stride = sz_b;
				start  = sz_b / 2;
			}
			else if ( tid == nt - 1 )
			{
				// Thread on second NUMA zone.
				for ( char* p = prev_max_aligned + start - 1; p < buf_end;
				            p += stride)
				{
					if ( p >= charbuf ) { *p = 'b'; }
				}

				stride = sz_b;
				start  = sz_b / 2;
			}

			// Sync point.
			_Pragma( "omp barrier" )
		}
	}
}

#endif

