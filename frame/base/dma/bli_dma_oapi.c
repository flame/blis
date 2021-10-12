/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, Kalray Inc.

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

#ifdef BLIS_ENABLE_DMA

static void bli_dma_get_check
     (
       obj_t*  a,
       obj_t*  p
     )
{
	err_t e_val;

	// Check object datatypes.
	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( p );
	bli_check_error_code( e_val );

	// Check object dimensions.
	e_val = bli_check_conformal_dims( a, p );
	bli_check_error_code( e_val );
}

#undef GENFRONT
#define GENFRONT( opname, dma_func ) \
\
static int PASTEMAC0(opname) \
     ( \
       obj_t*       a, \
       obj_t*       p, \
       dma_event_t* event, \
       thrinfo_t*   thread \
     ) \
{ \
	int ret = 0; \
\
	bli_dma_get_check( a, p ); \
\
	/* only thread chief triggers dma */ \
	if ( bli_thread_am_ochief( thread ) ) \
	{ \
		obj_t* global_root     = bli_obj_root( a ); \
		dim_t m_root           = bli_obj_length( global_root ); \
		dim_t n_root           = bli_obj_width( global_root ); \
		dim_t m_a              = bli_obj_length( a ); \
		dim_t n_a              = bli_obj_width( a ); \
		dim_t rs_a             = bli_obj_row_stride( a ); \
		dim_t cs_a             = bli_obj_col_stride( a ); \
\
		/* Query if object a is stored in row-major or col-major */ \
		bool is_row_major    = bli_is_row_stored_f( m_a, n_a, rs_a, cs_a ); \
\
		void*        global    = bli_obj_buffer( global_root ); \
		void*        local     = bli_obj_buffer( p ); \
		siz_t        elem_size = bli_obj_elem_size( p ); \
\
		point2d_t local_point  = { 0 }; \
		point2d_t global_point = { 0 }; \
\
		if ( is_row_major ) \
		{ \
			local_point.xpos = 0;                     /* xpos = zero in local panel */ \
			local_point.ypos = 0;                     /* ypos = zero in local panel */ \
			local_point.xdim = bli_obj_width( p );    /* xdim = width of local panel */ \
			local_point.ydim = bli_obj_length( p );   /* ydim = length of root */ \
\
			/* The "invisible" part of a: for any (alignment) reason, the real
			   allocated buffer of the root-matrix can be larger than its dimensions.
			   This is why we must use rs_a (or cs_a) instead of n_a (or m_a) for
			   global_point.xdim. */ \
			global_point.xpos = bli_obj_col_off( a ); /* xpos = col-offset of a in root */ \
			global_point.ypos = bli_obj_row_off( a ); /* ypos = row-offset of a in root */ \
			global_point.xdim = rs_a;                 /* xdim = row-stride of a */ \
			global_point.ydim = m_root;               /* ydim = length of root */ \
		} \
		else  /* is_col_major */ \
		{ \
			local_point.xpos = 0;                     /* xpos = zero in local panel */ \
			local_point.ypos = 0;                     /* ypos = zero in local panel */ \
			local_point.xdim = bli_obj_length( p );   /* xdim = length of local panel */ \
			local_point.ydim = bli_obj_width( p );    /* ydim = width of root */ \
\
			global_point.xpos = bli_obj_row_off( a ); /* xpos = row-offset of a in root */ \
			global_point.ypos = bli_obj_col_off( a ); /* ypos = col-offset of a in root */ \
			global_point.xdim = cs_a;                 /* xdim = col-stride of a */ \
			global_point.ydim = n_root;               /* ydim = width of root */ \
		} \
\
		/* trigger DMA transfer */ \
		ret  = dma_func(global,               /* global */ \
		                local,                /* local */ \
		                elem_size,            /* elem_size */ \
		                local_point.xdim,     /* width */ \
		                local_point.ydim,     /* height */ \
		                &global_point,        /* global_point */ \
		                &local_point,         /* local_point */ \
		                event                 /* event */ \
		               ); \
	} \
\
	return ret; \
}

GENFRONT( dma_get2D, bli_dma_backend_get2D )
GENFRONT( dma_put2D, bli_dma_backend_put2D )


static siz_t bli_dma_get_init ( obj_t* a, obj_t* p )
{
	siz_t elem_size   = bli_obj_elem_size( a );
	dim_t m_a         = bli_obj_length( a );
	dim_t n_a         = bli_obj_width( a );
	dim_t rs_a        = bli_obj_row_stride( a );
	dim_t cs_a        = bli_obj_col_stride( a );
	siz_t size_needed = 0;

	dim_t rs_p,   cs_p;
	dim_t offm_p, offn_p;
	dim_t m_p,    n_p;

	// We begin by copying the fields of A.
	bli_obj_alias_to( a, p );

	// If the object is marked as being filled with zeros, then we can skip
	// the dma operation entirely and return zero, otherwise
	if ( !bli_obj_is_zeros( a ) )
	{
		// Default offsets of p is zeros
		offm_p = 0;
		offn_p = 0;
		m_p = m_a;
		n_p = n_a;

		bool is_row_major = bli_is_row_stored_f( m_a, n_a, rs_a, cs_a );

		if( is_row_major )
		{
			rs_p = n_p;
			cs_p = 1;
		}
		else // is_col_major
		{
			rs_p = 1;
			cs_p = m_p;
		}

		bli_obj_set_dims( m_p, n_p, p );
		bli_obj_set_offs( offm_p, offn_p, p );
		bli_obj_set_strides( rs_p, cs_p, p );

		size_needed = m_p * n_p * elem_size;
	}

	return size_needed;
}

static void bli_dma_alloc
      (
        siz_t      size_needed,
        mem_t*     mem_p_dma,
        rntm_t*    rntm,
        thrinfo_t* thread
      )
{
	mem_t* local_mem_p;
	mem_t  local_mem_s;

	siz_t mem_size = 0;

	if ( !mem_p_dma )
	{
		fprintf( stderr, "%s:%d: mem_p_dma must not be NULL\n",
		         __FILE__, __LINE__ );
		bli_check_error_code( BLIS_NULL_POINTER );
	}

	if ( bli_mem_is_alloc( mem_p_dma ) ) {
		mem_size = bli_mem_size( mem_p_dma );
	}

	if ( mem_size < size_needed )
	{
		if ( bli_thread_am_ochief( thread ) )
		{
			packbuf_t dma_buf_type = bli_mem_buf_type( mem_p_dma );

			// The chief thread releases the existing block associated with
			// the mem_t entry in the control tree, and then re-acquires a
			// new block, saving the associated mem_t entry to local_mem_s.
			if ( bli_mem_is_alloc( mem_p_dma ) )
			{
				bli_pba_release
				(
				  rntm,
				  mem_p_dma
				);
			}
			bli_pba_acquire_m
			(
			  rntm,
			  size_needed,
			  dma_buf_type,
			  &local_mem_s
			);
		}

		// Broadcast the address of the chief thread's local mem_t entry to
		// all threads.
		local_mem_p = bli_thread_broadcast( thread, &local_mem_s );

		// Save the chief thread's local mem_t entry to the mem_t field in
		// this thread's control tree node.
		*mem_p_dma = *local_mem_p;
	}

	bli_thread_barrier( thread );
}

void bli_dma_get
      (
        obj_t*       a,
        obj_t*       p,
        mem_t*       mem_p_dma,
        dma_event_t* event,
        rntm_t*      rntm,
        thrinfo_t*   thread
      )
{
	bli_init_once();
	bli_thread_barrier( thread );

	siz_t size_needed = bli_dma_get_init( a, p );
	if ( size_needed > 0 )
	{
		// check for potentially need to increase the local DMA buffer
		bli_dma_alloc( size_needed, mem_p_dma, rntm, thread );

		// reset local buffer to p
		void* dma_buffer = bli_mem_buffer( mem_p_dma );
		bli_obj_set_buffer( dma_buffer, p );

		// call get2D backend
		int err = bli_dma_get2D( a, p, event, thread );
		if ( err ) {
			bli_check_error_code( BLIS_DMA_GET_FAILURE );
		}
	}

	bli_thread_barrier( thread );
}

void bli_dma_put
      (
        obj_t*       a,
        obj_t*       p,
        dma_event_t* event,
        thrinfo_t*   thread
      )
{
	bli_init_once();
	bli_thread_barrier( thread );

	// call put2D backend
	int err = bli_dma_put2D( a, p, event, thread );
	if ( err ) {
		bli_check_error_code( BLIS_DMA_PUT_FAILURE );
	}

	bli_thread_barrier( thread );
}

void bli_dma_wait( dma_event_t* event, thrinfo_t* thread )
{
	bli_thread_barrier( thread );

	// only thread chief waits
	if ( event && bli_thread_am_ochief( thread ) )
	{
		int err = bli_dma_backend_wait( event );
		if ( err ) {
			bli_check_error_code( BLIS_DMA_WAIT_FAILURE );
		}
	}

	// Barrier so that DMA is done before computation, or data sent after
	// computation.
	bli_thread_barrier( thread );
}

#endif // BLIS_ENABLE_DMA
