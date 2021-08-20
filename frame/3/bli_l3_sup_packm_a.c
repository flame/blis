/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       bool             will_pack, \
       packbuf_t        pack_buf_type, \
       dim_t            m, \
       dim_t            k, \
       dim_t            mr, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	/* Inspect whether we are going to be packing matrix A. */ \
	if ( will_pack == FALSE ) \
	{ \
	} \
	else /* if ( will_pack == TRUE ) */ \
	{ \
		/* NOTE: This "rounding up" of the last upanel is actually optional
		   for the rrc/crc cases, but absolutely necessary for the other cases
		   since we NEED that last micropanel to have the same ldim (cs_p) as
		   the other micropanels. Why? So that millikernels can use the same
		   upanel ldim for all iterations of the ir loop. */ \
		const dim_t m_pack = ( m / mr + ( m % mr ? 1 : 0 ) ) * mr; \
		const dim_t k_pack = k; \
\
		/* Barrier to make sure all threads are caught up and ready to begin
		   the packm stage. */ \
		bli_thread_barrier( thread ); \
\
		/* Compute the size of the memory block eneded. */ \
		siz_t size_needed = sizeof( ctype ) * m_pack * k_pack; \
\
		/* Check the mem_t entry provided by the caller. If it is unallocated,
		   then we need to acquire a block from the memory broker. */ \
		if ( bli_mem_is_unalloc( mem ) ) \
		{ \
			if ( bli_thread_am_ochief( thread ) ) \
			{ \
				/* Acquire directly to the chief thread's mem_t that was
				   passed in. It needs to be that mem_t struct, and not a
				   local (temporary) mem_t, since there is no barrier until
				   after packing is finished, which could allow a race
				   condition whereby the chief thread exits the current
				   function before the other threads have a chance to copy
				   from it. (A barrier would fix that race condition, but
				   then again, I prefer to keep barriers to a minimum.) */ \
				bli_pba_acquire_m \
				( \
				  rntm, \
				  size_needed, \
				  pack_buf_type, \
				  mem  \
				); \
			} \
\
			/* Broadcast the address of the chief thread's passed-in mem_t
			   to all threads. */ \
			mem_t* mem_p = bli_thread_broadcast( thread, mem ); \
\
			/* Non-chief threads: Copy the contents of the chief thread's
			   passed-in mem_t to the passed-in mem_t for this thread. (The
			   chief thread already has the mem_t, so it does not need to
			   perform any copy.) */ \
			if ( !bli_thread_am_ochief( thread ) ) \
			{ \
				*mem = *mem_p; \
			} \
		} \
		else /* if ( bli_mem_is_alloc( mem ) ) */ \
		{ \
			/* If the mem_t entry provided by the caller does NOT contain a NULL
			   buffer, then a block has already been acquired from the memory
			   broker and cached by the caller. */ \
\
			/* As a sanity check, we should make sure that the mem_t object isn't
			   associated with a block that is too small compared to the size of
			   the packed matrix buffer that is needed, according to the value
			   computed above. */ \
			siz_t mem_size = bli_mem_size( mem ); \
\
			if ( mem_size < size_needed ) \
			{ \
				if ( bli_thread_am_ochief( thread ) ) \
				{ \
					/* The chief thread releases the existing block associated
					   with the mem_t, and then re-acquires a new block, saving
					   the associated mem_t to its passed-in mem_t. (See coment
					   above for why the acquisition needs to be directly to
					   the chief thread's passed-in mem_t and not a local
					   (temporary) mem_t. */ \
					bli_pba_release \
					( \
					  rntm, \
					  mem \
					); \
					bli_pba_acquire_m \
					( \
					  rntm, \
					  size_needed, \
					  pack_buf_type, \
					  mem \
					); \
				} \
\
				/* Broadcast the address of the chief thread's passed-in mem_t
				   to all threads. */ \
				mem_t* mem_p = bli_thread_broadcast( thread, mem ); \
\
				/* Non-chief threads: Copy the contents of the chief thread's
				   passed-in mem_t to the passed-in mem_t for this thread. (The
				   chief thread already has the mem_t, so it does not need to
				   perform any copy.) */ \
				if ( !bli_thread_am_ochief( thread ) ) \
				{ \
					*mem = *mem_p; \
				} \
			} \
			else \
			{ \
				/* If the mem_t entry is already allocated and sufficiently large,
				   then we use it as-is. No action is needed. */ \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( packm_sup_init_mem_a )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       bool             did_pack, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	/* Inspect whether we previously packed matrix A. */ \
	if ( did_pack == FALSE ) \
	{ \
		/* If we didn't pack matrix A, there's nothing to be done. */ \
	} \
	else /* if ( did_pack == TRUE ) */ \
	{ \
		if ( thread != NULL ) \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			/* Check the mem_t entry provided by the caller. Only proceed if it
			   is allocated, which it should be. */ \
			if ( bli_mem_is_alloc( mem ) ) \
			{ \
				bli_pba_release \
				( \
				  rntm, \
				  mem \
				); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( packm_sup_finalize_mem_a )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       bool             will_pack, \
       stor3_t          stor_id, \
       pack_t* restrict schema, \
       dim_t            m, \
       dim_t            k, \
       dim_t            mr, \
       dim_t*  restrict m_max, \
       dim_t*  restrict k_max, \
       ctype*           x, inc_t           rs_x, inc_t           cs_x, \
       ctype**          p, inc_t* restrict rs_p, inc_t* restrict cs_p, \
                           dim_t* restrict pd_p, inc_t* restrict ps_p, \
       cntx_t* restrict cntx, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	/* Inspect whether we are going to be packing matrix A. */ \
	if ( will_pack == FALSE ) \
	{ \
		*m_max = m; \
		*k_max = k; \
\
		/* Set the parameters for use with no packing of A (ie: using the
		   source matrix A directly). */ \
		{ \
			/* Use the strides of the source matrix as the final values. */ \
			*rs_p = rs_x; \
			*cs_p = cs_x; \
\
			*pd_p = mr; \
			*ps_p = mr * rs_x; \
\
			/* Set the schema to "not packed" to indicate that packing will be
			   skipped. */ \
			*schema = BLIS_NOT_PACKED; \
		} \
\
		/* Since we won't be packing, simply update the buffer address provided
		   by the caller to point to source matrix. */ \
		*p = x; \
	} \
	else /* if ( will_pack == TRUE ) */ \
	{ \
		/* NOTE: This is "rounding up" of the last upanel is actually optional
		   for the rrc/crc cases, but absolutely necessary for the other cases
		   since we NEED that last micropanel to have the same ldim (cs_p) as
		   the other micropanels. Why? So that millikernels can use the same
		   upanel ldim for all iterations of the ir loop. */ \
		*m_max = ( m / mr + ( m % mr ? 1 : 0 ) ) * mr; \
		*k_max = k; \
\
		/* Determine the dimensions and strides for the packed matrix A. */ \
		if ( stor_id == BLIS_RRC || \
			 stor_id == BLIS_CRC ) \
		{ \
			/* stor3_t id values _RRC and _CRC: pack A to plain row storage. */ \
			*rs_p = k; \
			*cs_p = 1; \
\
			*pd_p = mr; \
			*ps_p = mr * k; \
\
			/* Set the schema to "row packed" to indicate packing to plain
			   row storage. */ \
			*schema = BLIS_PACKED_ROWS; \
		} \
		else \
		{ \
			/* All other stor3_t ids: pack A to column-stored row-panels. */ \
			*rs_p = 1; \
			*cs_p = mr; \
\
			*pd_p = mr; \
			*ps_p = mr * k; \
\
			/* Set the schema to "packed row panels" to indicate packing to
			   conventional column-stored row panels. */ \
			*schema = BLIS_PACKED_ROW_PANELS; \
		} \
\
		/* Set the buffer address provided by the caller to point to the
		   memory associated with the mem_t entry acquired from the memory
		   broker. */ \
		*p = bli_mem_buffer( mem ); \
	} \
}

INSERT_GENTFUNC_BASIC0( packm_sup_init_a )


//
// Define BLAS-like interfaces to the variant chooser.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       bool             will_pack, \
       packbuf_t        pack_buf_type, \
       stor3_t          stor_id, \
       trans_t          transc, \
       dim_t            m_alloc, \
       dim_t            k_alloc, \
       dim_t            m, \
       dim_t            k, \
       dim_t            mr, \
       ctype*  restrict kappa, \
       ctype*  restrict a, inc_t           rs_a, inc_t           cs_a, \
       ctype** restrict p, inc_t* restrict rs_p, inc_t* restrict cs_p, \
                                                 inc_t* restrict ps_p, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	pack_t schema; \
	dim_t  m_max; \
	dim_t  k_max; \
	dim_t  pd_p; \
\
	/* Prepare the packing destination buffer. If packing is not requested,
	   this function will reduce to a no-op. */ \
	PASTEMAC(ch,packm_sup_init_mem_a) \
	( \
	  will_pack, \
	  pack_buf_type, \
	  m_alloc, k_alloc, mr, \
	  cntx, \
	  rntm, \
	  mem, \
	  thread  \
	); \
\
	/* Determine the packing buffer and related parameters for matrix A. If A
	   will not be packed, then a_use will be set to point to a and the _a_use
	   strides will be set accordingly. */ \
	PASTEMAC(ch,packm_sup_init_a) \
	( \
	  will_pack, \
	  stor_id, \
	  &schema, \
	  m, k, mr, \
	  &m_max, &k_max, \
	  a, rs_a,  cs_a, \
	  p, rs_p,  cs_p, \
	     &pd_p, ps_p, \
	  cntx, \
	  mem, \
	  thread  \
	); \
\
	/* Inspect whether we are going to be packing matrix A. */ \
	if ( will_pack == FALSE ) \
	{ \
		/* If we aren't going to pack matrix A, then there's nothing to do. */ \
\
		/*
		printf( "blis_ packm_sup_a: not packing A.\n" ); \
		*/ \
	} \
	else /* if ( will_pack == TRUE ) */ \
	{ \
		if ( schema == BLIS_PACKED_ROWS ) \
		{ \
			/*
			printf( "blis_ packm_sup_a: packing A to rows.\n" ); \
			*/ \
\
			/* For plain packing by rows, use var2. */ \
			PASTEMAC(ch,packm_sup_var2) \
			( \
			  transc, \
			  schema, \
			  m, \
			  k, \
			  kappa, \
			  a,  rs_a,  cs_a, \
			  *p, *rs_p, *cs_p, \
			  cntx, \
			  thread  \
			); \
		} \
		else /* if ( schema == BLIS_PACKED_ROW_PANELS ) */ \
		{ \
			/*
			printf( "blis_ packm_sup_a: packing A to row panels.\n" ); \
			*/ \
\
			/* For packing to column-stored row panels, use var1. */ \
			PASTEMAC(ch,packm_sup_var1) \
			( \
			  transc, \
			  schema, \
			  m, \
			  k, \
			  m_max, \
			  k_max, \
			  kappa, \
			  a,  rs_a,  cs_a, \
			  *p, *rs_p, *cs_p, \
			      pd_p,  *ps_p, \
			  cntx, \
			  thread  \
			); \
		} \
\
		/* Barrier so that packing is done before computation. */ \
		bli_thread_barrier( thread ); \
	} \
}

INSERT_GENTFUNC_BASIC0( packm_sup_a )

