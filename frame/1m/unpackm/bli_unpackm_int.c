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

#include "blis.h"

#define FUNCPTR_T unpackm_fp

typedef void (*FUNCPTR_T)( obj_t*     p,
                           obj_t*     a,
                           unpackm_t* cntl );

static FUNCPTR_T vars[2][3] =
{
	// unblocked            optimized unblocked    blocked
	{ bli_unpackm_unb_var1, NULL,                  NULL,                 },
	{ NULL,                 NULL,                  bli_unpackm_blk_var2, },
};

void bli_unpackm_int( obj_t*     p,
                      obj_t*     a,
                      unpackm_t* cntl,
                      packm_thrinfo_t* thread )
{
	// The unpackm operation consists of an optional post-process: castm.
	// (This post-process is analogous to the castm pre-process in packm.)
	// Here are the following possible ways unpackm can execute:
	//  1. unpack and cast: Unpack to a temporary matrix c and then cast
	//     c to a.
	//  2. unpack only: Unpack directly to matrix a since typecasting is
	//     not needed.
	//  3. cast only: Not yet supported / not used.
	//  4. no-op: The control tree directs us to skip the unpack operation
	//     entirely. No action is taken.

	obj_t     c;

	varnum_t  n;
	impl_t    i;
	FUNCPTR_T f;

	// Sanity check; A should never have a zero dimension. If we must support
	// it, then we should fold it into the next alias-and-early-exit block.
	//if ( bli_obj_has_zero_dim( *a ) ) bli_abort();

	// First check if we are to skip this operation because the control tree
	// is NULL, and if so, simply return.
	if ( cntl_is_noop( cntl ) )
	{
		return;
	}

	// If p was aliased to a during the pack stage (because it was already
	// in an acceptable packed/contiguous format), then no unpack is actually
	// necessary, so we return.
	if ( bli_obj_is_alias_of( *p, *a ) )
	{
		return;
	}

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_unpackm_check( p, a, cntl );

	// Now, if we are not skipping the unpack operation, then the only
	// question left is whether we are to typecast matrix a after unpacking.
	if ( bli_obj_datatype( *p ) != bli_obj_datatype( *a ) )
		bli_abort();
/*
	if ( bli_obj_datatype( *p ) != bli_obj_datatype( *a ) )
	{
		// Initialize an object c for the intermediate typecast matrix.
		bli_unpackm_init_cast( p,
		                       a,
		                       &c );
	}
	else
*/
	{
		// If no cast is needed, then aliasing object c to the original
		// matrix serves as a minor optimization. This causes the unpackm
		// implementation to unpack directly into matrix a.
		bli_obj_alias_to( *a, c );
	}

	// Now we are ready to proceed with the unpacking.

	// Extract the variant number and implementation type.
	n = cntl_var_num( cntl );
	i = cntl_impl_type( cntl );

	// Index into the variant array to extract the correct function pointer.
	f = vars[n][i];

	// Invoke the variant.
    if( thread_am_ochief( thread ) ) {
        f( p,
           &c,
           cntl );
    }
    thread_obarrier( thread );

	// Now, if necessary, we cast the contents of c to matrix a. If casting
	// was not necessary, then we are done because the call to the unpackm
	// implementation would have unpacked directly to matrix a.
/*
	if ( bli_obj_datatype( *p ) != bli_obj_datatype( *a ) )
	{
		// Copy/typecast matrix c to matrix a.
		// NOTE: Here, we use copynzm instead of copym because, in the cases
		// where we are unpacking/typecasting a real matrix c to a complex
		// matrix a, we want to touch only the real components of a, rather
		// than also set the imaginary components to zero. This comes about
		// because of the fact that, if we are unpacking real-to-complex,
		// then it is because all of the computation occurred in the real
		// domain, and so we would want to leave whatever imaginary values
		// there are in matrix a untouched. Notice that for unpackings that
		// entail complex-to-complex data movements, the copynzm operation
		// behaves exactly as copym, so no use cases are lost (at least none
		// that I can think of).
		bli_copynzm( &c,
		             a );

		// NOTE: The above code/comment is outdated. What should happen is
		// as follows:
		// - If dt(a) is complex and dt(p) is real, then create an alias of
		//   a and then tweak it so that it looks like a real domain object.
		//   This will involve:
		//   - projecting the datatype to real domain
		//   - scaling both the row and column strides by 2
		//   ALL OF THIS should be done in the front-end, NOT here, as
		//   unpackm() won't even be needed in that case.
	}
*/
}

/*
void bli_unpackm_init_cast( obj_t*  p,
                            obj_t*  a,
                            obj_t*  c )
{
	// The idea here is that we want to create an object c that is identical
	// to object a, except that:
	//  (1) the storage datatype of c is equal to the target datatype of a,
	//      with the element size of c adjusted accordingly,
	//  (2) the view offset of c is reset to (0,0),
    //  (3) object c's main buffer is set to a new memory region acquired
    //      from the memory manager, or extracted from p if a mem entry is
    //      already available, (After acquring a mem entry from the memory
    //      manager, it is cached within p for quick access later on.)
	//  (4) object c is marked as being stored in a standard, contiguous
	//      format (ie: column-major order).
	// Any transposition encoded within object a will also be encoded in
	// object c. That way, unpackm handles any needed transposition during
	// the unpacking, and the only thing the cast stage needs to do is cast.

	num_t dt_targ_a    = bli_obj_target_datatype( *a );
	dim_t m_a          = bli_obj_length( *a );
	siz_t elem_size_c  = bli_datatype_size( dt_targ_a );

	inc_t  rs_c, cs_c;

	// We begin by copying the basic fields of a.
	bli_obj_alias_to( *a, *c );

	// Update datatype and element size fields.
	bli_obj_set_datatype( dt_targ_a, *c );
	bli_obj_set_elem_size( elem_size_c, *c );

    // Reset the view offsets to (0,0).
    bli_obj_set_offs( 0, 0, *c );

	// Check the mem_t entry of p associated with the cast buffer. If it is
	// NULL, then acquire memory sufficient to hold the object data and cache
	// it to p. (Otherwise, if it is non-NULL, then memory has already been
	// acquired from the memory manager and cached.) We then set the main
	// buffer of c to the cached address of the cast memory.
	bli_obj_set_buffer_with_cached_cast_mem( *p, *c );

	// Update the strides. We set the increments to reflect column-major order
	// storage. We start the leading dimension out as m(a) and increment it if
	// necessary so that the beginning of each column is aligned.
	cs_c = bli_align_dim_to_size( m_a, elem_size_c,
	                              BLIS_HEAP_STRIDE_ALIGN_SIZE );
    rs_c = 1;
    bli_obj_set_incs( rs_c, cs_c, *c );
}
*/
