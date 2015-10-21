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

static bool_t bli_thread_is_init = FALSE;

packm_thrinfo_t BLIS_PACKM_SINGLE_THREADED;
gemm_thrinfo_t BLIS_GEMM_SINGLE_THREADED;
herk_thrinfo_t BLIS_HERK_SINGLE_THREADED;
thread_comm_t BLIS_SINGLE_COMM;

void bli_thread_init( void )
{
	// If the API is already initialized, return early.
	if ( bli_thread_is_initialized() ) return;

	bli_setup_communicator( &BLIS_SINGLE_COMM, 1 );
	bli_setup_packm_single_threaded_info( &BLIS_PACKM_SINGLE_THREADED );
	bli_setup_gemm_single_threaded_info( &BLIS_GEMM_SINGLE_THREADED );
	bli_setup_herk_single_threaded_info( &BLIS_HERK_SINGLE_THREADED );

	// Mark API as initialized.
	bli_thread_is_init = TRUE;
}

void bli_thread_finalize( void )
{
	// Mark API as uninitialized.
	bli_thread_is_init = FALSE;
}

bool_t bli_thread_is_initialized( void )
{
	return bli_thread_is_init;
}

// -----------------------------------------------------------------------------

//*********** Stuff Specific to single-threaded *************
#ifndef BLIS_ENABLE_MULTITHREADING
void bli_barrier( thread_comm_t* communicator, dim_t t_id )
{
    return;
}

void bli_level3_thread_decorator( dim_t n_threads, 
                                  level3_int_t func, 
                                  obj_t* alpha, 
                                  obj_t* a, 
                                  obj_t* b, 
                                  obj_t* beta, 
                                  obj_t* c, 
                                  void* cntl, 
                                  void** thread )
{
        func( alpha, a, b, beta, c, cntl, thread[0] );
}


//Constructors and destructors for constructors
thread_comm_t* bli_create_communicator( dim_t n_threads )
{
    thread_comm_t* comm = (thread_comm_t*) bli_malloc( sizeof(thread_comm_t) );
    bli_setup_communicator( comm, n_threads );
    return comm;
}

void bli_setup_communicator( thread_comm_t* communicator, dim_t n_threads)
{
    if( communicator == NULL ) return;
    communicator->sent_object = NULL;
    communicator->n_threads = n_threads;
    communicator->barrier_sense = 0;
    communicator->barrier_threads_arrived = 0;
}

void bli_free_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
    bli_cleanup_communicator( communicator );
    bli_free( communicator );
}

void bli_cleanup_communicator( thread_comm_t* communicator )
{
    if( communicator == NULL ) return;
}

#endif

//Constructors and destructors for thread infos
thrinfo_t* bli_create_thread_info( thread_comm_t* ocomm, dim_t ocomm_id, thread_comm_t* icomm, dim_t icomm_id,
                             dim_t n_way, dim_t work_id )
{

        thrinfo_t* thr = (thrinfo_t*) bli_malloc( sizeof(thrinfo_t) );
        bli_setup_thread_info( thr, ocomm, ocomm_id, icomm, icomm_id, n_way, work_id );
        return thr;
}

void bli_setup_thread_info( thrinfo_t* thr, thread_comm_t* ocomm, dim_t ocomm_id, thread_comm_t* icomm, dim_t icomm_id,
                             dim_t n_way, dim_t work_id )
{
        thr->ocomm = ocomm;
        thr->ocomm_id = ocomm_id;
        thr->icomm = icomm;
        thr->icomm_id = icomm_id;

        thr->n_way = n_way;
        thr->work_id = work_id;
}

// Broadcast code
void* bli_broadcast_structure( thread_comm_t* communicator, dim_t id, void* to_send )
{   
    if( communicator == NULL || communicator->n_threads == 1 ) return to_send;

    if( id == 0 ) communicator->sent_object = to_send;

    bli_barrier( communicator, id );
    void * object = communicator->sent_object;
    bli_barrier( communicator, id );

    return object;
}

// Code for work assignments
void bli_get_range( void* thr, dim_t n, dim_t bf, bool_t handle_edge_low, dim_t* start, dim_t* end )
{
	thrinfo_t* thread     = ( thrinfo_t* )thr;
	dim_t      n_way      = thread->n_way;
	dim_t      work_id    = thread->work_id;

	dim_t      all_start  = 0;
	dim_t      all_end    = n;

	dim_t      size       = all_end - all_start;

	dim_t      n_bf_whole = size / bf;
	dim_t      n_bf_left  = size % bf;

	dim_t      n_bf_lo    = n_bf_whole / n_way;
	dim_t      n_bf_hi    = n_bf_whole / n_way;

	// In this function, we partition the space between all_start and
	// all_end into n_way partitions, each a multiple of block_factor
	// with the exception of the one partition that recieves the
	// "edge" case (if applicable).
	//
	// Here are examples of various thread partitionings, in units of
	// the block_factor, when n_way = 4. (A '+' indicates the thread
	// that receives the leftover edge case (ie: n_bf_left extra
	// rows/columns in its sub-range).
	//                                        (all_start ... all_end)
	// n_bf_whole  _left  hel  n_th_lo  _hi   thr0  thr1  thr2  thr3
	//         12     =0    f        0    4      3     3     3     3
	//         12     >0    f        0    4      3     3     3     3+
	//         13     >0    f        1    3      4     3     3     3+
	//         14     >0    f        2    2      4     4     3     3+
	//         15     >0    f        3    1      4     4     4     3+
	//         15     =0    f        3    1      4     4     4     3 
	//
	//         12     =0    t        4    0      3     3     3     3
	//         12     >0    t        4    0      3+    3     3     3
	//         13     >0    t        3    1      3+    3     3     4
	//         14     >0    t        2    2      3+    3     4     4
	//         15     >0    t        1    3      3+    4     4     4
	//         15     =0    t        1    3      3     4     4     4

	// As indicated by the table above, load is balanced as equally
	// as possible, even in the presence of an edge case.

	// First, we must differentiate between cases where the leftover
	// "edge" case (n_bf_left) should be allocated to a thread partition
	// at the low end of the index range or the high end.

	if ( handle_edge_low == FALSE )
	{
		// Notice that if all threads receive the same number of
		// block_factors, those threads are considered "high" and
		// the "low" thread group is empty.
		dim_t n_th_lo = n_bf_whole % n_way;
		//dim_t n_th_hi = n_way - n_th_lo;

		// If some partitions must have more block_factors than others
		// assign the slightly larger partitions to lower index threads.
		if ( n_th_lo != 0 ) n_bf_lo += 1;

		// Compute the actual widths (in units of rows/columns) of
		// individual threads in the low and high groups.
		dim_t size_lo = n_bf_lo * bf;
		dim_t size_hi = n_bf_hi * bf;

		// Precompute the starting indices of the low and high groups.
		dim_t lo_start = all_start;
		dim_t hi_start = all_start + n_th_lo * size_lo;

		// Compute the start and end of individual threads' ranges
		// as a function of their work_ids and also the group to which
		// they belong (low or high).
		if ( work_id < n_th_lo )
		{
			*start = lo_start + (work_id  ) * size_lo;
			*end   = lo_start + (work_id+1) * size_lo;
		}
		else // if ( n_th_lo <= work_id )
		{
			*start = hi_start + (work_id-n_th_lo  ) * size_hi;
			*end   = hi_start + (work_id-n_th_lo+1) * size_hi;

			// Since the edge case is being allocated to the high
			// end of the index range, we have to advance the last
			// thread's end.
			if ( work_id == n_way - 1 ) *end += n_bf_left;
		}
	}
	else // if ( handle_edge_low == TRUE )
	{
		// Notice that if all threads receive the same number of
		// block_factors, those threads are considered "low" and
		// the "high" thread group is empty.
		dim_t n_th_hi = n_bf_whole % n_way;
		dim_t n_th_lo = n_way - n_th_hi;

		// If some partitions must have more block_factors than others
		// assign the slightly larger partitions to higher index threads.
		if ( n_th_hi != 0 ) n_bf_hi += 1;

		// Compute the actual widths (in units of rows/columns) of
		// individual threads in the low and high groups.
		dim_t size_lo = n_bf_lo * bf;
		dim_t size_hi = n_bf_hi * bf;

		// Precompute the starting indices of the low and high groups.
		dim_t lo_start = all_start;
		dim_t hi_start = all_start + n_th_lo * size_lo
		                           + n_bf_left;

		// Compute the start and end of individual threads' ranges
		// as a function of their work_ids and also the group to which
		// they belong (low or high).
		if ( work_id < n_th_lo )
		{
			*start = lo_start + (work_id  ) * size_lo;
			*end   = lo_start + (work_id+1) * size_lo;

			// Since the edge case is being allocated to the low
			// end of the index range, we have to advance the
			// starts/ends accordingly.
			if ( work_id == 0 )   *end   += n_bf_left;
			else                { *start += n_bf_left;
			                      *end   += n_bf_left; }
		}
		else // if ( n_th_lo <= work_id )
		{
			*start = hi_start + (work_id-n_th_lo  ) * size_hi;
			*end   = hi_start + (work_id-n_th_lo+1) * size_hi;
		}
	}
}

siz_t bli_get_range_l2r( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	dim_t m = bli_obj_length_after_trans( *a );
	dim_t n = bli_obj_width_after_trans( *a );

	bli_get_range( thr, n, bf,
	               FALSE, start, end );

	return m * ( *end - *start );
}

siz_t bli_get_range_r2l( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	dim_t m = bli_obj_length_after_trans( *a );
	dim_t n = bli_obj_width_after_trans( *a );

	bli_get_range( thr, n, bf,
	               TRUE, start, end );

	return m * ( *end - *start );
}

siz_t bli_get_range_t2b( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	dim_t m = bli_obj_length_after_trans( *a );
	dim_t n = bli_obj_width_after_trans( *a );

	bli_get_range( thr, m, bf,
	               FALSE, start, end );

	return n * ( *end - *start );
}

siz_t bli_get_range_b2t( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	dim_t m = bli_obj_length_after_trans( *a );
	dim_t n = bli_obj_width_after_trans( *a );

	bli_get_range( thr, m, bf,
	               TRUE, start, end );

	return n * ( *end - *start );
}

dim_t bli_get_range_width_l( doff_t diagoff_j,
                             dim_t  m,
                             dim_t  n_j,
                             dim_t  j,
                             dim_t  n_way,
                             dim_t  bf,
                             dim_t  bf_left,
                             double area_per_thr,
                             bool_t handle_edge_low )
{
	dim_t width;

	// In this function, we assume that we are somewhere in the process of
	// partitioning an m x n lower-stored region (with arbitrary diagonal
	// offset) n_ways along the n dimension (into column panels). The value
	// j identifies the left-to-right subpartition index (from 0 to n_way-1)
	// of the subpartition whose width we are about to compute using the
	// area per thread determined by the caller. n_j is the number of
	// columns in the remaining region of the matrix being partitioned,
	// and diagoff_j is that region's diagonal offset.

	// If this is the last subpartition, the width is simply equal to n_j.
	// Note that this statement handles cases where the "edge case" (if
	// one exists) is assigned to the high end of the index range (ie:
	// handle_edge_low == FALSE).
	if ( j == n_way - 1 ) return n_j;

	// At this point, we know there are at least two subpartitions left.
	// We also know that IF the submatrix contains a completely dense
	// rectangular submatrix, it will occur BEFORE the triangular (or
	// trapezoidal) part.

	// Here, we implement a somewhat minor load balancing optimization
	// that ends up getting employed only for relatively small matrices.
	// First, recall that all subpartition widths will be some multiple
	// of the blocking factor bf, except perhaps either the first or last
	// subpartition, which will receive the edge case, if it exists.
	// Also recall that j represents the current thread (or thread group,
	// or "caucus") for which we are computing a subpartition width.
	// If n_j is sufficiently small that we can only allocate bf columns
	// to each of the remaining threads, then we set the width to bf. We
	// do not allow the subpartition width to be less than bf, so, under
	// some conditions, if n_j is small enough, some of the reamining
	// threads may not get any work. For the purposes of this lower bound
	// on work (ie: width >= bf), we allow the edge case to count as a
	// "full" set of bf columns.
	{
		dim_t n_j_bf = n_j / bf + ( bf_left > 0 ? 1 : 0 );

		if ( n_j_bf <= n_way - j )
		{
			if ( j == 0 && handle_edge_low )
				width = ( bf_left > 0 ? bf_left : bf );
			else
				width = bf;

			// Make sure that the width does not exceed n_j. This would
			// occur if and when n_j_bf < n_way - j; that is, when the
			// matrix being partitioned is sufficiently small relative to
			// n_way such that there is not even enough work for every
			// (remaining) thread to get bf (or bf_left) columns. The
			// net effect of this safeguard is that some threads may get
			// assigned empty ranges (ie: no work), which of course must
			// happen in some situations.
			if ( width > n_j ) width = n_j;

			return width;
		}
	}

	// This block computes the width assuming that we are entirely within
	// a dense rectangle that precedes the triangular (or trapezoidal)
	// part.
	{
		// First compute the width of the current panel under the
		// assumption that the diagonal offset would not intersect.
		width = ( dim_t )bli_round( ( double )area_per_thr / ( double )m );

		// Adjust the width, if necessary. Specifically, we may need
		// to allocate the edge case to the first subpartition, if
		// requested; otherwise, we just need to ensure that the
		// subpartition is a multiple of the blocking factor.
		if ( j == 0 && handle_edge_low )
		{
			if ( width % bf != bf_left ) width += bf_left - ( width % bf );
		}
		else // if interior case
		{
			// Round up to the next multiple of the blocking factor.
			//if ( width % bf != 0       ) width += bf      - ( width % bf );
			// Round to the nearest multiple of the blocking factor.
			if ( width % bf != 0       ) width = bli_round_to_mult( width, bf );
		}
	}

	// We need to recompute width if the panel, according to the width
	// as currently computed, would intersect the diagonal.
	if ( diagoff_j < width )
	{
		dim_t offm_inc, offn_inc;

		// Prune away the unstored region above the diagonal, if it exists.
		// Note that the entire region was pruned initially, so we know that
		// we don't need to try to prune the right side. (Also, we discard
		// the offset deltas since we don't need to actually index into the
		// subpartition.)
		bli_prune_unstored_region_top_l( diagoff_j, m, n_j, offm_inc );
		//bli_prune_unstored_region_right_l( diagoff_j, m, n_j, offn_inc );

		// We don't need offm_inc, offn_inc here. These statements should
		// prevent compiler warnings.
		( void )offm_inc;
		( void )offn_inc;

		// Solve a quadratic equation to find the width of the current (jth)
		// subpartition given the m dimension, diagonal offset, and area.
		// NOTE: We know that the +/- in the quadratic formula must be a +
		// here because we know that the desired solution (the subpartition
		// width) will be smaller than (m + diagoff), not larger. If you
		// don't believe me, draw a picture!
		const double a = -0.5;
		const double b = ( double )m + ( double )diagoff_j + 0.5;
		const double c = -0.5 * (   ( double )diagoff_j *
		                          ( ( double )diagoff_j + 1.0 )
		                        ) - area_per_thr;
		const double x = ( -b + sqrt( b * b - 4.0 * a * c ) ) / ( 2.0 * a );

		// Use the rounded solution as our width, but make sure it didn't
		// round to zero.
		width = ( dim_t )bli_round( x );
		if ( width == 0 ) width = 1;

		// Adjust the width, if necessary.
		if ( j == 0 && handle_edge_low )
		{
			if ( width % bf != bf_left ) width += bf_left - ( width % bf );
		}
		else // if interior case
		{
			// Round up to the next multiple of the blocking factor.
			//if ( width % bf != 0       ) width += bf      - ( width % bf );
			// Round to the nearest multiple of the blocking factor.
			if ( width % bf != 0       ) width = bli_round_to_mult( width, bf );
		}
	}

	// Make sure that the width, after being adjusted, does not cause the
	// subpartition to exceed n_j.
	if ( width > n_j ) width = n_j;

	return width;
}

siz_t bli_find_area_trap_l( dim_t m, dim_t n, doff_t diagoff )
{
	dim_t  offm_inc = 0;
	dim_t  offn_inc = 0;
	double tri_area;
	double area;

	// Prune away any rectangular region above where the diagonal
	// intersects the left edge of the subpartition, if it exists.
	bli_prune_unstored_region_top_l( diagoff, m, n, offm_inc );

	// Prune away any rectangular region to the right of where the
	// diagonal intersects the bottom edge of the subpartition, if
	// it exists. (This shouldn't ever be needed, since the caller
	// would presumably have already performed rightward pruning,
	// but it's here just in case.)
	bli_prune_unstored_region_right_l( diagoff, m, n, offn_inc );

	( void )offm_inc;
	( void )offn_inc;

	// Compute the area of the empty triangle so we can subtract it
	// from the area of the rectangle that bounds the subpartition.
	if ( bli_intersects_diag_n( diagoff, m, n ) )
	{
		double tri_dim = ( double )( n - diagoff - 1 );
		tri_area = tri_dim * ( tri_dim + 1.0 ) / 2.0;
	}
	else
	{
		// If the diagonal does not intersect the trapezoid, then
		// we can compute the area as a simple rectangle.
		tri_area = 0.0;
	}

	area = ( double )m * ( double )n - tri_area;

	return ( siz_t )area;
}

siz_t bli_get_range_weighted( void*  thr,
                              doff_t diagoff,
                              uplo_t uplo,
                              dim_t  m,
                              dim_t  n,
                              dim_t  bf,
                              bool_t handle_edge_low,
                              dim_t* j_start_thr,
                              dim_t* j_end_thr )
{
	thrinfo_t* thread  = ( thrinfo_t* )thr;

	dim_t      n_way   = thread->n_way;
	dim_t      my_id   = thread->work_id;

	dim_t      bf_left = n % bf;

	dim_t      j;

	dim_t      off_j;
	doff_t     diagoff_j;
	dim_t      n_left;

	dim_t      width_j;

	dim_t      offm_inc, offn_inc;

	double     tri_dim, tri_area;
	double     area_total, area_per_thr;

	siz_t      area = 0;

	// In this function, we assume that the caller has already determined
	// that (a) the diagonal intersects the submatrix, and (b) the submatrix
	// is either lower- or upper-stored.

	if ( bli_is_lower( uplo ) )
	{
		// Prune away the unstored region above the diagonal, if it exists,
		// and then to the right of where the diagonal intersects the bottom,
		// if it exists. (Also, we discard the offset deltas since we don't
		// need to actually index into the subpartition.)
		bli_prune_unstored_region_top_l( diagoff, m, n, offm_inc );
		bli_prune_unstored_region_right_l( diagoff, m, n, offn_inc );

		// We don't need offm_inc, offn_inc here. These statements should
		// prevent compiler warnings.
		( void )offm_inc;
		( void )offn_inc;

		// Now that pruning has taken place, we know that diagoff >= 0.

		// Compute the total area of the submatrix, accounting for the
		// location of the diagonal, and divide it by the number of ways
		// of parallelism.
		tri_dim      = ( double )( n - diagoff - 1 );
		tri_area     = tri_dim * ( tri_dim + 1.0 ) / 2.0;
		area_total   = ( double )m * ( double )n - tri_area;
		area_per_thr = area_total / ( double )n_way;

		// Initialize some variables prior to the loop: the offset to the
		// current subpartition, the remainder of the n dimension, and
		// the diagonal offset of the current subpartition.
		off_j     = 0;
		diagoff_j = diagoff;
		n_left    = n;

		// Iterate over the subpartition indices corresponding to each
		// thread/caucus participating in the n_way parallelism.
		for ( j = 0; j < n_way; ++j )
		{
			// Compute the width of the jth subpartition, taking the
			// current diagonal offset into account, if needed.
			width_j = bli_get_range_width_l( diagoff_j, m, n_left,
			                                 j, n_way,
			                                 bf, bf_left,
			                                 area_per_thr,
			                                 handle_edge_low );

			// If the current thread belongs to caucus j, this is his
			// subpartition. So we compute the implied index range and
			// end our search.
			if ( j == my_id )
			{
				*j_start_thr = off_j;
				*j_end_thr   = off_j + width_j;

				area = bli_find_area_trap_l( m, width_j, diagoff_j );

				break;
			}

			// Shift the current subpartition's starting and diagonal offsets,
			// as well as the remainder of the n dimension, according to the
			// computed width, and then iterate to the next subpartition.
			off_j     += width_j;
			diagoff_j -= width_j;
			n_left    -= width_j;
		}
	}
	else // if ( bli_is_upper( uplo ) )
	{
		// Express the upper-stored case in terms of the lower-stored case.

		// First, we convert the upper-stored trapezoid to an equivalent
		// lower-stored trapezoid by rotating it 180 degrees.
		bli_rotate180_trapezoid( diagoff, uplo );

		// Now that the trapezoid is "flipped" in the n dimension, negate
		// the bool that encodes whether to handle the edge case at the
		// low (or high) end of the index range.
		bli_toggle_bool( handle_edge_low );

		// Compute the appropriate range for the rotated trapezoid.
		area = bli_get_range_weighted( thr, diagoff, uplo, m, n, bf,
		                               handle_edge_low,
		                               j_start_thr, j_end_thr );

		// Reverse the indexing basis for the subpartition ranges so that
		// the indices, relative to left-to-right iteration through the
		// unrotated upper-stored trapezoid, map to the correct columns
		// (relative to the diagonal). This amounts to subtracting the
		// range from n.
		bli_reverse_index_direction( *j_start_thr, *j_end_thr, n );
	}

	return area;
}

siz_t bli_get_range_weighted_l2r( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	siz_t area;

	// This function assigns area-weighted ranges in the n dimension
	// where the total range spans 0 to n-1 with 0 at the left end and
	// n-1 at the right end.

	if ( bli_obj_intersects_diag( *a ) &&
	     bli_obj_is_upper_or_lower( *a ) )
	{
		doff_t diagoff = bli_obj_diag_offset( *a );
		uplo_t uplo    = bli_obj_uplo( *a );
		dim_t  m       = bli_obj_length( *a );
		dim_t  n       = bli_obj_width( *a );

		// Support implicit transposition.
		if ( bli_obj_has_trans( *a ) )
		{
			bli_reflect_about_diag( diagoff, uplo, m, n );
		}

		area = bli_get_range_weighted( thr, diagoff, uplo, m, n, bf,
		                               FALSE, start, end );
	}
	else // if dense or zeros
	{
		area = bli_get_range_l2r( thr, a, bf,
		                          start, end );
	}

	return area;
}

siz_t bli_get_range_weighted_r2l( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	siz_t area;

	// This function assigns area-weighted ranges in the n dimension
	// where the total range spans 0 to n-1 with 0 at the right end and
	// n-1 at the left end.

	if ( bli_obj_intersects_diag( *a ) &&
	     bli_obj_is_upper_or_lower( *a ) )
	{
		doff_t diagoff = bli_obj_diag_offset( *a );
		uplo_t uplo    = bli_obj_uplo( *a );
		dim_t  m       = bli_obj_length( *a );
		dim_t  n       = bli_obj_width( *a );

		// Support implicit transposition.
		if ( bli_obj_has_trans( *a ) )
		{
			bli_reflect_about_diag( diagoff, uplo, m, n );
		}

		bli_rotate180_trapezoid( diagoff, uplo );

		area = bli_get_range_weighted( thr, diagoff, uplo, m, n, bf,
		                               TRUE, start, end );
	}
	else // if dense or zeros
	{
		area = bli_get_range_r2l( thr, a, bf,
		                          start, end );
	}

	return area;
}

siz_t bli_get_range_weighted_t2b( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	siz_t area;

	// This function assigns area-weighted ranges in the m dimension
	// where the total range spans 0 to m-1 with 0 at the top end and
	// m-1 at the bottom end.

	if ( bli_obj_intersects_diag( *a ) &&
	     bli_obj_is_upper_or_lower( *a ) )
	{
		doff_t diagoff = bli_obj_diag_offset( *a );
		uplo_t uplo    = bli_obj_uplo( *a );
		dim_t  m       = bli_obj_length( *a );
		dim_t  n       = bli_obj_width( *a );

		// Support implicit transposition.
		if ( bli_obj_has_trans( *a ) )
		{
			bli_reflect_about_diag( diagoff, uplo, m, n );
		}

		bli_reflect_about_diag( diagoff, uplo, m, n );

		area = bli_get_range_weighted( thr, diagoff, uplo, m, n, bf,
		                               FALSE, start, end );
	}
	else // if dense or zeros
	{
		area = bli_get_range_t2b( thr, a, bf,
		                          start, end );
	}

	return area;
}

siz_t bli_get_range_weighted_b2t( void* thr, obj_t* a, dim_t bf, dim_t* start, dim_t* end )
{
	siz_t area;

	// This function assigns area-weighted ranges in the m dimension
	// where the total range spans 0 to m-1 with 0 at the bottom end and
	// m-1 at the top end.

	if ( bli_obj_intersects_diag( *a ) &&
	     bli_obj_is_upper_or_lower( *a ) )
	{
		doff_t diagoff = bli_obj_diag_offset( *a );
		uplo_t uplo    = bli_obj_uplo( *a );
		dim_t  m       = bli_obj_length( *a );
		dim_t  n       = bli_obj_width( *a );

		// Support implicit transposition.
		if ( bli_obj_has_trans( *a ) )
		{
			bli_reflect_about_diag( diagoff, uplo, m, n );
		}

		bli_reflect_about_diag( diagoff, uplo, m, n );

		bli_rotate180_trapezoid( diagoff, uplo );

		area = bli_get_range_weighted( thr, diagoff, uplo, m, n, bf,
		                               TRUE, start, end );
	}
	else // if dense or zeros
	{
		area = bli_get_range_b2t( thr, a, bf,
		                          start, end );
	}

	return area;
}


// Some utilities
dim_t bli_read_nway_from_env( char* env )
{
    dim_t number = 1;
    char* str = getenv( env );
    if( str != NULL )
    {   
        number = strtol( str, NULL, 10 );
    }   
    return number;
}

dim_t bli_gcd( dim_t x, dim_t y )
{
    while( y != 0 ) {
        dim_t t = y;
        y = x % y;
        x = t;
    }
    return x;
}

dim_t bli_lcm( dim_t x, dim_t y)
{
    return x * y / bli_gcd( x, y );
}
