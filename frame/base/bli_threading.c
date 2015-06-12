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
void bli_get_range( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, bool_t handle_edge_low, dim_t* start, dim_t* end )
{
	thrinfo_t* thread     = ( thrinfo_t* )thr;
	dim_t      n_way      = thread->n_way;
	dim_t      work_id    = thread->work_id;

	dim_t      size       = all_end - all_start;

	dim_t      n_bf_whole = size / block_factor;
	dim_t      n_bf_left  = size % block_factor;

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
		dim_t size_lo = n_bf_lo * block_factor;
		dim_t size_hi = n_bf_hi * block_factor;

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
		dim_t size_lo = n_bf_lo * block_factor;
		dim_t size_hi = n_bf_hi * block_factor;

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

void bli_get_range_l2r( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, dim_t* start, dim_t* end )
{
	bli_get_range( thr, all_start, all_end, block_factor,
	               FALSE, start, end );
}

void bli_get_range_r2l( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, dim_t* start, dim_t* end )
{
	bli_get_range( thr, all_start, all_end, block_factor,
	               TRUE, start, end );
}

void bli_get_range_t2b( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, dim_t* start, dim_t* end )
{
	bli_get_range( thr, all_start, all_end, block_factor,
	               FALSE, start, end );
}

void bli_get_range_b2t( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, dim_t* start, dim_t* end )
{
	bli_get_range( thr, all_start, all_end, block_factor,
	               TRUE, start, end );
}

void bli_get_range_weighted( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, uplo_t uplo, bool_t handle_edge_low, dim_t* start, dim_t* end )
{
	thrinfo_t* thread  = ( thrinfo_t* )thr;
	dim_t      n_way   = thread->n_way;
	dim_t      work_id = thread->work_id;
	dim_t      size    = all_end - all_start;
	dim_t      width;
	dim_t      block_fac_leftover = size % block_factor;
	dim_t      i;
	double     num;

	*start = 0;
	*end   = all_end - all_start;
	num    = size * size / ( double )n_way;

	if ( bli_is_lower( uplo ) )
	{
		dim_t cur_caucus = n_way - 1;
		dim_t len        = 0;

		// This loop computes subpartitions backwards, from the high end
		// of the index range to the low end. If the low end is assumed
		// to be on the left and the high end the right, this assignment
		// of widths is appropriate for n dimension partitioning of a
		// lower triangular matrix.
		for ( i = 0; TRUE; ++i )
		{
			width = ceil( sqrt( len*len + num ) ) - len;

			// If we need to allocate the edge case (assuming it exists)
			// to the high thread subpartition, adjust width so that it
			// contains the exact amount of leftover edge dimension so that
			// all remaining subpartitions can be multiples of block_factor.
			// If the edge case is to be allocated to the low subpartition,
			// or if there is no edge case, it is implicitly allocated to
			// the low subpartition by virtue of the fact that all other
			// subpartitions already assigned will be multiples of
			// block_factor.
			if ( i == 0 && !handle_edge_low )
			{
				if ( width % block_factor != block_fac_leftover )
					width += block_fac_leftover - ( width % block_factor );
			}
			else
			{
				if ( width % block_factor != 0 )
					width += block_factor - ( width % block_factor );
			}

			if ( cur_caucus == work_id )
			{
				*start = bli_max( 0, *end - width ) + all_start;
				*end   = *end + all_start;
				return;
			}
			else
			{
				*end -= width;
				len  += width;
				cur_caucus--;
			}
		}
	}
	else // if ( bli_is_upper( uplo ) )
	{
		// This loop computes subpartitions forwards, from the low end
		// of the index range to the high end. If the low end is assumed
		// to be on the left and the high end the right, this assignment
		// of widths is appropriate for n dimension partitioning of an
		// upper triangular matrix.
		for ( i = 0; TRUE; ++i )
		{
			width = ceil( sqrt( *start * *start + num ) ) - *start;

			if ( i == 0 && handle_edge_low )
			{
				if ( width % block_factor != block_fac_leftover )
					width += block_fac_leftover - ( width % block_factor );
			}
			else
			{
				if ( width % block_factor != 0 )
					width += block_factor - ( width % block_factor );
			}

			if ( work_id == 0 )
			{
				*start = *start + all_start;
				*end = bli_min( *start + width, all_end );
				return;
			}
			else
			{
				*start = *start + width;
				work_id--;
			}
		}
	}
}

void bli_get_range_weighted_l2r( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, uplo_t uplo, dim_t* start, dim_t* end )
{
	if ( bli_is_upper_or_lower( uplo ) )
	{
		bli_get_range_weighted( thr, all_start, all_end, block_factor,
		                        uplo, FALSE, start, end );
	}
	else // if dense or zeros
	{
		bli_get_range_l2r( thr, all_start, all_end, block_factor,
		                   start, end );
	}
}

void bli_get_range_weighted_r2l( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, uplo_t uplo, dim_t* start, dim_t* end )
{
	if ( bli_is_upper_or_lower( uplo ) )
	{
//printf( "bli_get_range_weighted_r2l: is upper or lower\n" );
		bli_toggle_uplo( uplo );
		bli_get_range_weighted( thr, all_start, all_end, block_factor,
		                        uplo, TRUE, start, end );
	}
	else // if dense or zeros
	{
//printf( "bli_get_range_weighted_r2l: is dense or zeros\n" );
		bli_get_range_r2l( thr, all_start, all_end, block_factor,
		                   start, end );
	}
}

void bli_get_range_weighted_t2b( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, uplo_t uplo, dim_t* start, dim_t* end )
{
	if ( bli_is_upper_or_lower( uplo ) )
	{
		bli_toggle_uplo( uplo );
		bli_get_range_weighted( thr, all_start, all_end, block_factor,
		                        uplo, FALSE, start, end );
	}
	else // if dense or zeros
	{
		bli_get_range_t2b( thr, all_start, all_end, block_factor,
		                   start, end );
	}
}

void bli_get_range_weighted_b2t( void* thr, dim_t all_start, dim_t all_end, dim_t block_factor, uplo_t uplo, dim_t* start, dim_t* end )
{
	if ( bli_is_upper_or_lower( uplo ) )
	{
		bli_get_range_weighted( thr, all_start, all_end, block_factor,
		                        uplo, TRUE, start, end );
	}
	else // if dense or zeros
	{
		bli_get_range_b2t( thr, all_start, all_end, block_factor,
		                   start, end );
	}
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
