/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016 Hewlett Packard Enterprise Development LP

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

#ifndef BLIS_THREAD_H
#define BLIS_THREAD_H

// Perform a sanity check to make sure the user doesn't try to enable
// both OpenMP and pthreads.
#if defined ( BLIS_ENABLE_OPENMP ) && \
    defined ( BLIS_ENABLE_PTHREADS )
  #error "BLIS_ENABLE_OPENMP and BLIS_ENABLE_PTHREADS may not be simultaneously defined."
#endif

// Here, we define BLIS_ENABLE_MULTITHREADING if either OpenMP
// or pthreads are enabled. This macro is useful in situations when
// we want to detect use of either OpenMP or pthreads (as opposed
// to neither being used).
#if defined ( BLIS_ENABLE_OPENMP ) || \
    defined ( BLIS_ENABLE_PTHREADS )
  #define BLIS_ENABLE_MULTITHREADING
#endif

// Include thread mutex (mtx_t) object definitions and prototypes.
#include "bli_mutex.h"

// Include thread communicator (thrcomm_t) object definitions and prototypes.
#include "bli_thrcomm.h"

// Include thread info (thrinfo_t) object definitions and prototypes.
#include "bli_thrinfo.h"

// Include some operation-specific thrinfo_t prototypes.
// Note that the bli_packm_thrinfo.h must be included before the others!
#include "bli_packm_thrinfo.h"
#include "bli_l3_thrinfo.h"

// Initialization-related prototypes.
void    bli_thread_init( void );
void    bli_thread_finalize( void );
bool_t  bli_thread_is_initialized( void );

// Thread range-related prototypes.
void bli_thread_get_range
     (
       thrinfo_t* thread,
       dim_t      n,
       dim_t      bf,
       bool_t     handle_edge_low,
       dim_t*     start,
       dim_t*     end
     );

#undef  GENPROT
#define GENPROT( opname ) \
\
siz_t PASTEMAC0( opname ) \
     ( \
       thrinfo_t* thr, \
       obj_t*     a, \
       blksz_t*   bmult, \
       dim_t*     start, \
       dim_t*     end  \
     );

GENPROT( thread_get_range_l2r )
GENPROT( thread_get_range_r2l )
GENPROT( thread_get_range_t2b )
GENPROT( thread_get_range_b2t )

GENPROT( thread_get_range_weighted_l2r )
GENPROT( thread_get_range_weighted_r2l )
GENPROT( thread_get_range_weighted_t2b )
GENPROT( thread_get_range_weighted_b2t )


dim_t bli_thread_get_range_width_l
     (
       doff_t diagoff_j,
       dim_t  m,
       dim_t  n_j,
       dim_t  j,
       dim_t  n_way,
       dim_t  bf,
       dim_t  bf_left,
       double area_per_thr,
       bool_t handle_edge_low
     );
siz_t bli_find_area_trap_l
     (
       dim_t  m,
       dim_t  n,
       doff_t diagoff
     );
siz_t bli_thread_get_range_weighted
     (
       thrinfo_t* thread,
       doff_t     diagoff,
       uplo_t     uplo,
       dim_t      m,
       dim_t      n,
       dim_t      bf,
       bool_t     handle_edge_low,
       dim_t*     j_start_thr,
       dim_t*     j_end_thr
     );



// Level-3 internal function type
typedef void (*l3_int_t)
     (
       obj_t* alpha,
       obj_t* a,
       obj_t* b,
       obj_t* beta,
       obj_t* c,
       void*  cntx,
       void*  cntl,
       void*  thread
     );

// Level-3 thread decorator prototype
void bli_l3_thread_decorator
     (
       dim_t    num_threads, 
       l3_int_t func, 
       obj_t*   alpha, 
       obj_t*   a,  
       obj_t*   b,  
       obj_t*   beta, 
       obj_t*   c,  
       void*    cntx, 
       void*    cntl, 
       void**   thread
     );

// Miscellaneous prototypes
dim_t bli_env_read_nway( char* env );
dim_t bli_gcd( dim_t x, dim_t y );
dim_t bli_lcm( dim_t x, dim_t y );

#endif

