/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
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

#ifndef BLIS_THREAD_RANGE_H
#define BLIS_THREAD_RANGE_H

// Thread range-related prototypes.

BLIS_EXPORT_BLIS void bli_thread_range_sub
     (
       const thrinfo_t* thread,
             dim_t      n,
             dim_t      bf,
             bool       handle_edge_low,
             dim_t*     start,
             dim_t*     end
     );

#undef  GENPROT
#define GENPROT( opname ) \
\
siz_t PASTEMAC0( opname ) \
     ( \
             dir_t      direct, \
       const thrinfo_t* thr, \
       const obj_t*     a, \
       const obj_t*     b, \
       const obj_t*     c, \
       const cntl_t*    cntl, \
       const cntx_t*    cntx, \
             dim_t*     start, \
             dim_t*     end  \
     );

GENPROT( thread_range_mdim )
GENPROT( thread_range_ndim )

#undef  GENPROT
#define GENPROT( opname ) \
\
siz_t PASTEMAC0( opname ) \
     ( \
       const thrinfo_t* thr, \
       const obj_t*     a, \
       const blksz_t*   bmult, \
             dim_t*     start, \
             dim_t*     end  \
     );

GENPROT( thread_range_l2r )
GENPROT( thread_range_r2l )
GENPROT( thread_range_t2b )
GENPROT( thread_range_b2t )

GENPROT( thread_range_weighted_l2r )
GENPROT( thread_range_weighted_r2l )
GENPROT( thread_range_weighted_t2b )
GENPROT( thread_range_weighted_b2t )


dim_t bli_thread_range_width_l
     (
       doff_t diagoff_j,
       dim_t  m,
       dim_t  n_j,
       dim_t  j,
       dim_t  n_way,
       dim_t  bf,
       dim_t  bf_left,
       double area_per_thr,
       bool   handle_edge_low
     );
siz_t bli_find_area_trap_l
     (
       doff_t diagoff,
       dim_t  m,
       dim_t  n,
       dim_t  bf
     );

siz_t bli_thread_range_weighted_sub
     (
       const thrinfo_t* thread,
             doff_t     diagoff,
             uplo_t     uplo,
             uplo_t     uplo_orig,
             dim_t      m,
             dim_t      n,
             dim_t      bf,
             bool       handle_edge_low,
             dim_t*     j_start_thr,
             dim_t*     j_end_thr
     );

#endif
