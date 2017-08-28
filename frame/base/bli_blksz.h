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

// -----------------------------------------------------------------------------

// blksz_t query

#define bli_blksz_get_def( dt, b ) \
\
	( (b)->v[ dt ] )

#define bli_blksz_get_max( dt, b ) \
\
	( (b)->e[ dt ] )

#define bli_blksz_get_def_max( def, max, dt, b ) \
{ \
	*(def) = bli_blksz_get_def( dt, b ); \
	*(max) = bli_blksz_get_max( dt, b ); \
}

// blksz_t modification

#define bli_blksz_set_def( val, dt, b ) \
{ \
	(b)->v[ dt ] = val; \
}

#define bli_blksz_set_max( val, dt, b ) \
{ \
	(b)->e[ dt ] = val; \
}

#define bli_blksz_set_def_max( def, max, dt, b ) \
{ \
	bli_blksz_set_def( def, dt, b ); \
	bli_blksz_set_max( max, dt, b ); \
}

#define bli_blksz_copy( b_src, b_dst ) \
{ \
	*(b_dst) = *(b_src); \
}

#define bli_blksz_copy_dt( dt_src, b_src, \
                           dt_dst, b_dst ) \
{ \
	const dim_t v_src = bli_blksz_get_def( dt_src, b_src ); \
	const dim_t e_src = bli_blksz_get_max( dt_src, b_src ); \
\
	bli_blksz_set_def( v_src, dt_dst, b_dst ); \
	bli_blksz_set_max( e_src, dt_dst, b_dst ); \
}

#define bli_blksz_scale_def( num, den, dt, b ) \
{ \
	(b)->v[ dt ] = ( (b)->v[ dt ] * num ) / den; \
}

#define bli_blksz_scale_max( num, den, dt, b ) \
{ \
	(b)->e[ dt ] = ( (b)->e[ dt ] * num ) / den; \
}

#if 0
#define bli_blksz_scale_dt_by( num, den, dt, b ) \
{ \
	(b)->v[ dt ] = ( (b)->v[ dt ] * num ) / den; \
	(b)->e[ dt ] = ( (b)->e[ dt ] * num ) / den; \
}
#endif

// -----------------------------------------------------------------------------

blksz_t* bli_blksz_create_ed
     (
       dim_t b_s, dim_t be_s,
       dim_t b_d, dim_t be_d,
       dim_t b_c, dim_t be_c,
       dim_t b_z, dim_t be_z
     );

blksz_t* bli_blksz_create
     (
       dim_t b_s,  dim_t b_d,  dim_t b_c,  dim_t b_z,
       dim_t be_s, dim_t be_d, dim_t be_c, dim_t be_z
     );

void bli_blksz_init_ed
     (
       blksz_t* b,
       dim_t    b_s, dim_t be_s,
       dim_t    b_d, dim_t be_d,
       dim_t    b_c, dim_t be_c,
       dim_t    b_z, dim_t be_z
     );

void bli_blksz_init
     (
       blksz_t* b,
       dim_t b_s,  dim_t b_d,  dim_t b_c,  dim_t b_z,
       dim_t be_s, dim_t be_d, dim_t be_c, dim_t be_z
     );

void bli_blksz_init_easy
     (
       blksz_t* b,
       dim_t b_s,  dim_t b_d,  dim_t b_c,  dim_t b_z
     );

void bli_blksz_free
     (
       blksz_t* b
     );

// -----------------------------------------------------------------------------

#if 0
void bli_blksz_reduce_dt_to
     (
       num_t dt_bm, blksz_t* bmult,
       num_t dt_bs, blksz_t* blksz
     );
#endif

void bli_blksz_reduce_def_to
     (
       num_t dt_bm, blksz_t* bmult,
       num_t dt_bs, blksz_t* blksz
     );

void bli_blksz_reduce_max_to
     (
       num_t dt_bm, blksz_t* bmult,
       num_t dt_bs, blksz_t* blksz
     );
// -----------------------------------------------------------------------------

dim_t bli_determine_blocksize
     (
       dir_t   direct,
       dim_t   i,
       dim_t   dim,
       obj_t*  obj,
       bszid_t bszid,
       cntx_t* cntx
     );

dim_t bli_determine_blocksize_f
     (
       dim_t   i,
       dim_t   dim,
       obj_t*  obj,
       bszid_t bszid,
       cntx_t* cntx
     );

dim_t bli_determine_blocksize_b
     (
       dim_t   i,
       dim_t   dim,
       obj_t*  obj,
       bszid_t bszid,
       cntx_t* cntx
     );

dim_t bli_determine_blocksize_f_sub
     (
       dim_t  i,
       dim_t  dim,
       dim_t  b_alg,
       dim_t  b_max
     );

dim_t bli_determine_blocksize_b_sub
     (
       dim_t  i,
       dim_t  dim,
       dim_t  b_alg,
       dim_t  b_max
     );

