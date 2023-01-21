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


struct packm_cntl_s
{
	cntl_t        cntl; // cntl field must be present and come first.
	packm_var_oft var;
    const void*   params;
};
typedef struct packm_cntl_s packm_cntl_t;

struct packm_def_cntl_s
{
	packm_cntl_t  cntl; // cntl field must be present and come first.
    packm_ker_vft ukr;
    dim_t         bmult_m_def;
    dim_t         bmult_m_pack;
    dim_t         bmult_n_def;
	bool          does_invert_diag;
	bool          rev_iter_if_upper;
	bool          rev_iter_if_lower;
	pack_t        pack_schema;
	packbuf_t     pack_buf_type;
    const void*   params;
};
typedef struct packm_def_cntl_s packm_def_cntl_t;

// -----------------------------------------------------------------------------

BLIS_INLINE packm_var_oft bli_packm_cntl_variant( const cntl_t* cntl )
{
	return ( ( packm_cntl_t* ) cntl )->var;
}

BLIS_INLINE const void* bli_packm_cntl_variant_params( const cntl_t* cntl )
{
	return ( ( packm_cntl_t* ) cntl )->params;
}

BLIS_INLINE dim_t bli_packm_def_cntl_bmult_m_def( const cntl_t* cntl )
{
	return ( ( packm_def_cntl_t* ) cntl )->bmult_m_def;
}

BLIS_INLINE dim_t bli_packm_def_cntl_bmult_m_pack( const cntl_t* cntl )
{
	return ( ( packm_def_cntl_t* ) cntl )->bmult_m_pack;
}

BLIS_INLINE dim_t bli_packm_def_cntl_bmult_n_def( const cntl_t* cntl )
{
	return ( ( packm_def_cntl_t* ) cntl )->bmult_n_def;
}

BLIS_INLINE bool bli_packm_def_cntl_does_invert_diag( const cntl_t* cntl )
{
	 return ( ( packm_def_cntl_t* ) cntl )->does_invert_diag;
}

BLIS_INLINE bool bli_packm_def_cntl_rev_iter_if_upper( const cntl_t* cntl )
{
	return ( ( packm_def_cntl_t* ) cntl )->rev_iter_if_upper;
}

BLIS_INLINE bool bli_packm_def_cntl_rev_iter_if_lower( const cntl_t* cntl )
{
	return ( ( packm_def_cntl_t* ) cntl )->rev_iter_if_lower;
}

BLIS_INLINE pack_t bli_packm_def_cntl_pack_schema( const cntl_t* cntl )
{
	return ( ( packm_def_cntl_t* ) cntl )->pack_schema;
}

BLIS_INLINE packbuf_t bli_packm_def_cntl_pack_buf_type( const cntl_t* cntl )
{
	return ( ( packm_def_cntl_t* ) cntl )->pack_buf_type;
}

BLIS_INLINE packm_ker_vft bli_packm_def_cntl_ukr( const cntl_t* cntl )
{
    return ( ( packm_def_cntl_t* ) cntl )->ukr;
}

BLIS_INLINE const void* bli_packm_def_cntl_ukr_params( const cntl_t* cntl )
{
    return ( ( packm_def_cntl_t* ) cntl )->params;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_packm_cntl_set_variant( packm_var_oft var, const cntl_t* cntl )
{
	( ( packm_cntl_t* ) cntl )->var = var;
}

BLIS_INLINE void bli_packm_cntl_set_variant_params( const void* params, const cntl_t* cntl )
{
	( ( packm_cntl_t* ) cntl )->params = params;
}

BLIS_INLINE void bli_packm_def_cntl_set_bmult_m_def( dim_t bmult_m_def, const cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->bmult_m_def = bmult_m_def;
}

BLIS_INLINE void bli_packm_def_cntl_set_bmult_m_pack( dim_t bmult_m_pack, const cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->bmult_m_pack = bmult_m_pack;
}

BLIS_INLINE void bli_packm_def_cntl_set_bmult_n_def( dim_t bmult_n_def, const cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->bmult_n_def = bmult_n_def;
}

BLIS_INLINE void bli_packm_def_cntl_set_does_invert_diag( bool does_invert_diag, const cntl_t* cntl )
{
	 ( ( packm_def_cntl_t* ) cntl )->does_invert_diag = does_invert_diag;
}

BLIS_INLINE void bli_packm_def_cntl_set_rev_iter_if_upper( bool rev_iter_if_upper, const cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->rev_iter_if_upper = rev_iter_if_upper;
}

BLIS_INLINE void bli_packm_def_cntl_set_rev_iter_if_lower( bool rev_iter_if_lower, const cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->rev_iter_if_lower = rev_iter_if_lower;
}

BLIS_INLINE void bli_packm_def_cntl_set_pack_schema( pack_t pack_schema, const cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->pack_schema = pack_schema;
}

BLIS_INLINE void bli_packm_def_cntl_set_pack_buf_type( packbuf_t pack_buf_type, const cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->pack_buf_type = pack_buf_type;
}

BLIS_INLINE void bli_packm_def_cntl_set_ukr( packm_ker_vft ukr, const cntl_t* cntl )
{
    ( ( packm_def_cntl_t* ) cntl )->ukr = ukr;
}

BLIS_INLINE void bli_packm_def_cntl_set_ukr_params( const void* params, const cntl_t* cntl )
{
    ( ( packm_def_cntl_t* ) cntl )->params = params;
}

// -----------------------------------------------------------------------------

void bli_packm_cntl_init_node
     (
       void_fp       var_func,
       packm_var_oft var,
       const void*   params,
       packm_cntl_t* cntl
     );

void bli_packm_def_cntl_init_node
     (
       void_fp           var_func,
       num_t             dt_a,
       num_t             dt_p,
       dim_t             bmult_m_def,
       dim_t             bmult_m_pack,
       dim_t             bmult_n_def,
       bool              does_invert_diag,
       bool              rev_iter_if_upper,
       bool              rev_iter_if_lower,
       pack_t            pack_schema,
       packbuf_t         pack_buf_type,
       packm_def_cntl_t* cntl
     );

