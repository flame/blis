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

struct gemm_var_cntl_s
{
    cntl_t       cntl; //this field must be present and come first
    gemm_ukr_vft ukr;
    const void*  params;
    bool         row_pref;
};
typedef struct gemm_var_cntl_s gemm_var_cntl_t;

struct gemm_cntl_s
{
         part_cntl_t part_jc;
         part_cntl_t part_pc;
    packm_def_cntl_t pack_b;
         part_cntl_t part_ic;
    packm_def_cntl_t pack_a;
     gemm_var_cntl_t ker;
              cntl_t ir_loop;
};
typedef struct gemm_cntl_s gemm_cntl_t;

// -----------------------------------------------------------------------------

BLIS_INLINE gemm_ukr_vft bli_gemm_var_cntl_ukr( const cntl_t* cntl )
{
    return ( ( const gemm_var_cntl_t* ) cntl )->ukr;
}

BLIS_INLINE bool bli_gemm_var_cntl_row_pref( const cntl_t* cntl )
{
    return ( ( const gemm_var_cntl_t* ) cntl )->row_pref;
}

BLIS_INLINE const void* bli_gemm_var_cntl_params( const cntl_t* cntl )
{
    return ( ( const gemm_var_cntl_t* ) cntl )->params;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_gemm_var_cntl_set_ukr( gemm_ukr_vft ukr, cntl_t* cntl )
{
    ( ( gemm_var_cntl_t* ) cntl )->ukr = ukr;
}

BLIS_INLINE void bli_gemm_var_cntl_set_row_pref( bool row_pref, cntl_t* cntl )
{
    ( ( gemm_var_cntl_t* ) cntl )->row_pref = row_pref;
}

BLIS_INLINE void bli_gemm_var_cntl_set_params( const void* params, cntl_t* cntl )
{
    ( ( gemm_var_cntl_t* ) cntl )->params = params;
}

// -----------------------------------------------------------------------------

void bli_gemm_var_cntl_init_node
     (
       void_fp          var_func,
       gemm_ukr_vft     ukr,
       bool             row_pref,
       gemm_var_cntl_t* cntl
     );

void bli_gemm_cntl_init
     (
             opid_t       family,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             pack_t       schema_a,
             pack_t       schema_b,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl
     );

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_gemm_cntl_set_ukr( gemm_ukr_vft ukr, bool row_pref, gemm_cntl_t* cntl )
{
    bli_gemm_var_cntl_set_ukr( ukr, ( cntl_t* )&cntl->ker );
    bli_gemm_var_cntl_set_row_pref( row_pref, ( cntl_t* )&cntl->ker );
}

BLIS_INLINE void bli_gemm_cntl_set_params( const void* params, gemm_cntl_t* cntl )
{
    bli_gemm_var_cntl_set_params( params, ( cntl_t* )&cntl->ker );
}

BLIS_INLINE void bli_gemm_cntl_set_var( l3_var_oft var, gemm_cntl_t* cntl )
{
    bli_cntl_set_var_func( ( void_fp )var, ( cntl_t* )&cntl->ker );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_ukr( packm_ker_vft ukr, gemm_cntl_t* cntl )
{
    bli_packm_def_cntl_set_ukr( ukr, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_params( const void* params, gemm_cntl_t* cntl )
{
    bli_packm_def_cntl_set_ukr_params( params, ( cntl_t* )&cntl->pack_a );
    bli_packm_cntl_set_variant_params( params, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_var( packm_var_oft var, gemm_cntl_t* cntl )
{
    bli_packm_cntl_set_variant( var, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_ukr( packm_ker_vft ukr, gemm_cntl_t* cntl )
{
    bli_packm_def_cntl_set_ukr( ukr, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_params( const void* params, gemm_cntl_t* cntl )
{
    bli_packm_def_cntl_set_ukr_params( params, ( cntl_t* )&cntl->pack_b );
    bli_packm_cntl_set_variant_params( params, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_var( packm_var_oft var, gemm_cntl_t* cntl )
{
    bli_packm_cntl_set_variant( var, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_mr( dim_t mr_def, dim_t mr_pack, gemm_cntl_t* cntl )
{
    bli_packm_def_cntl_set_bmult_m_def( mr_def, ( cntl_t* )&cntl->pack_a );
    bli_packm_def_cntl_set_bmult_m_pack( mr_pack, ( cntl_t* )&cntl->pack_a );
    bli_part_cntl_set_b_mult( mr_def, ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE void bli_gemm_cntl_set_nr( dim_t nr_def, dim_t nr_pack, gemm_cntl_t* cntl )
{
    bli_packm_def_cntl_set_bmult_m_def( nr_def, ( cntl_t* )&cntl->pack_b );
    bli_packm_def_cntl_set_bmult_m_pack( nr_pack, ( cntl_t* )&cntl->pack_b );
    bli_part_cntl_set_b_mult( nr_def, ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE void bli_gemm_cntl_set_mc( dim_t mc_def, dim_t mc_max, gemm_cntl_t* cntl )
{
    bli_part_cntl_set_b_alg( mc_def, ( cntl_t* )&cntl->part_ic );
    bli_part_cntl_set_b_max( mc_max, ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE void bli_gemm_cntl_set_nc( dim_t nc_def, dim_t nc_max, gemm_cntl_t* cntl )
{
    bli_part_cntl_set_b_alg( nc_def, ( cntl_t* )&cntl->part_jc );
    bli_part_cntl_set_b_max( nc_max, ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE void bli_gemm_cntl_set_kc( dim_t kc_def, dim_t kc_max, gemm_cntl_t* cntl )
{
    bli_part_cntl_set_b_alg( kc_def, ( cntl_t* )&cntl->part_pc );
    bli_part_cntl_set_b_max( kc_max, ( cntl_t* )&cntl->part_pc );
}

BLIS_EXPORT_BLIS void bli_gemm_cntl_finalize
     (
             opid_t       family,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             gemm_cntl_t* cntl
     );

