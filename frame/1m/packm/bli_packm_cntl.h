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

struct packm_params_s
{
    func2_t ukr;
};
typedef struct packm_params_s packm_params_t;

struct packm_cntl_s
{
	      bool      does_invert_diag;
	      bool      rev_iter_if_upper;
	      bool      rev_iter_if_lower;
	      pack_t    pack_schema;
	      packbuf_t pack_buf_type;
          num_t     dt_pack;
          dim_t     mr;
          dim_t     nr;
          dim_t     kr;
    const void*     params;
};
typedef struct packm_cntl_s packm_cntl_t;

BLIS_INLINE const blksz_t* bli_cntl_packm_mr( const cntl_t* cntl )
{
	const packm_params_t* ppp = ( ( packm_cntl_t* )cntl->params )->params; return &ppp->mr;
}

BLIS_INLINE const blksz_t* bli_cntl_packm_nr( const cntl_t* cntl )
{
	const packm_params_t* ppp = ( ( packm_cntl_t* )cntl->params )->params; return &ppp->nr;
}

BLIS_INLINE const blksz_t* bli_cntl_packm_kr( const cntl_t* cntl )
{
	const packm_params_t* ppp = ( ( packm_cntl_t* )cntl->params )->params; return &ppp->kr;
}

BLIS_INLINE bool bli_cntl_packm_does_invert_diag( const cntl_t* cntl )
{
	packm_cntl_t* ppp = ( packm_cntl_t* )cntl->params; return ppp->does_invert_diag;
}

BLIS_INLINE bool bli_cntl_packm_rev_iter_if_upper( const cntl_t* cntl )
{
	packm_cntl_t* ppp = ( packm_cntl_t* )cntl->params; return ppp->rev_iter_if_upper;
}

BLIS_INLINE bool bli_cntl_packm_rev_iter_if_lower( const cntl_t* cntl )
{
	packm_cntl_t* ppp = ( packm_cntl_t* )cntl->params; return ppp->rev_iter_if_lower;
}

BLIS_INLINE pack_t bli_cntl_packm_pack_schema( const cntl_t* cntl )
{
	packm_cntl_t* ppp = ( packm_cntl_t* )cntl->params; return ppp->pack_schema;
}

BLIS_INLINE packbuf_t bli_cntl_packm_pack_buf_type( const cntl_t* cntl )
{
	packm_cntl_t* ppp = ( packm_cntl_t* )cntl->params; return ppp->pack_buf_type;
}

BLIS_INLINE packm_ker_vft bli_cntl_packm_ukr_mr( num_t dt_c, num_t dt_p, const cntl_t* cntl )
{
	const packm_params_t* ppp = ( ( packm_cntl_t* )cntl->params )->params; return ppp->ukr_mr[ dt_c ][ dt_p ];
}

BLIS_INLINE packm_ker_vft bli_cntl_packm_ukr_nr( num_t dt_c, num_t dt_p, const cntl_t* cntl )
{
	const packm_params_t* ppp = ( ( packm_cntl_t* )cntl->params )->params; return ppp->ukr_nr[ dt_c ][ dt_p ];
}

