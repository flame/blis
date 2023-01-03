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


struct part_cntl_s
{
	cntl_t cntl; // cntl field must be present and come first.
	dim_t  b_alg;
	dim_t  b_max;
	dim_t  b_mult;
    dir_t  direct;
    bool   use_weighted;
};
typedef struct part_cntl_s part_cntl_t;

BLIS_INLINE dim_t bli_part_cntl_b_alg( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl; return ppp->b_alg;
}

BLIS_INLINE dim_t bli_part_cntl_b_max( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl; return ppp->b_max;
}

BLIS_INLINE dim_t bli_part_cntl_b_mult( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl; return ppp->b_mult;
}

BLIS_INLINE dir_t bli_part_cntl_direct( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl; return ppp->direct;
}

BLIS_INLINE bool bli_part_cntl_use_weighted( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl; return ppp->use_weighted;
}

// -----------------------------------------------------------------------------

void bli_part_cntl_init_node
     (
       void_fp      var_func,
       bszid_t      bszid,
       dim_t        b_alg,
       dim_t        b_max,
       dim_t        b_mult,
       dir_t        direct,
       bool         use_weighted,
       cntl_t*      sub_node,
       part_cntl_t* cntl
     );

