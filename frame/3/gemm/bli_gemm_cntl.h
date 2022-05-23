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

typedef struct
{
    dim_t blksz;
    dim_t blksz_max;
    dim_t bmult;
} part_cntl_t;

typedef struct
{
    blksz_t mr, nr;
	func_t ukr;
    mbool_t ukr_row_pref;
} gemm_params_t;

typedef struct
{
    const gemm_params_t* params;
    num_t dt_comp;
    bool ukr_row_pref;
} gemm_ker_cntl_t;

typedef struct
{
    cntl_t loop5;
    cntl_t loop4;
    cntl_t loop3;
    cntl_t loop2;
    cntl_t loop1;
    cntl_t packa;
    cntl_t packb;
    part_cntl_t mc;
    part_cntl_t nc;
    part_cntl_t kc;
    packm_cntl_t packa_params;
    packm_cntl_t packb_params;
    gemm_ker_cntl_t ker_params;
} goto_cntl_t;

BLIS_INLINE dim_t bli_cntl_part_blksz_def( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl->params; return ppp->blksz;
}

BLIS_INLINE dim_t bli_cntl_part_blksz_max( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl->params; return ppp->blksz_max;
}

BLIS_INLINE dim_t bli_cntl_part_bmult( const cntl_t* cntl )
{
	part_cntl_t* ppp = ( part_cntl_t* )cntl->params; return ppp->bmult;
}

// -----------------------------------------------------------------------------

cntl_t* bli_gemm_cntl_create
     (
             goto_cntl_t* cntl,
             opid_t       family,
             num_t        dt_comp,
             obj_t*       a,
             obj_t*       b,
             obj_t*       c,
       const cntx_t*      cntx
     );

