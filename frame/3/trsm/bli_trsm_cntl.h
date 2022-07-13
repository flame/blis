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
    cntl_t loop5;
    cntl_t loop4;
    cntl_t packb;
    cntl_t loop3;
    cntl_t gemm_packa;
    cntl_t gemm_loop2;
    cntl_t gemm_loop1;
    cntl_t trsm_packa;
    cntl_t trsm_loop2;
    cntl_t trsm_loop1;
    part_params_t nc;
    part_params_t kc;
    part_params_t mc;
    packm_params_t gemm_packa_params;
    packm_params_t trsm_packa_params;
    packm_params_t packb_params;
    gemm_params_t ker_params;
} trsm_cntl_t;

cntl_t* bli_trsm_cntl_create
     (
             trsm_cntl_t* cntl,
             side_t       side,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
       const cntx_t*      cntx
     );

cntl_t* bli_trsm_l_cntl_create
     (
             trsm_cntl_t* cntl,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
       const cntx_t*      cntx
     );

cntl_t* bli_trsm_r_cntl_create
     (
             trsm_cntl_t* cntl,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
       const cntx_t*      cntx
     );

