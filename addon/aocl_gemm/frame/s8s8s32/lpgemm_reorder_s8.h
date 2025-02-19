/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_REORDER_H_S8
#define LPGEMM_REORDER_H_S8

#include "lpgemm_types.h"

void unpackb_nr64_s8_reference
(
  int8_t*       b,
  int8_t*       unpack_b_buffer,
  const dim_t	  NC,
  const dim_t   KC,
  dim_t         rs_b,
  dim_t         cs_b
);

void unreorderb_nr64_s8s8s32os32_reference
(
  lpgemm_obj_t*  b,
  lpgemm_obj_t*  b_reorder,
  rntm_t*        rntm,
  lpgemm_cntx_t* lcntx
);

void reorderb_nr64_s8s8s32o32
(
  lpgemm_obj_t*  b,
  lpgemm_obj_t*  b_reorder,
  rntm_t*        rntm,
  lpgemm_cntx_t* lcntx
);

void reorderb_nr64_s8s8s32o32_sym_quant
     (
       lpgemm_obj_t*  b,
       lpgemm_obj_t*  b_reorder,
       rntm_t*        rntm,
       lpgemm_cntx_t* lcntx,
       dim_t          group_size
     );

void reordera_mr6_s8s8s32o32
(
  lpgemm_obj_t*  a,
  lpgemm_obj_t*  a_reorder,
  rntm_t*        rntm,
  lpgemm_cntx_t* lcntx
);

#endif //LPGEMM_REORDER_H_S8

