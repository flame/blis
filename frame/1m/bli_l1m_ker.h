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


//
// Define template prototypes for level-1m kernels.
//

// native packm kernels

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     );

INSERT_GENTPROT_BASIC( packm_2xk_ker_name )
INSERT_GENTPROT_BASIC( packm_3xk_ker_name )
INSERT_GENTPROT_BASIC( packm_4xk_ker_name )
INSERT_GENTPROT_BASIC( packm_6xk_ker_name )
INSERT_GENTPROT_BASIC( packm_8xk_ker_name )
INSERT_GENTPROT_BASIC( packm_10xk_ker_name )
INSERT_GENTPROT_BASIC( packm_12xk_ker_name )
INSERT_GENTPROT_BASIC( packm_14xk_ker_name )
INSERT_GENTPROT_BASIC( packm_16xk_ker_name )
INSERT_GENTPROT_BASIC( packm_24xk_ker_name )
INSERT_GENTPROT_BASIC( packm_30xk_ker_name )


// 3mis packm kernels

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     );

INSERT_GENTPROT_BASIC( packm_2xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_4xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_6xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_8xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_10xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_12xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_14xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_16xk_3mis_ker_name )
INSERT_GENTPROT_BASIC( packm_30xk_3mis_ker_name )


// 4mi packm kernels

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p, inc_t is_p, inc_t ldp  \
     );

INSERT_GENTPROT_BASIC( packm_2xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_4xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_6xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_8xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_10xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_12xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_14xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_16xk_4mi_ker_name )
INSERT_GENTPROT_BASIC( packm_30xk_4mi_ker_name )


// rih packm kernels

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     );

INSERT_GENTPROT_BASIC( packm_2xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_3xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_4xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_6xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_8xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_10xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_12xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_14xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_16xk_rih_ker_name )
INSERT_GENTPROT_BASIC( packm_30xk_rih_ker_name )

