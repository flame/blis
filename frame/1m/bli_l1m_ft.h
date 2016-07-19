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

#ifndef BLIS_L1M_FT_H
#define BLIS_L1M_FT_H


//
// -- Level-1m function types --------------------------------------------------
//

// packm

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH2(ch,opname,tsuf)) \
     ( \
       struc_t         strucc, \
       doff_t          diagoffc, \
       diag_t          diagc, \
       uplo_t          uploc, \
       conj_t          conjc, \
       pack_t          schema, \
       bool_t          invdiag, \
       dim_t           m_panel, \
       dim_t           n_panel, \
       dim_t           m_panel_max, \
       dim_t           n_panel_max, \
       ctype* restrict kappa, \
       ctype* restrict c, inc_t rs_c, inc_t cs_c, \
       ctype* restrict p, inc_t rs_p, inc_t cs_p, \
                          inc_t is_p, \
       cntx_t*         cntx  \
     );

INSERT_GENTDEF( packm )


// NOTE: the following macros generate packm kernel function type definitions
// that are "ctyped" and void-typed, for each of the floating-point datatypes.
// However, we will only make use of the void-typed definitions because the
// functions such as bli_?packm_cxk() (currently) use arrays of function
// pointers to store and access the function pointers for various unrolling
// (register blocksize) values, and therefore they must all be of the same
// type (hence the use of void* for kappa, a, and p).

// packm_ker

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH2(ch,opname,tsuf)) \
     ( \
       conj_t          conja, \
       dim_t           n, \
       ctype* restrict kappa, \
       ctype* restrict a, inc_t inca, inc_t lda, \
       ctype* restrict p,             inc_t ldp  \
     );

INSERT_GENTDEF( packm_cxk_ker )


// packm_3mis_ker

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH2(ch,opname,tsuf)) \
     ( \
       conj_t          conja, \
       dim_t           n, \
       ctype* restrict kappa, \
       ctype* restrict a, inc_t inca, inc_t lda, \
       ctype* restrict p, inc_t is_p, inc_t ldp  \
     );

INSERT_GENTDEF( packm_cxk_3mis_ker )
INSERT_GENTDEF( packm_cxk_4mi_ker )


// packm_rih_ker

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef void (*PASTECH2(ch,opname,tsuf)) \
     ( \
       conj_t          conja, \
       pack_t          schema, \
       dim_t           n, \
       ctype* restrict kappa, \
       ctype* restrict a, inc_t inca, inc_t lda, \
       ctype* restrict p,             inc_t ldp  \
     );

INSERT_GENTDEF( packm_cxk_rih_ker )



#endif

