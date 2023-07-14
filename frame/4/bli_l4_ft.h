/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

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

#ifndef BLIS_L4_FT_H
#define BLIS_L4_FT_H

//
// -- Level-4 function types ---------------------------------------------------
//

// chol, ttmm, hpdinv

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef err_t (*PASTECH2(ch,opname,tsuf)) \
     ( \
             uplo_t  uploa, \
             dim_t   m, \
             ctype*  a, inc_t rs_a, inc_t cs_a, \
       const cntx_t* cntx, \
             rntm_t* rntm  \
     );

INSERT_GENTDEF( chol )
INSERT_GENTDEF( ttmm )
INSERT_GENTDEF( hpdinv )

// trinv

#undef  GENTDEF
#define GENTDEF( ctype, ch, opname, tsuf ) \
\
typedef err_t (*PASTECH2(ch,opname,tsuf)) \
     ( \
             uplo_t  uploa, \
             diag_t  diaga, \
             dim_t   m, \
             ctype*  a, inc_t rs_a, inc_t cs_a, \
       const cntx_t* cntx, \
             rntm_t* rntm  \
     );

INSERT_GENTDEF( trinv )

// hevd

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef err_t (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
             bool     comp_evecs, \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
             ctype*   v, inc_t rs_v, inc_t cs_v, \
             ctype_r* e, inc_t ince, \
             ctype*   work, \
             dim_t    lwork, \
             ctype_r* rwork, \
       const cntx_t*  cntx, \
       const rntm_t*  rntm  \
     );

INSERT_GENTDEFR( hevd )

// rhevd

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef err_t (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   v, inc_t rs_v, inc_t cs_v, \
             ctype_r* e, inc_t ince, \
             ctype*   a, inc_t rs_a, inc_t cs_a, \
       const cntx_t*  cntx, \
       const rntm_t*  rntm  \
     );

INSERT_GENTDEFR( rhevd )

// hevpinv

#undef  GENTDEFR
#define GENTDEFR( ctype, ctype_r, ch, chr, opname, tsuf ) \
\
typedef err_t (*PASTECH3(ch,opname,BLIS_TAPI_EX_SUF,tsuf)) \
     ( \
             double   thresh, \
             uplo_t   uploa, \
             dim_t    m, \
       const ctype*   a, inc_t rs_a, inc_t cs_a, \
             ctype*   p, inc_t rs_p, inc_t cs_p, \
       const cntx_t*  cntx, \
       const rntm_t*  rntm  \
     );

INSERT_GENTDEFR( hevpinv )


#endif

