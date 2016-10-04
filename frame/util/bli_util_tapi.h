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
// Prototype BLAS-like interfaces with typed operands.
//

#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* asum, \
       cntx_t*  cntx  \
     );

INSERT_GENTPROTR_BASIC( asumv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     );

INSERT_GENTPROT_BASIC( mkherm )
INSERT_GENTPROT_BASIC( mksymm )
INSERT_GENTPROT_BASIC( mktrim )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx  \
     );

INSERT_GENTPROTR_BASIC( norm1v )
INSERT_GENTPROTR_BASIC( normfv )
INSERT_GENTPROTR_BASIC( normiv )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx  \
     );

INSERT_GENTPROTR_BASIC( norm1m )
INSERT_GENTPROTR_BASIC( normfm )
INSERT_GENTPROTR_BASIC( normim )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       char*  s1, \
       dim_t  n, \
       void*  x, inc_t incx, \
       char*  format, \
       char*  s2  \
     );

INSERT_GENTPROT_BASIC_I( printv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       char*  s1, \
       dim_t  m, \
       dim_t  n, \
       void*  x, inc_t rs_x, inc_t cs_x, \
       char*  format, \
       char*  s2  \
     );

INSERT_GENTPROT_BASIC_I( printm )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       cntx_t*  cntx  \
     );

INSERT_GENTPROT_BASIC( randv )
INSERT_GENTPROT_BASIC( randnv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       doff_t  diagoffx, \
       uplo_t  uplox, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       cntx_t* cntx  \
     );

INSERT_GENTPROT_BASIC( randm )
INSERT_GENTPROT_BASIC( randnm )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* scale, \
       ctype_r* sumsq, \
       cntx_t*  cntx  \
     );

INSERT_GENTPROTR_BASIC( sumsqv )


