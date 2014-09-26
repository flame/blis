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

#ifndef BLIS_KERNEL_TYPE_DEFS_H
#define BLIS_KERNEL_TYPE_DEFS_H


//
// -- BLIS kernel types --------------------------------------------------------
//

// Here we generate typedef statements that generate custom types for
// micro-kernel function pointers. Note that we use the function
// prototype-generating macro since it takes the same arguments we need
// to define our types.


// -- gemm micro-kernel --

#undef  GENTPROT
#define GENTPROT( ctype, ch, tname ) \
\
typedef void \
(*PASTECH(ch,tname))( \
                      dim_t           k, \
                      ctype* restrict alpha, \
                      ctype* restrict a, \
                      ctype* restrict b, \
                      ctype* restrict beta, \
                      ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                      auxinfo_t*      data  \
                    );

INSERT_GENTPROT_BASIC( gemm_ukr_t )


// -- trsm_l/u micro-kernels --

#undef  GENTPROT
#define GENTPROT( ctype, ch, tname ) \
\
typedef void \
(*PASTECH(ch,tname))( \
                      ctype* restrict a, \
                      ctype* restrict b, \
                      ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                      auxinfo_t*      data  \
                    );

INSERT_GENTPROT_BASIC( trsm_ukr_t )


// -- gemmtrsm_l/u micro-kernel --

#undef  GENTPROT
#define GENTPROT( ctype, ch, tname ) \
\
typedef void \
(*PASTECH(ch,tname))( \
                      dim_t           k, \
                      ctype* restrict alpha, \
                      ctype* restrict a1x, \
                      ctype* restrict a11, \
                      ctype* restrict bx1, \
                      ctype* restrict b11, \
                      ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
                      auxinfo_t*      data  \
                    );

INSERT_GENTPROT_BASIC( gemmtrsm_ukr_t )


// -- packm_struc_cxk kernel --

#undef  GENTPROT
#define GENTPROT( ctype, ch, tname ) \
\
typedef void \
(*PASTECH(ch,tname))( \
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
                      ctype* restrict p, inc_t rs_p, inc_t cs_p  \
                    );

INSERT_GENTPROT_BASIC( packm_ker_t )



#endif

