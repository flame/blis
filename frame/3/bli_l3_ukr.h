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

//
// Define template prototypes for level-3 micro-kernels.
//

// Note: Instead of defining function prototype macro templates and then
// instantiating those macros to define the individual function prototypes,
// we simply alias the official operations' prototypes as defined in
// bli_l3_ukr_prot.h.

#undef  GENTPROT
#define GENTPROT GEMM_UKR_PROT

INSERT_GENTPROT_BASIC0( gemm_ukr_name )


#undef  GENTPROT
#define GENTPROT GEMMTRSM_UKR_PROT

INSERT_GENTPROT_BASIC0( gemmtrsm_l_ukr_name )
INSERT_GENTPROT_BASIC0( gemmtrsm_u_ukr_name )


#undef  GENTPROT
#define GENTPROT TRSM_UKR_PROT

INSERT_GENTPROT_BASIC0( trsm_l_ukr_name )
INSERT_GENTPROT_BASIC0( trsm_u_ukr_name )

