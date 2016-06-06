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

#include "bli_l3_cntx.h"
#include "bli_l3_check.h"

#include "bli_l3_ft.h"
#include "bli_l3_oft.h"

#include "bli_l3_blocksize.h"
#include "bli_l3_prune.h"

// Prototype object APIs with and without contexts.
#include "bli_oapi_w_cntx.h"
#include "bli_l3_oapi.h"
#include "bli_oapi_wo_cntx.h"
#include "bli_l3_oapi.h"

#include "bli_l3_tapi.h"

#include "bli_l3_ukr_oapi.h"
#include "bli_l3_ukr_tapi.h"

// Prototype reference micro-kernels.
#include "bli_l3_ukr_ref.h"

// Operation-specific headers
#include "bli_gemm.h"
#include "bli_hemm.h"
#include "bli_herk.h"
#include "bli_her2k.h"
#include "bli_symm.h"
#include "bli_syrk.h"
#include "bli_syr2k.h"
#include "bli_trmm.h"
#include "bli_trmm3.h"
#include "bli_trsm.h"

