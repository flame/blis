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

// Redefine level-1m kernel API names to induce prototypes.

#undef  packm_2xk_ker_name
#define packm_2xk_ker_name  packm_2xk_ref
#undef  packm_3xk_ker_name
#define packm_3xk_ker_name  packm_3xk_ref
#undef  packm_4xk_ker_name
#define packm_4xk_ker_name  packm_4xk_ref
#undef  packm_6xk_ker_name
#define packm_6xk_ker_name  packm_6xk_ref
#undef  packm_8xk_ker_name
#define packm_8xk_ker_name  packm_8xk_ref
#undef  packm_10xk_ker_name
#define packm_10xk_ker_name packm_10xk_ref
#undef  packm_12xk_ker_name
#define packm_12xk_ker_name packm_12xk_ref
#undef  packm_14xk_ker_name
#define packm_14xk_ker_name packm_14xk_ref
#undef  packm_16xk_ker_name
#define packm_16xk_ker_name packm_16xk_ref
#undef  packm_24xk_ker_name
#define packm_24xk_ker_name packm_24xk_ref
#undef  packm_30xk_ker_name
#define packm_30xk_ker_name packm_30xk_ref

// Include the level-1m kernel API template.

#include "bli_l1m_ker.h"

