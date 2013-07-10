/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

// -- Level-1 BLAS --

#include "bla_amax.h"
#include "bla_asum.h"
#include "bla_axpy.h"
#include "bla_copy.h"
#include "bla_dot.h"
#include "bla_nrm2.h"
//#include "bla_rot.h"
//#include "bla_rotg.h"
//#include "bla_rotm.h"
//#include "bla_rotmg.h"
#include "bla_scal.h"
#include "bla_swap.h"


// -- Level-2 BLAS --

// dense

#include "bla_gemv.h"
#include "bla_ger.h"
#include "bla_hemv.h"
#include "bla_her.h"
#include "bla_her2.h"
#include "bla_symv.h"
#include "bla_syr.h"
#include "bla_syr2.h"
#include "bla_trmv.h"
#include "bla_trsv.h"

// packed

//#include "bla_hpmv.h"
//#include "bla_hpr.h"
//#include "bla_hpr2.h"
//#include "bla_spmv.h"
//#include "bla_spr.h"
//#include "bla_spr2.h"
//#include "bla_tpmv.h"
//#include "bla_tpsv.h"

// banded

//#include "bla_gbmv.h"
//#include "bla_hbmv.h"
//#include "bla_sbmv.h"
//#include "bla_tbmv.h"
//#include "bla_tbsv.h"


// -- Level-3 BLAS --

#include "bla_gemm.h"
#include "bla_hemm.h"
#include "bla_herk.h"
#include "bla_her2k.h"
#include "bla_symm.h"
#include "bla_syrk.h"
#include "bla_syr2k.h"
#include "bla_trmm.h"
#include "bla_trsm.h"


