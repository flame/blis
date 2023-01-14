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

#ifndef BLIS_LT_H
#define BLIS_LT_H


// lt (passed by value)

#define bli_slt( a, b )  (          (a) <          (b) )
#define bli_dlt( a, b )  (          (a) <          (b) )
#define bli_clt( a, b )  ( bli_creal(a) < bli_creal(b) )
#define bli_zlt( a, b )  ( bli_zreal(a) < bli_zreal(b) )
#define bli_ilt( a, b )  (          (a) <          (b) )

// lt0

#define bli_slt0( a )  (          (a) < 0.0F )
#define bli_dlt0( a )  (          (a) < 0.0  )
#define bli_clt0( a )  ( bli_creal(a) < 0.0F )
#define bli_zlt0( a )  ( bli_zreal(a) < 0.0  )

// gt (passed by value)

#define bli_sgt( a, b )  (          (a) >          (b) )
#define bli_dgt( a, b )  (          (a) >          (b) )
#define bli_cgt( a, b )  ( bli_creal(a) > bli_creal(b) )
#define bli_zgt( a, b )  ( bli_zreal(a) > bli_zreal(b) )
#define bli_igt( a, b )  (          (a) >          (b) )

// gt0

#define bli_sgt0( a )  (          (a) > 0.0F )
#define bli_dgt0( a )  (          (a) > 0.0  )
#define bli_cgt0( a )  ( bli_creal(a) > 0.0F )
#define bli_zgt0( a )  ( bli_zreal(a) > 0.0  )



#endif
