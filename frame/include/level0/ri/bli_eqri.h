/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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

#ifndef BLIS_EQRI_H
#define BLIS_EQRI_H


// eqri (passed by value)

#define bli_seqri( ar, ai, br, bi )  ( (ar) == (br) )
#define bli_deqri( ar, ai, br, bi )  ( (ar) == (br) )
#define bli_ceqri( ar, ai, br, bi )  ( (ar) == (br) && (ai) == (bi) )
#define bli_zeqri( ar, ai, br, bi )  ( (ar) == (br) && (ai) == (bi) )
#define bli_ieqri( ar, ai, br, bi )  ( (ar) == (br) )


// eq1ri

#define bli_seq1ri( ar, ai )  bli_seqri( (ar), (ai), 1.0F, 0.0F )
#define bli_deq1ri( ar, ai )  bli_deqri( (ar), (ai), 1.0,  0.0  )
#define bli_ceq1ri( ar, ai )  bli_ceqri( (ar), (ai), 1.0F, 0.0F )
#define bli_zeq1ri( ar, ai )  bli_zeqri( (ar), (ai), 1.0,  0.0  )
#define bli_ieq1ri( ar, ai )  bli_ieqri( (ar), (ai), 1,    0    )


// eq0ri

#define bli_seq0ri( ar, ai )  bli_seqri( (ar), (ai), 0.0F, 0.0F )
#define bli_deq0ri( ar, ai )  bli_deqri( (ar), (ai), 0.0,  0.0  )
#define bli_ceq0ri( ar, ai )  bli_ceqri( (ar), (ai), 0.0F, 0.0F )
#define bli_zeq0ri( ar, ai )  bli_zeqri( (ar), (ai), 0.0,  0.0  )
#define bli_ieq0ri( ar, ai )  bli_ieqri( (ar), (ai), 0,    0    )


// eqm1ri

#define bli_seqm1ri( ar, ai )  bli_seqri( (ar), (ai), -1.0F, 0.0F )
#define bli_deqm1ri( ar, ai )  bli_deqri( (ar), (ai), -1.0,  0.0  )
#define bli_ceqm1ri( ar, ai )  bli_ceqri( (ar), (ai), -1.0F, 0.0F )
#define bli_zeqm1ri( ar, ai )  bli_zeqri( (ar), (ai), -1.0,  0.0  )
#define bli_ieqm1ri( ar, ai )  bli_ieqri( (ar), (ai), -1,    0    )



#endif
