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

#ifndef BLIS_AUXINFO_MACRO_DEFS_H
#define BLIS_AUXINFO_MACRO_DEFS_H


// auxinfo_t field query

#define bli_auxinfo_schema_a( auxinfo )  ( (auxinfo)->schema_a )
#define bli_auxinfo_schema_b( auxinfo )  ( (auxinfo)->schema_b )

#define bli_auxinfo_next_a( auxinfo )    ( (auxinfo)->a_next )
#define bli_auxinfo_next_b( auxinfo )    ( (auxinfo)->b_next )

#define bli_auxinfo_is_a( auxinfo )      ( (auxinfo)->is_a )
#define bli_auxinfo_is_b( auxinfo )      ( (auxinfo)->is_b )


// auxinfo_t field modification

#define bli_auxinfo_set_schema_a( schema, auxinfo )   { (auxinfo).schema_a = schema; }
#define bli_auxinfo_set_schema_b( schema, auxinfo )   { (auxinfo).schema_b = schema; }

#define bli_auxinfo_set_next_a( a_p, auxinfo ) { (auxinfo).a_next = a_p; }
#define bli_auxinfo_set_next_b( b_p, auxinfo ) { (auxinfo).b_next = b_p; }

#define bli_auxinfo_set_next_ab( a_p, b_p, auxinfo ) \
{ \
	bli_auxinfo_set_next_a( a_p, auxinfo ); \
	bli_auxinfo_set_next_b( b_p, auxinfo ); \
}

#define bli_auxinfo_set_is_a( is, auxinfo )   { (auxinfo).is_a = is; }
#define bli_auxinfo_set_is_b( is, auxinfo )   { (auxinfo).is_b = is; }


#endif 

