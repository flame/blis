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


func_t* bli_func_obj_create( void* ptr_s, bool_t pref_s,
                             void* ptr_d, bool_t pref_d,
                             void* ptr_c, bool_t pref_c,
                             void* ptr_z, bool_t pref_z );

void bli_func_obj_init( func_t* f,
                        void*   ptr_s, bool_t pref_s,
                        void*   ptr_d, bool_t pref_d,
                        void*   ptr_c, bool_t pref_c,
                        void*   ptr_z, bool_t pref_z );

void bli_func_obj_free( func_t* f );


void* bli_func_obj_query( num_t   dt,
                          func_t* f );

bool_t bli_func_prefers_contig_rows( num_t   dt,
                                     func_t* f );

bool_t bli_func_prefers_contig_cols( num_t   dt,
                                     func_t* f );

bool_t bli_func_pref_is_sat_by( obj_t*  a,
                                func_t* f );

bool_t bli_func_pref_is_unsat_by( obj_t*  a,
                                  func_t* f );

