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


blksz_t* bli_blksz_obj_create( dim_t b_s, dim_t be_s,
                               dim_t b_d, dim_t be_d,
                               dim_t b_c, dim_t be_c,
                               dim_t b_z, dim_t be_z );

void bli_blksz_obj_init( blksz_t* b,
                         dim_t    b_s, dim_t be_s,
                         dim_t    b_d, dim_t be_d,
                         dim_t    b_c, dim_t be_c,
                         dim_t    b_z, dim_t be_z );

void bli_blksz_obj_attach_mult_to( blksz_t* br,
                                   blksz_t* bc );

void bli_blksz_obj_attach_mr_nr_to( blksz_t* bmr,
                                    blksz_t* bnr,
                                    blksz_t* bc );

void bli_blksz_obj_free( blksz_t* b );

void bli_blksz_set_def( dim_t    val,
                        num_t    dt,
                        blksz_t* b );

void bli_blksz_set_max( dim_t    val,
                        num_t    dt,
                        blksz_t* b );

void bli_blksz_set_def_max( dim_t    def,
                            dim_t    max,
                            num_t    dt,
                            blksz_t* b );

void bli_blksz_reduce_to_mult( blksz_t* b );

dim_t bli_blksz_get_def( num_t dt, blksz_t* b );
dim_t bli_blksz_get_max( num_t dt, blksz_t* b );

dim_t bli_blksz_get_def_for_obj( obj_t* obj, blksz_t* b );
dim_t bli_blksz_get_max_for_obj( obj_t* obj, blksz_t* b );

blksz_t* bli_blksz_mult( blksz_t* b );
dim_t bli_blksz_get_mult( num_t dt, blksz_t* b );
dim_t bli_blksz_get_mult_for_obj( obj_t* obj, blksz_t* b );

blksz_t* bli_blksz_mr( blksz_t* b );
blksz_t* bli_blksz_nr( blksz_t* b );

dim_t bli_blksz_get_mr( num_t dt, blksz_t* b );
dim_t bli_blksz_get_nr( num_t dt, blksz_t* b );

// -----------------------------------------------------------------------------

dim_t bli_determine_blocksize_f( dim_t    i,
                                 dim_t    dim,
                                 obj_t*   obj,
                                 blksz_t* bsize );
dim_t bli_determine_blocksize_f_sub( dim_t  i,
                                     dim_t  dim,
                                     dim_t  b_alg,
                                     dim_t  b_max );

dim_t bli_determine_blocksize_b( dim_t    i,
                                 dim_t    dim,
                                 obj_t*   obj,
                                 blksz_t* bsize );
dim_t bli_determine_blocksize_b_sub( dim_t  i,
                                     dim_t  dim,
                                     dim_t  b_alg,
                                     dim_t  b_max );

