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

struct packv_s
{
	impl_t         impl_type;
	varnum_t       var_num;
	pack_t         pack_schema;
	blksz_t*       mult_dim;
};
typedef struct packv_s packv_t;

#define cntl_mult_dim( cntl ) cntl->mult_dim

#define cntl_sub_packv( cntl )     cntl->sub_packv
#define cntl_sub_packv_x( cntl )   cntl->sub_packv_x
#define cntl_sub_packv_x1( cntl )  cntl->sub_packv_x1
#define cntl_sub_packv_y( cntl )   cntl->sub_packv_y
#define cntl_sub_packv_y1( cntl )  cntl->sub_packv_y1

void     bli_packv_cntl_init( void );
void     bli_packv_cntl_finalize( void );
packv_t* bli_packv_cntl_obj_create( impl_t     impl_type,
                                    varnum_t   var_num,
                                    blksz_t*   mult_dim,
                                    pack_t     pack_schema );
void bli_packv_cntl_obj_init( packv_t*   cntl,
                              impl_t     impl_type,
                              varnum_t   var_num,
                              blksz_t*   mult_dim,
                              pack_t     pack_schema );

