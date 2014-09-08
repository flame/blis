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

struct trmm_s
{
	impl_t             impl_type;
	varnum_t           var_num;
	blksz_t*           b;
	func_t*            gemm_ukrs;
	struct scalm_s*    sub_scalm;
	struct packm_s*    sub_packm_a;
	struct packm_s*    sub_packm_b;
	struct packm_s*    sub_packm_c;
	struct trmm_s*     sub_trmm;
	struct gemm_s*     sub_gemm;
	struct unpackm_s*  sub_unpackm_c;
};
typedef struct trmm_s trmm_t;

#define cntl_sub_gemm( cntl )      cntl->sub_trmm

void    bli_trmm_cntl_init( void );
void    bli_trmm_cntl_finalize( void );
trmm_t* bli_trmm_cntl_obj_create( impl_t       impl_type,
                                  varnum_t     var_num,
                                  blksz_t*     b,
                                  func_t*      gemm_ukrs,
                                  scalm_t*     sub_scalm,
                                  packm_t*     sub_pack_a,
                                  packm_t*     sub_pack_b,
                                  packm_t*     sub_pack_c,
                                  trmm_t*      sub_trmm,
                                  gemm_t*      sub_gemm,
                                  unpackm_t*   sub_unpack_c );

