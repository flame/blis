/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

#include "blis.h"
#include "bls_gemm_ukr.h"


static dim_t MR;
static dim_t NR;
static dim_t MC;
static dim_t NC;
static dim_t KC;


//
// -- gemm-like block-panel algorithm (object interface) -----------------------
//
//
//
BLIS_EXPORT_BLIS void bls_set_MC(int mr,
                                 int nr,
                                 int mc,
                                 int nc,
                                 int kc)
{
    MR = mr;
    NR = nr;
    MC = mc;
    NC = nc;
    KC = kc;
}


BLIS_EXPORT_BLIS void bls_gemm_bp_var1
     (
       const obj_t*     alpha,
       const obj_t*     a,
       const obj_t*     b,
       const obj_t*     beta,
       const obj_t*     c,
       const cntx_t*    cntx,
             thrinfo_t* thread
     )
{
    //printf("BLS_GEMM_BP_VAR!\n");
	const num_t  dt        = bli_obj_dt( c );
	//const dim_t  dt_size   = bli_dt_size( dt );
	const dim_t  dt_size   = 2;
    //printf("dt_size = %d\n", dt_size);

	const conj_t conja     = bli_obj_conj_status( a );
	const conj_t conjb     = bli_obj_conj_status( b );

	const dim_t  m         = bli_obj_length( c );
	const dim_t  n         = bli_obj_width( c );
	const dim_t  k         = bli_obj_width( a );
    //printf("***********m, n, k = %d, %d, %d \n", m, n, k);
    //printf("BLIS_M = %d", BLIS_M );
    //printf("bls_gemm_bp_var1 function, alpha, beta = %f, %f\n", *alpha, *beta);
	const char*  a_00      = bli_obj_buffer_at_off( a );
	const inc_t  rs_a      = bli_obj_row_stride( a );
	const inc_t  cs_a      = bli_obj_col_stride( a );

	const char*  b_00      = bli_obj_buffer_at_off( b );
	const inc_t  rs_b      = bli_obj_row_stride( b );
	const inc_t  cs_b      = bli_obj_col_stride( b );

	      char*  c_00      = bli_obj_buffer_at_off( c );
	const inc_t  rs_c      = bli_obj_row_stride( c );
	const inc_t  cs_c      = bli_obj_col_stride( c );


    _Float16* c_cast_use_bls_gemm     = ( _Float16* )c_00;

    //printf("\n\n\n");
    //printf("*********print C before the ukr runing in bls_gemm_bp_var1.c**********\n");
    //c_print(c_cast_use_bls_gemm, m, n, rs_c, cs_c);
    //printf("\n\n\n");

	const char*  alpha_buf = bli_obj_buffer_for_1x1( dt, alpha );
	const char*  beta_buf  = bli_obj_buffer_for_1x1( dt, beta );
	const char*  one       = bli_obj_buffer_for_1x1( dt, &BLIS_ONE );

	/* Query the context for various blocksizes. */
	//const dim_t  NR        = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); // hardcode it 
	//const dim_t  MR        = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
	//const dim_t  NC        = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );
	//const dim_t  MC        = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
	//const dim_t  KC        = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
	//const dim_t  NR        = 8; // hardcode it 
	//const dim_t  MR        = 12;
	//const dim_t  NC        = 1200;
	////const dim_t  MC        = 120;
	//const dim_t  KC        = 500;
    //printf("MC, NC, KC, MR, NR = %d, %d, %d, %d, %d\n", MC, NC, KC, MR, NR);

	/* Query the context for the microkernel address and cast it to its
	   function pointer type. */
	//gemm_ukr_ft  gemm_ukr  = bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); // hardcode it

	/* Compute partitioning step values for each matrix of each loop. */
	const inc_t jcstep_c = cs_c * dt_size;
	const inc_t jcstep_b = cs_b * dt_size;

	const inc_t pcstep_a = cs_a * dt_size;
	const inc_t pcstep_b = rs_b * dt_size;

	const inc_t icstep_c = rs_c * dt_size;
	const inc_t icstep_a = rs_a * dt_size;

	const inc_t jrstep_c = cs_c * NR * dt_size;
	const inc_t irstep_c = rs_c * MR * dt_size;

	thrinfo_t* thread_jc = bli_thrinfo_sub_node( 0, thread );    // NC loop 
	thrinfo_t* thread_pc = bli_thrinfo_sub_node( 0, thread_jc ); // KC loop
	thrinfo_t* thread_pb = bli_thrinfo_sub_node( 0, thread_pc ); // Packing B
	thrinfo_t* thread_ic = bli_thrinfo_sub_node( 0, thread_pb ); // MC Loop
	thrinfo_t* thread_pa = bli_thrinfo_sub_node( 0, thread_ic ); // Pakcing A
	thrinfo_t* thread_jr = bli_thrinfo_sub_node( 0, thread_pa ); // NR loop
	thrinfo_t* thread_ir = bli_thrinfo_sub_node( 0, thread_jr ); // MR loop

	/* Compute the JC loop thread range for the current thread. */
	dim_t jc_start, jc_end;
	dim_t jc_tid = bli_thrinfo_work_id( thread_jc );
	dim_t jc_nt  = bli_thrinfo_n_way( thread_jc );
	bli_thread_range_sub( jc_tid, jc_nt, n, NR, FALSE, &jc_start, &jc_end );
	const dim_t n_local = jc_end - jc_start;

	/* Compute number of primary and leftover components of the JC loop. */
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/
	const dim_t jc_left =   n_local % NC;


	/* Loop over the n dimension (NC rows/columns at a time). */
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
	{
        //printf("jj llops\n");
        //printf("jj = %d, jc_start, jc_end = %d, %d\n", jj, jc_start, jc_end);
		/* Calculate the thread's current JC block dimension. */
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left );

		const char* b_jc = b_00 + jj * jcstep_b;
		      char* c_jc = c_00 + jj * jcstep_c;

		/* Compute the PC loop thread range for the current thread. */
		const dim_t pc_start = 0, pc_end = k;
		const dim_t k_local = k;

		/* Compute number of primary and leftover components of the PC loop. */
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/
		const dim_t pc_left =   k_local % KC;

		/* Loop over the k dimension (KC rows/columns at a time). */
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC )
		{
            //printf("pp llops\n");
			/* Calculate the thread's current PC block dimension. */
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left );

			const char* a_pc = a_00 + pp * pcstep_a;
			const char* b_pc = b_jc + pp * pcstep_b;

			/* Only apply beta to the first iteration of the pc loop. */
			const char* beta_use = ( pp == 0 ? beta_buf : one );

			      void* b_use;
			      inc_t rs_b_use, cs_b_use, ps_b_use;

			/* Determine the packing buffer and related parameters for matrix
			   B. Then call the packm implementation. */
			bls_packm_b
			(
			  dt,
			  conjb,
			  KC,     NC,
			  kc_cur, nc_cur, NR,
			  one,
			  b_pc,   rs_b,      cs_b,
			  &b_use, &rs_b_use, &cs_b_use,
			                     &ps_b_use,
			  cntx,
			  thread_pb
			);

			/* Scale the panel stride of B by the data type size. */
			ps_b_use *= dt_size;

            //printf("ps_b_use = %d\n", ps_b_use);

            //_Float16* b_chaoy_use     = ( _Float16* )b_use;
            //printf("\n\n\n\----------------- B COPY PANEL-----------------\n");
            //for(int i_yc = 0; i_yc < KC; i_yc++)
            //{
            //    for(int j_yc = 0; j_yc < NR; j_yc++)
            //    {   
            //        printf("%f, ", (float)b_chaoy_use[i_yc * NR + j_yc]);
            //    }
            //    printf("\n");
            //}
            //printf("b_jr first ******** = %f\n", (float)b_chao_use[0]);       


			/* Alias b_use so that it's clear this is our current block of
			   matrix B. */
			const char* b_pc_use = b_use;

			/* Compute the IC loop thread range for the current thread. */
			dim_t ic_start, ic_end;
			dim_t ic_tid = bli_thrinfo_work_id( thread_ic );
			dim_t ic_nt  = bli_thrinfo_n_way( thread_ic );
            //printf("ic_tid, ic_nt, m, MR = %d, %d, %d, %d\n", ic_tid, ic_nt, m, MR);
            //
			bli_thread_range_sub( ic_tid, ic_nt, m, MR, FALSE, &ic_start, &ic_end );
			const dim_t m_local = ic_end - ic_start;
            //printf("ic_start, ic_end, m_local = %d, %d, %d\n", ic_start, ic_end, m_local);

			/* Compute number of primary and leftover components of the IC loop. */
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/
			const dim_t ic_left =   m_local % MC;

			/* Loop over the m dimension (MC rows at a time). */
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
			{
                //printf("ii llops\n");
                //printf("ii, icstep_c = %d, %d, ic_start, ic_end = %d, %d\n", ii, icstep_c,  ic_start, ic_end);
                //printf("MC, ic_end, ii, ic_left = %d, %d, %d, %d\n", MC, ic_end, ii, ic_left);
				/* Calculate the thread's current IC block dimension. */
				const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left );

				const char* a_ic = a_pc + ii * icstep_a;
				      char* c_ic = c_jc + ii * icstep_c;

				      void* a_use;
				      inc_t rs_a_use, cs_a_use, ps_a_use;

				/* Determine the packing buffer and related parameters for matrix
				   A. Then call the packm implementation. */
				bls_packm_a
				(
				  dt,
				  conja,
				  MC,     KC,
				  mc_cur, kc_cur, MR,
				  one,
				  a_ic,   rs_a,      cs_a,
				  &a_use, &rs_a_use, &cs_a_use,
				                     &ps_a_use,
				  cntx,
				  thread_pa
				);

				/* Scale the panel stride of A by the data type size. */
				ps_a_use *= dt_size;

				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */
				const char* a_ic_use = a_use;

				/* Query the number of threads and thread ids for the JR loop.
				   NOTE: These values are only needed when computing the next
				   micropanel of B. */
				const dim_t jr_nt  = bli_thrinfo_n_way( thread_jr );
				const dim_t jr_tid = bli_thrinfo_work_id( thread_jr );

				/* Compute number of primary and leftover components of the JR loop. */
				dim_t jr_iter = ( nc_cur + NR - 1 ) / NR;
				dim_t jr_left =   nc_cur % NR;

				/* Compute the JR loop thread range for the current thread. */
				dim_t jr_start, jr_end;
				bli_thread_range_sub( jr_tid, jr_nt, jr_iter, 1, FALSE, &jr_start, &jr_end );

				/* Loop over the n dimension (NR columns at a time). */
				for ( dim_t j = jr_start; j < jr_end; j += 1)
				{
					const dim_t nr_cur
					= ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left );
					const char* b_jr = b_pc_use + j * ps_b_use;
					      char* c_jr = c_ic     + j * jrstep_c;

                    //for(int i_chao = 0; i_chao < kc_cur; i_chao++)
                    //{
                    //    for(int j_chao = 0; j_chao < nc_cur; j_chao++)
                    //    {
                    //        printf("%f, ", b_jr[i_chao * rs_b_use + j_chao * cs_b_use]);
                    //    }
                    //    printf("\n");
                    //}

                    //printf("b_jr = %p, b_pc_use = %p, j = %d, ps_b_use = %d\n", b_jr,  b_pc_use, j, ps_b_use);
					/* Assume for now that our next panel of B to be the current panel
					   of B. */
					const char* b2 = b_jr;

					/* Query the number of threads and thread ids for the IR loop.
					   NOTE: These values are only needed when computing the next
					   micropanel of A. */
					const dim_t ir_nt  = bli_thrinfo_n_way( thread_ir );
					const dim_t ir_tid = bli_thrinfo_work_id( thread_ir );

					/* Compute number of primary and leftover components of the IR loop. */
					dim_t ir_iter = ( mc_cur + MR - 1 ) / MR;
					dim_t ir_left =   mc_cur % MR;

					/* Compute the IR loop thread range for the current thread. */
					dim_t ir_start, ir_end;
					bli_thread_range_sub( ir_tid, ir_nt, ir_iter, 1, FALSE, &ir_start, &ir_end );
                    //printf("ir_tid, ir_nt, ir_iter, = %d, %d, %d\n", ir_tid, ir_nt, ir_iter);

					/* Loop over the m dimension (MR rows at a time). */
					for ( dim_t i = ir_start; i < ir_end; i += 1)
					{
                        //printf("i, irstep_c = %d, %d, ir_start, ir_end = %d, %d\n", i, irstep_c,  ir_start, ir_end);
                        //printf("i, ir_iter, ir_left = %d, %d, %d\n!", i, ir_iter, ir_left);
						const dim_t mr_cur
						= ( bli_is_not_edge_f( i, ir_iter, ir_left ) ? MR : ir_left );

						const char* a_ir = a_ic_use + i * ps_a_use;
						      char* c_ir = c_jr     + i * irstep_c;

						const char* a2;

						/* Compute the addresses of the next micropanels of A and B. */
						a2 = bli_gemm_get_next_a_upanel( a_ir, ps_a_use, 1 );
						if ( bli_is_last_iter_slrr( i, ir_end, ir_tid, ir_nt ) )
						{
							a2 = a_ic_use;
							b2 = bli_gemm_get_next_b_upanel( b_jr, ps_b_use, 1 );
							if ( bli_is_last_iter_slrr( j, jr_end, jr_tid, jr_nt ) )
								b2 = b_pc_use;
						}


                    //_Float16* a_chao_use     = ( _Float16* )a_ir;
                    ////printf("\n\n\n\*******************print A COPY PANEL******************\n");
                    //for(int i_yc = 0; i_yc < MR; i_yc++)
                    //{
                    //    for(int j_yc = 0; j_yc < KC; j_yc++)
                    //    {   
                    //        printf("%f, ", (float)a_chao_use[i_yc + j_yc * MR]);
                    //    }
                    //    printf("\n");
                    //}



                    //_Float16* b_chao_use     = ( _Float16* )b_jr;
                    //printf("\n\n\n\*******************print B COPY PANEL******************\n");
                    //for(int i_yc = 0; i_yc < KC; i_yc++)
                    //{
                    //    for(int j_yc = 0; j_yc < NR; j_yc++)
                    //    {   
                    //        printf("%f, ", (float)b_chao_use[i_yc * NR + j_yc]);
                    //    }
                    //    printf("\n");
                    //}
                    //printf("b_jr first ******** = %f\n", (float)b_chao_use[0]);       

						/* Save the addresses of next micropanels of A and B to the
						   auxinfo_t object. */
						auxinfo_t aux;
						bli_auxinfo_set_next_a( a2, &aux );
						bli_auxinfo_set_next_b( b2, &aux );

						/* Invoke the gemm microkernel. */
						//gemm_ukr
                        //int total = jj * jcstep_c + ii * icstep_c + j * jrstep_c +  i * irstep_c;
                        //printf("jj = %d, ii = %d, j = %d, i = %d, ir_start = %d, ir_end = %d, c_offset = %x, c_addr = %p\n", jj, ii, j, i, ir_start, ir_end, total, c_ir);
                        //printf("mr_cur = %d, nr_cur = %d, kc_cur = %d, a_ir = %p, b_jr = %p, c_ir = %p\n", mr_cur, nr_cur, kc_cur, (void*)a_ir, (void*)b_jr, (void*)c_ir);
                        //printf("c_00, c_ir, jj, jcstep_c, ii, icstep_c, j, jrstep_c, i, irstep_c, total =  %p,%p, %d, %d, %d, %d, %d, %d, %d, %x\n", c_00, c_ir, jj, jcstep_c, ii, icstep_c, j, jrstep_c, i, irstep_c, jj * jcstep_c + ii * icstep_c + j * jrstep_c +  i * irstep_c);
                        //printf("c_00 = %p, c_ir = %p, jj = %d, jcstep_c = %d, ii = %d, icstep_c = %d, j = %d, jrstep_c = %d, i = %d, irstep_c =%d, total =%x\n", c_00, c_ir, jj, jcstep_c, ii, icstep_c, j, jrstep_c, i, irstep_c, jj * jcstep_c + ii * icstep_c + j * jrstep_c +  i * irstep_c);
                        //bli_hgemm_armv8a_asm_h24x8r
                        if (MR == 24 && NR == 8)
                        {
                            bli_hgemm_armv8a_asm_h24x8r
						    (
						      mr_cur,
						      nr_cur,
						      kc_cur,
						      alpha_buf,
						      a_ir,
						      b_jr,
						      beta_use,
						      c_ir, rs_c, cs_c,
						      &aux,
						      cntx
						    );
                        }
                        else if (MR == 12 && NR == 16)
                        {
                            bli_hgemm_armv8a_asm_h12x16r
						    (
						      mr_cur,
						      nr_cur,
						      kc_cur,
						      alpha_buf,
						      a_ir,
						      b_jr,
						      beta_use,
						      c_ir, rs_c, cs_c,
						      &aux,
						      cntx
						    );
                        }
                        else if (MR == 12 && NR == 8)
                        {
                            bli_hgemm_armv8a_asm_sh12x8r
						    (
						      mr_cur,
						      nr_cur,
						      kc_cur,
						      alpha_buf,
						      a_ir,
						      b_jr,
						      beta_use,
						      c_ir, rs_c, cs_c,
						      &aux,
						      cntx
						    );
                        }
    //printf("\n\n\n");
    //printf("*********print C inside the ukr runing in bls_gemm_bp_var1.c**********\n");
    //c_print(c_cast_use_bls_gemm, m, n, rs_c, cs_c);
    //printf("\n\n\n");
					}
				}
			}

			/* This barrier is needed to prevent threads from starting to pack
			   the next row panel of B before the current row panel is fully
			   computed upon. */
			bli_thrinfo_barrier( thread_pb );
		}
	}
    //printf("\n\n\n");
    //printf("*********print C after the ukr runing in bls_gemm_bp_var1.c**********\n");
    //c_print(c_cast_use_bls_gemm, m, n, rs_c, cs_c);
    //printf("\n\n\n");

/*
PASTEMAC(ch,fprintm)( stdout, "gemm_bp_var1: a1_packed", mr_cur, kc_cur, a_ir, rs_a_use, cs_a_use, "%5.2f", "" );
PASTEMAC(ch,fprintm)( stdout, "gemm_bp_var1: b1_packed", kc_cur, nr_cur, b_jr, rs_b_use, cs_b_use, "%5.2f", "" );
PASTEMAC(ch,fprintm)( stdout, "gemm_bp_var1: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%5.2f", "" );
*/
}

