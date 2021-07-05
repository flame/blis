/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc.

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
#include "immintrin.h"
#include "bli_gemm_sqp_kernels.h"

#define SQP_THREAD_ENABLE 0//currently disabled
#define BLI_SQP_MAX_THREADS 128
#define BLIS_LOADFIRST 0
#define MEM_ALLOC 1//malloc performs better than bli_malloc.

#define SET_TRANS(X,Y)\
    Y = BLIS_NO_TRANSPOSE;\
    if(bli_obj_has_trans( a ))\
    {\
        Y = BLIS_TRANSPOSE;\
        if(bli_obj_has_conj(a))\
        {\
            Y = BLIS_CONJ_TRANSPOSE;\
        }\
    }\
    else if(bli_obj_has_conj(a))\
    {\
        Y = BLIS_CONJ_NO_TRANSPOSE;\
    }

//Macro for 3m_sqp n loop
#define BLI_SQP_ZGEMM_N(MX)\
    int j=0;\
    for(; j<=(n-nx); j+= nx)\
    {\
        status = bli_sqp_zgemm_m8( m, nx, k, a, lda, b+(j*ldb), ldb, c+(j*ldc), ldc, alpha_real, beta_real, transa, MX,  p_istart, kx, &mem_3m_sqp);\
    }\
    if(j<n)\
    {\
        status = bli_sqp_zgemm_m8( m, n-j, k, a, lda, b+(j*ldb), ldb, c+(j*ldc), ldc, alpha_real, beta_real, transa, MX,  p_istart, kx, &mem_3m_sqp);\
    }

//Macro for sqp_dgemm n loop
#define BLI_SQP_DGEMM_N(MX)\
    int j=0;\
    for(; j<=(n-nx); j+= nx)\
    {\
        status = bli_sqp_dgemm_m8( m, nx, k, a, lda, b+(j*ldb), ldb, c+(j*ldc), ldc, isTransA, alpha, MX, p_istart, kx, a_aligned);\
    }\
    if(j<n)\
    {\
        status = bli_sqp_dgemm_m8( m, n-j, k, a, lda, b+(j*ldb), ldb, c+(j*ldc), ldc, isTransA, alpha, MX, p_istart, kx, a_aligned);\
    }

typedef struct  {
    siz_t data_size;
    siz_t size;
    void* alignedBuf;
    void* unalignedBuf;
}mem_block;

// 3m_sqp workspace data-structure
typedef struct {
    double *ar;
    double *ai;
    double *as;

    double *br;
    double *bi;
    double *bs;

    double *cr;
    double *ci;

    double *w;
    double *aPacked;

    double *ar_unaligned;
    double *ai_unaligned;
    double *as_unaligned;

    double *br_unaligned;
    double *bi_unaligned;
    double *bs_unaligned;

    double *cr_unaligned;
    double *ci_unaligned;

    double *w_unaligned;

}workspace_3m_sqp;

//sqp threading datastructure
typedef struct bli_sqp_thread_info
{
    gint_t i_start;
    gint_t i_end;
    gint_t m;
    gint_t n;
    gint_t k;
    gint_t kx;
    double* a;
    guint_t lda;
    double* b;
    guint_t ldb;
    double* c;
    guint_t ldc;
    bool isTransA;
    double alpha;
    gint_t mx;
    bool pack_on;
    double *aligned;
} bli_sqp_thread_info;

BLIS_INLINE err_t bli_sqp_zgemm( gint_t m,
                            gint_t n,
                            gint_t k,
                            double* a,
                            guint_t lda,
                            double* b,
                            guint_t ldb,
                            double* c,
                            guint_t ldc,
                            double alpha,
                            double beta,
                            trans_t transa,
                            dim_t nt);

BLIS_INLINE err_t bli_sqp_dgemm( gint_t m,
                            gint_t n,
                            gint_t k,
                            double* a,
                            guint_t lda,
                            double* b,
                            guint_t ldb,
                            double* c,
                            guint_t ldc,
                            double alpha,
                            double beta,
                            bool isTransA,
                            dim_t nt);

/*
* The bli_gemm_sqp (square packed) function performs dgemm and 3m zgemm.
* It focuses on square matrix sizes, where m=n=k. But supports non-square matrix sizes as well.
* Currently works for column major storage and kernels.
* It has custom dgemm 8mxn block column preferred kernels
* with single load and store of C matrix to perform dgemm,
* which is also used as real kernel in 3m complex gemm computation.
*/
err_t bli_gemm_sqp
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);

    // if row major format return.
    if ((bli_obj_row_stride( a ) != 1) ||
        (bli_obj_row_stride( b ) != 1) ||
        (bli_obj_row_stride( c ) != 1))
    {
        return BLIS_INVALID_ROW_STRIDE;
    }

    if(bli_obj_has_conj(b))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    if (bli_obj_has_trans( b ))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    num_t dt = bli_obj_dt(c);
    gint_t m = bli_obj_length( c ); // number of rows of Matrix C
    gint_t n = bli_obj_width( c );  // number of columns of Matrix C
    gint_t k = bli_obj_length( b ); // number of rows of Matrix B

    guint_t lda = bli_obj_col_stride( a ); // column stride of matrix OP(A), where OP(A) is Transpose(A) if transA enabled.
    guint_t ldb = bli_obj_col_stride( b ); // column stride of matrix OP(B), where OP(B) is Transpose(B) if transB enabled.
    guint_t ldc = bli_obj_col_stride( c ); // column stride of matrix C

    if((m==0)||(n==0)||(k==0))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    if((dt != BLIS_DCOMPLEX)&&(dt != BLIS_DOUBLE))
    {
        return BLIS_NOT_YET_IMPLEMENTED;
    }

    bool isTransA = false;
    if (bli_obj_has_trans( a ))
    {
        isTransA = true;
    }

    trans_t transa = BLIS_NO_TRANSPOSE;
    SET_TRANS(a,transa)

    dim_t nt = bli_thread_get_num_threads(); // get number of threads

    double* ap     = ( double* )bli_obj_buffer( a );
    double* bp     = ( double* )bli_obj_buffer( b );
    double* cp     = ( double* )bli_obj_buffer( c );

    if(dt==BLIS_DCOMPLEX)
    {
        dcomplex* alphap = ( dcomplex* )bli_obj_buffer( alpha );
        dcomplex* betap  = ( dcomplex* )bli_obj_buffer( beta );

        //alpha and beta both real are implemented. alpha and beta with imaginary component to be implemented.
        double alpha_real = alphap->real;
        double alpha_imag = alphap->imag;
        double beta_real  = betap->real;
        double beta_imag  = betap->imag;
        if( (alpha_imag!=0)||(beta_imag!=0) )
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
        //printf("zsqp ");
        return bli_sqp_zgemm( m, n, k, ap, lda, bp, ldb, cp, ldc, alpha_real, beta_real, transa, nt);
    }
    else if(dt == BLIS_DOUBLE)
    {
        double *alpha_cast, *beta_cast;
        alpha_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, alpha);
        beta_cast = bli_obj_buffer_for_1x1(BLIS_DOUBLE, beta);

        if((*beta_cast)!=1.0)
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
        if(((*alpha_cast)!=1.0)&&((*alpha_cast)!=-1.0))
        {
            return BLIS_NOT_YET_IMPLEMENTED;
        }
        //printf("dsqp ");
        // dgemm case only transpose or no-transpose is handled.
        // conjugate_transpose and conjugate no transpose are not applicable.
        return bli_sqp_dgemm( m, n, k, ap, lda, bp, ldb, cp, ldc, *alpha_cast, *beta_cast, isTransA, nt);
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
    return BLIS_NOT_YET_IMPLEMENTED;
};

//sqp_dgemm k partition
BLIS_INLINE void bli_sqp_dgemm_kx(  gint_t m,
                                    gint_t n,
                                    gint_t kx,
                                    gint_t p,
                                    double* a,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc,
                                    bool isTransA,
                                    double alpha,
                                    gint_t mx,
                                    gint_t i,
                                    bool pack_on,
                                    double *aligned)
{
    inc_t j = 0;
    double* ci = c + i;
    double* aPacked;
    //packing
    if(pack_on==true)
    {
        aPacked = aligned;
        double *pa = a + i + (p*lda);
        if(isTransA==true)
        {
            pa = a + (i*lda) + p;
        }
        bli_sqp_prepackA(pa, aPacked, kx, lda, isTransA, alpha, mx);
    }
    else
    {
        aPacked = a+i + (p*lda);
    }

    //compute
    if(mx==8)
    {
        //printf("\n mx8i:%3ld ", i);
        if (j <= (n - 6))
        {
            j = bli_sqp_dgemm_kernel_8mx6n(n, kx, j, aPacked, lda, b + p, ldb, ci, ldc);
        }
        if (j <= (n - 5))
        {
            j = bli_sqp_dgemm_kernel_8mx5n(n, kx, j, aPacked, lda, b + (j * ldb) + p, ldb, ci + (j * ldc), ldc);
        }
        if (j <= (n - 4))
        {
            j = bli_sqp_dgemm_kernel_8mx4n(n, kx, j, aPacked, lda, b + (j * ldb) + p, ldb, ci + (j * ldc), ldc);
        }
        if (j <= (n - 3))
        {
            j = bli_sqp_dgemm_kernel_8mx3n(n, kx, j, aPacked, lda, b + (j * ldb) + p, ldb, ci + (j * ldc), ldc);
        }
        if (j <= (n - 2))
        {
            j = bli_sqp_dgemm_kernel_8mx2n(n, kx, j, aPacked, lda, b + (j * ldb) + p, ldb, ci + (j * ldc), ldc);
        }
        if (j <= (n - 1))
        {
            j = bli_sqp_dgemm_kernel_8mx1n(n, kx, j, aPacked, lda, b + (j * ldb) + p, ldb, ci + (j * ldc), ldc);
        }
    }
    /* mx==4 to be implemented */
    else
    {
        // this residue kernel needs to be improved.
        j = bli_sqp_dgemm_kernel_mxn(n, kx, j, aPacked, lda, b + p, ldb, ci, ldc, mx);
    }
}

//sqp dgemm m loop
void bli_sqp_dgemm_m(   gint_t i_start,
                        gint_t i_end,
                        gint_t m,
                        gint_t n,
                        gint_t k,
                        gint_t kx,
                        double* a,
                        guint_t lda,
                        double* b,
                        guint_t ldb,
                        double* c,
                        guint_t ldc,
                        bool isTransA,
                        double alpha,
                        gint_t mx,
                        bool pack_on,
                        double *aligned )
{
#if SQP_THREAD_ENABLE
    if(pack_on==true)
    {
        //NEEDED IN THREADING CASE:
        aligned = (double*)bli_malloc_user(sizeof(double) * kx * mx);
        if(aligned==NULL)
        {
            return BLIS_MALLOC_RETURNED_NULL;// return to be removed
        }
    }
#endif//SQP_THREAD_ENABLE

    for (gint_t i = i_start; i <= (i_end-mx); i += mx) //this loop can be threaded. no of workitems = m/8
    {
        int p = 0;
        for(; p <= (k-kx); p += kx)
        {
            bli_sqp_dgemm_kx(m, n, kx, p, a,  lda, b, ldb, c, ldc, isTransA, alpha, mx, i, pack_on, aligned);
        }// k loop end

        if(p<k)//kx = k - p
        {
            bli_sqp_dgemm_kx(m, n, (k - p), p, a,  lda, b, ldb, c, ldc, isTransA, alpha, mx, i, pack_on, aligned);
        }
    }// i loop end

#if SQP_THREAD_ENABLE
    //NEEDED IN THREADING CASE:
    if(pack_on==true)
    {
        bli_free_user(aligned);
    }
#endif//SQP_THREAD_ENABLE
}

void bli_sqp_thread(void* info)
{
    bli_sqp_thread_info* arg = (bli_sqp_thread_info*) info;
    bli_sqp_dgemm_m(arg->i_start,
                    arg->i_end,
                    arg->m,
                    arg->n,
                    arg->k,
                    arg->kx,
                    arg->a,
                    arg->lda,
                    arg->b,
                    arg->ldb,
                    arg->c,
                    arg->ldc,
                    arg->isTransA,
                    arg->alpha,
                    arg->mx,
                    arg->pack_on,
                    arg->aligned);
}

// sqp_dgemm m loop
BLIS_INLINE err_t bli_sqp_dgemm_m8( gint_t m,
                                    gint_t n,
                                    gint_t k,
                                    double* a,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc,
                                    bool isTransA,
                                    double alpha,
                                    gint_t mx,
                                    gint_t* p_istart,
                                    gint_t kx,
                                    double *aligned)
{
    gint_t i;
    if(kx > k)
    {
        kx = k;
    }

    bool pack_on = false;
    if((m!=mx)||(m!=lda)||isTransA)
    {
        pack_on = true;
    }

#if 0//SQP_THREAD_ENABLE//ENABLE Threading
    gint_t status = 0;
    gint_t workitems = (m-(*p_istart))/mx;
    gint_t inputThreadCount = bli_thread_get_num_threads();
    inputThreadCount = bli_min(inputThreadCount, BLI_SQP_MAX_THREADS);
    inputThreadCount = bli_min(inputThreadCount,workitems);// limit input thread count when workitems are lesser.
    inputThreadCount = bli_max(inputThreadCount,1);
    gint_t num_threads;
    num_threads = bli_max(inputThreadCount,1);
    gint_t mx_per_thread = workitems/num_threads;//no of workitems per thread
    //printf("\nistart %d workitems %d inputThreadCount %d num_threads %d mx_per_thread %d mx %d " ,
    *p_istart, workitems,inputThreadCount,num_threads,mx_per_thread, mx);

    pthread_t ptid[BLI_SQP_MAX_THREADS];
    bli_sqp_thread_info thread_info[BLI_SQP_MAX_THREADS];

    //create threads
    for (gint_t t = 0; t < num_threads; t++)
    {
        //ptid[t].tid = t;
        gint_t i_end = ((mx_per_thread*(t+1))*mx)+(*p_istart);
        if(i_end>m)
        {
            i_end = m;
        }

        if(t==(num_threads-1))
        {
            if((i_end+mx)==m)
            {
                i_end = m;
            }

            if(mx==1)
            {
                i_end = m;
            }
        }

        thread_info[t].i_start = ((mx_per_thread*t)*mx)+(*p_istart);
        thread_info[t].i_end = i_end;
        //printf("\n threadid %d istart %d iend %d m %d mx %d", t, thread_info[t].i_start, i_end, m, mx);
        thread_info[t].m = m;
        thread_info[t].n = n;
        thread_info[t].k = k;
        thread_info[t].kx = kx;
        thread_info[t].a = a;
        thread_info[t].lda = lda;
        thread_info[t].b = b;
        thread_info[t].ldb = ldb;
        thread_info[t].c = c;
        thread_info[t].ldc = ldc;
        thread_info[t].isTransA = isTransA;
        thread_info[t].alpha = alpha;
        thread_info[t].mx = mx;
        thread_info[t].pack_on = pack_on;
        thread_info[t].aligned = aligned;
#if 1
        if ((status = pthread_create(&ptid[t], NULL, bli_sqp_thread, (void*)&thread_info[t])))
        {
            printf("error sqp pthread_create\n");
            return BLIS_FAILURE;
        }
#else
        //simulate thread for debugging..
        bli_sqp_thread((void*)&thread_info[t]);
#endif
    }

    //wait for completion
    for (gint_t t = 0; t < num_threads; t++)
    {
        pthread_join(ptid[t], NULL);
    }

    if(num_threads>0)
    {
        *p_istart = thread_info[(num_threads-1)].i_end;
    }
#else//SQP_THREAD_ENABLE

    if(pack_on==true)
    {
        //aligned = (double*)bli_malloc_user(sizeof(double) * kx * mx); // allocation moved to top.
        if(aligned==NULL)
        {
            return BLIS_MALLOC_RETURNED_NULL;
        }
    }

    for (i = (*p_istart); i <= (m-mx); i += mx) //this loop can be threaded. no of workitems = m/8
    {
        int p = 0;
        for(; p <= (k-kx); p += kx)
        {
            bli_sqp_dgemm_kx(m, n, kx, p, a,  lda, b, ldb, c, ldc,
                            isTransA, alpha, mx, i, pack_on, aligned);
        }// k loop end

        if(p<k)//kx = k - p
        {
            bli_sqp_dgemm_kx(m, n, (k - p), p, a,  lda, b, ldb, c, ldc,
                            isTransA, alpha, mx, i, pack_on, aligned);
        }
    }// i loop end
#endif//SQP_THREAD_ENABLE

    return BLIS_SUCCESS;
}

gint_t bli_getaligned(mem_block* mem_req)
{

    guint_t memSize = mem_req->data_size * mem_req->size;
    if (memSize == 0)
    {
        return -1;
    }
    memSize += 128;// extra 128 bytes added for alignment. Could be minimized to 64.
#if MEM_ALLOC
#ifdef BLIS_ENABLE_MEM_TRACING
    printf( "malloc(): size %ld\n",( long )memSize);
    fflush( stdout );
#endif
    mem_req->unalignedBuf = (double*)malloc(memSize);
    if (mem_req->unalignedBuf == NULL)
    {
        return -1;
    }

    int64_t address = (int64_t)mem_req->unalignedBuf;
    address += (-address) & 63; //64 bytes alignment done.
    mem_req->alignedBuf = (double*)address;
#else
    mem_req->alignedBuf = bli_malloc_user( memSize );
    if (mem_req->alignedBuf == NULL)
    {
        return -1;
    }
#endif
    return 0;
}

gint_t bli_allocateWorkspace(gint_t n, gint_t k, mem_block *mxr, mem_block *mxi, mem_block *msx)
{
    //allocate workspace
    mxr->data_size = mxi->data_size = msx->data_size = sizeof(double);
    mxr->size = mxi->size = n * k;
    msx->size = n * k;
    mxr->alignedBuf = mxi->alignedBuf = msx->alignedBuf = NULL;
    mxr->unalignedBuf = mxi->unalignedBuf = msx->unalignedBuf = NULL;

    if (!((bli_getaligned(mxr) == 0) && (bli_getaligned(mxi) == 0) && (bli_getaligned(msx) == 0)))
    {
#if MEM_ALLOC
        if(mxr->unalignedBuf)
        {
            free(mxr->unalignedBuf);
        }
        if(mxi->unalignedBuf)
        {
            free(mxi->unalignedBuf);
        }
        if(msx->unalignedBuf)
        {
            free(msx->unalignedBuf);
        }
#else
        bli_free_user(mxr->alignedBuf);
        bli_free_user(mxi->alignedBuf);
        bli_free_user(msx->alignedBuf);
#endif
        return -1;
    }
    return 0;
}

//3m_sqp k loop
BLIS_INLINE void bli_sqp_zgemm_kx(  gint_t m,
                                    gint_t n,
                                    gint_t kx,
                                    gint_t p,
                                    double* a,
                                    guint_t lda,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc,
                                    trans_t transa,
                                    double alpha,
                                    double beta,
                                    gint_t mx,
                                    gint_t i,
                                    double* ar,
                                    double* ai,
                                    double* as,
                                    double* br,
                                    double* bi,
                                    double* bs,
                                    double* cr,
                                    double* ci,
                                    double* w,
                                    double *a_aligned)
{
    gint_t j;

    ////////////// operation 1 /////////////////
    /* Split    a  (ar, ai) and
        compute  as = ar + ai   */
    double* par = ar;
    double* pai = ai;
    double* pas = as;

    /* a matrix real and imag packing and compute. */
    bli_3m_sqp_packA_real_imag_sum(a, i, kx+p, lda, par, pai, pas, transa, mx, p);

    double* pcr = cr;
    double* pci = ci;

    //Split Cr and Ci and beta multiplication done.
    double* pc = c + i;
    if(p==0)
    {
        bli_3m_sqp_packC_real_imag(pc, n, mx, ldc, pcr, pci, beta, mx);
    }
    //Ci := rgemm( SA, SB, Ci )
    gint_t istart = 0;
    gint_t* p_is = &istart;
    *p_is = 0;
    bli_sqp_dgemm_m8(mx, n, kx, as, mx, bs, ldb, ci, mx, false, 1.0, mx, p_is, kx, a_aligned);

    ////////////// operation 2 /////////////////
    //Wr: = dgemm_sqp(Ar, Br, 0)  // Wr output 8xn
    double* wr = w;
    for (j = 0; j < n; j++) {
        for (gint_t ii = 0; ii < mx; ii += 1) {
            *wr = 0;
            wr++;
        }
    }
    wr = w;

    *p_is = 0;
    bli_sqp_dgemm_m8(mx, n, kx, ar, mx, br, ldb, wr, mx, false, 1.0, mx, p_is, kx, a_aligned);
    //Cr : = addm(Wr, Cr)
    bli_add_m(mx, n, wr, cr);
    //Ci : = subm(Wr, Ci)
    bli_sub_m(mx, n, wr, ci);


    ////////////// operation 3 /////////////////
    //Wi : = dgemm_sqp(Ai, Bi, 0)  // Wi output 8xn
    double* wi = w;
    for (j = 0; j < n; j++) {
        for (gint_t ii = 0; ii < mx; ii += 1) {
            *wi = 0;
            wi++;
        }
    }
    wi = w;

    *p_is = 0;
    bli_sqp_dgemm_m8(mx, n, kx, ai, mx, bi, ldb, wi, mx, false, 1.0, mx, p_is, kx, a_aligned);
    //Cr : = subm(Wi, Cr)
    bli_sub_m(mx, n, wi, cr);
    //Ci : = subm(Wi, Ci)
    bli_sub_m(mx, n, wi, ci);

    pcr = cr;
    pci = ci;

    for (j = 0; j < n; j++)
    {
        for (gint_t ii = 0; ii < (mx*2); ii += 2)
        {
            c[(j * ldc) + i + ii]     = *pcr;
            c[(j * ldc) + i + ii + 1] = *pci;
            pcr++; pci++;
        }
    }
}

/**************************************************************/
/* workspace memory allocation for 3m_sqp algorithm for zgemm */
/**************************************************************/
err_t allocate_3m_Sqp_workspace(workspace_3m_sqp *mem_3m_sqp,
                                gint_t mx,
                                gint_t nx,
                                gint_t k,
                                gint_t kx )
{
    //3m_sqp workspace Memory allocation
    /* B matrix */
    // B matrix packed with n x k size. without kx smaller sizes for now.
    mem_block mbr, mbi, mbs;
    if(bli_allocateWorkspace(nx, k, &mbr, &mbi, &mbs)!=0)
    {
        return BLIS_FAILURE;
    }
    mem_3m_sqp->br = (double*)mbr.alignedBuf;
    mem_3m_sqp->bi = (double*)mbi.alignedBuf;
    mem_3m_sqp->bs = (double*)mbs.alignedBuf;
    mem_3m_sqp->br_unaligned = (double*)mbr.unalignedBuf;
    mem_3m_sqp->bi_unaligned = (double*)mbi.unalignedBuf;
    mem_3m_sqp->bs_unaligned = (double*)mbs.unalignedBuf;

    /* Workspace memory allocation currently done dynamically
    This needs to be taken from already allocated memory pool in application for better performance */
    /* A matrix */
    mem_block mar, mai, mas;
    if(bli_allocateWorkspace(mx, kx, &mar, &mai, &mas) !=0)
    {
        return BLIS_FAILURE;
    }
    mem_3m_sqp->ar = (double*)mar.alignedBuf;
    mem_3m_sqp->ai = (double*)mai.alignedBuf;
    mem_3m_sqp->as = (double*)mas.alignedBuf;
    mem_3m_sqp->ar_unaligned = (double*)mar.unalignedBuf;
    mem_3m_sqp->ai_unaligned = (double*)mai.unalignedBuf;
    mem_3m_sqp->as_unaligned = (double*)mas.unalignedBuf;

    /* w matrix */
    mem_block mw;
    mw.data_size = sizeof(double);
    mw.size = mx * nx;
    if (bli_getaligned(&mw) != 0)
    {
        return BLIS_FAILURE;
    }
    mem_3m_sqp->w = (double*)mw.alignedBuf;
    mem_3m_sqp->w_unaligned = (double*)mw.unalignedBuf;
    /* cr matrix */
    mem_block mcr;
    mcr.data_size = sizeof(double);
    mcr.size = mx * nx;
    if (bli_getaligned(&mcr) != 0)
    {
        return BLIS_FAILURE;
    }
    mem_3m_sqp->cr = (double*)mcr.alignedBuf;
    mem_3m_sqp->cr_unaligned = (double*)mcr.unalignedBuf;


    /* ci matrix */
    mem_block mci;
    mci.data_size = sizeof(double);
    mci.size = mx * nx;
    if (bli_getaligned(&mci) != 0)
    {
        return BLIS_FAILURE;
    }
    mem_3m_sqp->ci = (double*)mci.alignedBuf;
    mem_3m_sqp->ci_unaligned = (double*)mci.unalignedBuf;

    // A packing buffer
    mem_3m_sqp->aPacked = (double*)bli_malloc_user(sizeof(double) * kx * mx);
    if (mem_3m_sqp->aPacked == NULL)
    {
        return BLIS_FAILURE;
    }

    return BLIS_SUCCESS;
}

void free_3m_Sqp_workspace(workspace_3m_sqp *mem_3m_sqp)
{
    // A packing buffer free
    bli_free_user(mem_3m_sqp->aPacked);

#if MEM_ALLOC
    if(mem_3m_sqp->ar_unaligned)
    {
        free(mem_3m_sqp->ar_unaligned);
    }
    if(mem_3m_sqp->ai_unaligned)
    {
        free(mem_3m_sqp->ai_unaligned);
    }
    if(mem_3m_sqp->as_unaligned)
    {
        free(mem_3m_sqp->as_unaligned);
    }

    if(mem_3m_sqp->br_unaligned)
    {
        free(mem_3m_sqp->br_unaligned);
    }
    if(mem_3m_sqp->bi_unaligned)
    {
        free(mem_3m_sqp->bi_unaligned);
    }
    if(mem_3m_sqp->bs_unaligned)
    {
        free(mem_3m_sqp->bs_unaligned);
    }

    if(mem_3m_sqp->w_unaligned)
    {
        free(mem_3m_sqp->w_unaligned);
    }
    if(mem_3m_sqp->cr_unaligned)
    {
        free(mem_3m_sqp->cr_unaligned);
    }
    if(mem_3m_sqp->ci_unaligned)
    {
        free(mem_3m_sqp->ci_unaligned);
    }

#else//MEM_ALLOC
    /* free workspace buffers */
    bli_free_user(mem_3m_sqp->br);
    bli_free_user(mem_3m_sqp->bi);
    bli_free_user(mem_3m_sqp->bs);
    bli_free_user(mem_3m_sqp->ar);
    bli_free_user(mem_3m_sqp->ai);
    bli_free_user(mem_3m_sqp->as);
    bli_free_user(mem_3m_sqp->w);
    bli_free_user(mem_3m_sqp->cr);
    bli_free_user(mem_3m_sqp->ci);
#endif//MEM_ALLOC
}

//3m_sqp m loop
BLIS_INLINE err_t bli_sqp_zgemm_m8( gint_t m,
                                    gint_t n,
                                    gint_t k,
                                    double* a,
                                    guint_t lda,
                                    double* b,
                                    guint_t ldb,
                                    double* c,
                                    guint_t ldc,
                                    double alpha,
                                    double beta,
                                    trans_t transa,
                                    gint_t mx,
                                    gint_t* p_istart,
                                    gint_t kx,
                                    workspace_3m_sqp *mem_3m_sqp)
{
    inc_t m2 = m<<1;
    inc_t mxmul2 = mx<<1;

    if((*p_istart) > (m2-mxmul2))
    {
        return BLIS_SUCCESS;
    }
    inc_t i;
    gint_t max_m = (m2-mxmul2);

    //get workspace
    double* ar, * ai, * as;
    ar = mem_3m_sqp->ar;
    ai = mem_3m_sqp->ai;
    as = mem_3m_sqp->as;

    double* br, * bi, * bs;
    br = mem_3m_sqp->br;
    bi = mem_3m_sqp->bi;
    bs = mem_3m_sqp->bs;

    double* cr, * ci;
    cr = mem_3m_sqp->cr;
    ci = mem_3m_sqp->ci;

    double *w;
    w = mem_3m_sqp->w;

    double* a_aligned;
    a_aligned = mem_3m_sqp->aPacked;

    /* Split    b  (br, bi) and
       compute  bs = br + bi    */
    double* pbr = br;
    double* pbi = bi;
    double* pbs = bs;
    /* b matrix real and imag packing and compute. */
    bli_3m_sqp_packB_real_imag_sum(b, n, k, ldb, pbr, pbi, pbs, alpha, mx);

    for (i = (*p_istart); i <= max_m; i += mxmul2) //this loop can be threaded.
    {
#if KLP//kloop
        int p = 0;
        for(; p <= (k-kx); p += kx)
        {
            bli_sqp_zgemm_kx(m, n, kx, p, a, lda, k, c, ldc,
                            transa, alpha, beta, mx, i, ar, ai, as,
                            br + p, bi + p, bs + p, cr, ci, w, a_aligned);
        }// k loop end

        if(p<k)
        {
            bli_sqp_zgemm_kx(m, n, (k - p), p, a, lda, k, c, ldc,
                             transa, alpha, beta, mx, i, ar, ai, as,
                            br + p, bi + p, bs + p, cr, ci, w, a_aligned);
        }
#else//kloop
        ////////////// operation 1 /////////////////

        /* Split    a  (ar, ai) and
           compute  as = ar + ai   */
        double* par = ar;
        double* pai = ai;
        double* pas = as;

        /* a matrix real and imag packing and compute. */
        bli_3m_sqp_packA_real_imag_sum(a, i, k, lda, par, pai, pas, transa, mx, 0);

        double* pcr = cr;
        double* pci = ci;

        //Split Cr and Ci and beta multiplication done.
        double* pc = c + i;
        bli_3m_sqp_packC_real_imag(pc, n, mx, ldc, pcr, pci, beta, mx);

        //Ci := rgemm( SA, SB, Ci )
        gint_t istart = 0;
        gint_t* p_is = &istart;
        *p_is = 0;
        bli_sqp_dgemm_m8(mx, n, k, as, mx, bs, k, ci, mx, false, 1.0, mx, p_is, k);

        ////////////// operation 2 /////////////////
        //Wr: = dgemm_sqp(Ar, Br, 0)  // Wr output 8xn
        double* wr = w;
        for (j = 0; j < n; j++) {
            for (gint_t ii = 0; ii < mx; ii += 1) {
                *wr = 0;
                wr++;
            }
        }
        wr = w;

        *p_is = 0;
        bli_sqp_dgemm_m8(mx, n, k, ar, mx, br, k, wr, mx, false, 1.0, mx, p_is, k);
        //Cr : = addm(Wr, Cr)
        bli_add_m(mx, n, wr, cr);
        //Ci : = subm(Wr, Ci)
        bli_sub_m(mx, n, wr, ci);

        ////////////// operation 3 /////////////////
        //Wi : = dgemm_sqp(Ai, Bi, 0)  // Wi output 8xn
        double* wi = w;
        for (j = 0; j < n; j++) {
            for (gint_t ii = 0; ii < mx; ii += 1) {
                *wi = 0;
                wi++;
            }
        }
        wi = w;

        *p_is = 0;
        bli_sqp_dgemm_m8(mx, n, k, ai, mx, bi, k, wi, mx, false, 1.0, mx, p_is, k);
        //Cr : = subm(Wi, Cr)
        bli_sub_m(mx, n, wi, cr);
        //Ci : = subm(Wi, Ci)
        bli_sub_m(mx, n, wi, ci);


        pcr = cr;
        pci = ci;

        for (j = 0; j < n; j++)
        {
            for (gint_t ii = 0; ii < mxmul2; ii += 2)
            {
                c[(j * ldc) + i + ii]     = *pcr;
                c[(j * ldc) + i + ii + 1] = *pci;
                pcr++; pci++;
            }
        }
#endif//kloop
    }

    return BLIS_SUCCESS;
}

/****************************************************************************/
/************* 3m_sqp implementation   **************************************/
/****************************************************************************/
/* 3m_sqp implementation packs A, B and C matrix and uses dgemm_sqp
   real kernel implementation. 3m_sqp focuses mainly on square matrixes
   but also supports non-square matrix. Current support is limiteed to
   column storage.
*/
BLIS_INLINE err_t bli_sqp_zgemm(gint_t m,
                                gint_t n,
                                gint_t k,
                                double* a,
                                guint_t lda,
                                double* b,
                                guint_t ldb,
                                double* c,
                                guint_t ldc,
                                double alpha_real,
                                double beta_real,
                                trans_t transa,
                                dim_t nt)
{
    gint_t istart = 0;
    gint_t* p_istart = &istart;
    *p_istart = 0;
    err_t status = BLIS_SUCCESS;

    gint_t mx = 8;
    dim_t m8rem = m - ((m>>3)<<3);

    workspace_3m_sqp mem_3m_sqp;

    /* multiply lda, ldb and ldc by 2 to account for
     real & imaginary components per dcomplex. */
    lda = lda * 2;
    ldb = ldb * 2;
    ldc = ldc * 2;

    /* user can set BLIS_MULTI_INSTANCE macro for
    better performance while runing multi-instance use-case.
    */
    dim_t multi_instance = bli_env_get_var( "BLIS_MULTI_INSTANCE", -1 );
    gint_t nx = n;
    if(multi_instance>0)
    {
        //limited nx size helps in reducing memory footprint in multi-instance case.
        nx = 84;
        // 84 is derived based on tuning results
    }

    if(nx>n)
    {
        nx = n;
    }

    gint_t kx = k;// kx is configurable at run-time.
#if KLP
    if (kx > k)
    {
        kx = k;
    }
    // for tn case there is a bug in handling k parts. To be fixed.
    if(transa!=BLIS_NO_TRANSPOSE)
    {
        kx = k;
    }
#else
    kx = k;
#endif
    //3m_sqp workspace Memory allocation
    if(allocate_3m_Sqp_workspace(&mem_3m_sqp, mx, nx, k, kx)!=BLIS_SUCCESS)
    {
        return BLIS_FAILURE;
    }

    BLI_SQP_ZGEMM_N(mx)
    *p_istart = (m-m8rem)*2;

    if(m8rem!=0)
    {
        //complete residue m blocks
        BLI_SQP_ZGEMM_N(m8rem)
    }

    free_3m_Sqp_workspace(&mem_3m_sqp);
    return status;
}

/****************************************************************************/
/*********************** dgemm_sqp implementation****************************/
/****************************************************************************/
/* dgemm_sqp implementation packs A matrix based on lda and m size.
   dgemm_sqp focuses mainly on square matrixes but also supports non-square matrix.
   Current support is limiteed to m multiple of 8 and column storage.
   C = AxB and C = AtxB is handled in the design.
   AtxB case is done by transposing A matrix while packing A.
   In majority of use-case, alpha are +/-1, so instead of explicitly multiplying
   alpha its done during packing itself by changing sign.
*/
BLIS_INLINE err_t bli_sqp_dgemm(gint_t m,
                                gint_t n,
                                gint_t k,
                                double* a,
                                guint_t lda,
                                double* b,
                                guint_t ldb,
                                double* c,
                                guint_t ldc,
                                double alpha,
                                double beta,
                                bool isTransA,
                                dim_t nt)
{
    gint_t istart = 0;
    gint_t* p_istart = &istart;
    *p_istart = 0;
    err_t status = BLIS_SUCCESS;
    dim_t m8rem = m - ((m>>3)<<3);

    /* dgemm implementation with 8mx5n major kernel and column preferred storage */
    gint_t mx = 8;
    gint_t kx = k;
    double* a_aligned = NULL;

    if(nt<=1)//single pack buffer allocated for single thread case
    {
        a_aligned = (double*)bli_malloc_user(sizeof(double) * kx * mx);
    }

    gint_t nx = n;//MAX;
    if(nx>n)
    {
        nx = n;
    }

    //mx==8 case for dgemm.
    BLI_SQP_DGEMM_N(mx)
    *p_istart = (m-m8rem);

    if(nt>1)
    {
        //2nd level thread for mx=8
        gint_t rem_m = m - (*p_istart);
        if((rem_m>=mx)&&(status==BLIS_SUCCESS))
        {
            status = bli_sqp_dgemm_m8( m, n, k, a, lda, b, ldb, c, ldc,
            isTransA, alpha, mx, p_istart, kx, a_aligned);
        }
    }

    if(status==BLIS_SUCCESS)
    {
        if(m8rem!=0)
        {
            //complete residue m blocks
            BLI_SQP_DGEMM_N(m8rem)
        }
    }

    if(nt<=1)//single pack buffer allocated for single thread case
    {
        bli_free_user(a_aligned);
    }
    return status;
}