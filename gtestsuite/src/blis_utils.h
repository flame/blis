#ifndef BLIS_UTILS_H
#define BLIS_UTILS_H

#include <limits>
#include <math.h>

#include "blis_test.h"

using namespace std;

#define abscomplex(x)  (abs(x.real) + abs(x.imag))

#define mulr(x,y)  (( x.real * y.real ) - ( x.imag * y.imag ))
#define muli(x,y)  (( x.real * y.imag ) + ( x.imag * y.real ))

#define ELD    8
#define PAT32  0x78abcdef
#define PAT64  0x0123456789abcdef

template <typename T>
T real( T x )
{
    T r = { 0.0, 0.0 };
    r.real = x.real;
    r.imag = 0;
    return r;
}

template <typename T>
T conjugate( T x )
{
    T r = { 0.0, 0.0 };
    r.real =   x.real;
    r.imag = -(x.imag);
    return r;
}

template <typename T>
T addc( T xx, T yy ) {
  T r = { 0.0, 0.0 };
  r.real = xx.real + yy.real;
  r.imag = xx.imag + yy.imag;
  return r;
}

template <typename T>
T subc( T xx, T yy )
{
    T r = { 0.0, 0.0 };
    r.real = xx.real - yy.real;
    r.imag = xx.imag - yy.imag;
    return r;
}

template <typename T>
T mulc( T xx, T yy )
{
    T r = { 0.0, 0.0 };
    r.real = mulr( xx, yy );
    r.imag = muli( xx, yy );
    return r;
}

template <typename T, typename U>
T divc( T yy, T xx )
{
    T r = { 0.0, 0.0 };
    U s    = bli_fmaxabs( (xx.real),(xx.imag) );
    U xxrs = ( (xx.real)/s );
    U xxis = ( (xx.imag)/s );
    U deno = ( (xxrs * xx.real) + (xxis * xx.imag) );
    r.real = ( ((yy.real * xxrs) + (yy.imag * xxis))/deno );
    r.imag = ( ((yy.imag * xxrs) - (yy.real * xxis))/deno );
    return r;
}

template <typename T, typename U>
T divct( T yy, T xx )
{
    T r = { 0.0, 0.0 };
    U deno = ( (xx.real * xx.real) + (xx.imag * xx.imag) );
    r.real = ( ((xx.real * yy.real) + (xx.imag * yy.imag))/deno );
    r.imag = ( ((yy.imag * xx.real) - (yy.real * xx.imag))/deno );
    return r;
}

template <typename T>
double computediffrv( dim_t len, dim_t incy, T *act, T *ref )
{
    double resid = 0.0;
    unsigned int j,jy = 0;
    for( j = 0 ; j < len ; j++ )
    {
        auto av = ref[jy];
        auto xc = act[jy];
        resid += xc - av;
        jy = jy + incy;
    }
    return abs(resid);
}

template <typename T>
double computediffiv( dim_t len, dim_t incy, T *act, T *ref )
{
    double resid = 0.0;
    unsigned int j,jy = 0;
    double rr,ri;
    rr = ri = 0.0;
    for( j = 0 ; j < len ; j++ )
    {
        auto av = ref[jy];
        auto xc = act[jy];
        rr += xc.real - av.real;
        ri += xc.imag - av.imag;
        jy = jy + incy;
    }
    resid = rr + ri;
    return abs(resid);
}

template <typename T>
double computediffrm( dim_t m,dim_t n, T *act, T *ref, dim_t rsc, dim_t csc )
{
    double resid = 0.0;
    unsigned int i,j;
    for( i = 0 ; i < m ; i++ )
    {
        for( j = 0 ; j < n ; j++ )
        {
            auto av = ref[ i*rsc + j*csc ];
            auto xc = act[ i*rsc + j*csc ];
            resid += xc - av;
        }
    }
    return abs(resid);
}

template <typename T>
double computediffim( dim_t m,dim_t n, T *act, T *ref, dim_t rsc, dim_t csc )
{
    unsigned int i,j;
    double rr,ri;
    double resid = 0.0;
    rr = ri = 0.0;
    for( i = 0 ; i < m ; i++ )
    {
        for( j = 0 ; j < n ; j++ )
        {
            auto av = ref[ i*rsc + j*csc ];
            auto xc = act[ i*rsc + j*csc ];
            rr += xc.real - av.real;
            ri += xc.imag - av.imag;
        }
    }
    resid = rr + ri;
    return abs(resid);
}

template <typename T>
double libblis_vector_check_real( vflg_t flg, dim_t len, dim_t incy, T *buf )
{
    double resid = 0.0;
    unsigned int j,jy=0;
    T val = 0.0;
    if(flg == BLIS_OVERFLOW)
    {
        val = (std::numeric_limits<T>::max)();
        for( j = 0 ; j < len ; j++ )
        {
            auto res = buf[jy];
            if((isnan(res)) || (fabs(res) > val))
            {
                return abs(res);
            }
            jy = jy + incy;
        }
    }
    else
    {
        val = (std::numeric_limits<T>::min)();
        for( j = 0 ; j < len ; j++ ) {
            auto res = buf[jy];
            if((isnan(res)) || (fabs(res) < val))
            {
                return abs(res);
            }
            jy = jy + incy;
        }
    }
    return resid;
}

template <typename T, typename U>
double libblis_vector_check_cmplx( vflg_t flg, dim_t len, dim_t incy, T *buf )
{
    double resid = 0.0;
    unsigned int j,jy=0;
    U val = 0.0;
    if(flg == BLIS_OVERFLOW)
    {
        val = (std::numeric_limits<U>::max)();
        for( j = 0 ; j < len ; j++ )
        {
            auto res = buf[jy];
            if((isnan(res.real) || (isnan(res.imag))) ||
               (fabs(res.real) > val) || (fabs(res.imag) > val))
            {
                resid = (fabs(res.real) > fabs(res.imag)) ? res.real : res.imag;
                return abs(resid);
            }
            jy = jy + incy;
        }
    }
    else
    {
        val = (std::numeric_limits<U>::min)();
        for( j = 0 ; j < len ; j++ )
        {
            auto res = buf[jy];
            if((isnan(res.real) || (isnan(res.imag))) ||
               (fabs(res.real) < val) || (fabs(res.imag) < val))
            {
                resid = (fabs(res.real) < fabs(res.imag)) ? res.imag : res.real;
                return abs(resid);
            }
            jy = jy + incy;
        }
    }
    return resid;
}

template <typename T>
double libblis_matrix_check_real( vflg_t flg, T* buf, dim_t m, dim_t n,
                                  dim_t rsc, dim_t csc )
{
    double resid = 0.0;
    unsigned int i,j;
    T val = 0.0;
    if(flg == BLIS_OVERFLOW)
    {
        val = (std::numeric_limits<T>::max)();
        for( i = 0 ; i < m ; i++ )
        {
            for( j = 0 ; j < n ; j++ )
            {
                auto res = buf[ i*rsc + j*csc ];
                if((isnan(res)) || (fabs(res) > val))
                {
                    return abs(res);
                }
            }
        }
    }
    else
    {
        val = (std::numeric_limits<T>::min)();
        for( i = 0 ; i < m ; i++ )
        {
            for( j = 0 ; j < n ; j++ )
            {
                auto res = buf[ i*rsc + j*csc ];
                if((isnan(res)) || (fabs(res) < val))
                {
                    return abs(res);
                }
            }
        }
    }
    return resid;
}

template <typename T, typename U>
double libblis_matrix_check_cmplx(vflg_t flg, T* buf, dim_t m, dim_t n,
                                                     dim_t rsc, dim_t csc)
{
    double resid = 0.0;
    unsigned int i,j;
    U val = 0.0;
    if(flg == BLIS_OVERFLOW)
    {
        val = (std::numeric_limits<U>::max)();
        for( i = 0 ; i < m ; i++ )
        {
            for( j = 0 ; j < n ; j++ )
            {
                auto res = buf[ i*rsc + j*csc ];
                if((isnan(res.real) || (isnan(res.imag))) ||
                   (fabs(res.real) > val) || (fabs(res.imag) > val))
                {
                    resid = (fabs(res.real) > fabs(res.imag)) ? res.real : res.imag;
                    return abs(resid);
                }
            }
        }
    }
    else
    {
        val = (std::numeric_limits<U>::min)();
        for( i = 0 ; i < m ; i++ )
        {
            for( j = 0 ; j < n ; j++ )
            {
                auto res = buf[ i*rsc + j*csc ];
                if((isnan(res.real) || (isnan(res.imag))) ||
                   (fabs(res.real) < val) || (fabs(res.imag) < val))
                {
                    resid = (fabs(res.real) < fabs(res.imag)) ? res.imag : res.real;
                    return abs(resid);
                }
            }
        }
    }
    return resid;
}

template <typename T>
void conjugatematrix(T* X, dim_t m, dim_t n, dim_t rs, dim_t cs)
{
    dim_t  i,j;
    for( i = 0 ; i < m ; i++ )
    {
        for( j = 0 ; j < n ; j++ )
        {
            X[i*rs + j*cs] = conjugate<T>( X[i*rs + j*cs] );
        }
    }
    return;
}


template <typename T>
void test_mmfill( T* dst, T* src, f77_int m, f77_int n, f77_int ld, T val )
{
  f77_int i,j;
  f77_int ldm = ld-ELD;
  if( n == ldm )
  {
     f77_int tmp;
     tmp = n;
     n = m;
     m = tmp;
  }

  for( j = 0 ; j < (n+ELD) ; j++ ) {
    for( i = 0 ; i < (m+ELD) ; i++ ) {
      dst[ i + j*ld ] = val;
    }
  }

  for( j = 0 ; j < n ; j++ ) {
    for( i = 0 ; i < m ; i++ ) {
      dst[ i + j*ld ] = src[ i + j*ldm ];
    }
  }
/*
  for( j = 0 ; j < n ; j++ ) {
    for( i = m ; i < (m+ELD) ; i++ ) {
       dst[ i + j*ld ] = val;
    }
  }

  for( j = n ; j < (n+ELD) ; j++ ) {
    for( i = 0 ; i < (m+ELD) ; i++ ) {
      dst[ i + j*ld ] = val;
    }
  }
*/
}

template <typename T>
double test_mmchk( T* dst, f77_int m, f77_int n, f77_int ld, T val )
{
  f77_int i,j;
  f77_int ldm = ld-ELD;

  if( n == ldm )
  {
     f77_int tmp;
     tmp = n;
     n = m;
     m = tmp;
  }

  for( j = 0 ; j < n ; j++ ) {
    for( i = m ; i < (m+ELD) ; i++ ) {
      if( dst[ i + j*ld ] != val ) {
        cout << "Invalid Access" << endl;
        return val;
      }
    }
  }

  for( j = n ; j < (n+ELD) ; j++ ) {
    for( i = 0 ; i < (m+ELD) ; i++ ) {
      if( dst[ i + j*ld ] != val ) {
        cout << "Invalid Access" << endl;
        return val;
      }
    }
  }
  return 0;
}

template <typename T, typename U>
double test_mmchkc( T* dst, f77_int m, f77_int n, f77_int ld, U val )
{
  f77_int i,j;
  f77_int ldm = ld-ELD;

  if( n == ldm )
  {
     f77_int tmp;
     tmp = n;
     n = m;
     m = tmp;
  }

  for( j = 0 ; j < n ; j++ ) {
    for( i = m ; i < (m+ELD) ; i++ ) {
      T tmp = dst[ i + j*ld ];
      if((tmp.real != val) ||(tmp.imag != val)) {
        cout << "Invalid Access" << endl;
        return val;
      }
    }
  }

  for( j = n ; j < (n+ELD) ; j++ ) {
    for( i = 0 ; i < (m+ELD) ; i++ ) {
      T tmp = dst[ i + j*ld ];
      if((tmp.real != val) ||(tmp.imag != val)) {
        cout << "Invalid Access" << endl;
        return val;
      }
    }
  }
  return 0;
}

void conjugate_tensor( obj_t* aa, num_t dt );
void libblis_test_build_col_labels_string( test_params_t* params,
                                           test_op_t* op, char* l_str );
unsigned int libblis_test_get_n_dims_from_dimset( dimset_t dimset ) ;
void bli_param_map_char_to_blas_trans( char trans, trans_t* blas_trans );
void bli_param_map_char_to_herk_trans( char trans, trans_t* herk_trans );
void bli_param_map_char_to_syrk_trans( char trans, trans_t* syrk_trans );
void libblis_test_fprintf( FILE* output_stream, const char* message, ... );
ind_t ind_enable_get_str( test_params_t*, unsigned int d, unsigned int x,
                                                          test_op_t* op );

double libblis_test_vector_check( test_params_t* params, obj_t* y );
double libblis_test_matrix_check( test_params_t* params, obj_t* y );
double libblis_test_bitrp_vector( obj_t* c, obj_t* r, num_t dt );
double libblis_test_bitrp_matrix( obj_t* c, obj_t* r, num_t dt );
void libblis_test_mobj_irandomize( test_params_t* params, obj_t* x );
void libblis_test_vobj_irandomize( test_params_t* params, obj_t* x );
void test_fillbuffmem( obj_t* c, num_t dt );
void test_fillbuffmem_diag( obj_t* c, num_t dt );

int libblis_test_dt_str_has_sp_char_str( int n, char* str );
int libblis_test_dt_str_has_dp_char_str( int n, char* str );
int libblis_test_dt_str_has_rd_char_str( int n, char* str );

void bli_map_blis_to_netlib_trans( trans_t trans, char* blas_trans );
bool libblis_test_op_is_done( test_op_t* op );
int libblis_test_l3_is_disabled( test_op_t* op );
bool libblis_test_get_string_for_result( double resid, num_t dt,
                                         const thresh_t* thresh, char *r_val );

void libblis_test_read_next_line( char* buffer, FILE* input_stream );
void libblis_test_fopen_check_stream( char* filename_str, FILE* stream );
void libblis_test_read_section_override( test_ops_t*  ops,
                                         FILE* input_stream, int* override );
void libblis_test_read_op_info( test_ops_t*  ops, FILE* input_stream,
      opid_t opid, dimset_t dimset,  unsigned int n_params, test_op_t* op );
void libblis_test_output_section_overrides( FILE* os, test_ops_t* ops );
void libblis_test_output_params_struct( FILE* os, test_params_t* params );

param_t libblis_test_get_param_type_for_char( char p_type );
unsigned int libblis_test_count_combos ( unsigned int n_operands, char* spec_str, char** char_sets );
void libblis_test_fill_param_strings( char* p_spec_str, char** chars_for_param,
         unsigned int  n_params, unsigned int  n_param_combos, char** pc_str );
operand_t libblis_test_get_operand_type_for_char( char o_type ) ;
int libblis_test_dt_str_has_rd_char( test_params_t* params );
int libblis_test_dt_str_has_cd_char( test_params_t* params );
int libblis_test_dt_str_has_sp_char( test_params_t* params );
int libblis_test_dt_str_has_dp_char( test_params_t* params );
char libblis_test_proj_dtchar_to_precchar( char dt_char );

void libblis_test_printf_error( const char* message, ... );
void libblis_test_check_empty_problem( obj_t* c, double* resid );
void libblis_test_mobj_create( test_params_t* params, num_t dt, trans_t trans,
                                     char storage, dim_t m, dim_t n, obj_t* a );
void libblis_test_vobj_create( test_params_t* params, num_t dt,
                                                  char storage, dim_t m, obj_t* x );
void libblis_test_mobj_randomize( test_params_t* params, bool normalize, obj_t* a );
void libblis_test_mobj_load_diag( test_params_t* params, obj_t* a );
void libblis_test_vobj_randomize( test_params_t* params, bool normalize, obj_t* x );

void libblis_test_alloc_buffer( obj_t* a );
void libblis_test_obj_free( obj_t* a );

#endif