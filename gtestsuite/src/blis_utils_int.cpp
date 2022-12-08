#include <time.h>
#include "blis_utils.h"

//#define DEF_SRAND

void bli_isrands( float* alpha )
{
    /* 24*24*k < 23 bits max value to avoid
               rounding off errors */
    int32_t a = ( int32_t ) (rand() % 4);
    *alpha = ( float ) a ;
}

void bli_idrands( double* alpha )
{
    int64_t a = ( int64_t ) (rand() % 4);
    *alpha = ( double ) a ;
}

void bli_icrands( scomplex* alpha )
{
    bli_isrands( &(alpha->real) );
    bli_isrands( &(alpha->imag) );
}

void bli_izrands( dcomplex* alpha )
{
    bli_idrands( &(alpha->real) );
    bli_idrands( &(alpha->imag) );
}

void bli_israndv( int n, float* x, int incx )
{
    float* chi;
    int i;
#ifdef DEF_SRAND
    srand(time(0));
#endif
    for ( i = 0; i < n; ++i )	{
        chi = x + i*incx;
        bli_isrands( chi );
    }
}

void bli_idrandv( int n, double* x, int incx )
{
    double* chi;
    int i;
#ifdef DEF_SRAND
    srand(time(0));
#endif
    for ( i = 0; i < n; ++i )
    {
        chi = x + i*incx;
        bli_idrands( chi );
    }
}

void bli_icrandv( int n, scomplex* x, int incx )
{
    scomplex* chi;
    int       i;
#ifdef DEF_SRAND
    srand(time(0));
#endif
    for ( i = 0; i < n; ++i )
    {
        chi = x + i*incx;
        bli_icrands( chi );
    }
}

void bli_izrandv( int n, dcomplex* x, int incx )
{
    dcomplex* chi;
    int       i;
#ifdef DEF_SRAND
    srand(time(0));
#endif
    for ( i = 0; i < n; ++i )
    {
        chi = x + i*incx;
        bli_izrands( chi );
    }
}


void bli_israndm(test_params_t* params, int m, int n, float* a, int a_rs, int a_cs )
{
    float*    a_begin, *x;
    inc_t     inca, lda;
    inc_t     n_iter;
    inc_t     n_elem;
    int       i,j;

    // Return early if possible.
    if ( bli_zero_dim2( m, n ) ) return;

    // Initialize with optimal values for column-major storage.
    inca   = a_rs;
    lda    = a_cs;
    n_iter = n;
    n_elem = m;

    // An optimization: if A is row-major, then let's access the matrix by
    // rows instead of by columns for increased spatial locality.
    if ( bli_is_row_stored( a_rs, a_cs ) )
    {
        bli_swap_incs( &n_iter, &n_elem );
        bli_swap_incs( &lda, &inca );
    }

    if(1)   //if(params->oruflw == BLIS_DEFAULT)
    {
        for ( j = 0; j < n_iter; j++ )
        {
            a_begin = a + j*lda;
            bli_israndv( n_elem, a_begin, inca );
        }
    }
    else
    {
        float val;
        val = (std::numeric_limits<float>::max)();
        if(params->oruflw == BLIS_UNDERFLOW)
        {
            val = (std::numeric_limits<float>::min)();
        }
        for ( j = 0; j < n_iter; j++ )
        {
            x = a + j*lda;
            for ( i = 0; i < n_elem; ++i )
            {
                x[i*inca] = val ;
            }
        }
    }
}

void bli_idrandm(test_params_t* params, int m, int n, double* a, int a_rs, int a_cs )
{
    double*   a_begin, *x;
    inc_t     inca, lda;
    inc_t     n_iter;
    inc_t     n_elem;
    int       i,j;

    // Return early if possible.
    if ( bli_zero_dim2( m, n ) ) return;

    // Initialize with optimal values for column-major storage.
    inca   = a_rs;
    lda    = a_cs;
    n_iter = n;
    n_elem = m;

    // An optimization: if A is row-major, then let's access the matrix by
    // rows instead of by columns for increased spatial locality.
    if ( bli_is_row_stored( a_rs, a_cs ) )
    {
        bli_swap_incs( &n_iter, &n_elem );
        bli_swap_incs( &lda, &inca );
    }

    if(1)   //if(params->oruflw == BLIS_DEFAULT)
    {
        for ( j = 0; j < n_iter; j++ )
        {
            a_begin = a + j*lda;
            bli_idrandv( n_elem, a_begin, inca );
        }
    }
    else
    {
        double val;
        val = (std::numeric_limits<double>::max)();
        if(params->oruflw == BLIS_UNDERFLOW)
        {
            val = (std::numeric_limits<double>::min)();
        }
        for ( j = 0; j < n_iter; j++ )	{
            x = a + j*lda;
            for ( i = 0; i < n_elem; ++i )	{
                x[i*inca] = val ;
            }
        }
    }
}

void bli_icrandm(test_params_t* params, int m, int n, scomplex* a, int a_rs, int a_cs )
{
    scomplex* a_begin, *x;
    inc_t     inca, lda;
    inc_t     n_iter;
    inc_t     n_elem;
    int       i,j;

    // Return early if possible.
    if ( bli_zero_dim2( m, n ) ) return;

    // Initialize with optimal values for column-major storage.
    inca   = a_rs;
    lda    = a_cs;
    n_iter = n;
    n_elem = m;

    // An optimization: if A is row-major, then let's access the matrix by
    // rows instead of by columns for increased spatial locality.
    if ( bli_is_row_stored( a_rs, a_cs ) )
    {
        bli_swap_incs( &n_iter, &n_elem );
        bli_swap_incs( &lda, &inca );
    }

    if(1)   //if(params->oruflw == BLIS_DEFAULT)
    {
        for ( j = 0; j < n_iter; j++ )
        {
            a_begin = a + j*lda;
            bli_icrandv( n_elem, a_begin, inca );
        }
    }
    else
    {
        float val;
        val = (std::numeric_limits<float>::max)();
        if(params->oruflw == BLIS_UNDERFLOW)
        {
            val = (std::numeric_limits<float>::min)();
        }
        scomplex cval = {val, val};
        for ( j = 0; j < n_iter; j++ )
        {
            x = a + j*lda;
            for ( i = 0; i < n_elem; ++i )
            {
                x[i*inca] = cval ;
            }
        }
    }
}

void bli_izrandm(test_params_t* params, int m, int n, dcomplex* a, int a_rs, int a_cs ) {
    dcomplex* a_begin, *x;
    inc_t     inca, lda;
    inc_t     n_iter;
    inc_t     n_elem;
    int       i,j;

    // Return early if possible.
    if ( bli_zero_dim2( m, n ) ) return;

    // Initialize with optimal values for column-major storage.
    inca   = a_rs;
    lda    = a_cs;
    n_iter = n;
    n_elem = m;

    // An optimization: if A is row-major, then let's access the matrix by
    // rows instead of by columns for increased spatial locality.
    if ( bli_is_row_stored( a_rs, a_cs ) )
    {
        bli_swap_incs( &n_iter, &n_elem );
        bli_swap_incs( &lda, &inca );
    }

    if(1)   //if(params->oruflw == BLIS_DEFAULT)
    {
        for ( j = 0; j < n_iter; j++ )
        {
            a_begin = a + j*lda;
            bli_izrandv( n_elem, a_begin, inca );
        }
    }
    else
    {
        double val;
        val = (std::numeric_limits<double>::max)();
        if(params->oruflw == BLIS_UNDERFLOW)
        {
            val = (std::numeric_limits<double>::min)();
        }
        dcomplex cval = {val, val};
        for ( j = 0; j < n_iter; j++ )
        {
            x = a + j*lda;
            for ( i = 0; i < n_elem; ++i )
            {
                x[i*inca] = cval ;
            }
        }
    }
}

void libblis_test_mobj_irandomize(test_params_t* params, obj_t* x )
{
    num_t dt = bli_obj_dt( x );
    dim_t m  = bli_obj_length( x );
    dim_t n  = bli_obj_width( x );
    inc_t rs = bli_obj_row_stride( x );
    inc_t cs = bli_obj_col_stride( x );

    switch( dt )
    {
        case BLIS_FLOAT :
        {
            float *buff = ( float * ) bli_obj_buffer_at_off( x );
            bli_israndm(params, m, n, buff, rs, cs );
            break;
        }
        case BLIS_DOUBLE :
        {
            double *buff = ( double * ) bli_obj_buffer_at_off( x );
            bli_idrandm(params, m, n, buff, rs, cs );
            break;
        }
        case BLIS_SCOMPLEX :
        {
            scomplex *buff = ( scomplex * ) bli_obj_buffer_at_off( x );
            bli_icrandm(params, m, n, buff, rs, cs );
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex *buff = ( dcomplex * ) bli_obj_buffer_at_off( x );
            bli_izrandm(params, m, n, buff, rs, cs );
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
}

void libblis_test_vobj_irandomize(test_params_t* params, obj_t* x )
{
    num_t dt  = bli_obj_dt( x );
    dim_t n   = bli_obj_vector_dim( x );
    inc_t inx = bli_obj_vector_inc( x );
    int       i;

    switch( dt )
    {
        case BLIS_FLOAT :
        {
            float *buff = ( float * ) bli_obj_buffer_at_off( x );
            if(1)       //if(params->oruflw == BLIS_DEFAULT)
            {
                bli_israndv( n, buff, inx );
            }
            else
            {
                float val;
                if(params->oruflw == BLIS_OVERFLOW)
                {
                   val = (std::numeric_limits<float>::max)();
                }
                else
                {
                    val = (std::numeric_limits<float>::min)();
                }
                for ( i = 0; i < n; ++i )
                {
                    buff[i*inx] = val ;
                }
            }
            break;
        }
        case BLIS_DOUBLE :
        {
            double *buff = ( double * ) bli_obj_buffer_at_off( x );
            if(1)       //if(params->oruflw == BLIS_DEFAULT)
            {
                bli_idrandv( n, buff, inx );
            }
            else
            {
                double val;
                if(params->oruflw == BLIS_OVERFLOW)
                {
                    val = (std::numeric_limits<double>::max)();
                }
                else
                {
                    val = (std::numeric_limits<double>::min)();
                }
                for ( i = 0; i < n; ++i )
                {
                    buff[i*inx] = val ;
                }
            }
            break;
        }
        case BLIS_SCOMPLEX :
        {
            scomplex *buff = ( scomplex * ) bli_obj_buffer_at_off( x );;
            if(1)       //if(params->oruflw == BLIS_DEFAULT)
            {
                bli_icrandv( n, buff, inx );
            }
            else
            {
                float val;
                if(params->oruflw == BLIS_OVERFLOW)
                {
                    val = (std::numeric_limits<float>::max)();
                }
                else
                {
                    val = (std::numeric_limits<float>::min)();
                }
                scomplex cval = {val, val};
                for ( i = 0; i < n; ++i )
                {
                    buff[i*inx] = cval ;
                }
            }
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex *buff = ( dcomplex * ) bli_obj_buffer_at_off( x );
            if(1)       //if(params->oruflw == BLIS_DEFAULT)
            {
                bli_izrandv( n, buff, inx );
            }
            else
            {
                double val;
                if(params->oruflw == BLIS_OVERFLOW)
                {
                    val = (std::numeric_limits<double>::max)();
                }
                else
                {
                    val = (std::numeric_limits<double>::min)();
                }
                dcomplex cval = {val, val};
                for ( i = 0; i < n; ++i )
                {
                    buff[i*inx] = cval ;
                }
            }
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
}

///////////////////////////////////////////////////////////////////////////////
using namespace std;
template <typename T>
void fillcbuff( dim_t rsc, dim_t csc, obj_t* c )
{
    dim_t  M = bli_obj_length( c );
    dim_t  N = bli_obj_width( c );
    dim_t  i,j;
    T* C = (T*) bli_obj_buffer( c );
    T Nan =(T)NAN;

    for( i = 0 ; i < M ; i++ )
    {
        for( j = 0 ; j < N ; j++ )
        {
            C[ i*rsc + j*csc ] = ( Nan );
        }
    }
    return;
}

template <typename U, typename T>
void fillicbuff ( dim_t rsc, dim_t csc, obj_t* c )
{
    dim_t  M = bli_obj_length( c );
    dim_t  N = bli_obj_width( c );
    dim_t  i,j;
    U* C = (U*) bli_obj_buffer( c );
    T Nan =(T)NAN;

    U tv = {0,0};
    tv.real = Nan;
    tv.imag = Nan;
    for( i = 0 ; i < M ; i++ )
    {
      for( j = 0 ; j < N ; j++ )
      {
          C[ i*rsc + j*csc ] = tv ;
      }
    }
    return;
}

void test_fillbuffmem(obj_t* c, num_t dt )
{
    dim_t  rsc, csc;

    if( bli_obj_row_stride( c ) == 1 )
    {
      rsc = 1;
      csc = bli_obj_col_stride( c );
    }
    else
    {
      rsc = bli_obj_row_stride( c );
      csc = 1 ;
    }

    switch( dt )
    {
      case BLIS_FLOAT :
      {
          fillcbuff<float>( rsc, csc, c );
          break;
      }
      case BLIS_DOUBLE :
      {
          fillcbuff<double>( rsc, csc, c );
          break;
      }
      case BLIS_SCOMPLEX :
      {
          fillicbuff<scomplex, float>( rsc, csc, c );
          break;
      }
      case BLIS_DCOMPLEX :
      {
          fillicbuff<dcomplex, double>( rsc, csc, c );
          break;
      }
      default :
          bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
  return ;
}

///////////////////////////////////////////////////////////////////////////////
using namespace std;
template <typename T>
void fillcbuff_diag( dim_t rsc, dim_t csc, obj_t* c )
{
    dim_t  M = bli_obj_length( c );
    dim_t  N = bli_obj_width( c );
    dim_t  i,j;
    T* C  = (T*) bli_obj_buffer( c );
    T val = (T) 2.0;

    for( i = 0 ; i < M ; i++ )
    {
        for( j = 0 ; j < N ; j++ )
        {
            if(i == j)
            {
                C[ i*rsc + j*csc ] = ( val );
            }
        }
    }
    return;
}

template <typename T>
void fillicbuff_diag ( dim_t rsc, dim_t csc, obj_t* c )
{
    dim_t  M = bli_obj_length( c );
    dim_t  N = bli_obj_width( c );
    dim_t  i,j;
    T* C = (T*) bli_obj_buffer( c );

    T val = {2.0,2.0};
    for( i = 0 ; i < M ; i++ )
    {
        for( j = 0 ; j < N ; j++ )
        {
            if(i == j)
            {
              C[ i*rsc + j*csc ] = ( val );
            }
        }
    }
    return;
}

void test_fillbuffmem_diag( obj_t* c, num_t dt )
{
    dim_t  rsc, csc;

    if( bli_obj_row_stride( c ) == 1 )
    {
      rsc = 1;
      csc = bli_obj_col_stride( c );
    }
    else
    {
      rsc = bli_obj_row_stride( c );
      csc = 1 ;
    }

    switch( dt )
    {
        case BLIS_FLOAT :
        {
            fillcbuff_diag<float>( rsc, csc, c );
            break;
        }
        case BLIS_DOUBLE :
        {
            fillcbuff_diag<double>( rsc, csc, c );
            break;
        }
        case BLIS_SCOMPLEX :
        {
            fillicbuff_diag<scomplex>( rsc, csc, c );
            break;
        }
        case BLIS_DCOMPLEX :
        {
            fillicbuff_diag<dcomplex>( rsc, csc, c );
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }

    return ;
}
///////////////////////////////////////////////////////////////////////////////////////////
