#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cstdio>

#include "blis_test.h"

#ifdef BLIS_ENABLE_ADDONS

#define S8_MIN  (-128)
#define S8_MAX  (+127)

static inline int max (int a, int b)
{
	return ( a > b ? a : b );
}

static inline int min (int a, int b)
{
	return ( a < b ? a : b );
}

template <typename T>
void fill_array ( T* arr, dim_t size )
{
    T* temp_arr = ( T* ) arr;
    for ( dim_t i = 0; i < size; ++i )
    {
        temp_arr[i] = ( T )( i % 10 );
    }
}

template <typename T>
void fill_array_post_ops( T* arr, dim_t size )
{
    T* temp_arr = ( T* ) arr;
    for ( dim_t i = 0; i < size; ++i )
    {
        temp_arr[i] = ( T )( i % 20 );
    }
}

static void float_to_bf16( float* float_value, bfloat16* bf16_val )
{
    /*Set offset 2 to copy most significant 2 bytes of float
    to convert float values to bf16 values*/
    memcpy( ( bf16_val ), (char *)( float_value ) + 2, sizeof ( bfloat16 ) );
}

static inline void convert_float_arr_to_bf16( float* array, bfloat16* array_bf16, int size )
{
    for (int i = 0 ; i < size ; i++)
    {
        float_to_bf16( ( array + i ), ( array_bf16 + i ) );
    }
}

/* Only supports bias followed by RELU and vice versa for now.*/
template <typename X, typename Z>
aocl_post_op* lpgemm_create_post_ops_struct( dim_t m, dim_t n,
                  char* post_ops_str, bool dscale_out )
{
  aocl_post_op* post_ops = NULL;
  post_ops = ( aocl_post_op* ) malloc( sizeof( aocl_post_op ) );

  if ( ( post_ops == NULL ) && ( dscale_out ) )
  {
    return NULL;
  }

  /* Only supporting 3 post ops at max for now.*/
  dim_t max_post_ops_seq_length = 3;
  post_ops->seq_vector = ( AOCL_POST_OP_TYPE* )
        malloc( max_post_ops_seq_length * sizeof( AOCL_POST_OP_TYPE ) );

  if ( post_ops->seq_vector == NULL )
  {
    free( post_ops );
    return NULL;
  }

  /* Parse post ops list.*/
  dim_t cur_op_index = 0;
  /* Ensure the buffers that use NULL check in deinit code is properly set to NULL.*/
  post_ops->eltwise.algo.alpha = NULL;
  post_ops->bias.bias = NULL;
  post_ops->sum.scale_factor = NULL;
  if ( post_ops_str != NULL )
  {
    char* ops_tok = strtok(post_ops_str, ", " );
    bool is_param_relu = FALSE;
    while ( ops_tok )
    {
      if ( strcmp( ops_tok, "bias") == 0 )
      {
        post_ops->seq_vector[cur_op_index] = BIAS;
      }
      else if ( strcmp( ops_tok, "relu") == 0 )
      {
        post_ops->seq_vector[cur_op_index] = ELTWISE;
      }
      else if ( strcmp( ops_tok, "prelu") == 0 )
      {
        post_ops->seq_vector[cur_op_index] = ELTWISE;
        is_param_relu = TRUE;
      }
      ops_tok = strtok( NULL, ", " );
      cur_op_index++;
    }

    /* Allocate bias buffer, return early if alloc fails.*/
    post_ops->bias.bias = malloc( n * sizeof( X ) );
    if ( post_ops->bias.bias == NULL )
    {
      free( post_ops->seq_vector );
      free( post_ops );
      return NULL;
    }
    fill_array_post_ops<X>((X*)post_ops->bias.bias, n );

    post_ops->eltwise.is_power_of_2 = FALSE;
    post_ops->eltwise.scale_factor = NULL;
    post_ops->eltwise.algo.alpha = NULL;
    post_ops->eltwise.algo.algo_type = RELU;
    if ( is_param_relu == TRUE )
    {
      post_ops->eltwise.algo.alpha = malloc( sizeof( X ) );
      *( ( X* ) post_ops->eltwise.algo.alpha ) = ( X )6;
      post_ops->eltwise.algo.algo_type = PRELU;
    }
    post_ops->eltwise.algo.beta = NULL;
  }

  if ( dscale_out )
  {
    post_ops->seq_vector[cur_op_index] = SCALE;
    cur_op_index++;

    post_ops->sum.is_power_of_2 = FALSE;
    post_ops->sum.scale_factor = NULL;
    post_ops->sum.buff = NULL;
    post_ops->sum.zero_point = NULL;
    if ( dscale_out )
    {
      /* Allocate scale buffer, return early if alloc fails.*/
      post_ops->sum.scale_factor = malloc( n * sizeof( Z ) );
      if ( post_ops->sum.scale_factor == NULL )
      {
       free ( post_ops->bias.bias );
       free( post_ops->seq_vector );
       free( post_ops );
       return NULL;
      }
      /* Fill scale factor.*/
      Z* temp_dscale_ptr = ( Z* )post_ops->sum.scale_factor;
      for ( dim_t i = 0; i < n; ++i )
      {
        temp_dscale_ptr[i] = ( ( Z )1 )/ ( ( Z )1000 );
      }
    }
  }

  post_ops->seq_length = cur_op_index;

  return post_ops;
}

void lpgemm_destroy_post_ops_struct( aocl_post_op* post_ops );

template <typename B, typename Z, typename ST>
B mat_mul_accuracy_check_downscale( Z temp_accum, B out_temp_accum,
                                   aocl_post_op*  post_op,  dim_t j )
{
  out_temp_accum = ( B ) min ( max ( nearbyintf( ( ST )temp_accum *
      ( *( ( ST* )post_op->sum.scale_factor + j ) ) ), S8_MIN ), S8_MAX ) ;
  return 	out_temp_accum;
}

template <typename A, typename B, typename X, typename Z>
Z mat_mul_accuracy_check_accum
    (
      A*    a,
      B*    b,
      X*    c_ref,
      Z     temp_accum,
      Z     alpha,
      Z     beta,
      dim_t rs_a,
      dim_t rs_b,
      dim_t cs_a,
      dim_t cs_b,
      dim_t rs_c_ref,
      dim_t cs_c_ref,
      dim_t i,
      dim_t j,
      dim_t k
    )
{
  dim_t p;

  for( p = 0 ; p < k ; ++p )
  {
    temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) *
                    *( b + ( rs_b * p ) + ( cs_b * j ) ) );
  }

  temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) )
                 + ( alpha * temp_accum );
  return temp_accum;
}

template <typename A, typename B, typename X, typename Z, typename ST>
double mat_mul_accuracy_check_driver
    (
      dim_t   m,
      dim_t   n,
      dim_t   k,
      Z       alpha,
      A*      a,
      dim_t   rs_a,
      dim_t   cs_a,
      B*      b,
      dim_t   rs_b,
      dim_t   cs_b,
      Z       beta,
      X*      c,
      dim_t   rs_c,
      dim_t   cs_c,
      X*      c_ref,
      aocl_post_op*  post_op,
      bool    dscale_out
    )
{
  double resid = 0.0;
  dim_t rs_c_ref = rs_c;
  dim_t cs_c_ref = cs_c;
  dim_t i,j;

  for( i = 0 ; i < m ; ++i )
  {
    for( j = 0 ; j < n ; ++j )
    {
      Z temp_accum = 0;
      X out_temp_accum = 0;

      temp_accum = mat_mul_accuracy_check_accum<A,B,X,Z> (a, b, c_ref,
                   temp_accum, alpha, beta, rs_a, rs_b, cs_a, cs_b,
                   rs_c_ref, cs_c_ref, i, j, k);

      if ( post_op != NULL )
      {
        /* Apply bias followed by relu. */
        if ( post_op->seq_vector[0] == BIAS )
        {
          if ( post_op->seq_length >= 1 )
          {
            temp_accum += ( *( ( Z* )post_op->bias.bias + j ) );
          }
          if ( ( post_op->seq_length > 1 ) &&
            ( post_op->seq_vector[1] == ELTWISE ) )
          {
            if ( post_op->eltwise.algo.alpha != NULL ) /* PReLU*/
            {
              temp_accum = ( temp_accum > 0 ) ?
              temp_accum :
              ( temp_accum *
              *( ( Z* ) post_op->eltwise.algo.alpha ) );
            }
            else
            {
              temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ;
            }
          }
        }
        else if ( post_op->seq_vector[0] == ELTWISE )
        {
          if ( post_op->seq_length >= 1 )
          {
            if ( post_op->eltwise.algo.alpha != NULL ) /* PReLU*/
            {
              temp_accum = ( temp_accum > 0 ) ?
               temp_accum :
               ( temp_accum * *( ( Z* ) post_op->eltwise.algo.alpha ) );
            }
            else
            {
              temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ;
            }
          }
          if ( ( post_op->seq_length > 1 ) && ( post_op->seq_vector[1] == BIAS ) )
          {
            temp_accum += ( *( ( Z* )post_op->bias.bias + j ) );
          }
        }
      }
      if ( dscale_out )
      {
        out_temp_accum = mat_mul_accuracy_check_downscale<B, Z, ST>
                                     ( temp_accum, out_temp_accum, post_op, j);
      }
      else
      {
        out_temp_accum = ( X )temp_accum;
      }

      if( *( c + ( rs_c * i ) + ( cs_c * j ) ) != out_temp_accum )
      {
        auto tmp = *( c + ( rs_c * i ) + ( cs_c * j ) );
        resid += abs( tmp - out_temp_accum );
        //return resid;
      }
    }
  }
	 return resid;
}

bfloat16 mat_mul_accuracy_check_downscale_bf16
        ( float temp_accum,  bfloat16 out_temp_accum, aocl_post_op*  post_op, dim_t j);

float bf16_to_float( bfloat16 bf16_val );

float mat_mul_accuracy_check_accum_bf16
    (
      bfloat16* a,
      bfloat16* b,
      float*    c_ref,
      float     temp_accum,
      float     alpha,
      float     beta,
      dim_t     rs_a,
      dim_t     rs_b,
      dim_t     cs_a,
      dim_t     cs_b,
      dim_t     rs_c_ref,
      dim_t     cs_c_ref,
      dim_t     i,
      dim_t     j,
      dim_t     k
    );

float mat_mul_accuracy_check_accum_bf16
    (
      bfloat16* a,
      bfloat16* b,
      bfloat16* c_ref,
      float     temp_accum,
      float     alpha,
      float     beta,
      dim_t     rs_a,
      dim_t     rs_b,
      dim_t     cs_a,
      dim_t     cs_b,
      dim_t     rs_c_ref,
      dim_t     cs_c_ref,
      dim_t     i,
      dim_t     j,
      dim_t     k
    );

template <typename A, typename B, typename X, typename Z, typename ST>
double mat_mul_accuracy_check_driver_bf16
    (
      dim_t   m,
      dim_t   n,
      dim_t   k,
      Z       alpha,
      A*      a,
      dim_t   rs_a,
      dim_t   cs_a,
      B*      b,
      dim_t   rs_b,
      dim_t   cs_b,
      Z       beta,
      X*      c,
      dim_t   rs_c,
      dim_t   cs_c,
      X*      c_ref,
      aocl_post_op*  post_op,
      bool    dscale_out
    )
{
  double resid = 0.0;
  dim_t rs_c_ref = rs_c;
  dim_t cs_c_ref = cs_c;

  for ( dim_t i = 0; i < m; ++i )
  {
    for ( dim_t j = 0; j < n; ++j )
    {
      Z temp_accum = 0;
      X out_temp_accum = 0;

      temp_accum = mat_mul_accuracy_check_accum_bf16(a, b, c_ref,
                   temp_accum, alpha, beta, rs_a, rs_b, cs_a, cs_b,
                   rs_c_ref, cs_c_ref, i, j, k);

      if ( post_op != NULL )
      {
        /* Apply bias followed by relu. */
        if ( post_op->seq_vector[0] == BIAS )
        {
          if ( post_op->seq_length >= 1 )
          {
            temp_accum += ( *( ( Z* )post_op->bias.bias + j ) );
          }
          if ( ( post_op->seq_length > 1 ) &&
            ( post_op->seq_vector[1] == ELTWISE ) )
          {
            if ( post_op->eltwise.algo.alpha != NULL ) /* PReLU*/
            {
              temp_accum = ( temp_accum > 0 ) ?
              temp_accum :
              ( temp_accum *
              *( ( Z* ) post_op->eltwise.algo.alpha ) );
            }
            else
            {
              temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ;
            }
          }
        }
        else if ( post_op->seq_vector[0] == ELTWISE )
        {
          if ( post_op->seq_length >= 1 )
          {
            if ( post_op->eltwise.algo.alpha != NULL ) /* PReLU*/
            {
              temp_accum = ( temp_accum > 0 ) ?
               temp_accum :
               ( temp_accum * *( ( Z* ) post_op->eltwise.algo.alpha ) );
            }
            else
            {
              temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ;
            }
          }
          if ( ( post_op->seq_length > 1 ) && ( post_op->seq_vector[1] == BIAS ) )
          {
            temp_accum += ( *( ( Z* )post_op->bias.bias + j ) );
          }
        }
      }
      if ( dscale_out )
      {
        out_temp_accum = mat_mul_accuracy_check_downscale_bf16
                                     ( temp_accum, out_temp_accum, post_op, j);
      }
      else
      {
        out_temp_accum = ( X )temp_accum;
      }

      if( *( c + ( rs_c * i ) + ( cs_c * j ) ) != out_temp_accum )
      {
        auto tmp = *( c + ( rs_c * i ) + ( cs_c * j ) );
        resid += abs( tmp - out_temp_accum );
        //return resid;
      }
    }
  }
	 return resid;
}
#endif

template <typename T>
void print_matrics(T *x , dim_t mm, dim_t nn, dim_t ld)
{
    dim_t i,j;
    int32_t val;
    for ( i = 0; i < mm; ++i ) {
        for ( j = 0; j < nn; ++j ) {
             val = (int32_t)(x[i*ld + j]);
             printf("%9d", val);
        }
        cout << endl;
    }
    cout << endl;
    return;
}

