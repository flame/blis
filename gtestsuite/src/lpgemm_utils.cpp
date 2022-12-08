#include <iostream>
#include <string>
#include <tuple>
#include <vector>
#include <cstdio>
#include "lpgemm_utils.h"

#ifdef BLIS_ENABLE_ADDONS

bfloat16 mat_mul_accuracy_check_downscale_bf16( float temp_accum,  bfloat16 out_temp_accum,
                                   aocl_post_op*  post_op, dim_t j)
{
  float_to_bf16( ( &temp_accum ), ( &out_temp_accum ) );
  return out_temp_accum;
}

float bf16_to_float( bfloat16 bf16_val )
{
  int32_t inter_temp = *( ( int16_t* ) &bf16_val );
  inter_temp = inter_temp << 16;
  float float_value = *( float* ) ( &inter_temp );
  return float_value;
}

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
    )
{
  for ( dim_t p = 0; p < k; ++p)
  {
    float a_float = bf16_to_float( *( a + i * rs_a + p * cs_a ) );
    float b_float = bf16_to_float( *( b + p * rs_b + j * cs_b ) );
    temp_accum += ( ( a_float ) * ( b_float ) );
  }
  temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) )
                 + ( alpha * temp_accum );
  return temp_accum;
}

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
    )
{
  for ( dim_t p = 0; p < k; ++p)
  {
    float a_float = bf16_to_float( *( a + i*rs_a + p*cs_a ) );
    float b_float = bf16_to_float( *( b + p*rs_b + j*cs_b ) );
    temp_accum += ( ( a_float ) * ( b_float ) );
  }
  float c_ref_float = bf16_to_float( *( c_ref + i*rs_c_ref + j*cs_c_ref ) );
  temp_accum = ( beta * ( c_ref_float ) ) + ( alpha * temp_accum );

  return temp_accum;
}

void lpgemm_destroy_post_ops_struct( aocl_post_op* post_ops )
{
  if ( post_ops == NULL )
  {
    return;
  }

  if ( post_ops->eltwise.algo.alpha != NULL )
  {
    free( post_ops->eltwise.algo.alpha );
  }
  if ( post_ops->sum.scale_factor != NULL )
  {
    free( post_ops->sum.scale_factor );
  }
  if ( post_ops->bias.bias != NULL )
  {
    free( post_ops->bias.bias );
  }
  if( post_ops->seq_vector != NULL )
  {
    free( post_ops->seq_vector );
  }

  free( post_ops );
}
#endif