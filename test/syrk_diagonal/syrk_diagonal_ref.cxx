#include "syrk_diagonal_ref.h"
#include "complex_math.hpp"

typedef void (*syrk_diag_ref_vft)
    (
        uplo_t uplo,
        dim_t m,
        dim_t k,
        void* alpha,
        void* a, inc_t rs_a, inc_t cs_a,
        void* d, inc_t incd,
        void* beta,
        void* c, inc_t rs_c, inc_t cs_c
    );

template <typename T>
void syrk_diag_ref
    (
        uplo_t uplo,
        dim_t m,
        dim_t k,
        void* alpha,
        void* a, inc_t rs_a, inc_t cs_a,
        void* d, inc_t incd,
        void* beta,
        void* c, inc_t rs_c, inc_t cs_c
    )
{
    auto alpha_cast = *( T* )alpha;
    auto beta_cast  = *( T* )beta;
    auto a_cast     = ( T* )a;
    auto d_cast     = ( T* )d;
    auto c_cast     = ( T* )c;

    for ( dim_t i = 0; i < m; i++ )
    {
        dim_t j_min = uplo == BLIS_UPPER ? i : 0;
        dim_t j_max = uplo == BLIS_UPPER ? m : i+1;

        for ( dim_t j = j_min; j < j_max; j++ )
        {
            auto ada = convert<T>(0.0);

            for ( dim_t p = 0; p < k; p++ )
            {
                ada += a_cast[ i*rs_a + p*cs_a ] *
                       d_cast[          p*incd ] *
                       a_cast[ j*rs_a + p*cs_a ];
            }

            if ( beta_cast == convert<T>(0.0) )
            {
                c_cast[ i*rs_c + j*cs_c ] = alpha_cast * ada;
            }
            else
            {
                c_cast[ i*rs_c + j*cs_c ] = alpha_cast * ada +
                                             beta_cast * c_cast[ i*rs_c + j*cs_c ];
            }
        }
    }
}

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
static auto PASTEMAC(ch,op) = &syrk_diag_ref<ctype>;

INSERT_GENTFUNC_BASIC0(syrk_diag_ref);

static syrk_diag_ref_vft GENARRAY( syrk_diag_ref_impl, syrk_diag_ref );

void syrk_diag_ref( obj_t* alpha, obj_t* a, obj_t* d, obj_t* beta, obj_t* c )
{
    num_t dt = bli_obj_dt( a );

    dim_t m = bli_obj_length_after_trans( a );
    dim_t k = bli_obj_width_after_trans( a );

    inc_t rs_a = bli_obj_row_stride( a );
    inc_t cs_a = bli_obj_col_stride( a );
    inc_t rs_c = bli_obj_row_stride( c );
    inc_t cs_c = bli_obj_col_stride( c );
    inc_t incd = bli_obj_row_stride( d );

    if ( bli_obj_has_trans( a ) )
        bli_swap_incs( &rs_a, &cs_a );

    if ( bli_obj_has_trans( c ) )
        bli_swap_incs( &rs_c, &cs_c );

    syrk_diag_ref_impl[ dt ]
    (
      bli_obj_uplo( c ),
      m, k,
      bli_obj_buffer_for_1x1( dt, alpha ),
      bli_obj_buffer_at_off( a ), rs_a, cs_a,
      bli_obj_buffer_at_off( d ), incd,
      bli_obj_buffer_for_1x1( dt, beta ),
      bli_obj_buffer_at_off( c ), rs_c, cs_c
    );
}

