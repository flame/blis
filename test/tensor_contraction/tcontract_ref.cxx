#include "tcontract_ref.hpp"

template <typename T>
void tcontract_ref( const std::vector<dim_t>& m, const std::vector<dim_t>& n, const std::vector<dim_t>& k,
                    const void* alpha, const void* a, const std::vector<inc_t>& rs_a, const std::vector<inc_t>& cs_a,
                                       const void* b, const std::vector<inc_t>& rs_b, const std::vector<inc_t>& cs_b,
                    const void*  beta,       void* c, const std::vector<inc_t>& rs_c, const std::vector<inc_t>& cs_c )
{
    auto alpha_cast = *( T* )alpha;
    auto beta_cast  = *( T* )beta;
    auto a_cast     = ( T* )a;
    auto b_cast     = ( T* )b;
    auto c_cast     = ( T* )c;

    for_each(m.size(), m.data(), a_cast, rs_a.data(), c_cast, rs_c.data(),
    [&]
    {
        for_each(n.size(), n.data(), b_cast, cs_b.data(), c_cast, cs_c.data(),
        [&]
        {
            auto ab = convert<T>(0.0);

            for_each(k.size(), k.data(), a_cast, cs_a.data(), b_cast, rs_b.data(),
            [&]
            {
                ab += (*a_cast) * (*b_cast);
            });

            if ( beta_cast == convert<T>(0.0) )
            {
                *c_cast = alpha_cast * ab;
            }
            else
            {
                *c_cast = alpha_cast * ab + beta_cast * (*c_cast);
            }
        });

        assert(b_cast == b);
    });

    assert(a_cast == a);
    assert(c_cast == c);
}

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
static auto PASTEMAC(ch,op) = &tcontract_ref<ctype>;

INSERT_GENTFUNC_BASIC0(tcontract_ref);

static decltype(&tcontract_ref<void>) GENARRAY( tcontract_ref_impl, tcontract_ref );

void tcontract_ref( num_t dt, const std::vector<dim_t>& m, const std::vector<dim_t>& n, const std::vector<dim_t>& k,
                    const void* alpha, const void* a, const std::vector<inc_t>& rs_a, const std::vector<inc_t>& cs_a,
                                       const void* b, const std::vector<inc_t>& rs_b, const std::vector<inc_t>& cs_b,
                    const void*  beta,       void* c, const std::vector<inc_t>& rs_c, const std::vector<inc_t>& cs_c )
{
    tcontract_ref_impl[ dt ]
    (
      m, n, k,
      alpha, a, rs_a, cs_a,
             b, rs_b, cs_b,
       beta, c, rs_c, cs_c
    );
}

