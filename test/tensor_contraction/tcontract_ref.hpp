#include "blis.h"
#include "complex_math.hpp"

#include <vector>
#include <array>
#include <cassert>

inline void increment(inc_t, gint_t) {}

template <typename T, typename... Args>
void increment(inc_t n, gint_t i, T& off, const inc_t* s, Args&... args)
{
    off += s[i]*n;
    increment(n, i, args...);
}

template <typename Body, typename... Args>
void for_each_impl(gint_t ndim, const dim_t* n,
                   dim_t off, dim_t len,
                   Body& body,
                   Args&... args)
{
    std::array<dim_t,8> i = {};
    assert( ndim <= i.size() );

    if ( off )
    {
        for ( gint_t k = 0; k < ndim; k++ )
        {
            i[k] = off % n[k];
            off /= n[k];
            increment(i[k], k, args...);
        }
    }

    for ( dim_t pos = 0; pos < len; pos++ )
    {
        body();

        for ( gint_t k = 0; k < ndim; k++ )
        {
            if ( i[k] == n[k]-1 )
            {
                increment(-i[k], k, args...);
                i[k] = 0;
            }
            else
            {
                increment(1, k, args...);
                i[k]++;
                break;
            }
        }
    }
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              dim_t off, dim_t len,
              T& a, const inc_t* s_a,
              Body&& body)
{
    for_each_impl( ndim, n, off, len, body, a, s_a );
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              dim_t off, dim_t len,
              T& a, const inc_t* s_a,
              T& b, const inc_t* s_b,
              Body&& body)
{
    for_each_impl( ndim, n, off, len, body, a, s_a, b, s_b );
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              T& a, const inc_t* s_a,
              Body&& body)
{
    dim_t len = 1;
    for ( gint_t i = 0;i < ndim;i++ ) len *= n[i];
    for_each_impl( ndim, n, 0, len, body, a, s_a );
}

template <typename T, typename Body>
void for_each(gint_t ndim, const dim_t* n,
              T& a, const inc_t* s_a,
              T& b, const inc_t* s_b,
              Body&& body)
{
    dim_t len = 1;
    for ( gint_t i = 0;i < ndim;i++ ) len *= n[i];
    for_each_impl( ndim, n, 0, len, body, a, s_a, b, s_b );
}

void tcontract_ref( num_t dt, const std::vector<dim_t>& m, const std::vector<dim_t>& n, const std::vector<dim_t>& k,
                    const void* alpha, const void* a, const std::vector<inc_t>& rs_a, const std::vector<inc_t>& cs_a,
                                       const void* b, const std::vector<inc_t>& rs_b, const std::vector<inc_t>& cs_b,
                    const void*  beta,       void* c, const std::vector<inc_t>& rs_c, const std::vector<inc_t>& cs_c );
