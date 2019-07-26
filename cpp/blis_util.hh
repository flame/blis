#ifndef BLIS_UTIL_HH
#define BLIS_UTIL_HH

#include <complex>
#include <cstdarg>

namespace blis {

// -----------------------------------------------------------------------------
// Extend real, imag, conj to other datatypes.
template< typename T >
inline T real( T x ) { return x; }

template< typename T >
inline T imag( T x ) { return 0; }

template< typename T >
inline T conj( T x ) { return x; }

// -----------------------------------------------------------------------------
// 1-norm absolute value, |Re(x)| + |Im(x)|
template< typename T >
T abs1( T x )
{
    return std::abs( x );
}

template< typename T >
T abs1( std::complex<T> x )
{
    return std::abs( real(x) ) + std::abs( imag(x) );
}

// -----------------------------------------------------------------------------
// common_type_t is defined in C++14; here's a C++11 definition
#if __cplusplus >= 201402L
    using std::common_type_t;
    using std::decay_t;
#else
    template< typename... Ts >
    using common_type_t = typename std::common_type< Ts... >::type;

    template< typename... Ts >
    using decay_t = typename std::decay< Ts... >::type;
#endif

//------------------------------------------------------------------------------
/// True if T is std::complex<T2> for some type T2.
template <typename T>
struct is_complex:
    std::integral_constant<bool, false>
{};

// specialize for std::complex
template <typename T>
struct is_complex< std::complex<T> >:
    std::integral_constant<bool, true>
{};

// -----------------------------------------------------------------------------
// Based on C++14 common_type implementation from
// http://www.cplusplus.com/reference/type_traits/common_type/
// Adds promotion of complex types based on the common type of the associated
// real types. This fixes various cases:
//
// std::common_type_t< double, complex<float> > is complex<float>  (wrong)
//        scalar_type< double, complex<float> > is complex<double> (right)
//
// std::common_type_t< int, complex<long> > is not defined (compile error)
//        scalar_type< int, complex<long> > is complex<long> (right)

// for zero types
template< typename... Types >
struct scalar_type_traits;

// define scalar_type<> type alias
template< typename... Types >
using scalar_type = typename scalar_type_traits< Types... >::type;

// for one type
template< typename T >
struct scalar_type_traits< T >
{
    using type = decay_t<T>;
};

// for two types
// relies on type of ?: operator being the common type of its two arguments
template< typename T1, typename T2 >
struct scalar_type_traits< T1, T2 >
{
    using type = decay_t< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >;
};

// for either or both complex,
// find common type of associated real types, then add complex
template< typename T1, typename T2 >
struct scalar_type_traits< std::complex<T1>, T2 >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template< typename T1, typename T2 >
struct scalar_type_traits< T1, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template< typename T1, typename T2 >
struct scalar_type_traits< std::complex<T1>, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

// for three or more types
template< typename T1, typename T2, typename... Types >
struct scalar_type_traits< T1, T2, Types... >
{
    using type = scalar_type< scalar_type< T1, T2 >, Types... >;
};

// -----------------------------------------------------------------------------
// for any combination of types, determine associated real, scalar,
// and complex types.
//
// real_type< float >                               is float
// real_type< float, double, complex<float> >       is double
//
// scalar_type< float >                             is float
// scalar_type< float, complex<float> >             is complex<float>
// scalar_type< float, double, complex<float> >     is complex<double>
//
// complex_type< float >                            is complex<float>
// complex_type< float, double >                    is complex<double>
// complex_type< float, double, complex<float> >    is complex<double>

// for zero types
template< typename... Types >
struct real_type_traits;

// define real_type<> type alias
template< typename... Types >
using real_type = typename real_type_traits< Types... >::real_t;

// define complex_type<> type alias
template< typename... Types >
using complex_type = std::complex< real_type< Types... > >;

// for one type
template< typename T >
struct real_type_traits<T>
{
    using real_t = T;
};

// for one complex type, strip complex
template< typename T >
struct real_type_traits< std::complex<T> >
{
    using real_t = T;
};

// for two or more types
template< typename T1, typename... Types >
struct real_type_traits< T1, Types... >
{
    using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
};

// -----------------------------------------------------------------------------
// max that works with different data types: int64_t = max( int, int64_t )
// and any number of arguments: max( a, b, c, d )

// one argument
template< typename T >
T max( T x )
{
    return x;
}

// two arguments
template< typename T1, typename T2 >
scalar_type< T1, T2 >
    max( T1 x, T2 y )
{
    return (x >= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
scalar_type< T1, Types... >
    max( T1 first, Types... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
// min that works with different data types: int64_t = min( int, int64_t )
// and any number of arguments: min( a, b, c, d )

// one argument
template< typename T >
T min( T x )
{
    return x;
}

// two arguments
template< typename T1, typename T2 >
scalar_type< T1, T2 >
    min( T1 x, T2 y )
{
    return (x <= y ? x : y);
}

// three or more arguments
template< typename T1, typename... Types >
scalar_type< T1, Types... >
    min( T1 first, Types... args )
{
    return min( first, min( args... ) );
}

}  // namespace blis

#endif        //  #ifndef BLIS_UTIL_HH
