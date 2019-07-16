#ifndef BLIS_UTIL_HH
#define BLIS_UTIL_HH

#include <exception>
#include <complex>
#include <cstdarg>

#include <assert.h>

namespace blis {

// -----------------------------------------------------------------------------
enum class Layout : char { ColMajor = 'C', RowMajor = 'R' };
enum class Op     : char { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper    = 'U', Lower    = 'L', General   = 'G' };
enum class Diag   : char { NonUnit  = 'N', Unit     = 'U' };
enum class Side   : char { Left     = 'L', Right    = 'R' };

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style char.
inline char layout2char( Layout layout ) { return char(layout); }
inline char     op2char( Op     op     ) { return char(op);     }
inline char   uplo2char( Uplo   uplo   ) { return char(uplo);   }
inline char   diag2char( Diag   diag   ) { return char(diag);   }
inline char   side2char( Side   side   ) { return char(side);   }

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style string.
inline const char* layout2str( Layout layout )
{
    switch (layout) {
        case Layout::ColMajor: return "col";
        case Layout::RowMajor: return "row";
    }
    return "";
}

inline const char* op2str( Op op )
{
    switch (op) {
        case Op::NoTrans:   return "notrans";
        case Op::Trans:     return "trans";
        case Op::ConjTrans: return "conj";
    }
    return "";
}

inline const char* uplo2str( Uplo uplo )
{
    switch (uplo) {
        case Uplo::Lower:   return "lower";
        case Uplo::Upper:   return "upper";
        case Uplo::General: return "general";
    }
    return "";
}

inline const char* diag2str( Diag diag )
{
    switch (diag) {
        case Diag::NonUnit: return "nonunit";
        case Diag::Unit:    return "unit";
    }
    return "";
}

inline const char* side2str( Side side )
{
    switch (side) {
        case Side::Left:  return "left";
        case Side::Right: return "right";
    }
    return "";
}

// -----------------------------------------------------------------------------
// Convert LAPACK-style char to enum.
inline Layout char2layout( char layout )
{
    layout = (char) toupper( layout );
    assert( layout == 'C' || layout == 'R' );
    return Layout( layout );
}

inline Op char2op( char op )
{
    op = (char) toupper( op );
    assert( op == 'N' || op == 'T' || op == 'C' );
    return Op( op );
}

inline Uplo char2uplo( char uplo )
{
    uplo = (char) toupper( uplo );
    assert( uplo == 'L' || uplo == 'U' || uplo == 'G' );
    return Uplo( uplo );
}

inline Diag char2diag( char diag )
{
    diag = (char) toupper( diag );
    assert( diag == 'N' || diag == 'U' );
    return Diag( diag );
}

inline Side char2side( char side )
{
    side = (char) toupper( side );
    assert( side == 'L' || side == 'R' );
    return Side( side );
}

// -----------------------------------------------------------------------------
/// Exception class for BLIS errors.
class Error: public std::exception {
public:
    /// Constructs BLIS error
    Error():
        std::exception()
    {}

    /// Constructs BLIS error with message
    Error( std::string const& msg ):
        std::exception(),
        msg_( msg )
    {}

    /// Constructs BLIS error with message: "msg, in function func"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// Returns BLIS error message
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
};

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


namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by blis_error_if macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
        throw Error( condstr, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by blis_error_if_msg macro
// condstr is ignored, but differentiates this from other version.
inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
    __attribute__((format( printf, 4, 5 )));

inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );
        throw Error( buf, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; aborts if cond is true
// uses printf-style format for error message
// called by blis_error_if_msg macro
inline void abort_if( bool cond, const char* func,  const char* format, ... )
    __attribute__((format( printf, 3, 4 )));

inline void abort_if( bool cond, const char* func,  const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );

        fprintf( stderr, "Error: %s, in function %s\n", buf, func );
        abort();
    }
}

} // namespace internal

// -----------------------------------------------------------------------------
// internal macros to handle error checks
#if defined(BLIS_ERROR_NDEBUG) || (defined(BLIS_ERROR_ASSERT) && defined(NDEBUG))

    // blispp does no error checking;
    // lower level BLIS may still handle errors via xerbla
    #define blis_error_if( cond ) \
        ((void)0)

    #define blis_error_if_msg( cond, ... ) \
        ((void)0)

#elif defined(BLIS_ERROR_ASSERT)

    // blispp aborts on error
    #define blis_error_if( cond ) \
        blis::internal::abort_if( cond, __func__, "%s", #cond )

    #define blis_error_if_msg( cond, ... ) \
        blis::internal::abort_if( cond, __func__, __VA_ARGS__ )

#else

    // blispp throws errors (default)
    // internal macro to get string #cond; throws Error if cond is true
    // ex: blis_error_if( a < b );
    #define blis_error_if( cond ) \
        blis::internal::throw_if( cond, #cond, __func__ )

    // internal macro takes cond and printf-style format for error message.
    // throws Error if cond is true.
    // ex: blis_error_if_msg( a < b, "a %d < b %d", a, b );
    #define blis_error_if_msg( cond, ... ) \
        blis::internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

#endif

}  // namespace blis

#endif        //  #ifndef BLIS_UTIL_HH
