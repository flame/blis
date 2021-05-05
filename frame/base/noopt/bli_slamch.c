#include "blis.h"

#include <float.h>
#include <fenv.h>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

bla_real bli_slamch(bla_character *cmach, ftnlen cmach_len)
{
/*          = 'E' or 'e',   SLAMCH := eps */
/*          = 'S' or 's ,   SLAMCH := sfmin */
/*          = 'B' or 'b',   SLAMCH := base */
/*          = 'P' or 'p',   SLAMCH := eps*base */
/*          = 'N' or 'n',   SLAMCH := t */
/*          = 'R' or 'r',   SLAMCH := rnd */
/*          = 'M' or 'm',   SLAMCH := emin */
/*          = 'U' or 'u',   SLAMCH := rmin */
/*          = 'L' or 'l',   SLAMCH := emax */
/*          = 'O' or 'o',   SLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */
	
	switch ( toupper( *cmach ) )
	{
		case 'E': return FLT_EPSILON;
		case 'S': return FLT_MIN;
		case 'B': return FLT_RADIX;
		case 'P': return FLT_RADIX*FLT_EPSILON;
		case 'N': return FLT_MANT_DIG;
		case 'R': return FLT_ROUNDS == FE_TONEAREST ? 1.0f : 0.0f;
		case 'M': return FLT_MIN_EXP;
		case 'U': return FLT_MIN;
		case 'L': return FLT_MAX_EXP;
		case 'O': return FLT_MAX;
	}
	
	return 0.0f;
}

#ifdef __cplusplus
}
#endif
