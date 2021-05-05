#include "blis.h"

#include <float.h>
#include <fenv.h>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

bla_double bli_dlamch(bla_character *cmach, ftnlen cmach_len)
{
/*          = 'E' or 'e',   DLAMCH := eps */
/*          = 'S' or 's ,   DLAMCH := sfmin */
/*          = 'B' or 'b',   DLAMCH := base */
/*          = 'P' or 'p',   DLAMCH := eps*base */
/*          = 'N' or 'n',   DLAMCH := t */
/*          = 'R' or 'r',   DLAMCH := rnd */
/*          = 'M' or 'm',   DLAMCH := emin */
/*          = 'U' or 'u',   DLAMCH := rmin */
/*          = 'L' or 'l',   DLAMCH := emax */
/*          = 'O' or 'o',   DLAMCH := rmax */

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
		case 'E': return DBL_EPSILON;
		case 'S': return DBL_MIN;
		case 'B': return FLT_RADIX;
		case 'P': return FLT_RADIX*DBL_EPSILON;
		case 'N': return DBL_MANT_DIG;
		case 'R': return FLT_ROUNDS == FE_TONEAREST ? 1.0 : 0.0;
		case 'M': return DBL_MIN_EXP;
		case 'U': return DBL_MIN;
		case 'L': return DBL_MAX_EXP;
		case 'O': return DBL_MAX;
	}
	
	return 0.0;
}

#ifdef __cplusplus
}
#endif
