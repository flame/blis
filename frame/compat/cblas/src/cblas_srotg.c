#include "bli_config.h"
#include "bli_system.h"
#include "bli_type_defs.h"
#include "bli_cblas.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 * cblas_srotg.c
 *
 * The program is a C interface to srotg.
 *
 * Written by Keita Teranishi.  2/11/1998
 *
 */
#include "cblas.h"
#include "cblas_f77.h"
void cblas_srotg(  float *a, float *b, float *c, float *s)
{
   F77_srotg(a,b,c,s);    
}
#endif
