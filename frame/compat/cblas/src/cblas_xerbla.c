#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "cblas.h"
#include "cblas_f77.h"

// The global rntm_t structure. (The definition resides in bli_rntm.c.)
extern rntm_t global_rntm;

// Make thread settings local to each thread calling BLIS routines.
// (The definition resides in bli_rntm.c.)
extern BLIS_THREAD_LOCAL rntm_t tl_rntm;

void cblas_xerbla(f77_int info, const char *rout, const char *form, ...)
{
   extern int RowMajorStrg;
   char empty[1] = "";

   if (RowMajorStrg)
   {
      if (strstr(rout,"gemm") != 0)
      {
         if      (info == 5 ) info =  4;
         else if (info == 4 ) info =  5;
         else if (info == 11) info =  9;
         else if (info == 9 ) info = 11;
      }
      else if (strstr(rout,"symm") != 0 || strstr(rout,"hemm") != 0)
      {
         if      (info == 5 ) info =  4;
         else if (info == 4 ) info =  5;
      }
      else if (strstr(rout,"trmm") != 0 || strstr(rout,"trsm") != 0)
      {
         if      (info == 7 ) info =  6;
         else if (info == 6 ) info =  7;
      }
      else if (strstr(rout,"gemv") != 0)
      {
         if      (info == 4)  info = 3;
         else if (info == 3)  info = 4;
      }
      else if (strstr(rout,"gbmv") != 0)
      {
         if      (info == 4)  info = 3;
         else if (info == 3)  info = 4;
         else if (info == 6)  info = 5;
         else if (info == 5)  info = 6;
      }
      else if (strstr(rout,"ger") != 0)
      {
         if      (info == 3) info = 2;
         else if (info == 2) info = 3;
         else if (info == 8) info = 6;
         else if (info == 6) info = 8;
      }
      else if ( (strstr(rout,"her2") != 0 || strstr(rout,"hpr2") != 0)
                 && strstr(rout,"her2k") == 0 )
      {
         if      (info == 8) info = 6;
         else if (info == 6) info = 8;
      }
   }

   if (info)
   {
      // Make sure rntm variables are initialized.
      bli_init_once();

      // Store info value in thread-local rntm data structure.
      gint_t info_value = (gint_t) info;
      bli_rntm_set_info_value_only( info_value, &tl_rntm );

      bool print_on_error = bli_rntm_print_on_error( &global_rntm );
      if (print_on_error)
      {
         va_list argptr;
         va_start(argptr, form);

         fprintf(stderr, "Parameter %d to routine %s was incorrect\n", (int)info, rout);
         vfprintf(stderr, form, argptr);
         va_end(argptr);
      }

      bool stop_on_error = bli_rntm_stop_on_error( &global_rntm );
      if (stop_on_error)
      {
         bli_abort();
      }

      if (info && !info) 
         F77_xerbla(empty, &info, 0); /* Force link of our F77 error handler */
   }
}
#endif

