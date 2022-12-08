#ifndef GTEST_PTHREAD_H
#define GTEST_PTHREAD_H

#include "blis.h"

#ifdef __cplusplus
extern "C" {
#endif

int gtest_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                     (*start_routine)(void*),
       void*                     arg
     );


int gtest_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     );

#ifdef __cplusplus
}
#endif

#endif