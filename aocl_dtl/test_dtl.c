/*===================================================================
 * File Name :  test_dtl.c
 * 
 * Description : Unit test cases for dtl.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc
 * 
 *==================================================================*/

#if 0 // Disable this for normal build.

#include "aocltpdef.h"
#include "aocldtl.h"

int aocl_allocate(double**A, double** B, double** C, int N)
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO, " aocl_allocate()");

	*A = (double*)malloc(sizeof(double) * N);
	if (*A == NULL)
	{
		AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Error allocating memory to A");
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO, " aocl_allocate()");
		return 1;
	}

	*B = (double*)malloc(sizeof(double) * N);
	if (*B == NULL)
	{
		AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Error allocating memory to B");
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO, " aocl_allocate()");
		return 1;
	}

	*C = (double*)malloc(sizeof(double) * N);
	if (*C == NULL)
	{
		AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Error allocating memory to C");
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO, " aocl_allocate()");
		return 1;
	}

	for (int i = 0; i < N; i++)
	{
		(*A)[i] = (double)((i + 1) * 1.0);
		(*B)[i] = (double)((i - 1) * 1.0);
		(*C)[i] = (double)((i) * 1.0);
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO, " aocl_allocate()");
	return 0;
}

void sumV(double* A, double* B, double* C, int N)
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_INFO, "sumV()");
	if ((A == NULL) || (B == NULL) || (C == NULL))
	{
		AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Invalid Pointers");
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO, " sumV()");
		return;
	}
	for (int i = 0; i < N; i++)
	{
		C[i] += A[i] + B[i];
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_INFO, "sumV()");
}

int main(void)
{
	int status = 0;
	double* A = NULL;
	double* B = NULL;
	double* C = NULL;

	printf("Initializing\n");
	AOCL_DTL_INITIALIZE(AOCL_DTL_LEVEL_ALL);

	AOCL_DTL_TRACE_ENTRY(AOCL_TRACE_LEVEL_1, "Main function()");

	status = aocl_allocate(&A, &B, &C, 120);
	if (status != 0)
	{
		printf("Error allocating memory\n");

		AOCL_DTL_LOG(AOCL_DTL_LEVEL_MAJOR, "Error in function aocl_allocate()");
		AOCL_DTL_TRACE_EXIT(AOCL_TRACE_LEVEL_1, "Main function()");
		AOCL_DTL_UNINITIALIZE();
		exit(1);
	}

	sumV(A, B, C, 120);
	
	AOCL_DTL_TRACE_EXIT(AOCL_TRACE_LEVEL_1, "Main function()");
	AOCL_DTL_UNINITIALIZE();

	return 0;
}
#endif