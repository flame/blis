/*=======================================================================
 * File Name :  aocldtl.c
 *
 * Description : This file contains main logging functions.
 *               These functions are invoked though macros by
 *               end user.
 *
 * Copyright (C) 2020-2022, Advanced Micro Devices, Inc. All rights reserved.
 *
 *=======================================================================*/
#include "blis.h"
#include "aocltpdef.h"
#include "aocldtl.h"
#include "aoclfal.h"
#include "aocldtlcf.h"
#include "aoclflist.h"
#include "aoclos.h"

#ifdef AOCL_DTL_AUTO_TRACE_ENABLE
#if defined(__linux__)
#define __USE_GNU
#include <dlfcn.h>
#endif
#endif

/*
 * Client should provide this function, it should return
 * number of threads used by the API
 */
extern dim_t AOCL_get_requested_threads_count(void);

/* By default the trace level will be set to ALL User can configure this
      parameter at run time using command line argument */
uint32 gui32TraceLogLevel = AOCL_DTL_TRACE_LEVEL;

/*
 * Time elapsed in the function will be logged from main thread only,
 * we will save the main thread id. This will be compared with the id
 * of the logging thread.
 */
AOCL_TID gtidMainThreadID = -1;

/* The user can configure the file name in which he wants to dump the data */
#if AOCL_DTL_TRACE_ENABLE
/* The file name for storing traced log added manually in the code */
static char *pchDTL_TRACE_FILE = AOCL_DTL_TRACE_FILE;

/* Global file pointer for trace logging */
AOCL_FLIST_Node *gpTraceFileList = NULL;

#endif

#if (AOCL_DTL_LOG_ENABLE || AOCL_DTL_DUMP_ENABLE)
/* The file name for storing log data */
static char *pchDTL_LOG_FILE = AOCL_DTL_LOG_FILE;

/* Global file pointer for logging the results */
AOCL_FLIST_Node *gpLogFileList = NULL;


/* Global flag to check if logging is enabled or not */
Bool gbIsLoggingEnabled = TRUE;
#endif

#if AOCL_DTL_AUTO_TRACE_ENABLE

/* The file name for storing execution trace,
   These files are used by compiler assisted execution testing */
static char *pchDTL_AUTO_TRACE_FILE = AOCL_DTL_AUTO_TRACE_FILE;

/* Global file pointer for logging the results */
AOCL_FLIST_Node *gpAutoTraceFileList = NULL;
#endif

/*===================================================================
*  Function Name           :  DTL_Initialize
*  Description             :  Creates/Opens log file and initializes the
*                             global trace log level
*  Input Parameter(s)      :  ui32CurrentLogLevel - current log level
*                             which user can configure at run time
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#ifdef AOCL_DTL_INITIALIZE_ENABLE

void DTL_Initialize(
    uint32 ui32CurrentLogLevel)
{
    /*
     * This function can be invoked multiple times either via library
     * initialization function (e.g. bli_init()) or when user changes
     * logging state using API. However we want it to run only once
     * This flag ensure that it is executed only once.
     * 
     * DTL can be used with many libraries hence it needs its own
     * method to ensure this.
     */

    static Bool bIsDTLInitDone = FALSE;
    
    if (bIsDTLInitDone) 
    {
        return;
    }

    /* If user selects invalid trace log level then the dafault trace log level
      will be AOCL_DTL_LEVEL_ALL */
    if ((ui32CurrentLogLevel < 1) || (ui32CurrentLogLevel > AOCL_DTL_LEVEL_ALL))
    {
        gui32TraceLogLevel = AOCL_DTL_LEVEL_ALL;
    }
    else
    {
        /* Assign the user requested log level to the global trace log level */
        gui32TraceLogLevel = ui32CurrentLogLevel;
    }

#if AOCL_DTL_TRACE_ENABLE
    /* Create/Open the file to log the traced data */
    AOCL_FLIST_AddFile(pchDTL_TRACE_FILE, &gpTraceFileList, AOCL_gettid());

    if (NULL == gpTraceFileList)
    {
        /* Unable to open the specified file.*/
        AOCL_DEBUGPRINT("Unable to create the trace file %s\n", pchDTL_TRACE_FILE);
        return;
    }
#endif

#if (AOCL_DTL_LOG_ENABLE || AOCL_DTL_DUMP_ENABLE)
    
    /* Check if DTL logging is requested via envoronment variable */ 
    gbIsLoggingEnabled = bli_env_get_var( "AOCL_VERBOSE", TRUE );
#endif

#if AOCL_DTL_AUTO_TRACE_ENABLE
    /* Create/Open the file to log the log data */
    AOCL_FLIST_AddFile(pchDTL_AUTO_TRACE_FILE, &gpAutoTraceFileList, AOCL_gettid());

    if (NULL == gpAutoTraceFileList)
    {
        /* Unable to open the specified file.*/
        AOCL_DEBUGPRINT("Unable to create the log file %s\n", pchDTL_AUTO_TRACE_FILE);
        return;
    }
#endif

    /* Save Id for main thread */
    gtidMainThreadID = AOCL_gettid();

    // Ensure that this function is executed only once
    bIsDTLInitDone = TRUE;

} /* DTL_Initialize */
#endif

/*===================================================================
*  Function Name           :  DTL_Uninitialize
*  Description             :  Close all the log files
*  Input Parameter(s)      :  void
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#ifdef AOCL_DTL_INITIALIZE_ENABLE
void DTL_Uninitialize(void)
{
#if AOCL_DTL_TRACE_ENABLE
    /* Close the trace file */
    AOCL_FLIST_CloseAll(gpTraceFileList);
#endif

#if (AOCL_DTL_LOG_ENABLE || AOCL_DTL_DUMP_ENABLE)
    /* Close the log file */
    AOCL_FLIST_CloseAll(gpLogFileList);
#endif

#if AOCL_DTL_AUTO_TRACE_ENABLE
    /* Close the log file */
    AOCL_FLIST_CloseAll(gpAutoTraceFileList);
#endif
    return;
} /* DTL_Uninitialise */
#endif

/*===================================================================
*  Function Name           :  DTL_Trace
*  Description             :  This is common lowest level function
*                             to log the event to a file, This function
*                             will take case of choosing correct file
*                             according to the current thread and
*                             log the event as per format requested.

*  Input Parameter(s)      :  ui8LogLevel - Log Level
*                             ui8LogType - Identify log type (entry, exit etc)
*                             pi8FileName.- File name
*                             pi8FunctionName - Function Name
*                             ui32LineNumber - Line number
*                             pi8Message - Message to be printed
*
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#if (AOCL_DTL_TRACE_ENABLE || AOCL_DTL_LOG_ENABLE)
void DTL_Trace(
    uint8 ui8LogLevel,
    uint8 ui8LogType,
    const int8 *pi8FileName,
    const int8 *pi8FunctionName,
    uint32 ui32LineNumber,
    const int8 *pi8Message)
{
    uint8 i = 0;
    AOCL_FAL_FILE *pOutFile = NULL;
    
#if AOCL_DTL_LOG_ENABLE
    /* 
     * For performance reasons we check the logging state in end user
     * macros, this is just an additional check in case the function
     * is invoked from any other context.
     */
    if (gbIsLoggingEnabled == FALSE && ui8LogType == TRACE_TYPE_LOG)
    {
        return;
    }
#endif
    
    uint64 u64EventTime = AOCL_getTimestamp();
    dim_t u64RequestedThreadsCount = AOCL_get_requested_threads_count();

    bli_init_auto();

    if (ui8LogType == TRACE_TYPE_LOG || ui8LogType == TRACE_TYPE_RAW)
    {
#if AOCL_DTL_LOG_ENABLE
        pOutFile = AOCL_FLIST_GetFile(gpLogFileList, AOCL_gettid());

        /* If trace file pointer is equal to NULL then return with out dumping data
         to the file */
        if (NULL == pOutFile)
        {
            /* It might be the first call from the current thread, try to create
         new trace for this thread. */
            pOutFile = AOCL_FLIST_AddFile(pchDTL_LOG_FILE, &gpLogFileList, AOCL_gettid());

            if (NULL == pOutFile)
            {
                AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
                return;
            }
        }
#endif /* Logging enabled */
    }
    else
    {
#if AOCL_DTL_TRACE_ENABLE
     pOutFile = AOCL_FLIST_GetFile(gpTraceFileList, AOCL_gettid());

        /* If trace file pointer is equal to NULL then return with out dumping data
         to file */
        if (NULL == pOutFile)
        {
            /* It might be the first call from the current thread, try to create
         new trace for this thread. */
            pOutFile = AOCL_FLIST_AddFile(pchDTL_TRACE_FILE, &gpTraceFileList, AOCL_gettid());

            if (NULL == pOutFile)
            {
                AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
                return;
            }
        }
#endif /* Trace Enabled */
    }

    /* Log the message only if the log level is less than or equal to global log
      level set while initialization */
    if (ui8LogLevel <= gui32TraceLogLevel)
    {
        /* Indent as per level if is function call trace */
        if ((ui8LogLevel >= AOCL_DTL_LEVEL_TRACE_1) &&
            (ui8LogLevel <= AOCL_DTL_LEVEL_TRACE_8))
        {
            /* this loop is for formating the output log file */
            for (i = 0; i < (ui8LogLevel - AOCL_DTL_LEVEL_TRACE_1); i++)
            {
                /* print tabs in the output file */
                fprintf(pOutFile, "\t");
            }
        }

        switch (ui8LogType)
        {
        case TRACE_TYPE_FENTRY:
            fprintf(pOutFile, "nt=%ld,ts=%ld: In %s()...\n",
                    u64RequestedThreadsCount,
                    u64EventTime,
                    pi8FunctionName);
            break;

        case TRACE_TYPE_FEXIT:
            if (pi8Message == NULL)
            { /* Function returned successfully */
                fprintf(pOutFile, "ts=%ld: Out of %s()\n",
                        u64EventTime,
                        pi8FunctionName);
            }
            else
            { /* Function failed to complete, use message to get error */
                fprintf(pOutFile, "ts=%ld: Out of %s() with error %s\n",
                        u64EventTime,
                        pi8FunctionName,
                        pi8Message);
            }
            break;

        case TRACE_TYPE_LOG:
                fprintf(pOutFile, "%s %s",
                        pi8FileName,
                        pi8Message
                        );

            break;

        case TRACE_TYPE_RAW:
            fprintf(pOutFile, "%s\n",
                    pi8Message);
            break;
        }
	fflush(pOutFile);
    }
} /* DTL_Data_Trace_Entry */
#endif

/*===================================================================
*  Function Name           :  DTL_DumpData
*  Description             :  This function is mainly used for dumping
*                             the data into the file
*  Input Parameter(s)      :  pui8Buffer - the buffer to be dumped
*                             ui32BufferSize.- the no. of bytes to be dumped
*                             ui8DataType - the data type char/int32/int32
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
#if AOCL_DTL_DUMP_ENABLE
void DTL_DumpData(
    uint8 ui8LogLevel,
    void *pvBuffer,
    uint32 ui32BufferSize,
    uint8 ui8DataType,
    int8 *pi8Message,
    int8 i8OutputType)
{
    uint32 j;

    /* Pointer to store the buffer */
    uint32 *pui32Array, ui32LocalData;
    uint16 *pui16Array;
    uint8 *pui8CharArray;
    int8 *pi8CharString;
    AOCL_FAL_FILE *pDumpFile;

#if (AOCL_DTL_DUMP_ENABLE)
    /* If dump (log) file pointer is equal to NULL return with out dumping data to file */
    pDumpFile = AOCL_FLIST_GetFile(gpLogFileList, AOCL_gettid());
    /* If trace file pointer is equal to NULL then return with out dumping data
       to the file */
    if (NULL == pDumpFile)
    {
        /* It might be the first call from the current thread, try to create
           new trace for this thread. */
        pDumpFile = AOCL_FLIST_AddFile(pchDTL_LOG_FILE, &gpLogFileList, AOCL_gettid());

        if (NULL == pDumpFile)
        {
            AOCL_DEBUGPRINT("File does not exists to dump the raw data \n");
            return;
        }
    }
#endif /* Dump enabled */
    /* Log the message only if the log level is less than or equal to global log
      level set while initialization */
    if (ui8LogLevel > gui32TraceLogLevel)
    {
        return;
    }

    /* The string message */
    if (pi8Message != NULL)
    {
        fprintf(pDumpFile, "%s :", pi8Message);
    }

    /* Assuming that if the Data type for character = 1
   * the Data type for uint32 = 2
   * the data type for uint32 = 4
   * the data type for string = 3
   */
    if (ui8DataType == AOCL_STRING_DATA_TYPE)
    {
        /* Typecast the void buffer to character buffer */
        pi8CharString = (int8 *)pvBuffer;
        fprintf(pDumpFile, "%s", pi8CharString);
        fprintf(pDumpFile, "\n");
    }

    if (ui8DataType == AOCL_CHAR_DATA_TYPE)
    {
        /* Typecast the void buffer to character buffer */
        pui8CharArray = (uint8 *)pvBuffer;

        for (j = 0; j < ui32BufferSize; j++)
        {
            if (i8OutputType == AOCL_LOG_HEX_VALUE)
            {
                fprintf(pDumpFile, "\n\t%5d:0x%x", j, pui8CharArray[j]);
            }
            else
            {
                fprintf(pDumpFile, "\n\t%5d:%u", j, pui8CharArray[j]);
            }
        }
        fprintf(pDumpFile, "\n");
    }

    if (ui8DataType == AOCL_UINT16_DATA_TYPE)
    {
        /* Typecast the void buffer to uint32 bit buffer */
        pui16Array = (uint16 *)pvBuffer;

        /* dump the data in the file line by line */
        for (j = 0; j < ui32BufferSize; j++)
        {
            if (i8OutputType == AOCL_LOG_HEX_VALUE)
            {
                fprintf(pDumpFile, "\n\t%5d:0x%x", j, pui16Array[j]);
            }
            else
            {
                fprintf(pDumpFile, "\n\t%5d:%u", j, pui16Array[j]);
            }
        }
        fprintf(pDumpFile, "\n");

    } /* End of if */

    if (ui8DataType == AOCL_UINT32_DATA_TYPE)
    {
        /* Typecast the void buffer to uint32 buffer */
        pui32Array = (uint32 *)pvBuffer;

        /* dump the data in the file line by line */
        for (j = 0; j < ui32BufferSize; j++)
        {
            ui32LocalData = pui32Array[j];

            if (i8OutputType == AOCL_LOG_HEX_VALUE)
            {
                fprintf(pDumpFile, "\n\t%5d:0x%x", j, ui32LocalData);
            }
            else
            {
                fprintf(pDumpFile, "\n\t%5d:%u", j, ui32LocalData);
            }
        }
        fprintf(pDumpFile, "\n");
    } /* End of if */
    fflush(pDumpFile);

} /* DTL_DumpData */
#endif

#if (AOCL_DTL_LOG_ENABLE)
void AOCL_DTL_start_perf_timer(void)
{
    AOCL_TID current_thread = AOCL_gettid();

    // Automatic duration calulation is currently
    // supported from main thread only, in other words
    // at BLAS interface.
    if (current_thread != gtidMainThreadID) {
        return;
    }

    AOCL_FLIST_Node *pFileNode = AOCL_FLIST_GetNode(gpLogFileList, current_thread);

    if (NULL == pFileNode) {
        /* It might be the first call from the current thread, try to create
        new trace for this thread. */
        AOCL_FAL_FILE *pOutFile = AOCL_FLIST_AddFile(pchDTL_LOG_FILE, &gpLogFileList, current_thread);

        if (NULL == pOutFile)
        {
            AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
            return;
        } else {
            pFileNode = AOCL_FLIST_GetNode(gpLogFileList, current_thread);
        }
    }

    pFileNode->u64SavedTimeStamp = AOCL_getTimestamp();
    fflush(stdout);
}


uint64 AOCL_DTL_get_time_spent(void)
{
    AOCL_TID current_thread = AOCL_gettid();

    // Automatic duration calulation is currently
    // supported from main thread only, in other words
    // at BLAS interface.
    if (current_thread != gtidMainThreadID) {
        return 0;
    }

    uint64 u64CurrentTimeStamp = AOCL_getTimestamp();
    AOCL_FLIST_Node *pFileNode = AOCL_FLIST_GetNode(gpLogFileList, AOCL_gettid());

    if (NULL == pFileNode) {
        /* It might be the first call from the current thread, try to create
        new trace for this thread. */
        AOCL_FAL_FILE *pOutFile = AOCL_FLIST_AddFile(pchDTL_LOG_FILE, &gpLogFileList, AOCL_gettid());

        if (NULL == pOutFile)
        {
            AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
            return 0;
        } else {
            pFileNode = AOCL_FLIST_GetNode(gpLogFileList, AOCL_gettid());
        }
    }

    return (u64CurrentTimeStamp - pFileNode->u64SavedTimeStamp);
}

#endif

/* This is enabled by passing ETRACE_ENABLE=1 to make */
#ifdef AOCL_DTL_AUTO_TRACE_ENABLE

/*
    Disable intrumentation for these functions as they will also be
    called from compiler generated instumation code to trace
   function execution.

    It needs to be part of declration in the C file so can't be
    moved to header file.

    WARNING: These functions are automatically invoked. however any function
    called from this should have instumtation disable to avoid recursive
    calls which results in hang/crash.
   */
void __cyg_profile_func_enter(void *this_fn, void *call_site) __attribute__((no_instrument_function));
void __cyg_profile_func_exit(void *this_fn, void *call_site) __attribute__((no_instrument_function));

/*===================================================================
*  Function Name           :  __cyg_profile_func_enter
*  Description             :  This function is automatically invoked
*                             by compiler instrumntation when the flow
*                             enters a function.
*  Input Parameter(s)      :  pvThisFunc - Address of function entered.
*                             call_site.- Address of the caller
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
void __cyg_profile_func_enter(void *pvThisFunc, void *pvCaller)
{
    Dl_info info;
    dladdr(pvThisFunc, &info);

    AOCL_FAL_FILE *pOutFile = NULL;

    pOutFile = AOCL_FLIST_GetFile(gpAutoTraceFileList, AOCL_gettid());

    /* If trace file pointer is equal to NULL then return with out dumping data
        to the file */
    if (NULL == pOutFile)
    {
        /* It might be the first call from the current thread, try to create
        new trace for this thread. */
        pOutFile = AOCL_FLIST_AddFile(pchDTL_AUTO_TRACE_FILE, &gpAutoTraceFileList, AOCL_gettid());

        if (NULL == pOutFile)
        {
            AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
            return;
        }
    }

    fprintf(pOutFile, "\n%lu:+:%p",
            AOCL_getTimestamp(),
            (void *)(pvThisFunc - info.dli_fbase));
    fflush(pOutFile);
}

/*===================================================================
*  Function Name           :  __cyg_profile_func_exit
*  Description             :  This function is automatically invoked
*                             by compiler before returing from a
*                             function.
*  Input Parameter(s)      :  pvThisFunc - Address of function to be existed.
*                             call_site.- Address of the caller
*  Output Parameter(s)     :  None
*  Return parameter(s)     :  None
*==================================================================*/
void __cyg_profile_func_exit(void *pvThisFunc, void *pvCaller)
{
    Dl_info info;
    dladdr(pvThisFunc, &info);
    AOCL_FAL_FILE *pOutFile = NULL;

    pOutFile = AOCL_FLIST_GetFile(gpAutoTraceFileList, AOCL_gettid());

    /* If trace file pointer is equal to NULL then return with out dumping data
        to the file */
    if (NULL == pOutFile)
    {
        /* It might be the first call from the current thread, try to create
        new trace for this thread. */
        pOutFile = AOCL_FLIST_AddFile(pchDTL_AUTO_TRACE_FILE, &gpAutoTraceFileList, AOCL_gettid());

        if (NULL == pOutFile)
        {
            AOCL_DEBUGPRINT("File does not exists to dump the trace data \n");
            return;
        }
    }

    fprintf(pOutFile, "\n%lu:-:%p",
            AOCL_getTimestamp(),
            (void *)(pvThisFunc - info.dli_fbase));
    fflush(pOutFile);
}

#endif /* AOCL_AUTO_TRACE_ENABLE */

/* ------------------ End of aocldtl.c ---------------------- */
