

/*

################################################################################
################################################################################

    MACROS FOR 16x4

################################################################################
################################################################################

*/


#define ZERO_OUT_VREG \
 "xxlxor           %%vs0, %%vs0, %%vs0          \n\t" \
 "xxlxor           %%vs1, %%vs1, %%vs1          \n\t" \
 "xxlxor           %%vs2, %%vs2, %%vs2          \n\t" \
 "xxlxor           %%vs3, %%vs3, %%vs3          \n\t" \
 "xxlxor           %%vs4, %%vs4, %%vs4          \n\t" \
 "xxlxor           %%vs5, %%vs5, %%vs5          \n\t" \
 "xxlxor           %%vs6, %%vs6, %%vs6          \n\t" \
 "xxlxor           %%vs7, %%vs7, %%vs7          \n\t" \
 "xxlxor           %%vs8, %%vs8, %%vs8          \n\t" \
 "xxlxor           %%vs9, %%vs9, %%vs9          \n\t" \
 "xxlxor           %%vs10, %%vs10, %%vs10       \n\t" \
 "xxlxor           %%vs11, %%vs11, %%vs11       \n\t" \
 "xxlxor           %%vs12, %%vs12, %%vs12       \n\t" \
 "xxlxor           %%vs13, %%vs13, %%vs13       \n\t" \
 "xxlxor           %%vs14, %%vs14, %%vs14       \n\t" \
 "xxlxor           %%vs15, %%vs15, %%vs15       \n\t" \
 "xxlxor           %%vs16, %%vs16, %%vs16       \n\t" \
 "xxlxor           %%vs17, %%vs17, %%vs17       \n\t" \
 "xxlxor           %%vs18, %%vs18, %%vs18       \n\t" \
 "xxlxor           %%vs19, %%vs19, %%vs19       \n\t" \
 "xxlxor           %%vs20, %%vs20, %%vs20       \n\t" \
 "xxlxor           %%vs21, %%vs21, %%vs21       \n\t" \
 "xxlxor           %%vs22, %%vs22, %%vs22       \n\t" \
 "xxlxor           %%vs23, %%vs23, %%vs23       \n\t" \
 "xxlxor           %%vs24, %%vs24, %%vs24       \n\t" \
 "xxlxor           %%vs25, %%vs25, %%vs25       \n\t" \
 "xxlxor           %%vs26, %%vs26, %%vs26       \n\t" \
 "xxlxor           %%vs27, %%vs27, %%vs27       \n\t" \
 "xxlxor           %%vs28, %%vs28, %%vs28       \n\t" \
 "xxlxor           %%vs29, %%vs29, %%vs29       \n\t" \
 "xxlxor           %%vs30, %%vs30, %%vs30       \n\t" \
 "xxlxor           %%vs31, %%vs31, %%vs31       \n\t" \
 "xxlxor           %%vs32, %%vs32, %%vs32       \n\t" \
 "xxlxor           %%vs33, %%vs33, %%vs33       \n\t" \
 "xxlxor           %%vs34, %%vs34, %%vs34       \n\t" \
 "xxlxor           %%vs35, %%vs35, %%vs35       \n\t" 











#define DSCALE_ALPHA \
"xvmuldp         %%vs0, %%vs0, %%vs60           \n\t" \
"xvmuldp         %%vs1, %%vs1, %%vs60           \n\t" \
"xvmuldp         %%vs2, %%vs2, %%vs60           \n\t" \
"xvmuldp         %%vs3, %%vs3, %%vs60           \n\t" \
"xvmuldp         %%vs4, %%vs4, %%vs60           \n\t" \
"xvmuldp         %%vs5, %%vs5, %%vs60           \n\t" \
"xvmuldp         %%vs6, %%vs6, %%vs60           \n\t" \
"xvmuldp         %%vs7, %%vs7, %%vs60           \n\t" \
"xvmuldp         %%vs8, %%vs8, %%vs60           \n\t" \
"xvmuldp         %%vs9, %%vs9, %%vs60           \n\t" \
"xvmuldp         %%vs10, %%vs10, %%vs60         \n\t" \
"xvmuldp         %%vs11, %%vs11, %%vs60         \n\t" \
"xvmuldp         %%vs12, %%vs12, %%vs60         \n\t" \
"xvmuldp         %%vs13, %%vs13, %%vs60         \n\t" \
"xvmuldp         %%vs14, %%vs14, %%vs60         \n\t" \
"xvmuldp         %%vs15, %%vs15, %%vs60         \n\t" \
"xvmuldp         %%vs16, %%vs16, %%vs60         \n\t" \
"xvmuldp         %%vs17, %%vs17, %%vs60         \n\t" \
"xvmuldp         %%vs18, %%vs18, %%vs60         \n\t" \
"xvmuldp         %%vs19, %%vs19, %%vs60         \n\t" \
"xvmuldp         %%vs20, %%vs20, %%vs60         \n\t" \
"xvmuldp         %%vs21, %%vs21, %%vs60         \n\t" \
"xvmuldp         %%vs22, %%vs22, %%vs60         \n\t" \
"xvmuldp         %%vs23, %%vs23, %%vs60         \n\t" \
"xvmuldp         %%vs24, %%vs24, %%vs60         \n\t" \
"xvmuldp         %%vs25, %%vs25, %%vs60         \n\t" \
"xvmuldp         %%vs26, %%vs26, %%vs60         \n\t" \
"xvmuldp         %%vs27, %%vs27, %%vs60         \n\t" \
"xvmuldp         %%vs28, %%vs28, %%vs60         \n\t" \
"xvmuldp         %%vs29, %%vs29, %%vs60         \n\t" \
"xvmuldp         %%vs30, %%vs30, %%vs60         \n\t" \
"xvmuldp         %%vs31, %%vs31, %%vs60         \n\t" \
"xvmuldp         %%vs32, %%vs32, %%vs60         \n\t" \
"xvmuldp         %%vs33, %%vs33, %%vs60         \n\t" \
"xvmuldp         %%vs34, %%vs34, %%vs60         \n\t" \
"xvmuldp         %%vs35, %%vs35, %%vs60         \n\t"








#define DPRELOAD_A_B \
"lxv              %%vs36, 0(%%r7)               \n\t" \
"lxv              %%vs37, 16(%%r7)              \n\t" \
"lxv              %%vs38, 32(%%r7)              \n\t" \
"lxv              %%vs39, 48(%%r7)              \n\t" \
"                                               \n\t" \
"lxv              %%vs54, 0(%%r8)               \n\t" \
"lxv              %%vs56, 16(%%r8)              \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2     \n\t" \
"xxpermdi         %%vs57, %%vs56, %%vs56, 2     \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 64(%%r7)              \n\t" \
"lxv              %%vs41, 80(%%r7)              \n\t" \
"lxv              %%vs42, 96(%%r7)              \n\t" \
"lxv              %%vs43, 112(%%r7)             \n\t" \
"lxv              %%vs44, 128(%%r7)             \n\t" 











#define DLOAD_UPDATE_16 \
"                                               \n\t" \
"lxv              %%vs45, 0(%%r7)               \n\t" \
"lxv              %%vs46, 16(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs36, %%vs54         \n\t" \
"xvmaddadp        %%vs1, %%vs37, %%vs54         \n\t" \
"xvmaddadp        %%vs2, %%vs38, %%vs54         \n\t" \
"xvmaddadp        %%vs3, %%vs39, %%vs54         \n\t" \
"xvmaddadp        %%vs4, %%vs40, %%vs54         \n\t" \
"xvmaddadp        %%vs5, %%vs41, %%vs54         \n\t" \
"                                               \n\t" \
"lxv              %%vs47, 32(%%r7)              \n\t" \
"lxv              %%vs48, 48(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs6, %%vs42, %%vs54         \n\t" \
"xvmaddadp        %%vs7, %%vs43, %%vs54         \n\t" \
"xvmaddadp        %%vs8, %%vs44, %%vs54         \n\t" \
"xvmaddadp        %%vs9, %%vs36, %%vs55         \n\t" \
"xvmaddadp        %%vs10, %%vs37, %%vs55        \n\t" \
"xvmaddadp        %%vs11, %%vs38, %%vs55        \n\t" \
"                                               \n\t" \
"lxv              %%vs58, 0(%%r8)               \n\t" \
"lxv              %%vs60, 16(%%r8)              \n\t" \
"xxpermdi         %%vs59, %%vs58, %%vs58, 2     \n\t" \
"xxpermdi         %%vs61, %%vs60, %%vs60, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs39, %%vs55        \n\t" \
"xvmaddadp        %%vs13, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs14, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs15, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs16, %%vs43, %%vs55        \n\t" \
"xvmaddadp        %%vs17, %%vs44, %%vs55        \n\t" \
"                                               \n\t" \
"lxv              %%vs49, 64(%%r7)              \n\t" \
"lxv              %%vs50, 80(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs18, %%vs36, %%vs56        \n\t" \
"xvmaddadp        %%vs19, %%vs37, %%vs56        \n\t" \
"xvmaddadp        %%vs20, %%vs38, %%vs56        \n\t" \
"xvmaddadp        %%vs21, %%vs39, %%vs56        \n\t" \
"xvmaddadp        %%vs22, %%vs40, %%vs56        \n\t" \
"xvmaddadp        %%vs23, %%vs41, %%vs56        \n\t" \
"                                               \n\t" \
"lxv              %%vs51, 96(%%r7)              \n\t" \
"lxv              %%vs52, 112(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs42, %%vs56        \n\t" \
"xvmaddadp        %%vs25, %%vs43, %%vs56        \n\t" \
"xvmaddadp        %%vs26, %%vs44, %%vs56        \n\t" \
"xvmaddadp        %%vs27, %%vs36, %%vs57        \n\t" \
"xvmaddadp        %%vs28, %%vs37, %%vs57        \n\t" \
"xvmaddadp        %%vs29, %%vs38, %%vs57        \n\t" \
"                                               \n\t" \
"lxv              %%vs53, 128(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs30, %%vs39, %%vs57        \n\t" \
"xvmaddadp        %%vs31, %%vs40, %%vs57        \n\t" \
"xvmaddadp        %%vs32, %%vs41, %%vs57        \n\t" \
"xvmaddadp        %%vs33, %%vs42, %%vs57        \n\t" \
"xvmaddadp        %%vs34, %%vs43, %%vs57        \n\t" \
"xvmaddadp        %%vs35, %%vs44, %%vs57        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs45, %%vs58         \n\t" \
"xvmaddadp        %%vs1, %%vs46, %%vs58         \n\t" \
"xvmaddadp        %%vs2, %%vs47, %%vs58         \n\t" \
"xvmaddadp        %%vs3, %%vs48, %%vs58         \n\t" \
"xvmaddadp        %%vs4, %%vs49, %%vs58         \n\t" \
"xvmaddadp        %%vs5, %%vs50, %%vs58         \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 144(%%r7)             \n\t" \
"lxv              %%vs37, 160(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs6, %%vs51, %%vs58         \n\t" \
"xvmaddadp        %%vs7, %%vs52, %%vs58         \n\t" \
"xvmaddadp        %%vs8, %%vs53, %%vs58         \n\t" \
"xvmaddadp        %%vs9, %%vs45, %%vs59         \n\t" \
"xvmaddadp        %%vs10, %%vs46, %%vs59        \n\t" \
"xvmaddadp        %%vs11, %%vs47, %%vs59        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 176(%%r7)             \n\t" \
"lxv              %%vs39, 192(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs48, %%vs59        \n\t" \
"xvmaddadp        %%vs13, %%vs49, %%vs59        \n\t" \
"xvmaddadp        %%vs14, %%vs50, %%vs59        \n\t" \
"xvmaddadp        %%vs15, %%vs51, %%vs59        \n\t" \
"xvmaddadp        %%vs16, %%vs52, %%vs59        \n\t" \
"xvmaddadp        %%vs17, %%vs53, %%vs59        \n\t" \
"                                               \n\t" \
"lxv              %%vs54, 32(%%r8)              \n\t" \
"lxv              %%vs56, 48(%%r8)              \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2     \n\t" \
"xxpermdi         %%vs57, %%vs56, %%vs56, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs18, %%vs45, %%vs60        \n\t" \
"xvmaddadp        %%vs19, %%vs46, %%vs60        \n\t" \
"xvmaddadp        %%vs20, %%vs47, %%vs60        \n\t" \
"xvmaddadp        %%vs21, %%vs48, %%vs60        \n\t" \
"xvmaddadp        %%vs22, %%vs49, %%vs60        \n\t" \
"xvmaddadp        %%vs23, %%vs50, %%vs60        \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 208(%%r7)             \n\t" \
"lxv              %%vs41, 224(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs51, %%vs60        \n\t" \
"xvmaddadp        %%vs25, %%vs52, %%vs60        \n\t" \
"xvmaddadp        %%vs26, %%vs53, %%vs60        \n\t" \
"xvmaddadp        %%vs27, %%vs45, %%vs61        \n\t" \
"xvmaddadp        %%vs28, %%vs46, %%vs61        \n\t" \
"xvmaddadp        %%vs29, %%vs47, %%vs61        \n\t" \
"                                               \n\t" \
"lxv              %%vs42, 240(%%r7)             \n\t" \
"lxv              %%vs43, 256(%%r7)             \n\t" \
"lxv              %%vs44, 272(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs30, %%vs48, %%vs61        \n\t" \
"xvmaddadp        %%vs31, %%vs49, %%vs61        \n\t" \
"xvmaddadp        %%vs32, %%vs50, %%vs61        \n\t" \
"xvmaddadp        %%vs33, %%vs51, %%vs61        \n\t" \
"xvmaddadp        %%vs34, %%vs52, %%vs61        \n\t" \
"xvmaddadp        %%vs35, %%vs53, %%vs61        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"addi             %%r8, %%r8, 64                \n\t" \
"addi             %%r7, %%r7, 288               \n\t" \



#define DLOAD_UPDATE_1 \
"                                                \n\t" \
"xvmaddadp        %%vs0, %%vs36, %%vs54          \n\t" \
"xvmaddadp        %%vs1, %%vs37, %%vs54          \n\t" \
"xvmaddadp        %%vs2, %%vs38, %%vs54          \n\t" \
"xvmaddadp        %%vs3, %%vs39, %%vs54          \n\t" \
"xvmaddadp        %%vs4, %%vs40, %%vs54          \n\t" \
"xvmaddadp        %%vs5, %%vs41, %%vs54          \n\t" \
"xvmaddadp        %%vs6, %%vs42, %%vs54          \n\t" \
"xvmaddadp        %%vs7, %%vs43, %%vs54          \n\t" \
"xvmaddadp        %%vs8, %%vs44, %%vs54          \n\t" \
"                                                \n\t" \
"xvmaddadp        %%vs9, %%vs36, %%vs55          \n\t" \
"xvmaddadp        %%vs10, %%vs37, %%vs55         \n\t" \
"xvmaddadp        %%vs11, %%vs38, %%vs55         \n\t" \
"xvmaddadp        %%vs12, %%vs39, %%vs55         \n\t" \
"xvmaddadp        %%vs13, %%vs40, %%vs55         \n\t" \
"xvmaddadp        %%vs14, %%vs41, %%vs55         \n\t" \
"xvmaddadp        %%vs15, %%vs42, %%vs55         \n\t" \
"xvmaddadp        %%vs16, %%vs43, %%vs55         \n\t" \
"xvmaddadp        %%vs17, %%vs44, %%vs55         \n\t" \
"                                                \n\t" \
"xvmaddadp        %%vs18, %%vs36, %%vs56         \n\t" \
"xvmaddadp        %%vs19, %%vs37, %%vs56         \n\t" \
"xvmaddadp        %%vs20, %%vs38, %%vs56         \n\t" \
"xvmaddadp        %%vs21, %%vs39, %%vs56         \n\t" \
"xvmaddadp        %%vs22, %%vs40, %%vs56         \n\t" \
"xvmaddadp        %%vs23, %%vs41, %%vs56         \n\t" \
"xvmaddadp        %%vs24, %%vs42, %%vs56         \n\t" \
"xvmaddadp        %%vs25, %%vs43, %%vs56         \n\t" \
"xvmaddadp        %%vs26, %%vs44, %%vs56         \n\t" \
"                                                \n\t" \
"xvmaddadp        %%vs27, %%vs36, %%vs57         \n\t" \
"xvmaddadp        %%vs28, %%vs37, %%vs57         \n\t" \
"xvmaddadp        %%vs29, %%vs38, %%vs57         \n\t" \
"xvmaddadp        %%vs30, %%vs39, %%vs57         \n\t" \
"xvmaddadp        %%vs31, %%vs40, %%vs57         \n\t" \
"xvmaddadp        %%vs32, %%vs41, %%vs57         \n\t" \
"xvmaddadp        %%vs33, %%vs42, %%vs57         \n\t" \
"xvmaddadp        %%vs34, %%vs43, %%vs57         \n\t" \
"xvmaddadp        %%vs35, %%vs44, %%vs57         \n\t" \
"                                                \n\t" \
"lxv              %%vs54, 0(%%r8)                \n\t" \
"lxv              %%vs56, 16(%%r8)               \n\t" \
"                                                \n\t" \
"lxv              %%vs36, 0(%%r7)                \n\t" \
"lxv              %%vs37, 16(%%r7)               \n\t" \
"lxv              %%vs38, 32(%%r7)               \n\t" \
"lxv              %%vs39, 48(%%r7)               \n\t" \
"lxv              %%vs40, 64(%%r7)               \n\t" \
"lxv              %%vs41, 80(%%r7)               \n\t" \
"lxv              %%vs42, 96(%%r7)               \n\t" \
"lxv              %%vs43, 112(%%r7)              \n\t" \
"lxv              %%vs44, 128(%%r7)              \n\t" \
"                                                \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2      \n\t" \
"xxpermdi         %%vs57, %%vs56, %%vs56, 2      \n\t" \
"                                                \n\t" \
"addi             %%r8, %%r8, 32                 \n\t" \
"addi             %%r7, %%r7, 144                \n\t" \
"                                                \n\t" \
"                                                \n\t" 










#define DCOL_ADD_TO_C \
"xvadddp          %%vs48, %%vs48, %%vs36 	      \n\t" \
"xvadddp          %%vs49, %%vs49, %%vs37    	  \n\t" \
"xvadddp          %%vs50, %%vs50, %%vs38   	    \n\t" \
"xvadddp          %%vs51, %%vs51, %%vs39     	  \n\t" \
"xvadddp          %%vs52, %%vs52, %%vs40   	    \n\t" \
"xvadddp          %%vs53, %%vs53, %%vs41   	    \n\t" \
"xvadddp          %%vs54, %%vs54, %%vs42   	    \n\t" \
"xvadddp          %%vs55, %%vs55, %%vs43        \n\t" \
"xvadddp          %%vs56, %%vs56, %%vs44 	      \n\t"














#define DCOL_SCALE_BETA \
"xvmuldp         %%vs36, %%vs36, %%vs59         \n\t" \
"xvmuldp         %%vs37, %%vs37, %%vs59         \n\t" \
"xvmuldp         %%vs38, %%vs38, %%vs59         \n\t" \
"xvmuldp         %%vs39, %%vs39, %%vs59         \n\t" \
"xvmuldp         %%vs40, %%vs40, %%vs59         \n\t" \
"xvmuldp         %%vs41, %%vs41, %%vs59         \n\t" \
"xvmuldp         %%vs42, %%vs42, %%vs59         \n\t" \
"xvmuldp         %%vs43, %%vs43, %%vs59         \n\t" \
"xvmuldp         %%vs44, %%vs44, %%vs59         \n\t" 











#define DGEN_BETA_SCALE \
"xvmuldp          %%vs36, %%vs36, %%vs59   	    \n\t" \
"xvmuldp          %%vs37, %%vs37, %%vs59   	    \n\t" \
"xvmuldp          %%vs38, %%vs38, %%vs59   	    \n\t" \
"xvmuldp          %%vs39, %%vs39, %%vs59   	    \n\t" \
"xvmuldp          %%vs40, %%vs40, %%vs59   	    \n\t" \
"xvmuldp          %%vs41, %%vs41, %%vs59   	    \n\t" \
"xvmuldp          %%vs42, %%vs42, %%vs59   	    \n\t" \
"xvmuldp          %%vs43, %%vs43, %%vs59   	    \n\t" \
"xvmuldp          %%vs44, %%vs44, %%vs59   	    \n\t" 









#define DGEN_LOAD_C \
"lxsdx     %%vs32, %%r9, %%r22                  \n\t" \
"lxsdx     %%vs33, 0, %%r22                     \n\t" \
"lxsdx     %%vs34, %%r9, %%r23                  \n\t" \
"lxsdx     %%vs35, 0, %%r23                     \n\t" \
"lxsdx     %%vs36, %%r9, %%r24                  \n\t" \
"lxsdx     %%vs37, 0, %%r24                     \n\t" \
"lxsdx     %%vs38, %%r9, %%r25                  \n\t" \
"lxsdx     %%vs39, 0, %%r25                     \n\t" \
"lxsdx     %%vs40, %%r9, %%r26                  \n\t" \
"lxsdx     %%vs41, 0, %%r26                     \n\t" \
"lxsdx     %%vs42, %%r9, %%r27                  \n\t" \
"lxsdx     %%vs43, 0, %%r27                     \n\t" \
"lxsdx     %%vs44, %%r9, %%r28                  \n\t" \
"lxsdx     %%vs45, 0, %%r28                     \n\t" \
"lxsdx     %%vs46, %%r9, %%r29                  \n\t" \
"lxsdx     %%vs47, 0, %%r29                     \n\t" \
"xxpermdi    %%vs32, %%vs32, %%vs33, 0          \n\t" \
"xxpermdi    %%vs33, %%vs34, %%vs35, 0          \n\t" \
"xxpermdi    %%vs34, %%vs36, %%vs37, 0          \n\t" \
"xxpermdi    %%vs35, %%vs38, %%vs39, 0          \n\t" \
"xxpermdi    %%vs36, %%vs40, %%vs41, 0          \n\t" \
"xxpermdi    %%vs37, %%vs42, %%vs43, 0          \n\t" \
"xxpermdi    %%vs38, %%vs44, %%vs45, 0          \n\t" \
"xxpermdi    %%vs39, %%vs46, %%vs47, 0          \n\t" 









#define DGEN_NEXT_COL_C \
"add             %%r22, %%r22, %%r10            \n\t" \
"add             %%r23, %%r23, %%r10            \n\t" \
"add             %%r24, %%r24, %%r10            \n\t" \
"add             %%r25, %%r25, %%r10            \n\t" \
"add             %%r26, %%r26, %%r10            \n\t" \
"add             %%r27, %%r27, %%r10            \n\t" \
"add             %%r28, %%r28, %%r10            \n\t" \
"add             %%r29, %%r29, %%r10            \n\t"











#define DGEN_ADD_STORE \
"xvadddp      %%vs40, %%vs40, %%vs32            \n\t" \
"xvadddp      %%vs41, %%vs41, %%vs33            \n\t" \
"xvadddp      %%vs42, %%vs42, %%vs34            \n\t" \
"xvadddp      %%vs43, %%vs43, %%vs35            \n\t" \
"xvadddp      %%vs44, %%vs44, %%vs36            \n\t" \
"xvadddp      %%vs45, %%vs45, %%vs37            \n\t" \
"xvadddp      %%vs46, %%vs46, %%vs38            \n\t" \
"xvadddp      %%vs47, %%vs47, %%vs39            \n\t" \
"                                              	\n\t" \
"stxsdx       %%vs40, %%r9, %%r22               \n\t" \
"stxsdx       %%vs41, %%r9, %%r23               \n\t" \
"stxsdx       %%vs42, %%r9, %%r24               \n\t" \
"stxsdx       %%vs43, %%r9, %%r25               \n\t" \
"stxsdx       %%vs44, %%r9, %%r26               \n\t" \
"stxsdx       %%vs45, %%r9, %%r27               \n\t" \
"stxsdx       %%vs46, %%r9, %%r28               \n\t" \
"stxsdx       %%vs47, %%r9, %%r29               \n\t" \
"xxswapd      %%vs40, %%vs40                    \n\t" \
"xxswapd      %%vs41, %%vs41                    \n\t" \
"xxswapd      %%vs42, %%vs42                    \n\t" \
"xxswapd      %%vs43, %%vs43                    \n\t" \
"xxswapd      %%vs44, %%vs44                    \n\t" \
"xxswapd      %%vs45, %%vs45                    \n\t" \
"xxswapd      %%vs46, %%vs46                    \n\t" \
"xxswapd      %%vs47, %%vs47                    \n\t" \
"stxsdx       %%vs40, 0, %%r22                  \n\t" \
"stxsdx       %%vs41, 0, %%r23                  \n\t" \
"stxsdx       %%vs42, 0, %%r24                  \n\t" \
"stxsdx       %%vs43, 0, %%r25                  \n\t" \
"stxsdx       %%vs44, 0, %%r26                  \n\t" \
"stxsdx       %%vs45, 0, %%r27                  \n\t" \
"stxsdx       %%vs46, 0, %%r28                  \n\t" \
"stxsdx       %%vs47, 0, %%r29                  \n\t" \







#define DGEN_LOAD_SCALE \
  DGEN_LOAD_C   \
  DGEN_BETA_SCALE






  