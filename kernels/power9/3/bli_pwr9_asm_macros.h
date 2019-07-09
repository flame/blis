



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
 "xxlxor           %%vs31, %%vs31, %%vs31       \n\t"  











#define SCALE_ALPHA \
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
"xvmuldp         %%vs31, %%vs31, %%vs60         \n\t"








#define PRELOAD_A_B \
"lxv              %%vs48, 0(%%r8)               \n\t" \
"lxv              %%vs50, 16(%%r8)              \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 0(%%r7)               \n\t" \
"lxv              %%vs33, 16(%%r7)              \n\t" \
"lxv              %%vs34, 32(%%r7)              \n\t" \
"lxv              %%vs35, 48(%%r7)              \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 64(%%r7)              \n\t" \
"lxv              %%vs37, 80(%%r7)              \n\t" \
"lxv              %%vs38, 96(%%r7)              \n\t" \
"lxv              %%vs39, 112(%%r7)             \n\t" 











#define LOAD_UPDATE_16 \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs32, %%vs48         \n\t" \
"xvmaddadp        %%vs1, %%vs33, %%vs48         \n\t" \
"xvmaddadp        %%vs2, %%vs34, %%vs48         \n\t" \
"xvmaddadp        %%vs3, %%vs35, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 0(%%r7)               \n\t" \
"lxv              %%vs41, 16(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" \
"xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" \
"xvmaddadp        %%vs6, %%vs38, %%vs48         \n\t" \
"xvmaddadp        %%vs7, %%vs39, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs42, 32(%%r7)              \n\t" \
"lxv              %%vs43, 48(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs32, %%vs49         \n\t" \
"xvmaddadp        %%vs9, %%vs33, %%vs49         \n\t" \
"xvmaddadp        %%vs10, %%vs34, %%vs49        \n\t" \
"xvmaddadp        %%vs11, %%vs35, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs52, 0(%%r8)               \n\t" \
"lxv              %%vs54, 16(%%r8)              \n\t" \
"xxpermdi         %%vs53, %%vs52, %%vs52, 2     \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs36, %%vs49        \n\t" \
"xvmaddadp        %%vs13, %%vs37, %%vs49        \n\t" \
"xvmaddadp        %%vs14, %%vs38, %%vs49        \n\t" \
"xvmaddadp        %%vs15, %%vs39, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs44, 64(%%r7)              \n\t" \
"lxv              %%vs45, 80(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs32, %%vs50        \n\t" \
"xvmaddadp        %%vs17, %%vs33, %%vs50        \n\t" \
"xvmaddadp        %%vs18, %%vs34, %%vs50        \n\t" \
"xvmaddadp        %%vs19, %%vs35, %%vs50        \n\t" \
"                                               \n\t" \
"lxv              %%vs46, 96(%%r7)              \n\t" \
"lxv              %%vs47, 112(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs36, %%vs50        \n\t" \
"xvmaddadp        %%vs21, %%vs37, %%vs50        \n\t" \
"xvmaddadp        %%vs22, %%vs38, %%vs50        \n\t" \
"xvmaddadp        %%vs23, %%vs39, %%vs50        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs32, %%vs51        \n\t" \
"xvmaddadp        %%vs25, %%vs33, %%vs51        \n\t" \
"xvmaddadp        %%vs26, %%vs34, %%vs51        \n\t" \
"xvmaddadp        %%vs27, %%vs35, %%vs51        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs36, %%vs51        \n\t" \
"xvmaddadp        %%vs29, %%vs37, %%vs51        \n\t" \
"xvmaddadp        %%vs30, %%vs38, %%vs51        \n\t" \
"xvmaddadp        %%vs31, %%vs39, %%vs51        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs40, %%vs52         \n\t" \
"xvmaddadp        %%vs1, %%vs41, %%vs52         \n\t" \
"xvmaddadp        %%vs2, %%vs42, %%vs52         \n\t" \
"xvmaddadp        %%vs3, %%vs43, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 32(%%r8)              \n\t" \
"lxv              %%vs50, 48(%%r8)              \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs44, %%vs52         \n\t" \
"xvmaddadp        %%vs5, %%vs45, %%vs52         \n\t" \
"xvmaddadp        %%vs6, %%vs46, %%vs52         \n\t" \
"xvmaddadp        %%vs7, %%vs47, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 128(%%r7)             \n\t" \
"lxv              %%vs33, 144(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs40, %%vs53         \n\t" \
"xvmaddadp        %%vs9, %%vs41, %%vs53         \n\t" \
"xvmaddadp        %%vs10, %%vs42, %%vs53        \n\t" \
"xvmaddadp        %%vs11, %%vs43, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs34, 160(%%r7)             \n\t" \
"lxv              %%vs35, 176(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs44, %%vs53        \n\t" \
"xvmaddadp        %%vs13, %%vs45, %%vs53        \n\t" \
"xvmaddadp        %%vs14, %%vs46, %%vs53        \n\t" \
"xvmaddadp        %%vs15, %%vs47, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 192(%%r7)             \n\t" \
"lxv              %%vs37, 208(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs40, %%vs54        \n\t" \
"xvmaddadp        %%vs17, %%vs41, %%vs54        \n\t" \
"xvmaddadp        %%vs18, %%vs42, %%vs54        \n\t" \
"xvmaddadp        %%vs19, %%vs43, %%vs54        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 224(%%r7)             \n\t" \
"lxv              %%vs39, 240(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs44, %%vs54        \n\t" \
"xvmaddadp        %%vs21, %%vs45, %%vs54        \n\t" \
"xvmaddadp        %%vs22, %%vs46, %%vs54        \n\t" \
"xvmaddadp        %%vs23, %%vs47, %%vs54        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs25, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs26, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs27, %%vs43, %%vs55        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs44, %%vs55        \n\t" \
"xvmaddadp        %%vs29, %%vs45, %%vs55        \n\t" \
"xvmaddadp        %%vs30, %%vs46, %%vs55        \n\t" \
"xvmaddadp        %%vs31, %%vs47, %%vs55        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs32, %%vs48         \n\t" \
"xvmaddadp        %%vs1, %%vs33, %%vs48         \n\t" \
"xvmaddadp        %%vs2, %%vs34, %%vs48         \n\t" \
"xvmaddadp        %%vs3, %%vs35, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 256(%%r7)             \n\t" \
"lxv              %%vs41, 272(%%r7)                \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" \
"xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" \
"xvmaddadp        %%vs6, %%vs38, %%vs48         \n\t" \
"xvmaddadp        %%vs7, %%vs39, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs42, 288(%%r7)              \n\t" \
"lxv              %%vs43, 304(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs32, %%vs49         \n\t" \
"xvmaddadp        %%vs9, %%vs33, %%vs49         \n\t" \
"xvmaddadp        %%vs10, %%vs34, %%vs49        \n\t" \
"xvmaddadp        %%vs11, %%vs35, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs52, 64(%%r8)               \n\t" \
"lxv              %%vs54, 80(%%r8)              \n\t" \
"xxpermdi         %%vs53, %%vs52, %%vs52, 2     \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs36, %%vs49        \n\t" \
"xvmaddadp        %%vs13, %%vs37, %%vs49        \n\t" \
"xvmaddadp        %%vs14, %%vs38, %%vs49        \n\t" \
"xvmaddadp        %%vs15, %%vs39, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs44, 320(%%r7)              \n\t" \
"lxv              %%vs45, 336(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs32, %%vs50        \n\t" \
"xvmaddadp        %%vs17, %%vs33, %%vs50        \n\t" \
"xvmaddadp        %%vs18, %%vs34, %%vs50        \n\t" \
"xvmaddadp        %%vs19, %%vs35, %%vs50        \n\t" \
"                                               \n\t" \
"lxv              %%vs46, 352(%%r7)              \n\t" \
"lxv              %%vs47, 368(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs36, %%vs50        \n\t" \
"xvmaddadp        %%vs21, %%vs37, %%vs50        \n\t" \
"xvmaddadp        %%vs22, %%vs38, %%vs50        \n\t" \
"xvmaddadp        %%vs23, %%vs39, %%vs50        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs32, %%vs51        \n\t" \
"xvmaddadp        %%vs25, %%vs33, %%vs51        \n\t" \
"xvmaddadp        %%vs26, %%vs34, %%vs51        \n\t" \
"xvmaddadp        %%vs27, %%vs35, %%vs51        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs36, %%vs51        \n\t" \
"xvmaddadp        %%vs29, %%vs37, %%vs51        \n\t" \
"xvmaddadp        %%vs30, %%vs38, %%vs51        \n\t" \
"xvmaddadp        %%vs31, %%vs39, %%vs51        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs40, %%vs52         \n\t" \
"xvmaddadp        %%vs1, %%vs41, %%vs52         \n\t" \
"xvmaddadp        %%vs2, %%vs42, %%vs52         \n\t" \
"xvmaddadp        %%vs3, %%vs43, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 96(%%r8)              \n\t" \
"lxv              %%vs50, 112(%%r8)             \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs44, %%vs52         \n\t" \
"xvmaddadp        %%vs5, %%vs45, %%vs52         \n\t" \
"xvmaddadp        %%vs6, %%vs46, %%vs52         \n\t" \
"xvmaddadp        %%vs7, %%vs47, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 384(%%r7)             \n\t" \
"lxv              %%vs33, 400(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs40, %%vs53         \n\t" \
"xvmaddadp        %%vs9, %%vs41, %%vs53         \n\t" \
"xvmaddadp        %%vs10, %%vs42, %%vs53        \n\t" \
"xvmaddadp        %%vs11, %%vs43, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs34, 416(%%r7)             \n\t" \
"lxv              %%vs35, 432(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs44, %%vs53        \n\t" \
"xvmaddadp        %%vs13, %%vs45, %%vs53        \n\t" \
"xvmaddadp        %%vs14, %%vs46, %%vs53        \n\t" \
"xvmaddadp        %%vs15, %%vs47, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 448(%%r7)             \n\t" \
"lxv              %%vs37, 464(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs40, %%vs54        \n\t" \
"xvmaddadp        %%vs17, %%vs41, %%vs54        \n\t" \
"xvmaddadp        %%vs18, %%vs42, %%vs54        \n\t" \
"xvmaddadp        %%vs19, %%vs43, %%vs54        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 480(%%r7)             \n\t" \
"lxv              %%vs39, 496(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs44, %%vs54        \n\t" \
"xvmaddadp        %%vs21, %%vs45, %%vs54        \n\t" \
"xvmaddadp        %%vs22, %%vs46, %%vs54        \n\t" \
"xvmaddadp        %%vs23, %%vs47, %%vs54        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs25, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs26, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs27, %%vs43, %%vs55        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs44, %%vs55        \n\t" \
"xvmaddadp        %%vs29, %%vs45, %%vs55        \n\t" \
"xvmaddadp        %%vs30, %%vs46, %%vs55        \n\t" \
"xvmaddadp        %%vs31, %%vs47, %%vs55        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs32, %%vs48         \n\t" \
"xvmaddadp        %%vs1, %%vs33, %%vs48         \n\t" \
"xvmaddadp        %%vs2, %%vs34, %%vs48         \n\t" \
"xvmaddadp        %%vs3, %%vs35, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 512(%%r7)             \n\t" \
"lxv              %%vs41, 528(%%r7)                \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" \
"xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" \
"xvmaddadp        %%vs6, %%vs38, %%vs48         \n\t" \
"xvmaddadp        %%vs7, %%vs39, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs42, 544(%%r7)              \n\t" \
"lxv              %%vs43, 560(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs32, %%vs49         \n\t" \
"xvmaddadp        %%vs9, %%vs33, %%vs49         \n\t" \
"xvmaddadp        %%vs10, %%vs34, %%vs49        \n\t" \
"xvmaddadp        %%vs11, %%vs35, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs52, 128(%%r8)             \n\t" \
"lxv              %%vs54, 144(%%r8)             \n\t" \
"xxpermdi         %%vs53, %%vs52, %%vs52, 2     \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs36, %%vs49        \n\t" \
"xvmaddadp        %%vs13, %%vs37, %%vs49        \n\t" \
"xvmaddadp        %%vs14, %%vs38, %%vs49        \n\t" \
"xvmaddadp        %%vs15, %%vs39, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs44, 576(%%r7)             \n\t" \
"lxv              %%vs45, 592(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs32, %%vs50        \n\t" \
"xvmaddadp        %%vs17, %%vs33, %%vs50        \n\t" \
"xvmaddadp        %%vs18, %%vs34, %%vs50        \n\t" \
"xvmaddadp        %%vs19, %%vs35, %%vs50        \n\t" \
"                                               \n\t" \
"lxv              %%vs46, 608(%%r7)             \n\t" \
"lxv              %%vs47, 624(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs36, %%vs50        \n\t" \
"xvmaddadp        %%vs21, %%vs37, %%vs50        \n\t" \
"xvmaddadp        %%vs22, %%vs38, %%vs50        \n\t" \
"xvmaddadp        %%vs23, %%vs39, %%vs50        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs32, %%vs51        \n\t" \
"xvmaddadp        %%vs25, %%vs33, %%vs51        \n\t" \
"xvmaddadp        %%vs26, %%vs34, %%vs51        \n\t" \
"xvmaddadp        %%vs27, %%vs35, %%vs51        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs36, %%vs51        \n\t" \
"xvmaddadp        %%vs29, %%vs37, %%vs51        \n\t" \
"xvmaddadp        %%vs30, %%vs38, %%vs51        \n\t" \
"xvmaddadp        %%vs31, %%vs39, %%vs51        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs40, %%vs52         \n\t" \
"xvmaddadp        %%vs1, %%vs41, %%vs52         \n\t" \
"xvmaddadp        %%vs2, %%vs42, %%vs52         \n\t" \
"xvmaddadp        %%vs3, %%vs43, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 160(%%r8)             \n\t" \
"lxv              %%vs50, 176(%%r8)             \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs44, %%vs52         \n\t" \
"xvmaddadp        %%vs5, %%vs45, %%vs52         \n\t" \
"xvmaddadp        %%vs6, %%vs46, %%vs52         \n\t" \
"xvmaddadp        %%vs7, %%vs47, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 640(%%r7)             \n\t" \
"lxv              %%vs33, 656(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs40, %%vs53         \n\t" \
"xvmaddadp        %%vs9, %%vs41, %%vs53         \n\t" \
"xvmaddadp        %%vs10, %%vs42, %%vs53        \n\t" \
"xvmaddadp        %%vs11, %%vs43, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs34, 672(%%r7)             \n\t" \
"lxv              %%vs35, 688(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs44, %%vs53        \n\t" \
"xvmaddadp        %%vs13, %%vs45, %%vs53        \n\t" \
"xvmaddadp        %%vs14, %%vs46, %%vs53        \n\t" \
"xvmaddadp        %%vs15, %%vs47, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 704(%%r7)             \n\t" \
"lxv              %%vs37, 720(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs40, %%vs54        \n\t" \
"xvmaddadp        %%vs17, %%vs41, %%vs54        \n\t" \
"xvmaddadp        %%vs18, %%vs42, %%vs54        \n\t" \
"xvmaddadp        %%vs19, %%vs43, %%vs54        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 736(%%r7)             \n\t" \
"lxv              %%vs39, 752(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs44, %%vs54        \n\t" \
"xvmaddadp        %%vs21, %%vs45, %%vs54        \n\t" \
"xvmaddadp        %%vs22, %%vs46, %%vs54        \n\t" \
"xvmaddadp        %%vs23, %%vs47, %%vs54        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs25, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs26, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs27, %%vs43, %%vs55        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs44, %%vs55        \n\t" \
"xvmaddadp        %%vs29, %%vs45, %%vs55        \n\t" \
"xvmaddadp        %%vs30, %%vs46, %%vs55        \n\t" \
"xvmaddadp        %%vs31, %%vs47, %%vs55        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs0, %%vs32, %%vs48	        \n\t" \
"xvmaddadp	      %%vs1, %%vs33, %%vs48	        \n\t" \
"xvmaddadp	      %%vs2, %%vs34, %%vs48	        \n\t" \
"xvmaddadp	      %%vs3, %%vs35, %%vs48	        \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 768(%%r7)             \n\t" \
"lxv              %%vs41, 784(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs4, %%vs36, %%vs48	        \n\t" \
"xvmaddadp	      %%vs5, %%vs37, %%vs48	        \n\t" \
"xvmaddadp	      %%vs6, %%vs38, %%vs48	        \n\t" \
"xvmaddadp	      %%vs7, %%vs39, %%vs48	        \n\t" \
"                                               \n\t" \
"lxv              %%vs42, 800(%%r7)             \n\t" \
"lxv              %%vs43, 816(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs8, %%vs32, %%vs49	        \n\t" \
"xvmaddadp	      %%vs9, %%vs33, %%vs49	        \n\t" \
"xvmaddadp	      %%vs10, %%vs34, %%vs49	      \n\t" \
"xvmaddadp	      %%vs11, %%vs35, %%vs49	      \n\t" \
"                                               \n\t" \
"lxv 	            %%vs52, 192(%%r8)	            \n\t" \
"lxv 	            %%vs54, 208(%%r8)	            \n\t" \
"xxpermdi         %%vs53, %%vs52, %%vs52, 2     \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2	    \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs12, %%vs36, %%vs49	      \n\t" \
"xvmaddadp	      %%vs13, %%vs37, %%vs49	      \n\t" \
"xvmaddadp	      %%vs14, %%vs38, %%vs49	      \n\t" \
"xvmaddadp	      %%vs15, %%vs39, %%vs49	      \n\t" \
"                                               \n\t" \
"lxv              %%vs44, 832(%%r7)             \n\t" \
"lxv              %%vs45, 848(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs16, %%vs32, %%vs50	      \n\t" \
"xvmaddadp	      %%vs17, %%vs33, %%vs50	      \n\t" \
"xvmaddadp	      %%vs18, %%vs34, %%vs50	      \n\t" \
"xvmaddadp	      %%vs19, %%vs35, %%vs50	      \n\t" \
"                                               \n\t" \
"lxv              %%vs46, 864(%%r7)             \n\t" \
"lxv              %%vs47, 880(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs20, %%vs36, %%vs50	      \n\t" \
"xvmaddadp	      %%vs21, %%vs37, %%vs50	      \n\t" \
"xvmaddadp	      %%vs22, %%vs38, %%vs50	      \n\t" \
"xvmaddadp	      %%vs23, %%vs39, %%vs50	      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs24, %%vs32, %%vs51	      \n\t" \
"xvmaddadp	      %%vs25, %%vs33, %%vs51	      \n\t" \
"xvmaddadp	      %%vs26, %%vs34, %%vs51	      \n\t" \
"xvmaddadp	      %%vs27, %%vs35, %%vs51	      \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs28, %%vs36, %%vs51	      \n\t" \
"xvmaddadp	      %%vs29, %%vs37, %%vs51	      \n\t" \
"xvmaddadp	      %%vs30, %%vs38, %%vs51	      \n\t" \
"xvmaddadp	      %%vs31, %%vs39, %%vs51	      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs40, %%vs52         \n\t" \
"xvmaddadp        %%vs1, %%vs41, %%vs52         \n\t" \
"xvmaddadp        %%vs2, %%vs42, %%vs52         \n\t" \
"xvmaddadp        %%vs3, %%vs43, %%vs52         \n\t" \
"                                               \n\t" \
"lxv	            %%vs48, 224(%%r8)	            \n\t" \
"lxv              %%vs50, 240(%%r8)             \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs44, %%vs52         \n\t" \
"xvmaddadp        %%vs5, %%vs45, %%vs52         \n\t" \
"xvmaddadp        %%vs6, %%vs46, %%vs52         \n\t" \
"xvmaddadp        %%vs7, %%vs47, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 896(%%r7)             \n\t" \
"lxv              %%vs33, 912(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs40, %%vs53         \n\t" \
"xvmaddadp        %%vs9, %%vs41, %%vs53         \n\t" \
"xvmaddadp        %%vs10, %%vs42, %%vs53        \n\t" \
"xvmaddadp        %%vs11, %%vs43, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs34, 928(%%r7)             \n\t" \
"lxv              %%vs35, 944(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs44, %%vs53        \n\t" \
"xvmaddadp        %%vs13, %%vs45, %%vs53        \n\t" \
"xvmaddadp        %%vs14, %%vs46, %%vs53        \n\t" \
"xvmaddadp        %%vs15, %%vs47, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 960(%%r7)             \n\t" \
"lxv              %%vs37, 976(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs40, %%vs54        \n\t" \
"xvmaddadp        %%vs17, %%vs41, %%vs54        \n\t" \
"xvmaddadp        %%vs18, %%vs42, %%vs54        \n\t" \
"xvmaddadp        %%vs19, %%vs43, %%vs54        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 992(%%r7)             \n\t" \
"lxv              %%vs39, 1008(%%r7)            \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs44, %%vs54        \n\t" \
"xvmaddadp        %%vs21, %%vs45, %%vs54        \n\t" \
"xvmaddadp        %%vs22, %%vs46, %%vs54        \n\t" \
"xvmaddadp        %%vs23, %%vs47, %%vs54        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs25, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs26, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs27, %%vs43, %%vs55        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs44, %%vs55        \n\t" \
"xvmaddadp        %%vs29, %%vs45, %%vs55        \n\t" \
"xvmaddadp        %%vs30, %%vs46, %%vs55        \n\t" \
"xvmaddadp        %%vs31, %%vs47, %%vs55        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs0, %%vs32, %%vs48	        \n\t" \
"xvmaddadp	      %%vs1, %%vs33, %%vs48	        \n\t" \
"xvmaddadp	      %%vs2, %%vs34, %%vs48	        \n\t" \
"xvmaddadp	      %%vs3, %%vs35, %%vs48	        \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 1024(%%r7)            \n\t" \
"lxv              %%vs41, 1040(%%r7)            \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs4, %%vs36, %%vs48	        \n\t" \
"xvmaddadp	      %%vs5, %%vs37, %%vs48	        \n\t" \
"xvmaddadp	      %%vs6, %%vs38, %%vs48	        \n\t" \
"xvmaddadp	      %%vs7, %%vs39, %%vs48	        \n\t" \
"                                               \n\t" \
"lxv           %%vs42, 1056(%%r7)               \n\t" \
"lxv           %%vs43, 1072(%%r7)               \n\t" \
"                                               \n\t" \
"xvmaddadp      	%%vs8, %%vs32, %%vs49	        \n\t" \
"xvmaddadp      	%%vs9, %%vs33, %%vs49	        \n\t" \
"xvmaddadp	      %%vs10, %%vs34, %%vs49	      \n\t" \
"xvmaddadp	      %%vs11, %%vs35, %%vs49	      \n\t" \
"                                               \n\t" \
"lxv 	            %%vs52, 256(%%r8)	            \n\t" \
"lxv 	            %%vs54, 272(%%r8)	            \n\t" \
"xxpermdi         %%vs53, %%vs52, %%vs52, 2     \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2	    \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs12, %%vs36, %%vs49	      \n\t" \
"xvmaddadp	      %%vs13, %%vs37, %%vs49	      \n\t" \
"xvmaddadp	      %%vs14, %%vs38, %%vs49	      \n\t" \
"xvmaddadp	      %%vs15, %%vs39, %%vs49	      \n\t" \
"                                               \n\t" \
"lxv           %%vs44, 1088(%%r7)               \n\t" \
"lxv           %%vs45, 1104(%%r7)               \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs16, %%vs32, %%vs50	      \n\t" \
"xvmaddadp	      %%vs17, %%vs33, %%vs50	      \n\t" \
"xvmaddadp	      %%vs18, %%vs34, %%vs50	      \n\t" \
"xvmaddadp	      %%vs19, %%vs35, %%vs50	      \n\t" \
"                                               \n\t" \
"lxv              %%vs46, 1120(%%r7)            \n\t" \
"lxv              %%vs47, 1136(%%r7)            \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs20, %%vs36, %%vs50	      \n\t" \
"xvmaddadp	      %%vs21, %%vs37, %%vs50	      \n\t" \
"xvmaddadp	      %%vs22, %%vs38, %%vs50	      \n\t" \
"xvmaddadp	      %%vs23, %%vs39, %%vs50	      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs24, %%vs32, %%vs51	      \n\t" \
"xvmaddadp	      %%vs25, %%vs33, %%vs51	      \n\t" \
"xvmaddadp	      %%vs26, %%vs34, %%vs51	      \n\t" \
"xvmaddadp	      %%vs27, %%vs35, %%vs51	      \n\t" \
"                                               \n\t" \
"xvmaddadp	      %%vs28, %%vs36, %%vs51	      \n\t" \
"xvmaddadp	      %%vs29, %%vs37, %%vs51	      \n\t" \
"xvmaddadp	      %%vs30, %%vs38, %%vs51	      \n\t" \
"xvmaddadp	      %%vs31, %%vs39, %%vs51	      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs40, %%vs52         \n\t" \
"xvmaddadp        %%vs1, %%vs41, %%vs52         \n\t" \
"xvmaddadp        %%vs2, %%vs42, %%vs52         \n\t" \
"xvmaddadp        %%vs3, %%vs43, %%vs52         \n\t" \
"                                               \n\t" \
"lxv	            %%vs48, 288(%%r8)	            \n\t" \
"lxv              %%vs50, 304(%%r8)             \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs44, %%vs52         \n\t" \
"xvmaddadp        %%vs5, %%vs45, %%vs52         \n\t" \
"xvmaddadp        %%vs6, %%vs46, %%vs52         \n\t" \
"xvmaddadp        %%vs7, %%vs47, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 1152(%%r7)            \n\t" \
"lxv              %%vs33, 1168(%%r7)            \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs40, %%vs53         \n\t" \
"xvmaddadp        %%vs9, %%vs41, %%vs53         \n\t" \
"xvmaddadp        %%vs10, %%vs42, %%vs53        \n\t" \
"xvmaddadp        %%vs11, %%vs43, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs34, 1184(%%r7)            \n\t" \
"lxv              %%vs35, 1200(%%r7)            \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs44, %%vs53        \n\t" \
"xvmaddadp        %%vs13, %%vs45, %%vs53        \n\t" \
"xvmaddadp        %%vs14, %%vs46, %%vs53        \n\t" \
"xvmaddadp        %%vs15, %%vs47, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 1216(%%r7)            \n\t" \
"lxv              %%vs37, 1232(%%r7)            \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs40, %%vs54        \n\t" \
"xvmaddadp        %%vs17, %%vs41, %%vs54        \n\t" \
"xvmaddadp        %%vs18, %%vs42, %%vs54        \n\t" \
"xvmaddadp        %%vs19, %%vs43, %%vs54        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 1248(%%r7)            \n\t" \
"lxv              %%vs39, 1264(%%r7)            \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs44, %%vs54        \n\t" \
"xvmaddadp        %%vs21, %%vs45, %%vs54        \n\t" \
"xvmaddadp        %%vs22, %%vs46, %%vs54        \n\t" \
"xvmaddadp        %%vs23, %%vs47, %%vs54        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs25, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs26, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs27, %%vs43, %%vs55        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs44, %%vs55        \n\t" \
"xvmaddadp        %%vs29, %%vs45, %%vs55        \n\t" \
"xvmaddadp        %%vs30, %%vs46, %%vs55        \n\t" \
"xvmaddadp        %%vs31, %%vs47, %%vs55        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	       %%vs0, %%vs32, %%vs48	      \n\t" \
"xvmaddadp	       %%vs1, %%vs33, %%vs48	      \n\t" \
"xvmaddadp	       %%vs2, %%vs34, %%vs48	      \n\t" \
"xvmaddadp	       %%vs3, %%vs35, %%vs48	      \n\t" \
"                                               \n\t" \
"lxv               %%vs40, 1280(%%r7)           \n\t" \
"lxv               %%vs41, 1296(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp	       %%vs4, %%vs36, %%vs48	      \n\t" \
"xvmaddadp	       %%vs5, %%vs37, %%vs48	      \n\t" \
"xvmaddadp	       %%vs6, %%vs38, %%vs48	      \n\t" \
"xvmaddadp	       %%vs7, %%vs39, %%vs48	      \n\t" \
"                                               \n\t" \
"lxv               %%vs42, 1312(%%r7)           \n\t" \
"lxv               %%vs43, 1328(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp	       %%vs8, %%vs32, %%vs49	      \n\t" \
"xvmaddadp	       %%vs9, %%vs33, %%vs49	      \n\t" \
"xvmaddadp	       %%vs10, %%vs34, %%vs49	      \n\t" \
"xvmaddadp	       %%vs11, %%vs35, %%vs49	      \n\t" \
"                                               \n\t" \
"lxv 	             %%vs52, 320(%%r8)	          \n\t" \
"lxv 	             %%vs54, 336(%%r8)	          \n\t" \
"xxpermdi          %%vs53, %%vs52, %%vs52, 2    \n\t" \
"xxpermdi          %%vs55, %%vs54, %%vs54, 2    \n\t" \
"                                               \n\t" \
"xvmaddadp    	   %%vs12, %%vs36, %%vs49	      \n\t" \
"xvmaddadp    	   %%vs13, %%vs37, %%vs49	      \n\t" \
"xvmaddadp    	   %%vs14, %%vs38, %%vs49	      \n\t" \
"xvmaddadp    	   %%vs15, %%vs39, %%vs49	      \n\t" \
"                                               \n\t" \
"lxv               %%vs44, 1344(%%r7)           \n\t" \
"lxv               %%vs45, 1360(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp	       %%vs16, %%vs32, %%vs50	      \n\t" \
"xvmaddadp	       %%vs17, %%vs33, %%vs50	      \n\t" \
"xvmaddadp	       %%vs18, %%vs34, %%vs50	      \n\t" \
"xvmaddadp	       %%vs19, %%vs35, %%vs50	      \n\t" \
"                                               \n\t" \
"lxv               %%vs46, 1376(%%r7)           \n\t" \
"lxv               %%vs47, 1392(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp	       %%vs20, %%vs36, %%vs50      	\n\t" \
"xvmaddadp	       %%vs21, %%vs37, %%vs50      	\n\t" \
"xvmaddadp	       %%vs22, %%vs38, %%vs50      	\n\t" \
"xvmaddadp	       %%vs23, %%vs39, %%vs50      	\n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	       %%vs24, %%vs32, %%vs51	      \n\t" \
"xvmaddadp	       %%vs25, %%vs33, %%vs51	      \n\t" \
"xvmaddadp	       %%vs26, %%vs34, %%vs51	      \n\t" \
"xvmaddadp	       %%vs27, %%vs35, %%vs51	      \n\t" \
"                                               \n\t" \
"xvmaddadp	       %%vs28, %%vs36, %%vs51	      \n\t" \
"xvmaddadp	       %%vs29, %%vs37, %%vs51	      \n\t" \
"xvmaddadp	       %%vs30, %%vs38, %%vs51	      \n\t" \
"xvmaddadp	       %%vs31, %%vs39, %%vs51	      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs0, %%vs40, %%vs52        \n\t" \
"xvmaddadp         %%vs1, %%vs41, %%vs52        \n\t" \
"xvmaddadp         %%vs2, %%vs42, %%vs52        \n\t" \
"xvmaddadp         %%vs3, %%vs43, %%vs52        \n\t" \
"                                               \n\t" \
"lxv	             %%vs48, 352(%%r8)	          \n\t" \
"lxv               %%vs50, 368(%%r8)            \n\t" \
"xxpermdi          %%vs49, %%vs48, %%vs48, 2    \n\t" \
"xxpermdi          %%vs51, %%vs50, %%vs50, 2    \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs4, %%vs44, %%vs52        \n\t" \
"xvmaddadp         %%vs5, %%vs45, %%vs52        \n\t" \
"xvmaddadp         %%vs6, %%vs46, %%vs52        \n\t" \
"xvmaddadp         %%vs7, %%vs47, %%vs52        \n\t" \
"                                               \n\t" \
"lxv               %%vs32, 1408(%%r7)           \n\t" \
"lxv               %%vs33, 1424(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs8, %%vs40, %%vs53        \n\t" \
"xvmaddadp         %%vs9, %%vs41, %%vs53        \n\t" \
"xvmaddadp         %%vs10, %%vs42, %%vs53       \n\t" \
"xvmaddadp         %%vs11, %%vs43, %%vs53       \n\t" \
"                                               \n\t" \
"lxv               %%vs34, 1440(%%r7)           \n\t" \
"lxv               %%vs35, 1456(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs12, %%vs44, %%vs53       \n\t" \
"xvmaddadp         %%vs13, %%vs45, %%vs53       \n\t" \
"xvmaddadp         %%vs14, %%vs46, %%vs53       \n\t" \
"xvmaddadp         %%vs15, %%vs47, %%vs53       \n\t" \
"                                               \n\t" \
"lxv               %%vs36, 1472(%%r7)           \n\t" \
"lxv               %%vs37, 1488(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs16, %%vs40, %%vs54       \n\t" \
"xvmaddadp         %%vs17, %%vs41, %%vs54       \n\t" \
"xvmaddadp         %%vs18, %%vs42, %%vs54       \n\t" \
"xvmaddadp         %%vs19, %%vs43, %%vs54       \n\t" \
"                                               \n\t" \
"lxv               %%vs38, 1504(%%r7)           \n\t" \
"lxv               %%vs39, 1520(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs20, %%vs44, %%vs54       \n\t" \
"xvmaddadp         %%vs21, %%vs45, %%vs54       \n\t" \
"xvmaddadp         %%vs22, %%vs46, %%vs54       \n\t" \
"xvmaddadp         %%vs23, %%vs47, %%vs54       \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs24, %%vs40, %%vs55       \n\t" \
"xvmaddadp         %%vs25, %%vs41, %%vs55       \n\t" \
"xvmaddadp         %%vs26, %%vs42, %%vs55       \n\t" \
"xvmaddadp         %%vs27, %%vs43, %%vs55       \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs28, %%vs44, %%vs55       \n\t" \
"xvmaddadp         %%vs29, %%vs45, %%vs55       \n\t" \
"xvmaddadp         %%vs30, %%vs46, %%vs55       \n\t" \
"xvmaddadp         %%vs31, %%vs47, %%vs55       \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs0, %%vs32, %%vs48       \n\t" \
"xvmaddadp	        %%vs1, %%vs33, %%vs48       \n\t" \
"xvmaddadp	        %%vs2, %%vs34, %%vs48       \n\t" \
"xvmaddadp	        %%vs3, %%vs35, %%vs48       \n\t" \
"                                               \n\t" \
"lxv                %%vs40, 1536(%%r7)          \n\t" \
"lxv                %%vs41, 1552(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs4, %%vs36, %%vs48       \n\t" \
"xvmaddadp	        %%vs5, %%vs37, %%vs48       \n\t" \
"xvmaddadp	        %%vs6, %%vs38, %%vs48       \n\t" \
"xvmaddadp	        %%vs7, %%vs39, %%vs48       \n\t" \
"                                               \n\t" \
"lxv                %%vs42, 1568(%%r7)          \n\t" \
"lxv                %%vs43, 1584(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs8, %%vs32, %%vs49       \n\t" \
"xvmaddadp	        %%vs9, %%vs33, %%vs49       \n\t" \
"xvmaddadp	        %%vs10, %%vs34, %%vs49      \n\t" \
"xvmaddadp	        %%vs11, %%vs35, %%vs49      \n\t" \
"                                               \n\t" \
"lxv 	              %%vs52, 384(%%r8)	          \n\t" \
"lxv 	              %%vs54, 400(%%r8)	          \n\t" \
"xxpermdi           %%vs53, %%vs52, %%vs52, 2   \n\t" \
"xxpermdi           %%vs55, %%vs54, %%vs54, 2   \n\t" \
"                                               \n\t" \
"xvmaddadp     	    %%vs12, %%vs36, %%vs49	    \n\t" \
"xvmaddadp     	    %%vs13, %%vs37, %%vs49	    \n\t" \
"xvmaddadp     	    %%vs14, %%vs38, %%vs49	    \n\t" \
"xvmaddadp     	    %%vs15, %%vs39, %%vs49	    \n\t" \
"                                               \n\t" \
"lxv                %%vs44, 1600(%%r7)          \n\t" \
"lxv                %%vs45, 1616(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs16, %%vs32, %%vs50	    \n\t" \
"xvmaddadp	        %%vs17, %%vs33, %%vs50	    \n\t" \
"xvmaddadp	        %%vs18, %%vs34, %%vs50	    \n\t" \
"xvmaddadp	        %%vs19, %%vs35, %%vs50	    \n\t" \
"                                               \n\t" \
"lxv                %%vs46, 1632(%%r7)          \n\t" \
"lxv                %%vs47, 1648(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs20, %%vs36, %%vs50     	\n\t" \
"xvmaddadp	        %%vs21, %%vs37, %%vs50     	\n\t" \
"xvmaddadp	        %%vs22, %%vs38, %%vs50     	\n\t" \
"xvmaddadp	        %%vs23, %%vs39, %%vs50     	\n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs24, %%vs32, %%vs51	    \n\t" \
"xvmaddadp	        %%vs25, %%vs33, %%vs51	    \n\t" \
"xvmaddadp	        %%vs26, %%vs34, %%vs51	    \n\t" \
"xvmaddadp	        %%vs27, %%vs35, %%vs51	    \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs28, %%vs36, %%vs51	    \n\t" \
"xvmaddadp	        %%vs29, %%vs37, %%vs51	    \n\t" \
"xvmaddadp	        %%vs30, %%vs38, %%vs51	    \n\t" \
"xvmaddadp	        %%vs31, %%vs39, %%vs51	    \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs0, %%vs40, %%vs52       \n\t" \
"xvmaddadp          %%vs1, %%vs41, %%vs52       \n\t" \
"xvmaddadp          %%vs2, %%vs42, %%vs52       \n\t" \
"xvmaddadp          %%vs3, %%vs43, %%vs52       \n\t" \
"                                               \n\t" \
"lxv	              %%vs48, 416(%%r8)	          \n\t" \
"lxv                %%vs50, 432(%%r8)           \n\t" \
"xxpermdi           %%vs49, %%vs48, %%vs48, 2   \n\t" \
"xxpermdi           %%vs51, %%vs50, %%vs50, 2   \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs4, %%vs44, %%vs52       \n\t" \
"xvmaddadp          %%vs5, %%vs45, %%vs52       \n\t" \
"xvmaddadp          %%vs6, %%vs46, %%vs52       \n\t" \
"xvmaddadp          %%vs7, %%vs47, %%vs52       \n\t" \
"                                               \n\t" \
"lxv                %%vs32, 1664(%%r7)          \n\t" \
"lxv                %%vs33, 1680(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs8, %%vs40, %%vs53       \n\t" \
"xvmaddadp          %%vs9, %%vs41, %%vs53       \n\t" \
"xvmaddadp          %%vs10, %%vs42, %%vs53      \n\t" \
"xvmaddadp          %%vs11, %%vs43, %%vs53      \n\t" \
"                                               \n\t" \
"lxv                %%vs34, 1696(%%r7)          \n\t" \
"lxv                %%vs35, 1712(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs12, %%vs44, %%vs53      \n\t" \
"xvmaddadp          %%vs13, %%vs45, %%vs53      \n\t" \
"xvmaddadp          %%vs14, %%vs46, %%vs53      \n\t" \
"xvmaddadp          %%vs15, %%vs47, %%vs53      \n\t" \
"                                               \n\t" \
"lxv                %%vs36, 1728(%%r7)          \n\t" \
"lxv                %%vs37, 1744(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs16, %%vs40, %%vs54      \n\t" \
"xvmaddadp          %%vs17, %%vs41, %%vs54      \n\t" \
"xvmaddadp          %%vs18, %%vs42, %%vs54      \n\t" \
"xvmaddadp          %%vs19, %%vs43, %%vs54      \n\t" \
"                                               \n\t" \
"lxv                %%vs38, 1760(%%r7)          \n\t" \
"lxv                %%vs39, 1776(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs20, %%vs44, %%vs54      \n\t" \
"xvmaddadp          %%vs21, %%vs45, %%vs54      \n\t" \
"xvmaddadp          %%vs22, %%vs46, %%vs54      \n\t" \
"xvmaddadp          %%vs23, %%vs47, %%vs54      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs24, %%vs40, %%vs55      \n\t" \
"xvmaddadp          %%vs25, %%vs41, %%vs55      \n\t" \
"xvmaddadp          %%vs26, %%vs42, %%vs55      \n\t" \
"xvmaddadp          %%vs27, %%vs43, %%vs55      \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs28, %%vs44, %%vs55      \n\t" \
"xvmaddadp          %%vs29, %%vs45, %%vs55      \n\t" \
"xvmaddadp          %%vs30, %%vs46, %%vs55      \n\t" \
"xvmaddadp          %%vs31, %%vs47, %%vs55      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs0, %%vs32, %%vs48	      \n\t" \
"xvmaddadp	        %%vs1, %%vs33, %%vs48	      \n\t" \
"xvmaddadp	        %%vs2, %%vs34, %%vs48	      \n\t" \
"xvmaddadp	        %%vs3, %%vs35, %%vs48	      \n\t" \
"                                               \n\t" \
"lxv                %%vs40, 1792(%%r7)          \n\t" \
"lxv                %%vs41, 1808(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs4, %%vs36, %%vs48	      \n\t" \
"xvmaddadp	        %%vs5, %%vs37, %%vs48	      \n\t" \
"xvmaddadp	        %%vs6, %%vs38, %%vs48	      \n\t" \
"xvmaddadp	        %%vs7, %%vs39, %%vs48	      \n\t" \
"                                               \n\t" \
"lxv                %%vs42, 1824(%%r7)          \n\t" \
"lxv                %%vs43, 1840(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs8, %%vs32, %%vs49	      \n\t" \
"xvmaddadp	        %%vs9, %%vs33, %%vs49	      \n\t" \
"xvmaddadp	        %%vs10, %%vs34, %%vs49	    \n\t" \
"xvmaddadp	        %%vs11, %%vs35, %%vs49	    \n\t" \
"                                               \n\t" \
"lxv 	              %%vs52, 448(%%r8)	          \n\t" \
"lxv 	              %%vs54, 464(%%r8)	          \n\t" \
"xxpermdi           %%vs53, %%vs52, %%vs52, 2   \n\t" \
"xxpermdi           %%vs55, %%vs54, %%vs54, 2   \n\t" \
"                                               \n\t" \
"xvmaddadp     	    %%vs12, %%vs36, %%vs49	    \n\t" \
"xvmaddadp     	    %%vs13, %%vs37, %%vs49	    \n\t" \
"xvmaddadp     	    %%vs14, %%vs38, %%vs49	    \n\t" \
"xvmaddadp     	    %%vs15, %%vs39, %%vs49	    \n\t" \
"                                               \n\t" \
"lxv                %%vs44, 1856(%%r7)          \n\t" \
"lxv                %%vs45, 1872(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs16, %%vs32, %%vs50	    \n\t" \
"xvmaddadp	        %%vs17, %%vs33, %%vs50	    \n\t" \
"xvmaddadp	        %%vs18, %%vs34, %%vs50	    \n\t" \
"xvmaddadp	        %%vs19, %%vs35, %%vs50	    \n\t" \
"                                               \n\t" \
"lxv                %%vs46, 1888(%%r7)          \n\t" \
"lxv                %%vs47, 1904(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs20, %%vs36, %%vs50     	\n\t" \
"xvmaddadp	        %%vs21, %%vs37, %%vs50     	\n\t" \
"xvmaddadp	        %%vs22, %%vs38, %%vs50     	\n\t" \
"xvmaddadp	        %%vs23, %%vs39, %%vs50     	\n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs24, %%vs32, %%vs51	    \n\t" \
"xvmaddadp	        %%vs25, %%vs33, %%vs51	    \n\t" \
"xvmaddadp	        %%vs26, %%vs34, %%vs51	    \n\t" \
"xvmaddadp	        %%vs27, %%vs35, %%vs51	    \n\t" \
"                                               \n\t" \
"xvmaddadp	        %%vs28, %%vs36, %%vs51      \n\t" \
"xvmaddadp	        %%vs29, %%vs37, %%vs51      \n\t" \
"xvmaddadp	        %%vs30, %%vs38, %%vs51      \n\t" \
"xvmaddadp	        %%vs31, %%vs39, %%vs51      \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs0, %%vs40, %%vs52       \n\t" \
"xvmaddadp          %%vs1, %%vs41, %%vs52       \n\t" \
"xvmaddadp          %%vs2, %%vs42, %%vs52       \n\t" \
"xvmaddadp          %%vs3, %%vs43, %%vs52       \n\t" \
"                                               \n\t" \
"lxv	              %%vs48, 480(%%r8)	          \n\t" \
"lxv                %%vs50, 496(%%r8)           \n\t" \
"xxpermdi           %%vs49, %%vs48, %%vs48, 2   \n\t" \
"xxpermdi           %%vs51, %%vs50, %%vs50, 2   \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs4, %%vs44, %%vs52       \n\t" \
"xvmaddadp          %%vs5, %%vs45, %%vs52       \n\t" \
"xvmaddadp          %%vs6, %%vs46, %%vs52       \n\t" \
"xvmaddadp          %%vs7, %%vs47, %%vs52       \n\t" \
"                                               \n\t" \
"lxv                %%vs32, 1920(%%r7)          \n\t" \
"lxv                %%vs33, 1936(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp          %%vs8, %%vs40, %%vs53       \n\t" \
"xvmaddadp          %%vs9, %%vs41, %%vs53       \n\t" \
"xvmaddadp          %%vs10, %%vs42, %%vs53      \n\t" \
"xvmaddadp          %%vs11, %%vs43, %%vs53      \n\t" \
"                                               \n\t" \
"lxv                %%vs34, 1952(%%r7)          \n\t" \
"lxv                %%vs35, 1968(%%r7)          \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs12, %%vs44, %%vs53       \n\t" \
"xvmaddadp         %%vs13, %%vs45, %%vs53       \n\t" \
"xvmaddadp         %%vs14, %%vs46, %%vs53       \n\t" \
"xvmaddadp         %%vs15, %%vs47, %%vs53       \n\t" \
"                                               \n\t" \
"lxv               %%vs36, 1984(%%r7)           \n\t" \
"lxv               %%vs37, 2000(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs16, %%vs40, %%vs54       \n\t" \
"xvmaddadp         %%vs17, %%vs41, %%vs54       \n\t" \
"xvmaddadp         %%vs18, %%vs42, %%vs54       \n\t" \
"xvmaddadp         %%vs19, %%vs43, %%vs54       \n\t" \
"                                               \n\t" \
"lxv               %%vs38, 2016(%%r7)           \n\t" \
"lxv               %%vs39, 2032(%%r7)           \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs20, %%vs44, %%vs54       \n\t" \
"xvmaddadp         %%vs21, %%vs45, %%vs54       \n\t" \
"xvmaddadp         %%vs22, %%vs46, %%vs54       \n\t" \
"xvmaddadp         %%vs23, %%vs47, %%vs54       \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs24, %%vs40, %%vs55       \n\t" \
"xvmaddadp         %%vs25, %%vs41, %%vs55       \n\t" \
"xvmaddadp         %%vs26, %%vs42, %%vs55       \n\t" \
"xvmaddadp         %%vs27, %%vs43, %%vs55       \n\t" \
"                                               \n\t" \
"xvmaddadp         %%vs28, %%vs44, %%vs55       \n\t" \
"xvmaddadp         %%vs29, %%vs45, %%vs55       \n\t" \
"xvmaddadp         %%vs30, %%vs46, %%vs55       \n\t" \
"xvmaddadp         %%vs31, %%vs47, %%vs55       \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"addi              %%r8, %%r8, 512              \n\t" \
"addi              %%r7, %%r7, 2048             \n\t" \
"                                               \n\t" \
"                                               \n\t"





#define LOAD_UPDATE_2 \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs32, %%vs48         \n\t" \
"xvmaddadp        %%vs1, %%vs33, %%vs48         \n\t" \
"xvmaddadp        %%vs2, %%vs34, %%vs48         \n\t" \
"xvmaddadp        %%vs3, %%vs35, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 0(%%r7)               \n\t" \
"lxv              %%vs41, 16(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" \
"xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" \
"xvmaddadp        %%vs6, %%vs38, %%vs48         \n\t" \
"xvmaddadp        %%vs7, %%vs39, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs42, 32(%%r7)              \n\t" \
"lxv              %%vs43, 48(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs32, %%vs49         \n\t" \
"xvmaddadp        %%vs9, %%vs33, %%vs49         \n\t" \
"xvmaddadp        %%vs10, %%vs34, %%vs49        \n\t" \
"xvmaddadp        %%vs11, %%vs35, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs52, 0(%%r8)               \n\t" \
"lxv              %%vs54, 16(%%r8)              \n\t" \
"xxpermdi         %%vs53, %%vs52, %%vs52, 2     \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs36, %%vs49        \n\t" \
"xvmaddadp        %%vs13, %%vs37, %%vs49        \n\t" \
"xvmaddadp        %%vs14, %%vs38, %%vs49        \n\t" \
"xvmaddadp        %%vs15, %%vs39, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs44, 64(%%r7)              \n\t" \
"lxv              %%vs45, 80(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs32, %%vs50        \n\t" \
"xvmaddadp        %%vs17, %%vs33, %%vs50        \n\t" \
"xvmaddadp        %%vs18, %%vs34, %%vs50        \n\t" \
"xvmaddadp        %%vs19, %%vs35, %%vs50        \n\t" \
"                                               \n\t" \
"lxv              %%vs46, 96(%%r7)              \n\t" \
"lxv              %%vs47, 112(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs36, %%vs50        \n\t" \
"xvmaddadp        %%vs21, %%vs37, %%vs50        \n\t" \
"xvmaddadp        %%vs22, %%vs38, %%vs50        \n\t" \
"xvmaddadp        %%vs23, %%vs39, %%vs50        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs32, %%vs51        \n\t" \
"xvmaddadp        %%vs25, %%vs33, %%vs51        \n\t" \
"xvmaddadp        %%vs26, %%vs34, %%vs51        \n\t" \
"xvmaddadp        %%vs27, %%vs35, %%vs51        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs36, %%vs51        \n\t" \
"xvmaddadp        %%vs29, %%vs37, %%vs51        \n\t" \
"xvmaddadp        %%vs30, %%vs38, %%vs51        \n\t" \
"xvmaddadp        %%vs31, %%vs39, %%vs51        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs40, %%vs52         \n\t" \
"xvmaddadp        %%vs1, %%vs41, %%vs52         \n\t" \
"xvmaddadp        %%vs2, %%vs42, %%vs52         \n\t" \
"xvmaddadp        %%vs3, %%vs43, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 32(%%r8)              \n\t" \
"lxv              %%vs50, 48(%%r8)              \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs44, %%vs52         \n\t" \
"xvmaddadp        %%vs5, %%vs45, %%vs52         \n\t" \
"xvmaddadp        %%vs6, %%vs46, %%vs52         \n\t" \
"xvmaddadp        %%vs7, %%vs47, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 128(%%r7)             \n\t" \
"lxv              %%vs33, 144(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs40, %%vs53         \n\t" \
"xvmaddadp        %%vs9, %%vs41, %%vs53         \n\t" \
"xvmaddadp        %%vs10, %%vs42, %%vs53        \n\t" \
"xvmaddadp        %%vs11, %%vs43, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs34, 160(%%r7)             \n\t" \
"lxv              %%vs35, 176(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs44, %%vs53        \n\t" \
"xvmaddadp        %%vs13, %%vs45, %%vs53        \n\t" \
"xvmaddadp        %%vs14, %%vs46, %%vs53        \n\t" \
"xvmaddadp        %%vs15, %%vs47, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 192(%%r7)             \n\t" \
"lxv              %%vs37, 208(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs40, %%vs54        \n\t" \
"xvmaddadp        %%vs17, %%vs41, %%vs54        \n\t" \
"xvmaddadp        %%vs18, %%vs42, %%vs54        \n\t" \
"xvmaddadp        %%vs19, %%vs43, %%vs54        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 224(%%r7)             \n\t" \
"lxv              %%vs39, 240(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs44, %%vs54        \n\t" \
"xvmaddadp        %%vs21, %%vs45, %%vs54        \n\t" \
"xvmaddadp        %%vs22, %%vs46, %%vs54        \n\t" \
"xvmaddadp        %%vs23, %%vs47, %%vs54        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs25, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs26, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs27, %%vs43, %%vs55        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs44, %%vs55        \n\t" \
"xvmaddadp        %%vs29, %%vs45, %%vs55        \n\t" \
"xvmaddadp        %%vs30, %%vs46, %%vs55        \n\t" \
"xvmaddadp        %%vs31, %%vs47, %%vs55        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"addi             %%r8, %%r8, 64                \n\t" \
"addi             %%r7, %%r7, 256               \n\t" \
"                                               \n\t" \
"                                               \n\t" 





#define LOAD_UPDATE_1 \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs32, %%vs48         \n\t" \
"xvmaddadp        %%vs1, %%vs33, %%vs48         \n\t" \
"xvmaddadp        %%vs2, %%vs34, %%vs48         \n\t" \
"xvmaddadp        %%vs3, %%vs35, %%vs48         \n\t" \
"xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" \
"xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" \
"xvmaddadp        %%vs6, %%vs38, %%vs48         \n\t" \
"xvmaddadp        %%vs7, %%vs39, %%vs48         \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs32, %%vs49         \n\t" \
"xvmaddadp        %%vs9, %%vs33, %%vs49         \n\t" \
"xvmaddadp        %%vs10, %%vs34, %%vs49        \n\t" \
"xvmaddadp        %%vs11, %%vs35, %%vs49        \n\t" \
"xvmaddadp        %%vs12, %%vs36, %%vs49        \n\t" \
"xvmaddadp        %%vs13, %%vs37, %%vs49        \n\t" \
"xvmaddadp        %%vs14, %%vs38, %%vs49        \n\t" \
"xvmaddadp        %%vs15, %%vs39, %%vs49        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 0(%%r8)               \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs32, %%vs50        \n\t" \
"xvmaddadp        %%vs17, %%vs33, %%vs50        \n\t" \
"xvmaddadp        %%vs18, %%vs34, %%vs50        \n\t" \
"xvmaddadp        %%vs19, %%vs35, %%vs50        \n\t" \
"xvmaddadp        %%vs20, %%vs36, %%vs50        \n\t" \
"xvmaddadp        %%vs21, %%vs37, %%vs50        \n\t" \
"xvmaddadp        %%vs22, %%vs38, %%vs50        \n\t" \
"xvmaddadp        %%vs23, %%vs39, %%vs50        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs32, %%vs51        \n\t" \
"xvmaddadp        %%vs25, %%vs33, %%vs51        \n\t" \
"xvmaddadp        %%vs26, %%vs34, %%vs51        \n\t" \
"xvmaddadp        %%vs27, %%vs35, %%vs51        \n\t" \
"xvmaddadp        %%vs28, %%vs36, %%vs51        \n\t" \
"xvmaddadp        %%vs29, %%vs37, %%vs51        \n\t" \
"xvmaddadp        %%vs30, %%vs38, %%vs51        \n\t" \
"xvmaddadp        %%vs31, %%vs39, %%vs51        \n\t" \
"                                               \n\t" \
"lxv              %%vs50, 16(%%r8)              \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 0(%%r7)               \n\t" \
"lxv              %%vs33, 16(%%r7)              \n\t" \
"lxv              %%vs34, 32(%%r7)              \n\t" \
"lxv              %%vs35, 48(%%r7)              \n\t" \
"lxv              %%vs36, 64(%%r7)              \n\t" \
"lxv              %%vs37, 80(%%r7)              \n\t" \
"lxv              %%vs38, 96(%%r7)              \n\t" \
"lxv              %%vs39, 112(%%r7)             \n\t" \
"                                               \n\t" \
"addi             %%r8, %%r8, 32                \n\t" \
"addi             %%r7, %%r7, 128               \n\t" \
"                                               \n\t" \
"                                               \n\t" 










#define COL_ADD_TO_C \
"xvadddp          %%vs48, %%vs48, %%vs32   	    \n\t" \
"xvadddp          %%vs49, %%vs49, %%vs33   	    \n\t" \
"xvadddp          %%vs50, %%vs50, %%vs34   	    \n\t" \
"xvadddp          %%vs51, %%vs51, %%vs35        \n\t" \
"xvadddp          %%vs52, %%vs52, %%vs36 	      \n\t" \
"xvadddp          %%vs53, %%vs53, %%vs37    	  \n\t" \
"xvadddp          %%vs54, %%vs54, %%vs38   	    \n\t" \
"xvadddp          %%vs55, %%vs55, %%vs39     	  \n\t" \
"xvadddp          %%vs56, %%vs56, %%vs40   	    \n\t" \
"xvadddp          %%vs57, %%vs57, %%vs41   	    \n\t" \
"xvadddp          %%vs58, %%vs58, %%vs42   	    \n\t" \
"xvadddp          %%vs59, %%vs59, %%vs43        \n\t" \
"xvadddp          %%vs60, %%vs60, %%vs44 	      \n\t" \
"xvadddp          %%vs61, %%vs61, %%vs45    	  \n\t" \
"xvadddp          %%vs62, %%vs62, %%vs46   	    \n\t" \
"xvadddp          %%vs63, %%vs63, %%vs47     	  \n\t" \
"            	                                  \n\t" \
"            	                                  \n\t" 














#define COL_SCALE_BETA \
"xvmuldp         %%vs32, %%vs32, %%vs59         \n\t" \
"xvmuldp         %%vs33, %%vs33, %%vs59         \n\t" \
"xvmuldp         %%vs34, %%vs34, %%vs59         \n\t" \
"xvmuldp         %%vs35, %%vs35, %%vs59         \n\t" \
"xvmuldp         %%vs36, %%vs36, %%vs59         \n\t" \
"xvmuldp         %%vs37, %%vs37, %%vs59         \n\t" \
"xvmuldp         %%vs38, %%vs38, %%vs59         \n\t" \
"xvmuldp         %%vs39, %%vs39, %%vs59         \n\t" \
"xvmuldp         %%vs40, %%vs40, %%vs59         \n\t" \
"xvmuldp         %%vs41, %%vs41, %%vs59         \n\t" \
"xvmuldp         %%vs42, %%vs42, %%vs59         \n\t" \
"xvmuldp         %%vs43, %%vs43, %%vs59         \n\t" \
"xvmuldp         %%vs44, %%vs44, %%vs59         \n\t" \
"xvmuldp         %%vs45, %%vs45, %%vs59         \n\t" \
"xvmuldp         %%vs46, %%vs46, %%vs59         \n\t" \
"xvmuldp         %%vs47, %%vs47, %%vs59         \n\t"  











#define GEN_BETA_SCALE \
"xvmuldp          %%vs32, %%vs32, %%vs59   	    \n\t" \
"xvmuldp          %%vs33, %%vs33, %%vs59   	    \n\t" \
"xvmuldp          %%vs34, %%vs34, %%vs59   	    \n\t" \
"xvmuldp          %%vs35, %%vs35, %%vs59   	    \n\t" \
"xvmuldp          %%vs36, %%vs36, %%vs59   	    \n\t" \
"xvmuldp          %%vs37, %%vs37, %%vs59   	    \n\t" \
"xvmuldp          %%vs38, %%vs38, %%vs59   	    \n\t" \
"xvmuldp          %%vs39, %%vs39, %%vs59   	    \n\t" 









#define GEN_LOAD_C \
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









#define GEN_NEXT_COL_C \
"add             %%r22, %%r22, %%r10            \n\t" \
"add             %%r23, %%r23, %%r10            \n\t" \
"add             %%r24, %%r24, %%r10            \n\t" \
"add             %%r25, %%r25, %%r10            \n\t" \
"add             %%r26, %%r26, %%r10            \n\t" \
"add             %%r27, %%r27, %%r10            \n\t" \
"add             %%r28, %%r28, %%r10            \n\t" \
"add             %%r29, %%r29, %%r10            \n\t"











#define GEN_ADD_STORE \
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
"                                              	\n\t" \
"xxpermdi     %%vs40, %%vs40, %%vs40, 2         \n\t" \
"xxpermdi     %%vs41, %%vs41, %%vs41, 2         \n\t" \
"xxpermdi     %%vs42, %%vs42, %%vs42, 2         \n\t" \
"xxpermdi     %%vs43, %%vs43, %%vs43, 2         \n\t" \
"xxpermdi     %%vs44, %%vs44, %%vs44, 2         \n\t" \
"xxpermdi     %%vs45, %%vs45, %%vs45, 2         \n\t" \
"xxpermdi     %%vs46, %%vs46, %%vs46, 2         \n\t" \
"xxpermdi     %%vs47, %%vs47, %%vs47, 2         \n\t" \
"                                              	\n\t" \
"stxsdx       %%vs40, 0, %%r22                  \n\t" \
"stxsdx       %%vs41, 0, %%r23                  \n\t" \
"stxsdx       %%vs42, 0, %%r24                  \n\t" \
"stxsdx       %%vs43, 0, %%r25                  \n\t" \
"stxsdx       %%vs44, 0, %%r26                  \n\t" \
"stxsdx       %%vs45, 0, %%r27                  \n\t" \
"stxsdx       %%vs46, 0, %%r28                  \n\t" \
"stxsdx       %%vs47, 0, %%r29                  \n\t" \







#define GEN_LOAD_SCALE \
  GEN_LOAD_C   \
  GEN_BETA_SCALE











#define PERMUTE_ALL_VREG \
  "xxpermdi   %%vs32, %%vs8, %%vs0, 1           \n\t" \
  "xxpermdi   %%vs33, %%vs9, %%vs1, 1           \n\t" \
  "xxpermdi   %%vs34, %%vs10, %%vs2, 1          \n\t" \
  "xxpermdi   %%vs35, %%vs11, %%vs3, 1          \n\t" \
  "xxpermdi   %%vs36, %%vs12, %%vs4, 1          \n\t" \
  "xxpermdi   %%vs37, %%vs13, %%vs5, 1          \n\t" \
  "xxpermdi   %%vs38, %%vs14, %%vs6, 1          \n\t" \
  "xxpermdi   %%vs39, %%vs15, %%vs7, 1          \n\t" \
  "xxpermdi   %%vs40, %%vs0, %%vs8, 1           \n\t" \
  "xxpermdi   %%vs41, %%vs1, %%vs9, 1           \n\t" \
  "xxpermdi   %%vs42, %%vs2, %%vs10, 1          \n\t" \
  "xxpermdi   %%vs43, %%vs3, %%vs11, 1          \n\t" \
  "xxpermdi   %%vs44, %%vs4, %%vs12, 1          \n\t" \
  "xxpermdi   %%vs45, %%vs5, %%vs13, 1          \n\t" \
  "xxpermdi   %%vs46, %%vs6, %%vs14, 1          \n\t" \
  "xxpermdi   %%vs47, %%vs7, %%vs15, 1          \n\t" \
  "xxpermdi   %%vs48, %%vs24, %%vs16, 1         \n\t" \
  "xxpermdi   %%vs49, %%vs25, %%vs17, 1         \n\t" \
  "xxpermdi   %%vs50, %%vs26, %%vs18, 1         \n\t" \
  "xxpermdi   %%vs51, %%vs27, %%vs19, 1         \n\t" \
  "xxpermdi   %%vs52, %%vs28, %%vs20, 1         \n\t" \
  "xxpermdi   %%vs53, %%vs29, %%vs21, 1         \n\t" \
  "xxpermdi   %%vs54, %%vs30, %%vs22, 1         \n\t" \
  "xxpermdi   %%vs55, %%vs31, %%vs23, 1         \n\t" \
  "xxpermdi   %%vs56, %%vs16, %%vs24, 1         \n\t" \
  "xxpermdi   %%vs57, %%vs17, %%vs25, 1         \n\t" \
  "xxpermdi   %%vs58, %%vs18, %%vs26, 1         \n\t" \
  "xxpermdi   %%vs59, %%vs19, %%vs27, 1         \n\t" \
  "xxpermdi   %%vs60, %%vs20, %%vs28, 1         \n\t" \
  "xxpermdi   %%vs61, %%vs21, %%vs29, 1         \n\t" \
  "xxpermdi   %%vs62, %%vs22, %%vs30, 1         \n\t" \
  "xxpermdi   %%vs63, %%vs23, %%vs31, 1         \n\t"











#define COL_BZ_STORE_C \
  "stxv              %%vs32, 0(%%r22)           \n\t" \
  "stxv              %%vs33, 16(%%r22)          \n\t" \
  "stxv              %%vs34, 32(%%r22)          \n\t" \
  "stxv              %%vs35, 48(%%r22)          \n\t" \
  "stxv              %%vs36, 64(%%r22)          \n\t" \
  "stxv              %%vs37, 80(%%r22)          \n\t" \
  "stxv              %%vs38, 96(%%r22)          \n\t" \
  "stxv              %%vs39, 112(%%r22)         \n\t" \
  "stxv              %%vs40, 0(%%r23)           \n\t" \
  "stxv              %%vs41, 16(%%r23)          \n\t" \
  "stxv              %%vs42, 32(%%r23)          \n\t" \
  "stxv              %%vs43, 48(%%r23)          \n\t" \
  "stxv              %%vs44, 64(%%r23)          \n\t" \
  "stxv              %%vs45, 80(%%r23)          \n\t" \
  "stxv              %%vs46, 96(%%r23)          \n\t" \
  "stxv              %%vs47, 112(%%r23)         \n\t" \
  "stxv              %%vs48, 0(%%r24)           \n\t" \
  "stxv              %%vs49, 16(%%r24)          \n\t" \
  "stxv              %%vs50, 32(%%r24)          \n\t" \
  "stxv              %%vs51, 48(%%r24)          \n\t" \
  "stxv              %%vs52, 64(%%r24)          \n\t" \
  "stxv              %%vs53, 80(%%r24)          \n\t" \
  "stxv              %%vs54, 96(%%r24)          \n\t" \
  "stxv              %%vs55, 112(%%r24)         \n\t" \
  "stxv              %%vs56, 0(%%r25)           \n\t" \
  "stxv              %%vs57, 16(%%r25)          \n\t" \
  "stxv              %%vs58, 32(%%r25)          \n\t" \
  "stxv              %%vs59, 48(%%r25)          \n\t" \
  "stxv              %%vs60, 64(%%r25)          \n\t" \
  "stxv              %%vs61, 80(%%r25)          \n\t" \
  "stxv              %%vs62, 96(%%r25)          \n\t" \
  "stxv              %%vs63, 112(%%r25)         \n\t" 



