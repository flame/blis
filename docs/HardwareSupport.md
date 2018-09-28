## Introduction

This wiki is intended to track the support for various hardware types within the BLIS framework source distribution.

We apologize if this wiki falls out of date. For the latest support, we recommend peeking inside of the relevant sub-configuration (specifically, in the `bli_cntx_init_<configname>.c` file) and looking at which kernels are registered. You may also contact the [blis-devel](http://groups.google.com/group/blis-devel) mailing list.


## Level-3 micro-kernels

The following table lists architectures for which there exist optimized level-3 micro-kernels, which micro-kernels are optimized, the name of the author or maintainer, and the current status of the micro-kernels.

A few remarks / reminders:
  * Optimizing only the [gemm micro-kernel](KernelsHowTo.md#gemm-micro-kernel) will result in optimal performance for all [level-3 operations](BLISTypedAPI#level-3-operations) except `trsm` (which will typically achieve 60 - 80% of attainable peak performance).
  * The [trsm](BLISTypedAPI#trsm) operation needs the [gemmtrsm micro-kernel(s)](KernelsHowTo.md#gemmtrsm-micro-kernels), in addition to the aforementioned [gemm micro-kernel](KernelsHowTo.md#gemm-micro-kernel), in order reach optimal performance.
  * Induced complex (1m) implementations are employed in all situations where the real domain [gemm micro-kernel](KernelsHowTo.md#gemm-micro-kernel) of the corresponding precision is available. Please see our [ACM TOMS article on the 1m method](https://github.com/flame/blis#citations) for more info on this topic.
  * Some microarchitectures use the same sub-configuration. This is not a typo. For example, Haswell and Broadwell systems as well as "desktop" (non-server) versions of Skylake, Kabylake, and Coffeelake all use the `haswell` sub-configuration and the kernels registered therein.
  * Remember that you (usually) don't have to choose your sub-configuration manually! Instead, you can always request configure-time hardware detection via `./configure auto`. This will defer to internal logic (based on CPUID for x86_64 systems) that will attempt to choose the appropriate sub-configuration automatically.

| Vendor/Microarchitecture             | BLIS sub-configuration | `gemm` | `gemmtrsm` |
|:-------------------------------------|:-----------------------|:-------|:-----------|
| AMD Bulldozer (AVX/FMA4)             | `bulldozer`            | `sdcz` |            |
| AMD Piledriver (AVX/FMA3)            | `piledriver`           | `sdcz` |            |
| AMD Steamroller (AVX/FMA3)           | `steamroller`          | `sdcz` |            |
| AMD Excavator (AVX/FMA3)             | `excavator`            | `sdcz` |            |
| AMD Zen (AVX/FMA3)                   | `zen`                  | `sdcz` |  `sd`      |
| Intel Core2 (SSE3)                   | `penryn`               | `sd`   |  `d`       |
| Intel Sandy/Ivy Bridge (AVX/FMA3)    | `sandybridge`          | `sdcz` |            |
| Intel Haswell, Broadwell (AVX/FMA3)  | `haswell`              | `sdcz` |  `sd`      |
| Intel Sky/Kaby/Coffeelake (AVX/FMA3) | `haswell`              | `sdcz` |  `sd`      |
| Intel Knights Landing (AVX-512/FMA3) | `knl`                  | `sd`   |            |
| Intel SkylakeX (AVX-512/FMA3)        | `skx`                  | `sd`   |            |
| ARMv7 Cortex-A9/A15 (NEON)           | `cortex-a9,-a15`       | `sd`   |            |
| ARMv8 Cortex-A57 (NEON)              | `cortex-a57`           | `sd`   |            |
| IBM Blue Gene/Q (QPX int)            | `bgq`                  |  `d`   |            |
| IBM Power7 (QPX int)                 | `power7`               |  `d`   |            |
| template (C99)                       | `template`             | `sdcz` | `sdcz`     |

## Level-1f kernels

Not yet written. Please see the relevant sub-configuration (`bli_cntx_init_<configname>.c`) to determine which kernels are implemented/registered.

## Level-1v kernels

Not yet written. Please see the relevant sub-configuration (`bli_cntx_init_<configname>.c`) to determine which kernels are implemented/registered.
