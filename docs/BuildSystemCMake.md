# Building BLIS with CMake

BLIS can be built with CMake (≥ 3.20) instead of `configure` + `make`. The
CMake build ports the full feature surface: configuration families with
hardware auto-detection, per-kernel-set ISA compilation, threading models
(single/pthreads/OpenMP/HPX), BLAS and CBLAS compatibility layers, addons,
sandboxes, static and shared libraries, the testsuite, the reference-BLAS
(blastest) tests, and installation with pkg-config + `find_package` support.

## Quick start

```sh
cmake -S . -B out -DBLIS_CONFIG=auto -DBLIS_THREADING=openmp
cmake --build out -j
ctest --test-dir out -L fast          # fast correctness check
ctest --test-dir out                  # full testsuite + BLAS tests
ctest --test-dir out -L blastest      # netlib BLAS tests only
cmake --install out --prefix /opt/blis
```

`cmake --build out --target uninstall` removes a previous installation.
It runs the four `uninstall-*` targets and also removes `lib/cmake/blis`
and `blis.pc`.

## Option mapping (`configure` → CMake)

| configure                    | CMake cache variable              | default |
|------------------------------|-----------------------------------|---------|
| `[config name]` / `auto`     | `BLIS_CONFIG=<name>` (`z15`, `zarch` for s390x) | `auto`  |
| `-t {openmp,pthreads,single,hpx,auto[,...]}` | `BLIS_THREADING="openmp;pthreads"`      | `single` |
| `-p PREFIX` / `--prefix`     | `CMAKE_INSTALL_PREFIX`            | system default |
| `--libdir/--includedir/...`  | `CMAKE_INSTALL_LIBDIR/...` (GNUInstallDirs) | |
| `--exec-prefix`              | `BLIS_INSTALL_EXEC_PREFIX`        | empty |
| | Libraries install to `${BLIS_INSTALL_EXEC_PREFIX}/${CMAKE_INSTALL_LIBDIR}`. |
| | `blis.pc`, `config.mk`, `uninstall-old`, and Darwin install name all use this dir.
| `--sharedir`                 | `BLIS_INSTALL_SHAREDIR`           | empty   |
| `make install` layout        | `BLIS_INSTALL_LAYOUT={make,gnu}`  | `make`  |
| `--enable/disable-static`    | `BLIS_ENABLE_STATIC`              | `ON`    |
| `--enable/disable-shared`    | `BLIS_ENABLE_SHARED`              | `ON`    |
| `-e {public,all}`            | `BLIS_EXPORT_SHARED`              | `public`|
| `--enable/disable-rpath`     | `BLIS_DARWIN_RPATH` (macOS)       | `OFF`   |
| `--int-size={32,64}`         | `BLIS_INT_SIZE`                   | `auto`  |
| `-b {32,64}`                 | `BLIS_BLAS_INT_SIZE`              | `32`    |
| `--enable/disable-blas`      | `BLIS_ENABLE_BLAS`                | `ON`    |
| `--enable/disable-cblas`     | `BLIS_ENABLE_CBLAS`               | `OFF`   |
| `--disable-sup-handling`     | `BLIS_ENABLE_SUP_HANDLING=OFF`    | `ON`    |
| `--enable/disable-trsm-preinversion` | `BLIS_ENABLE_TRSM_PREINVERSION` | `ON`    |
| `--enable/disable-mixed-precision-norm` | `BLIS_ENABLE_MIXED_PRECISION_NORM` | `ON`  |
| `--enable-lapack-compat`     | `BLIS_ENABLE_LAPACK_COMPAT`       | `OFF`   |
| `--enable-scalapack-compat`  | `BLIS_ENABLE_SCALAPACK_COMPAT`    | `OFF`   |
| `--omit-symbols=a,b`         | `BLIS_OMIT_SYMBOLS="a,b"`         | empty   |
| `-d {noopt,opt,sde}`         | `BLIS_DEBUG_TYPE=...`             | `off`   |
| `--enable-asan`              | `BLIS_ENABLE_ASAN`                | `OFF`   |
| `--enable-mem-tracing`       | `BLIS_ENABLE_MEM_TRACING`         | `OFF`   |
| `--disable-pba-pools`        | `BLIS_ENABLE_PBA_POOLS=OFF`       | `ON`    |
| `--disable-sba-pools`        | `BLIS_ENABLE_SBA_POOLS=OFF`       | `ON`    |
| `--with/without-memkind`     | `BLIS_ENABLE_MEMKIND=on/off/auto` | `auto`  |
| `--complex-return={gnu,intel}` | `BLIS_COMPLEX_RETURN`           | `auto`  |
| `-a NAME`                    | `BLIS_ENABLE_ADDONS="a;b"`        | empty   |
| `-s NAME`                    | `BLIS_SANDBOX=name`               | empty   |
| `--enable-amd-frame-tweaks`  | `BLIS_ENABLE_AMD_FRAME_TWEAKS`    | `OFF`   |
| `--enable/disable-arg-max-hack` | `BLIS_ENABLE_ARG_MAX_HACK`     | `OFF`   |
| `--enable/disable-verbose-make` | `BLIS_ENABLE_VERBOSE_MAKE`     | `OFF`   |
| `--show-config-list`         | `BLIS_SHOW_CONFIG_LIST`         | `OFF`   |
| `--disable-system/-tls`      | `BLIS_ENABLE_SYSTEM/BLIS_ENABLE_TLS` | `ON` |
| `-r {slab,rr,tlb}`           | `BLIS_THREAD_JRIR_PARTITION`      | `slab`  |
| `--force-version=x.y.z`      | `BLIS_FORCE_VERSION`              | unset   |
| `CC/CFLAGS/LDFLAGS` env      | `CMAKE_C_COMPILER`, `CMAKE_C_FLAGS` (env-seeded), env `LDFLAGS` | |
| `AR`, `RANLIB`, `PYTHON` env | `CMAKE_AR`, `CMAKE_RANLIB`, `Python3_EXECUTABLE` (seeded from the environment at the first configure with arguments. The raw value stays cached on reconfigure. A new environment value replaces it. An explicit `-D` value wins and drops the cached arguments) | |
| `TESTSUITE_WRAPPER` env      | `TESTSUITE_WRAPPER` (cache entry, seeded from env, replaces `CMAKE_CROSSCOMPILING_EMULATOR` for the test commands) | empty |
| `make checkblis[-fast,-md,-salt]` | `ctest -R blis-test-{full,fast,mixed,salt}` | |
| `make checkblas`             | `ctest -L blastest`               |         |
| `make showconfig`            | `showconfig` target               |         |
| `make check`                 | `check` (ctest `fast` and `blastest` labels) |         |
| `make blastest-bin`          | `blastest-bin` (f2c, all BLAT drivers) |        |
| `make uninstall-old`         | `uninstall-old` (-libs, -symlinks, -headers) |     |
| `make install-{headers,helper-headers,libs,lib-symlinks,share}` | same names as targets; see the install components below | |
| `make uninstall-{headers,libs,lib-symlinks,share}` | same names as targets | |
| other make target names      | one CMake target each; see the table below | |

Notes on deliberate differences from the make build:

- Per-flag compilation semantics are identical (compile lines were verified
  file-for-file against a `make V=1` reference build). Environment `CFLAGS`
  is applied via `CMAKE_C_FLAGS` (CMake's own mechanism, seeded from the
  same environment), not re-injected as the make system's "preset".
- The old Makefile's helper targets map as follows: `make checkblis*` →
  `ctest -R blis-test-*`, `make checkblas` → `ctest -L blastest`,
  `make symbols` → `ninja symbols` (or the `symbols` build target),
  `make changelog` → `changelog` target, `make showconfig` → configure-time
  summary, `make distclean` → `distclean` target,
  `make uninstall` → `uninstall` target, `make install INSTALL_HH=yes` →
  `-DBLIS_INSTALL_VENDOR_CPP_HEADERS=ON`.
- testsuite/, blastest/ and examples/ additionally work as *standalone*
  CMake projects against an installed BLIS (`cmake -S <dir> -B out
  -DCMAKE_PREFIX_PATH=<prefix>`), replacing their share/blis-based
  standalone Makefiles.
- Out-of-tree *plugins* are ordinary CMake projects that
  `find_package(blis)` and use the installed helper functions
  (`blis_plugin_frame_flags`, `blis_plugin_kernel_flags`,
  `blis_plugin_refkern_flags`); see docs/PluginHowTo.md. The install puts
  the make-based `configure-plugin` fragments under `<prefix>/share/blis`,
  with `config.mk`, the gen-make-frags scripts, the plugin templates, and
  the per-config `make_defs.mk` and kernel files.
- `-lrt` is linked on Linux/glibc (skipped on Android/Bionic); memkind is
  auto-detected, forceable with `BLIS_ENABLE_MEMKIND=on`.
- HPX threading enables C++ and requires `pkg-config hpx_component`.
  `blis.pc` has the HPX `LDFLAGS` from `pkg-config`.
- The `LDFLAGS` from `pkg-config` (HPX) do not change: `-framework X`
  stays together, and `-pthread` stays `-pthread` where `make` writes
  it raw.
- A config that calls `$(error ...)` in `make_defs.mk` stops at build start.
  The `blis_defer_error()` helper does this.
- `cmake --build DIR --target install` builds the code, then installs it.
- `cmake --install DIR` installs the built code.
- `cmake --install DIR --component X` installs one component. The components
  are `headers`, `helper-headers`, `libs`, `lib-symlinks`, `share` and
  `cmake-package`. The target `install-X` does the same for the first five
  names, after it builds the needed files. `install-headers` also installs
  `helper-headers`.
- The installed CMake package is relocatable. It works after `--prefix`,
  `DESTDIR` and `BLIS_INSTALL_EXEC_PREFIX` change the install location.
- The uninstall targets are `uninstall-headers`, `uninstall-libs`,
  `uninstall-lib-symlinks` and `uninstall-share`. They get the file names
  from the install layout. Each `uninstall` and `uninstall-*` target
  removes files from the prefix and the `DESTDIR` of the last install
  from the build dir. The install record holds the prefix and `DESTDIR`
  as absolute paths. No uninstall path resolves against the current directory.
  Before the first install, uninstall uses the configured prefix and
  the `DESTDIR` environment, as `make uninstall` does. It stops with
  an error when `DESTDIR` is relative.
  One build dir records only its last install.
- With HPX threading, the C++ sources of `frame/` build with the HPX
  compile flags. The HPX include directories apply to all sources.
- The first configure uses `AR`, `RANLIB` and `PYTHON` from the environment,
  as `configure` does. Each value can have arguments. The first word is
  the program. The other words are its arguments. `config.mk` records
  the raw string. A missing program named by `AR`, `RANLIB`
  or `PYTHON` stops the configure step with the message of `configure`.
  An explicit `-DCMAKE_AR`, `-DCMAKE_RANLIB` or `-DPython3_EXECUTABLE`
  wins and skips the check. Do not add arguments to it.
- `TESTSUITE_WRAPPER` replaces `CMAKE_CROSSCOMPILING_EMULATOR`. Each test has
  one launcher.
- With `BLIS_IS_MSVC` (cl, clang-cl or clang targeting windows-msvc) the
  DLL is `libblis.<major>.dll` and its import library is `libblis.lib`.
  The static library is `libblis_static.lib` when the shared library also
  builds, and `libblis.lib` in a static-only build.

## Make targets as CMake targets

Each make target name is also a CMake target. Build it with
`cmake --build DIR --target NAME`.

| make targets | CMake behavior |
|--------------|----------------|
| `libblis`, `libs` | build the static and shared libraries |
| `flat-header`, `flat-cblas-header` | make `include/blis.h` (and `cblas.h` when CBLAS is `ON`) |
| `check-env`, `check-env-mk`, `check-env-fragments`, `check-env-make-defs` | do nothing, because the configure step does the checks |
| `testsuite-bin`, `blastest-f2c` | build `test_libblis.x`, `libf2c.a` |
| `testsuite-run[-fast,-md,-salt]` | run ctest on the label `full`, `fast`, `mixed` or `salt` |
| `testsuite`, `testblis`, `checkblis` | run `testsuite-run` |
| `testblis-{fast,md,salt}`, `checkblis-{fast,md,salt}` | run `testsuite-run-{fast,md,salt}` |
| `blastest-run`, `testblas`, `checkblas` | run ctest on the label `blastest` |
| `run-DRIVER` (`run-sblat1` to `run-zblat3`) | run one blastest driver |
| `cleanh`, `cleanlib`, `cleanmk` | remove the flat headers, the objects and libraries, and nothing (CMake writes no fragments) |
| `cleanblastest`, `cleanblistest`, `cleantest` | remove the test objects and binaries |
| `distclean` | run the clean targets, then remove the configure outputs, `lib/` and `include/` |

Known differences from make:

- `cleanblistest` keeps the ctest `Testing/` directory, where the test output is.
- `cleanlib` also removes the soname links of the shared library.
- After `distclean`, the next build runs the configure step again. Make needs
  a manual `./configure`.
- If `BUILD_TESTING` is `OFF` or blastest is off, the test targets stop
  with an error that names the missing target.
- With Ninja, `cmake --build DIR --target help` lists only the top-level
  targets. `ninja -C DIR -t targets all` lists all names. All names build
  with `--target`.
- For an `hpx` build with static libraries off, `make` writes build-tree
  rpaths with an empty base path (`-Wl,-rpath,'/lib/haswell'`) into
  `Libs.private`. The CMake `blis.pc` does not.
