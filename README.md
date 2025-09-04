# AOCL-BLAS library

AOCL-BLAS is AMD's optimized version of BLAS targeted for AMD EPYC and Ryzen CPUs. It is developed as a forked version of BLIS (https://github.com/flame/blis), which is developed by members of the [Science of High-Performance Computing](http://shpc.oden.utexas.edu/) (SHPC) group in the [Institute for Computational Engineering and Sciences](https://www.oden.utexas.edu/) at [The University of Texas at Austin](https://www.utexas.edu/) and other collaborators (including AMD). All known features and functionalities of BLIS are retained and supported in AOCL-BLAS library. AOCL-BLAS is regularly updated with the improvements from the upstream repository.

AOCL BLAS is optimized with SSE2, AVX2, AVX512 instruction sets which would be enabled based on the target Zen architecture using the dynamic dispatch feature. All prominent Level 3, Level 2 and Level 1 APIs are designed and optimized for specific paths targeting different size spectrums e.g., Small, Medium and Large sizes. These algorithms are designed and customized to exploit the architectural improvements of the target platform.

For detailed instructions on how to configure, build, install, and link against AOCL-BLAS on AMD CPUs, please refer to the AOCL User Guide located on AMD developer [portal](https://www.amd.com/en/developer/aocl.html).

The upstream repository (https://github.com/flame/blis) contains further information on BLIS, including background information on BLIS design, usage examples, and a complete BLIS API reference.

AOCL-BLAS is developed and maintained by AMD. You can contact us on the email-id toolchainsupport@amd.com. You can also raise any issue/suggestion on the git-hub repository at https://github.com/amd/blis/issues.
