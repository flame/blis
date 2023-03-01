#if __riscv_xlen >= 64
#if __riscv_vector
rv64iv
#else
rv64i
#endif
#elif __riscv_xlen >= 32
#if __riscv_vector
rv32iv
#else
rv32i
#endif
#else
generic
#endif
