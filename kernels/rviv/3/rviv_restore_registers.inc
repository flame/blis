#if RISCV_SIZE == 64
    ld s7, 56(sp)
    ld s6, 48(sp)
    ld s5, 40(sp)
    ld s4, 32(sp)
    ld s3, 24(sp)
    ld s2, 16(sp)
    ld s1,  8(sp)
    ld s0,  0(sp)
#else
    lw s7, 56(sp)
    lw s6, 48(sp)
    lw s5, 40(sp)
    lw s4, 32(sp)
    lw s3, 24(sp)
    lw s2, 16(sp)
    lw s1,  8(sp)
    lw s0,  0(sp)
#endif

    addi sp,sp,64
