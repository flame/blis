    addi sp,sp,-64

#if RISCV_SIZE == 64
    sd s7, 56(sp)
    sd s6, 48(sp)
    sd s5, 40(sp)
    sd s4, 32(sp)
    sd s3, 24(sp)
    sd s2, 16(sp)
    sd s1,  8(sp)
    sd s0,  0(sp)
#else
    sw s7, 56(sp)
    sw s6, 48(sp)
    sw s5, 40(sp)
    sw s4, 32(sp)
    sw s3, 24(sp)
    sw s2, 16(sp)
    sw s1,  8(sp)
    sw s0,  0(sp)
#endif
