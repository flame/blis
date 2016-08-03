#ifndef BLIS_AVX512_MACROS_H
#define BLIS_AVX512_MACROS_H

//
// Assembly macros to make AVX-512 with AT&T syntax somewhat less painful
//

#define COMMENT_BEGIN "#"
#define COMMENT_END

#define STRINGIFY(...) #__VA_ARGS__
#define ASM(...) STRINGIFY(__VA_ARGS__) "\n\t"
#define LABEL(label) STRINGIFY(label) ":\n\t"

#define XMM(x) %% xmm##x
#define YMM(x) %% ymm##x
#define ZMM(x) %% zmm##x
#define EAX %%eax
#define EBX %%ebx
#define ECX %%ecx
#define EDX %%edx
#define EBP %%ebp
#define EDI %%edi
#define ESI %%esi
#define RAX %%rax
#define RBX %%rbx
#define RCX %%rcx
#define RDX %%rdx
#define RBP %%rbp
#define RDI %%rdi
#define RSI %%rsi
#define K(x) %% k##x
#define R(x) %% r##x
#define R8 %%r8
#define R9 %%r9
#define R10 %%r10
#define R11 %%r11
#define R12 %%r12
#define R13 %%r13
#define R14 %%r14
#define R15 %%r15
#define RD(x) %% r##x##d
#define R8D %%r8d
#define R9D %%r9d
#define R10D %%r10d
#define R11D %%r11d
#define R12D %%r12d
#define R13D %%r13d
#define R14D %%r14d
#define R15D %%r15d
#define IMM(x) $##x
#define VAR(x) %[x]

#define MEM_4(reg,off,scale,disp) disp(reg,off,scale)
#define MEM_3(reg,off,scale) (reg,off,scale)
#define MEM_2(reg,disp) disp(reg)
#define MEM_1(reg) (reg)

#define MEM_1TO8_4(reg,off,scale,disp) MEM(reg,off,scale,disp) %{1to8%}
#define MEM_1TO8_3(reg,off,scale) MEM(reg,off,scale) %{1to8%}
#define MEM_1TO8_2(reg,disp) MEM(reg,disp) %{1to8%}
#define MEM_1TO8_1(reg) MEM(reg) %{1to8%}

#define MEM_1TO16_4(reg,off,scale,disp) MEM(reg,off,scale,disp) %{1to16%}
#define MEM_1TO16_3(reg,off,scale) MEM(reg,off,scale) %{1to16%}
#define MEM_1TO16_2(reg,disp) MEM(reg,disp) %{1to16%}
#define MEM_1TO16_1(reg) MEM(reg) %{1to16%}

#define GET_MACRO(_1,_2,_3,_4,NAME,...) NAME
#define MEM(...) GET_MACRO(__VA_ARGS__,MEM_4,MEM_3,MEM_2,MEM_1)(__VA_ARGS__)
#define MEM_1TO8(...) GET_MACRO(__VA_ARGS__,MEM_1TO8_4,MEM_1TO8_3,MEM_1TO8_2,MEM_1TO8_1)(__VA_ARGS__)
#define MEM_1TO16(...) GET_MACRO(__VA_ARGS__,MEM_1TO16_4,MEM_1TO16_3,MEM_1TO16_2,MEM_1TO16_1)(__VA_ARGS__)

#define MASK_K(n) %{%% k##n %}
#define MASK_KZ(n) %{%% k##n %} %{z%}
#define KMOV(to,from) ASM(kmovw from, to)
#define JKNZD(kreg,label) \
    ASM(kortestw kreg, kreg) \
    ASM(jnz label)
#define KXNORW(_0, _1, _2) ASM(kxnorw _2, _1, _0)
#define KSHIFTRW(_0, _1, _2) ASM(kshiftrw _2, _1, _0)

#define ALIGN16 ASM(.p2align 4)
#define ALIGN32 ASM(.p2align 5)
#define RDTSC ASM(rdstc)
#define MOV(_0, _1) ASM(mov _1, _0)
#define MOVD(_0, _1) ASM(movd _1, _0)
#define MOVL(_0, _1) ASM(movl _1, _0)
#define MOVQ(_0, _1) ASM(movq _1, _0)
#define VMOVD(_0, _1) ASM(vmovd _1, _0)
#define VMOVQ(_0, _1) ASM(vmovq _1, _0)
#define CMP(_0, _1) ASM(cmp _1, _0)
#define AND(_0, _1) ASM(and _1, _0)
#define ADD(_0, _1) ASM(add _1, _0)
#define SUB(_0, _1) ASM(sub _1, _0)
#define SAL(_0, _1) ASM(sal _1, _0)
#define SHLX(_0, _1, _2) ASM(shlx _2, _1, _0)
#define SAR(_0, _1) ASM(sar _1, _0)
#define SAL1(_0) ASM(sal _0)
#define SAR1(_0) ASM(sar _0)
#define LEA(_0, _1) ASM(lea _1, _0)
#define TEST(_0, _1) ASM(test _1, _0)
//#define DEC(_0) ASM(dec _0)
#define DEC(_0) SUB(_0, IMM(1))
#define JLE(_0) ASM(jle _0)
#define JNZ(_0) ASM(jnz _0)
#define JZ(_0) ASM(jz _0)
#define JNE(_0) ASM(jne _0)
#define JE(_0) ASM(je _0)
#define JNC(_0) ASM(jnc _0)
#define JC(_0) ASM(jc _0)
#define JMP(_0) ASM(jmp _0)
#define VGATHERDPS(_0, _1) ASM(vgatherdps _1, _0)
#define VSCATTERDPS(_0, _1) ASM(vscatterdps _1, _0)
#define VGATHERDPD(_0, _1) ASM(vgatherdpd _1, _0)
#define VSCATTERDPD(_0, _1) ASM(vscatterdpd _1, _0)
#define VMULPS(_0, _1, _2) ASM(vmulps _2, _1, _0)
#define VMULPD(_0, _1, _2) ASM(vmulpd _2, _1, _0)
#define VPMULLD(_0, _1, _2) ASM(vpmulld _2, _1, _0)
#define VPADDD(_0, _1, _2) ASM(vpaddd _2, _1, _0)
#define VPSLLD(_0, _1, _2) ASM(vpslld _2, _1, _0)
#define VPXORD(_0, _1, _2) ASM(vpxord _2, _1, _0)
#define VFMADD132PS(_0, _1, _2) ASM(vfmadd132ps _2, _1, _0)
#define VFMADD213PS(_0, _1, _2) ASM(vfmadd213ps _2, _1, _0)
#define VFMADD231PS(_0, _1, _2) ASM(vfmadd231ps _2, _1, _0)
#define VFMADD132PD(_0, _1, _2) ASM(vfmadd132pd _2, _1, _0)
#define VFMADD213PD(_0, _1, _2) ASM(vfmadd213pd _2, _1, _0)
#define VFMADD231PD(_0, _1, _2) ASM(vfmadd231pd _2, _1, _0)
#define VMOVAPS(_0, _1) ASM(vmovaps _1, _0)
#define VMOVUPS(_0, _1) ASM(vmovups _1, _0)
#define VMOVAPD(_0, _1) ASM(vmovapd _1, _0)
#define VMOVUPD(_0, _1) ASM(vmovupd _1, _0)
#define VBROADCASTSS(_0, _1) ASM(vbroadcastss _1, _0)
#define VBROADCASTSD(_0, _1) ASM(vbroadcastsd _1, _0)
#define VPBROADCASTD(_0, _1) ASM(vpbroadcastd _1, _0)
#define VPBROADCASTQ(_0, _1) ASM(vpbroadcastq _1, _0)
#define VBROADCASTF64X4(_0, _1) ASM(vbroadcastf64x4 _1, _0)
#define VINSERTF64X4(_0, _1, _2, _3) ASM(vinsertf64x4 _3, _2, _1, _0)
#define VUNPCKLPD(_0, _1, _2) ASM(vunpcklpd _2, _1, _0)
#define VUNPCKHPD(_0, _1, _2) ASM(vunpckhpd _2, _1, _0)
#define VSHUFF64X2(_0, _1, _2, _3) ASM(vshuff64x2 _3, _2, _1, _0)
#define PREFETCH(LEVEL,ADDRESS) ASM(prefetcht##LEVEL ADDRESS)
#define PREFETCHW0(ADDRESS) ASM(prefetchw ADDRESS)
#define PREFETCHW1(ADDRESS) ASM(prefetchwt1 ADDRESS)
#define VGATHERPFDPS(LEVEL,ADDRESS) ASM(vgatherpf##LEVEL##dps ADDRESS)
#define VSCATTERPFDPS(LEVEL,ADDRESS) ASM(vscatterpf##LEVEL##dps ADDRESS)
#define VGATHERPFDPD(LEVEL,ADDRESS) ASM(vgatherpf##LEVEL##dpd ADDRESS)
#define VSCATTERPFDPD(LEVEL,ADDRESS) ASM(vscatterpf##LEVEL##dpd ADDRESS)

#endif
