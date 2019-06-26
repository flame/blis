	.file	"testasm.c"
	.machine power8
	.abiversion 2
	.section	".text"
	.section	.rodata
	.align 3
.LC0:
	.string	"hello world"
	.section	".text"
	.align 2
	.globl main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
.LCF0:
0:	addis 2,12,.TOC.-.LCF0@ha
	addi 2,2,.TOC.-.LCF0@l
	.localentry	main,.-main
	mflr 0
	std 0,16(1)
	std 31,-8(1)
	stdu 1,-48(1)
	.cfi_def_cfa_offset 48
	.cfi_offset 65, 16
	.cfi_offset 31, -8
	mr 31,1
	.cfi_def_cfa_register 31
	addis 3,2,.LC0@toc@ha
	addi 3,3,.LC0@toc@l
	bl puts
	nop
	li 9,0
	mr 3,9
	addi 1,31,48
	.cfi_def_cfa 1, 0
	ld 0,16(1)
	mtlr 0
	ld 31,-8(1)
	blr
	.long 0
	.byte 0,0,0,1,128,1,0,1
	.cfi_endproc
.LFE2:
	.size	main,.-main
	.ident	"GCC: (GNU) 8.2.0"
	.section	.note.GNU-stack,"",@progbits
