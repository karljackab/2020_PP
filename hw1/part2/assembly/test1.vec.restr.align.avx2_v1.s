	.text
	.file	"test1.cpp"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function _Z5test1PfS_S_i
.LCPI0_0:
	.quad	4472406533629990549     # double 1.0000000000000001E-9
	.text
	.globl	_Z5test1PfS_S_i
	.p2align	4, 0x90
	.type	_Z5test1PfS_S_i,@function
_Z5test1PfS_S_i:                        # @_Z5test1PfS_S_i
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
	.cfi_def_cfa_offset 96
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movq	%rdx, %r15
	movq	%rsi, %r14
	movq	%rdi, %rbx
	leaq	16(%rsp), %rsi
	movl	$1, %edi
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_13
# %bb.1:
	movq	16(%rsp), %rax
	movq	%rax, 32(%rsp)          # 8-byte Spill
	movq	24(%rsp), %rax
	movq	%rax, 8(%rsp)           # 8-byte Spill
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_2:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
	movq	$-8, %rcx
	.p2align	4, 0x90
.LBB0_3:                                #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovaps	32(%r14,%rcx,4), %ymm0
	vandps	32(%rbx,%rcx,4), %ymm0, %ymm0
	vmovaps	40(%r14,%rcx,4), %ymm1
	vandps	40(%rbx,%rcx,4), %ymm1, %ymm1
	vmovaps	%ymm0, 32(%r15,%rcx,4)
	vmovapd	48(%r14,%rcx,4), %ymm0
	vandpd	48(%rbx,%rcx,4), %ymm0, %ymm0
	vmovaps	%ymm1, 40(%r15,%rcx,4)
	vmovaps	56(%r14,%rcx,4), %ymm1
	vandps	56(%rbx,%rcx,4), %ymm1, %ymm1
	vmovapd	%ymm0, 48(%r15,%rcx,4)
	vmovaps	%ymm1, 56(%r15,%rcx,4)
	addq	$8, %rcx
	cmpq	$1016, %rcx             # imm = 0x3F8
	jb	.LBB0_3
# %bb.4:                                #   in Loop: Header=BB0_2 Depth=1
	addl	$1, %eax
	cmpl	$20000000, %eax         # imm = 0x1312D00
	jne	.LBB0_2
# %bb.5:
	leaq	16(%rsp), %rsi
	movl	$1, %edi
	vzeroupper
	callq	clock_gettime
	testl	%eax, %eax
	jne	.LBB0_13
# %bb.6:
	movq	16(%rsp), %rbp
	movq	24(%rsp), %r13
	xorl	%r14d, %r14d
	jmp	.LBB0_7
	.p2align	4, 0x90
.LBB0_10:                               #   in Loop: Header=BB0_7 Depth=1
	movq	%rbx, %rdi
	callq	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%rbx), %rax
	movq	%rbx, %rdi
	movl	$10, %esi
	callq	*48(%rax)
.LBB0_11:                               #   in Loop: Header=BB0_7 Depth=1
	movsbl	%al, %esi
	movq	%r12, %rdi
	callq	_ZNSo3putEc
	movq	%rax, %rdi
	callq	_ZNSo5flushEv
	addq	$1, %r14
	cmpq	$1024, %r14             # imm = 0x400
	je	.LBB0_12
.LBB0_7:                                # =>This Inner Loop Header: Depth=1
	movl	$_ZSt4cout, %edi
	movl	%r14d, %esi
	callq	_ZNSolsEi
	movq	%rax, %rbx
	movl	$.L.str, %esi
	movl	$1, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	vmovss	(%r15,%r14,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	movq	%rbx, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_
	movq	%rax, %r12
	movq	(%rax), %rax
	movq	-24(%rax), %rax
	movq	240(%r12,%rax), %rbx
	testq	%rbx, %rbx
	je	.LBB0_14
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=1
	cmpb	$0, 56(%rbx)
	je	.LBB0_10
# %bb.9:                                #   in Loop: Header=BB0_7 Depth=1
	movzbl	67(%rbx), %eax
	jmp	.LBB0_11
.LBB0_12:
	subq	32(%rsp), %rbp          # 8-byte Folded Reload
	vcvtsi2sd	%rbp, %xmm2, %xmm0
	subq	8(%rsp), %r13           # 8-byte Folded Reload
	vcvtsi2sd	%r13, %xmm2, %xmm1
	vmulsd	.LCPI0_0(%rip), %xmm1, %xmm1
	vaddsd	%xmm0, %xmm1, %xmm0
	vmovsd	%xmm0, 8(%rsp)          # 8-byte Spill
	movl	$_ZSt4cout, %edi
	movl	$.L.str.1, %esi
	movl	$47, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movl	$_ZSt4cout, %edi
	vmovsd	8(%rsp), %xmm0          # 8-byte Reload
                                        # xmm0 = mem[0],zero
	callq	_ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rbx
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$1024, %esi             # imm = 0x400
	callq	_ZNSolsEi
	movq	%rax, %rbx
	movl	$.L.str.3, %esi
	movl	$5, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	movq	%rbx, %rdi
	movl	$20000000, %esi         # imm = 0x1312D00
	callq	_ZNSolsEi
	movl	$.L.str.4, %esi
	movl	$2, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	addq	$40, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_14:
	.cfi_def_cfa_offset 96
	callq	_ZSt16__throw_bad_castv
.LBB0_13:
	movl	$.L.str.5, %edi
	movl	$.L.str.6, %esi
	movl	$.L__PRETTY_FUNCTION__._ZL7gettimev, %ecx
	movl	$75, %edx
	callq	__assert_fail
.Lfunc_end0:
	.size	_Z5test1PfS_S_i, .Lfunc_end0-_Z5test1PfS_S_i
	.cfi_endproc
                                        # -- End function
	.section	.text.startup,"ax",@progbits
	.p2align	4, 0x90         # -- Begin function _GLOBAL__sub_I_test1.cpp
	.type	_GLOBAL__sub_I_test1.cpp,@function
_GLOBAL__sub_I_test1.cpp:               # @_GLOBAL__sub_I_test1.cpp
	.cfi_startproc
# %bb.0:
	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$_ZStL8__ioinit, %edi
	callq	_ZNSt8ios_base4InitC1Ev
	movl	$_ZNSt8ios_base4InitD1Ev, %edi
	movl	$_ZStL8__ioinit, %esi
	movl	$__dso_handle, %edx
	popq	%rax
	.cfi_def_cfa_offset 8
	jmp	__cxa_atexit            # TAILCALL
.Lfunc_end1:
	.size	_GLOBAL__sub_I_test1.cpp, .Lfunc_end1-_GLOBAL__sub_I_test1.cpp
	.cfi_endproc
                                        # -- End function
	.type	_ZStL8__ioinit,@object  # @_ZStL8__ioinit
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	" "
	.size	.L.str, 2

	.type	.L.str.1,@object        # @.str.1
.L.str.1:
	.asciz	"Elapsed execution time of the loop in test1():\n"
	.size	.L.str.1, 48

	.type	.L.str.2,@object        # @.str.2
.L.str.2:
	.asciz	"sec (N: "
	.size	.L.str.2, 9

	.type	.L.str.3,@object        # @.str.3
.L.str.3:
	.asciz	", I: "
	.size	.L.str.3, 6

	.type	.L.str.4,@object        # @.str.4
.L.str.4:
	.asciz	")\n"
	.size	.L.str.4, 3

	.type	.L.str.5,@object        # @.str.5
.L.str.5:
	.asciz	"r == 0"
	.size	.L.str.5, 7

	.type	.L.str.6,@object        # @.str.6
.L.str.6:
	.asciz	"./fasttime.h"
	.size	.L.str.6, 13

	.type	.L__PRETTY_FUNCTION__._ZL7gettimev,@object # @__PRETTY_FUNCTION__._ZL7gettimev
.L__PRETTY_FUNCTION__._ZL7gettimev:
	.asciz	"fasttime_t gettime()"
	.size	.L__PRETTY_FUNCTION__._ZL7gettimev, 21

	.section	.init_array,"aw",@init_array
	.p2align	3
	.quad	_GLOBAL__sub_I_test1.cpp
	.ident	"clang version 10.0.0-4ubuntu1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _GLOBAL__sub_I_test1.cpp
	.addrsig_sym _ZStL8__ioinit
	.addrsig_sym __dso_handle
	.addrsig_sym _ZSt4cout
