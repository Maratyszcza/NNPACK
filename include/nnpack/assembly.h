#ifdef __ELF__
	.macro BEGIN_FUNCTION name
		.text
		.align 2
		.global \name
		.type \name, %function
		.func \name
		\name:
	.endm

	.macro END_FUNCTION name
		.endfunc
		.size \name, .-\name
	.endm
#else
	.macro BEGIN_FUNCTION name
		.text
		.align 2
		.global \name
		.func \name
		\name:
	.endm

	.macro END_FUNCTION name
		.endfunc
	.endm
#endif
