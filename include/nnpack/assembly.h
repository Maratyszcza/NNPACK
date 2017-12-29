#ifdef __ELF__
	.macro BEGIN_FUNCTION name
		.text
		.align 2
		.global \name
		.type \name, %function
		\name:
	.endm

	.macro END_FUNCTION name
		.size \name, .-\name
	.endm
#else
	.macro BEGIN_FUNCTION name
		.text
		.align 2
		.global \name
		\name:
	.endm

	.macro END_FUNCTION name
	.endm
#endif
