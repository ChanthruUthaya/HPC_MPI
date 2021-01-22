 stencil: stencil.c
	mpiicc -Ofast -qopenmp -std=c99 -Wall $^ -o $@

