all: stencil

%: %.cu
	nvcc $^ -DNUM_ELEMENTS=1000 -DRADIUS=3 -o $@
