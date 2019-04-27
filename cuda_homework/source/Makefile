CUDA_SOURCES := $(shell find . -name '*.cu')
HEADERS := $(shell find . -name '*.h')
CC := nvcc
CCFLAGS := --gpu-architecture=compute_60 --gpu-code=sm_60 -rdc=true --compiler-options=-Wall -lineinfo -g

all: astar_gpu heap_cpu

pathfinding.o: pathfinding.cu pathfinding.cuh errors.h
	$(CC) $(CCFLAGS) -c $@ $<

slidingpuzzle.o: slidingpuzzle.cu slidingpuzzle.cuh errors.h
	$(CC) $(CCFLAGS) -c $@ $<

astar_gpu: astar_gpu.cu pathfinding.o slidingpuzzle.o solver.cuh hashtable.cuh heap.cuh
	$(CC) $(CCFLAGS) -o $@ $< pathfinding.o slidingpuzzle.o

heap_cpu: heap_cpu.cpp
	clang++ -o $@ $^ --std=c++11

clean:
	rm astar_gpu heap_cpu pathfinding.o
